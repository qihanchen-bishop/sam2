#!/usr/bin/env python3
"""Remove source-bright background islands mislabeled as a dark instrument.

This is a conservative, provisional correction for front-view episodes. It
does not change object, cloth, region, or source SAM2 masks in place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from semantic_point_seed_prompts import clustered_masks, find_tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompt = json.loads(args.prompt.read_text())
    if prompt["task"] not in {"o_n", "b_n", "g_n"}:
        raise ValueError(f"Unsupported front-view task {prompt['task']}")
    total = int(prompt["frame_count"])
    source_masks = args.mask_root / "mask_frames"
    if len(list(source_masks.glob("*.png"))) != total:
        raise ValueError("Incomplete input masks")
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Cannot open original video")
    target = args.output / "mask_frames"
    target.mkdir(parents=True, exist_ok=True)
    removed = {"lefttool": 0, "righttool": 0}
    recovered_righttool = 0
    removed_offaxis_components = 0
    affected: list[list[int]] = []
    for frame_idx in range(total):
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Cannot read original video frame {frame_idx}")
        mask = cv2.imread(str(source_masks / f"{frame_idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != frame.shape[:2]:
            raise ValueError(f"Invalid mask at frame {frame_idx}")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        source_dark = (value < 115).astype(np.uint8)
        distance = cv2.distanceTransform(1 - source_dark, cv2.DIST_L2, 3)
        # Only visibly bright neutral source pixels well away from any dark
        # physical shaft may be removed. Metal jaws near the shaft remain.
        upper_rows = np.arange(mask.shape[0])[:, None] < 580
        bright_table = (saturation < 38) & (value > 175)
        neutral_upper_background = upper_rows & (saturation < 75) & (value > 115)
        impossible_tool = (bright_table | neutral_upper_background) & (distance > 17)
        detected = find_tools(frame, clustered_masks(frame, prompt["task"])["object"], "extra")
        right_offaxis = np.zeros(mask.shape, dtype=bool)
        right_source_shaft = np.zeros(mask.shape, dtype=bool)
        if "righttool" in detected:
            top, bottom = np.asarray(detected["righttool"]["line"], dtype=float)
            direction = bottom - top
            length = float(np.linalg.norm(direction))
            if length > 150:
                direction /= length
                corridor = np.zeros(mask.shape, dtype=np.uint8)
                cv2.line(corridor, tuple((top - direction * 100).astype(int)),
                         tuple((bottom + direction * 45).astype(int)), 255, 58)
                rows = np.arange(mask.shape[0])[:, None]
                right_offaxis = ((corridor == 0) & (rows < bottom[1] + 40)
                                 & (saturation < 80) & (value > 90))
                narrow = np.zeros(mask.shape, dtype=np.uint8)
                cv2.line(narrow, tuple((top - direction * 100).astype(int)),
                         tuple((bottom + direction * 30).astype(int)), 255, 20)
                right_source_shaft = ((narrow > 0) & (value < 130)
                                      & (saturation < 85))
        new = mask.copy()
        counts = []
        for class_id, name in ((4, "lefttool"), (5, "righttool")):
            pixels = (new == class_id) & (impossible_tool |
                                            (right_offaxis if name == "righttool" else False))
            count = int(pixels.sum())
            removed[name] += count
            new[pixels] = 0
            counts.append(count)
        add_shaft = (new == 0) & right_source_shaft
        recovered_righttool += int(add_shaft.sum())
        new[add_shaft] = 5
        if "righttool" in detected and float(np.linalg.norm(bottom - top)) > 150:
            count, labels, stats, centers = cv2.connectedComponentsWithStats(
                (new == 5).astype(np.uint8), 8)
            slope = float((bottom[0] - top[0]) / max(bottom[1] - top[1], 1))
            for component in range(1, count):
                x, y, width, height, area = stats[component]
                cx, cy = centers[component]
                source_x = float(top[0] + slope * (cy - top[1]))
                if (area >= 80 and y + height < bottom[1] + 65
                        and abs(cx - source_x) > 42):
                    pixels = labels == component
                    removed_offaxis_components += int(pixels.sum())
                    new[pixels] = 0
        if any(counts):
            affected.append([frame_idx, *counts])
        if not cv2.imwrite(str(target / f"{frame_idx:05d}.png"), new):
            raise RuntimeError(f"Cannot write output frame {frame_idx}")
        if frame_idx % 100 == 0:
            print(f"bright_tool_filter {frame_idx + 1}/{total}", flush=True)
    cap.release()
    report = {
        "status": "provisional_not_accepted",
        "source_mask_root": str(args.mask_root.resolve()),
        "source_prompt": str(args.prompt.resolve()),
        "frame_count": total,
        "removed_pixels": removed,
        "recovered_righttool_pixels": recovered_righttool,
        "removed_offaxis_component_pixels": removed_offaxis_components,
        "affected_frame_counts": affected,
        "rule": "bright-neutral background plus neutral righttool pixels outside 29px source-detected shaft corridor above distal tip",
    }
    (args.output / "bright_tool_filter.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"provisional_bright_tool_filter={args.output} removed={removed}")


if __name__ == "__main__":
    main()
