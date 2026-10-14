#!/usr/bin/env python3
"""Add visible dark upper tool rods missed by SAM2, without replacing any class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from semantic_point_refine_masks import interpolated_line
from semantic_point_seed_prompts import find_tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text())
    prompt = json.loads(args.prompt.read_text())
    total = int(prompt["frame_count"])
    anchors = {name: {} for name in ("lefttool", "righttool")}
    for frame_text, record in proposal["frames"].items():
        for name in anchors:
            if name in record["classes"]:
                anchors[name][int(frame_text)] = np.asarray(record["classes"][name]["line"], float)
    if any(not values for values in anchors.values()):
        raise ValueError("Missing tool-line anchors")
    mask_dir = args.mask_root / "mask_frames"
    if len(list(mask_dir.glob("*.png"))) != total:
        raise ValueError("Incomplete source mask set")
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Could not open source video")
    out_dir = args.output / "mask_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {name: 0 for name in anchors}
    frame_counts = {name: 0 for name in anchors}
    for index in range(total):
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Video ended early at f{index}")
        source = cv2.imread(str(mask_dir / f"{index:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if source is None or source.shape != frame.shape[:2] or int(source.max()) > 5:
            raise ValueError(f"Invalid source mask f{index}")
        mask = source.copy()
        h, w = mask.shape
        empty_object = np.zeros((h, w), np.uint8)
        detected = find_tools(frame, empty_object, "extra")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)
        dark = (value < 115) & (saturation < 110)
        for class_id, name in ((4, "lefttool"), (5, "righttool")):
            expected = interpolated_line(anchors[name], index)
            line = (np.asarray(detected[name]["line"], float)
                    if name in detected else expected)
            if np.linalg.norm(line[1] - expected[1]) > 115:
                line = expected
            top, bottom = line
            direction = bottom - top
            direction /= max(np.linalg.norm(direction), 1e-6)
            if direction[1] < 0.5:
                continue
            # Hough can find the already-visible black upper rod, whereas
            # the prompt anchor begins at the colored lower shaft. Extend
            # across that entire missing interval but cap any Hough outlier.
            expected_y = float(expected[0, 1])
            start_y = float(np.clip(max(top[1], expected_y),
                                    expected_y, expected_y + 85))
            top = top + direction * ((start_y - top[1]) / direction[1])
            distance = min(360.0, max(0.0, (top[1] + 10) / direction[1]))
            upper = top - direction * distance
            lower = top + direction * 30
            line_mask = np.zeros((h, w), np.uint8)
            cv2.line(line_mask, tuple(np.rint(upper).astype(int)),
                     tuple(np.rint(lower).astype(int)), 255, 25)
            candidate = (line_mask > 0) & dark & (mask == 0)
            # A narrow line through the known tool is required. Never paint
            # cloth, object, tray or the other instrument.
            pixels = int(candidate.sum())
            if pixels:
                mask[candidate] = class_id
                totals[name] += pixels
                frame_counts[name] += 1
        if not cv2.imwrite(str(out_dir / f"{index:05d}.png"), mask):
            raise RuntimeError(f"Could not write f{index}")
        if index % 100 == 0:
            print(f"rod_complete {index + 1}/{total}", flush=True)
    cap.release()
    report = {"status": "provisional_not_accepted",
              "source_mask_root": str(args.mask_root.resolve()),
              "source_prompt": str(args.prompt.resolve()),
              "frames": total, "added_dark_rod_pixels": totals,
              "frames_changed": frame_counts}
    (args.output / "rod_completion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"rod_completion={args.output} added_pixels={totals}")


if __name__ == "__main__":
    main()
