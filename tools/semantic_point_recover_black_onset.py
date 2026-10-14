#!/usr/bin/env python3
"""Recover a source-reviewed, tiny first-exposure black target in a fixed ROI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--roi", type=int, nargs=4, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--value-threshold", type=int, default=110)
    parser.add_argument("--min-area", type=int, default=8)
    args = parser.parse_args()
    prompt = json.loads(args.prompt.read_text())
    if prompt["task"] != "b_n":
        raise ValueError("Only black-target episodes are eligible")
    total = int(prompt["frame_count"])
    if not 0 <= args.start_frame <= args.end_frame < total:
        raise ValueError("Invalid frame range")
    x1, y1, x2, y2 = args.roi
    source = args.mask_root / "mask_frames"
    if len(list(source.glob("*.png"))) != total:
        raise ValueError("Incomplete source masks")
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Cannot open source video")
    target = args.output / "mask_frames"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.png")):
        os.link(path, target / path.name)
    frames = []
    for frame_idx in range(args.start_frame, args.end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Cannot read frame {frame_idx}")
        mask = cv2.imread(str(source / f"{frame_idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != frame.shape[:2]:
            raise ValueError(f"Invalid mask f{frame_idx}")
        new = mask.copy()
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        dark = (hsv[:, :, 2] < args.value_threshold).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        choices = [(index, int(stats[index, cv2.CC_STAT_AREA]))
                   for index in range(1, count)
                   if args.min_area <= int(stats[index, cv2.CC_STAT_AREA]) <= 1000]
        if not choices:
            raise RuntimeError(f"No reviewed dark component at f{frame_idx}")
        winner = max(choices, key=lambda item: item[1])[0]
        component = components == winner
        crop = new[y1:y2, x1:x2]
        crop[component] = 2
        target_path = target / f"{frame_idx:05d}.png"
        temporary = target_path.with_suffix(".tmp.png")
        if not cv2.imwrite(str(temporary), new):
            raise RuntimeError(f"Cannot write frame {frame_idx}")
        os.replace(temporary, target_path)
        frames.append([frame_idx, int(component.sum())])
    cap.release()
    report = {"status": "provisional_not_accepted", "frame_count": total,
              "source_mask_root": str(args.mask_root.resolve()),
              "source_prompt": str(args.prompt.resolve()), "roi": args.roi,
              "source_value_threshold": args.value_threshold,
              "source_min_component_area": args.min_area,
              "recovered_frames": frames,
              "note": "Only selected frames changed; all other masks are hard-linked to source."}
    (args.output / "black_onset_recovery.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"provisional_black_onset_recovery={args.output} frames={frames}")


if __name__ == "__main__":
    main()
