#!/usr/bin/env python3
"""Propose recovering a dark block after it reaches the colored target region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from semantic_point_seed_prompts import clustered_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    args = parser.parse_args()
    prompt = json.loads(args.prompt.read_text())
    if prompt["task"] != "b_n":
        raise ValueError("Only black target episodes are eligible")
    total = int(prompt["frame_count"])
    if not 0 <= args.start_frame < total:
        raise ValueError("Invalid start frame")
    source = args.mask_root / "mask_frames"
    if len(list(source.glob("*.png"))) != total:
        raise ValueError("Incomplete input masks")
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Cannot open original video")
    target = args.output / "mask_frames"
    target.mkdir(parents=True, exist_ok=True)
    report = {"status": "provisional_not_accepted", "frame_count": total,
              "start_frame": args.start_frame,
              "source_mask_root": str(args.mask_root.resolve()),
              "source_prompt": str(args.prompt.resolve()),
              "recovered_frames": [], "skipped_frames": []}
    for frame_idx in range(total):
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Cannot read original video frame {frame_idx}")
        old = cv2.imread(str(source / f"{frame_idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if old is None or old.shape != frame.shape[:2]:
            raise ValueError(f"Invalid mask at frame {frame_idx}")
        new = old.copy()
        if frame_idx >= args.start_frame:
            colors = clustered_masks(frame, "b_n")
            core = colors["object"] > 0
            overlap = int((core & (old == 2)).sum())
            area = int(core.sum())
            # A source-dark component is accepted only when it touches the
            # propagated object. Dark grippers and table marks are excluded.
            if 250 <= area <= 13000 and overlap >= 50:
                guard = cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                candidate = core | ((old == 2) & guard)
                candidate = cv2.morphologyEx(candidate.astype(np.uint8),
                                             cv2.MORPH_CLOSE,
                                             np.ones((5, 5), np.uint8)) > 0
                old_object = new == 2
                new[old_object] = np.where(colors["region"][old_object] > 0, 3, 0)
                new[candidate] = 2
                report["recovered_frames"].append(
                    [frame_idx, area, overlap, int(old_object.sum()), int(candidate.sum())])
            else:
                report["skipped_frames"].append([frame_idx, area, overlap])
        if not cv2.imwrite(str(target / f"{frame_idx:05d}.png"), new):
            raise RuntimeError(f"Cannot write frame {frame_idx}")
        if frame_idx % 100 == 0:
            print(f"black_region_recovery {frame_idx + 1}/{total}", flush=True)
    cap.release()
    (args.output / "black_region_recovery.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"provisional_black_region_recovery={args.output} "
          f"recovered={len(report['recovered_frames'])} "
          f"skipped={len(report['skipped_frames'])}")


if __name__ == "__main__":
    main()
