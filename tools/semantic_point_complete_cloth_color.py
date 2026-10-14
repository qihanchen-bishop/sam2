#!/usr/bin/env python3
"""Propose filling a source-color-supported lifted cloth flap missed by SAM2.

Only background pixels in the source's largest connected skin-colored cloth
component are changed. Object, tools, tray, and all existing labels are kept.
The output remains provisional until contact and whole-video visual review.
"""

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
    args = parser.parse_args()
    prompt = json.loads(args.prompt.read_text())
    total = int(prompt["frame_count"])
    source = args.mask_root / "mask_frames"
    if len(list(source.glob("*.png"))) != total:
        raise ValueError("Incomplete source mask sequence")
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Could not open source video")
    target = args.output / "mask_frames"
    target.mkdir(parents=True, exist_ok=True)
    report = {"status": "provisional_not_accepted", "episode_index": prompt["episode_index"],
              "source_mask_root": str(args.mask_root.resolve()),
              "source_prompt": str(args.prompt.resolve()), "frame_count": total,
              "frames_changed": 0, "added_occluder_pixels": 0,
              "large_addition_frames": [], "frame_summary": []}
    for frame_idx in range(total):
        okay, frame = cap.read()
        old = cv2.imread(str(source / f"{frame_idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if not okay or old is None or old.shape != frame.shape[:2]:
            raise ValueError(f"Bad video/mask pair at f{frame_idx}")
        new = old.copy()
        colors = clustered_masks(frame, prompt["task"])
        cloth = colors["cloth"] > 0
        # The tray begins at y~650. Keep its class and exclude its source
        # color even if it touches the cloth component in a particular frame.
        cloth[650:] = False
        cloth &= colors["region"] == 0
        added = cloth & (new == 0)
        n_added = int(added.sum())
        new[added] = 1
        report["added_occluder_pixels"] += n_added
        report["frames_changed"] += int(n_added > 0)
        if n_added > 5000:
            report["large_addition_frames"].append(frame_idx)
        report["frame_summary"].append({"frame": frame_idx,
                                        "source_cloth_pixels": int(cloth.sum()),
                                        "added_pixels": n_added})
        if not cv2.imwrite(str(target / f"{frame_idx:05d}.png"), new):
            raise RuntimeError(f"Could not write f{frame_idx}")
        if frame_idx % 100 == 0:
            print(f"cloth_completion {frame_idx + 1}/{total}", flush=True)
    cap.release()
    (args.output / "cloth_completion.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"provisional_cloth_completion={args.output} "
          f"added_pixels={report['added_occluder_pixels']} "
          f"large_addition_frames={len(report['large_addition_frames'])}", flush=True)


if __name__ == "__main__":
    main()
