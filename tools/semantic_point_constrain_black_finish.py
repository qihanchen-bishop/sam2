#!/usr/bin/env python3
"""Trim selected black-object finish frames to the source dark component.

The source frames are first inspected; this is a conservative, reviewable repair
for a SAM2 mask that spills from the black block onto the brown tray.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from semantic_point_seed_prompts import clustered_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--dilation", type=int, default=2)
    args = parser.parse_args()
    prompt = json.loads(args.prompt.read_text())
    if prompt["task"] != "b_n":
        raise ValueError("Only black-object episodes are eligible")
    total = int(prompt["frame_count"])
    if any(not 0 <= index < total for index in args.frames):
        raise ValueError("Repair frame outside source video")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    source = args.mask_root / "mask_frames"
    paths = sorted(source.glob("*.png"))
    if len(paths) != total:
        raise ValueError("Source masks are not complete")
    target = args.output / "mask_frames"
    target.mkdir(parents=True)
    for path in paths:
        os.link(path, target / path.name)
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise ValueError("Could not decode original video")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * args.dilation + 1, 2 * args.dilation + 1))
    rows = []
    try:
        for index in sorted(set(args.frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            okay, frame = cap.read()
            if not okay:
                raise RuntimeError(f"Cannot decode source f{index}")
            mask = cv2.imread(str(source / f"{index:05d}.png"),
                              cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != frame.shape[:2]:
                raise ValueError(f"Invalid mask f{index}")
            colors = clustered_masks(frame, "b_n")
            core = colors["object"] > 0
            old = mask == 2
            core_area = int(core.sum())
            overlap = int((core & old).sum())
            if not 250 <= core_area <= 13000 or overlap < core_area * .75:
                raise ValueError(f"Unreliable source dark component f{index}: "
                                 f"area={core_area} overlap={overlap}")
            allowed = cv2.dilate(core.astype(np.uint8), kernel) > 0
            keep = old & allowed
            if keep.sum() < core_area * .75:
                raise ValueError(f"Repair would remove black core at f{index}")
            removed = old & ~keep
            repaired = mask.copy()
            repaired[removed] = np.where(colors["region"][removed] > 0, 3, 0)
            path = target / f"{index:05d}.png"
            temporary = path.with_suffix(".tmp.png")
            if not cv2.imwrite(str(temporary), repaired):
                raise IOError(f"Could not write {temporary}")
            os.replace(temporary, path)
            rows.append({"frame": index, "source_core_pixels": core_area,
                         "before": int(old.sum()), "after": int(keep.sum()),
                         "removed": int(removed.sum())})
    finally:
        cap.release()
    report = {"status": "provisional_requires_visual_review",
              "source_mask_root": str(args.mask_root.resolve()),
              "source_prompt": str(args.prompt.resolve()),
              "dilation": args.dilation, "rows": rows,
              "note": "Only selected frames changed; all other masks are hard-linked to source."}
    (args.output / "black_finish_constraint.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"black_finish_constraint={args.output} frames={len(rows)}")


if __name__ == "__main__":
    main()
