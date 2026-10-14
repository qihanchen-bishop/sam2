#!/usr/bin/env python3
"""Constrain a reviewed one-frame terminal object spill to the preceding silhouette.

This is a narrow post-propagation repair, not a general object recovery: it only
removes new object pixels far from the immediately preceding accepted silhouette.
Removed pixels recover their previous-frame non-object label. The source root is
never modified and the area jump must exceed the configured trigger.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--object-id", type=int, default=2)
    parser.add_argument("--dilation", type=int, default=7,
                        help="Maximum one-frame silhouette motion in pixels")
    parser.add_argument("--minimum-area-ratio", type=float, default=1.4)
    parser.add_argument("--exact-previous", action="store_true",
                        help="Use only when adjacent source frames show a stationary object")
    args = parser.parse_args()
    if args.frame < 1 or args.output.exists():
        raise ValueError("Frame must be positive and output must not exist")
    source = args.mask_root / "mask_frames"
    previous = cv2.imread(str(source / f"{args.frame - 1:05d}.png"),
                          cv2.IMREAD_GRAYSCALE)
    current = cv2.imread(str(source / f"{args.frame:05d}.png"),
                         cv2.IMREAD_GRAYSCALE)
    if previous is None or current is None or previous.shape != current.shape:
        raise ValueError("Missing or incompatible adjacent mask frames")
    old_object = previous == args.object_id
    new_object = current == args.object_id
    old_area = int(old_object.sum())
    new_area = int(new_object.sum())
    if not old_area or new_area / old_area < args.minimum_area_ratio:
        raise ValueError(f"No qualifying area jump: {old_area} -> {new_area}")
    size = 2 * args.dilation + 1
    near_old = cv2.dilate(old_object.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (size, size))).astype(bool)
    removed = new_object & ~near_old
    if not removed.any() or not (new_object & near_old).any():
        raise ValueError("Terminal constraint would remove none or all object pixels")
    repaired = current.copy()
    repaired[removed] = previous[removed]
    if args.exact_previous:
        revised = new_object & ~old_object
        repaired[revised] = previous[revised]
        repaired[old_object] = args.object_id
    if np.any(repaired[removed] == args.object_id):
        raise AssertionError("Removed pixels must recover a non-object label")
    target = args.output / "mask_frames"
    target.mkdir(parents=True)
    for item in sorted(source.glob("*.png")):
        if item.name != f"{args.frame:05d}.png":
            shutil.copy2(item, target / item.name)
    if not cv2.imwrite(str(target / f"{args.frame:05d}.png"), repaired):
        raise RuntimeError("Could not write repaired terminal mask")
    report = {
        "source_root": str(args.mask_root.resolve()),
        "frame": args.frame,
        "reference_frame": args.frame - 1,
        "object_id": args.object_id,
        "dilation": args.dilation,
        "exact_previous": args.exact_previous,
        "old_area": old_area,
        "new_area": new_area,
        "repaired_area": int((repaired == args.object_id).sum()),
        "removed_pixels": int(removed.sum()),
        "status": "provisional_needs_visual_review",
    }
    (args.output / "terminal_object_constraint.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"terminal_constraint={args.output} {old_area}->{new_area}->"
          f"{report['repaired_area']} removed={report['removed_pixels']}")


if __name__ == "__main__":
    main()
