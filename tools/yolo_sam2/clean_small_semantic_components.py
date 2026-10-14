#!/usr/bin/env python3
"""Remove tiny disconnected components from one class in semantic PNG masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=Path, action="append", required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--max-area", type=int, default=19)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changes: list[dict[str, object]] = []
    for mask_dir in args.mask_dir:
        paths = sorted(mask_dir.glob("*.png"))
        if not paths:
            raise RuntimeError(f"No PNG masks found in {mask_dir}")
        for path in paths:
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                raise RuntimeError(f"Could not read single-channel mask: {path}")
            binary = (mask == args.class_id).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            removed_areas: list[int] = []
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area <= args.max_area:
                    mask[labels == component] = 0
                    removed_areas.append(area)
            if removed_areas:
                if not cv2.imwrite(str(path), mask):
                    raise RuntimeError(f"Could not write {path}")
                changes.append(
                    {
                        "mask_dir": str(mask_dir),
                        "frame": int(path.stem),
                        "removed_component_areas": removed_areas,
                    }
                )
    report = {
        "class_id": args.class_id,
        "max_removed_area": args.max_area,
        "changed_masks": changes,
        "changed_count": len(changes),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
