#!/usr/bin/env python3
"""Repair isolated holes and optional endpoint class dropouts from neighbors.

This only adds unanimous neighboring occluder pixels to background. It never
overwrites object, tool, or region pixels, and always writes a separate copy.
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
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--occluder-drop-ratio", type=float, default=0.84)
    parser.add_argument("--occluder-spike-ratio", type=float, default=1.015)
    parser.add_argument("--repair-endpoints", action="store_true")
    parser.add_argument("--endpoint-drop-ratio", type=float, default=0.96)
    args = parser.parse_args()
    prompt = json.loads(args.prompt.read_text())
    total = int(prompt["frame_count"])
    source = args.mask_root / "mask_frames"
    files = [source / f"{frame_idx:05d}.png" for frame_idx in range(total)]
    if any(not path.is_file() for path in files):
        raise ValueError("Incomplete source masks")
    areas = []
    for path in files:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or int(mask.max()) > 5:
            raise ValueError(f"Invalid mask: {path}")
        areas.append(int((mask == 1).sum()))
    targets = []
    spike_candidates = []
    for index in range(2, total - 2):
        neighbors = np.array([areas[index - 2], areas[index - 1],
                              areas[index + 1], areas[index + 2]])
        reference = float(np.median(neighbors))
        if (reference > 1000 and areas[index] < reference * args.occluder_drop_ratio
                and neighbors.min() >= 0.92 * reference):
            targets.append(index)
        if (reference > 1000 and areas[index] > reference * args.occluder_spike_ratio
                and neighbors.min() >= 0.95 * reference
                and neighbors.max() <= 1.05 * reference):
            spike_candidates.append(index)
    endpoint_targets = {0, total - 1} if args.repair_endpoints else set()
    out_dir = args.output / "mask_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    repairs = []
    for index, path in enumerate(files):
        target = out_dir / path.name
        if index not in targets and index not in spike_candidates and index not in endpoint_targets:
            shutil.copy2(path, target)
            continue
        current = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if index in targets:
            before = cv2.imread(str(files[index - 1]), cv2.IMREAD_GRAYSCALE)
            after = cv2.imread(str(files[index + 1]), cv2.IMREAD_GRAYSCALE)
            consensus = (before == 1) & (after == 1)
            recovered = consensus & (current == 0)
            current[recovered] = 1
            repairs.append({"frame": index, "class_id": 1,
                            "occluder_area_before": areas[index],
                            "neighbor_median": int(np.median([areas[index - 1], areas[index + 1]])),
                            "added_pixels": int(recovered.sum()),
                            "status": "repair_not_yet_reviewed"})
            print(f"isolated_repair f{index} pixels={int(recovered.sum())}", flush=True)
        if index in spike_candidates:
            before = cv2.imread(str(files[index - 1]), cv2.IMREAD_GRAYSCALE)
            after = cv2.imread(str(files[index + 1]), cv2.IMREAD_GRAYSCALE)
            neighbor_union = ((before == 1) | (after == 1)).astype(np.uint8)
            supported = cv2.dilate(neighbor_union, np.ones((9, 9), np.uint8)) > 0
            excess = (current == 1) & ~supported
            if int(excess.sum()) >= 1000:
                current[excess] = np.where((before[excess] == 3) & (after[excess] == 3),
                                           3, 0)
                repairs.append({"frame": index, "class_id": 1,
                                "area_before": areas[index],
                                "neighbor_median": int(np.median([areas[index - 1], areas[index + 1]])),
                                "removed_pixels": int(excess.sum()),
                                "status": "repair_not_yet_reviewed"})
                print(f"isolated_spike_trim f{index} pixels={int(excess.sum())}", flush=True)
        if index in endpoint_targets:
            neighbors = (1, 2) if index == 0 else (total - 2, total - 3)
            neighbor_masks = [cv2.imread(str(files[neighbor]), cv2.IMREAD_GRAYSCALE)
                              for neighbor in neighbors]
            for class_id in (1, 3):
                reference = float(np.median([int((mask == class_id).sum())
                                             for mask in neighbor_masks]))
                current_area = int((current == class_id).sum())
                if reference <= 1000 or current_area >= reference * args.endpoint_drop_ratio:
                    continue
                consensus = (neighbor_masks[0] == class_id) & (neighbor_masks[1] == class_id)
                recovered = consensus & (current == 0)
                current[recovered] = class_id
                repairs.append({"frame": index, "class_id": class_id,
                                "area_before": current_area, "neighbor_median": int(reference),
                                "added_pixels": int(recovered.sum()),
                                "status": "repair_not_yet_reviewed"})
                print(f"endpoint_repair f{index} class={class_id} pixels={int(recovered.sum())}",
                      flush=True)
        if not cv2.imwrite(str(target), current):
            raise RuntimeError(f"Could not save {target}")
    summary = {"source_mask_root": str(args.mask_root.resolve()),
               "prompt": str(args.prompt.resolve()), "frame_count": total,
               "repairs": repairs, "status": "provisional_not_accepted"}
    (args.output / "isolated_repair.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"repaired_masks={out_dir} frames={len(repairs)}", flush=True)


if __name__ == "__main__":
    main()
