#!/usr/bin/env python3
"""Read-only, frame-level search for occluder holes and endpoint dropouts.

Metrics are review candidates, not automatic corrections. Other semantic
classes never count as holes; only background pixels do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2 or int(mask.max()) > 5:
        raise ValueError(f"Invalid semantic mask: {path}")
    return mask


def enclosed_background(mask: np.ndarray) -> tuple[int, int, list[list[int]]]:
    foreground = (mask == 1).astype(np.uint8)
    contours, hierarchy = cv2.findContours(foreground, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0, 0, []
    total = 0
    largest = 0
    boxes = []
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] < 0 or cv2.contourArea(contour) < 20:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        inside = np.zeros((h, w), np.uint8)
        cv2.drawContours(inside, [contour - [x, y]], 0, 1, -1)
        pixels = int(((inside > 0) & (mask[y:y+h, x:x+w] == 0)).sum())
        if pixels < 20:
            continue
        total += pixels
        largest = max(largest, pixels)
        boxes.append([x, y, x + w, y + h, pixels])
    boxes.sort(key=lambda row: row[4], reverse=True)
    return total, largest, boxes[:4]


def temporal_missing(before: np.ndarray, now: np.ndarray,
                     after: np.ndarray) -> tuple[int, int, list[int] | None]:
    consensus = ((before == 1) & (after == 1)).astype(np.uint8)
    consensus = cv2.erode(consensus, np.ones((5, 5), np.uint8))
    missing = (consensus > 0) & (now == 0)
    count, _, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), 8)
    if count <= 1:
        return 0, 0, None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = map(int, stats[largest])
    return int(missing.sum()), area, [x, y, x + w, y + h]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-temporal", type=int, default=80)
    parser.add_argument("--min-enclosed", type=int, default=150)
    args = parser.parse_args()
    complete = json.loads((args.mask_root / "complete.json").read_text())
    count = int(complete["frame_count"])
    files = [args.mask_root / "mask_frames" / f"{i:05d}.png" for i in range(count)]
    if any(not path.is_file() for path in files):
        raise ValueError("The propagation is missing mask frames")
    rows = []
    previous = None
    current = load(files[0])
    following = load(files[1]) if count > 1 else None
    for frame in range(count):
        area = int((current == 1).sum())
        hole_total, hole_largest, hole_boxes = enclosed_background(current)
        gap_total, gap_largest, gap_box = (temporal_missing(previous, current, following)
                                           if previous is not None and following is not None
                                           else (0, 0, None))
        rows.append({"frame": frame, "occluder_area": area,
                     "enclosed_background_pixels": hole_total,
                     "largest_enclosed_hole": hole_largest,
                     "hole_boxes_xyxy_area": hole_boxes,
                     "temporal_missing_pixels": gap_total,
                     "largest_temporal_patch": gap_largest,
                     "temporal_patch_box_xyxy": gap_box})
        previous, current = current, following
        following = load(files[frame + 2]) if frame + 2 < count else None
        if (frame + 1) % 250 == 0:
            print(f"scanned={frame+1}/{count}", flush=True)
    areas = np.array([row["occluder_area"] for row in rows], dtype=np.float64)
    for i, row in enumerate(rows):
        neighborhood = np.concatenate((areas[max(0, i-5):i], areas[i+1:min(count, i+6)]))
        reference = float(np.median(neighborhood)) if len(neighborhood) else float(areas[i])
        row["neighbor_median_area"] = round(reference, 1)
        row["area_deficit_pixels"] = int(max(0, reference - areas[i]))
        row["area_deficit_pct"] = round(100 * max(0, reference - areas[i]) /
                                         max(reference, 1), 2)
    candidates = [row for row in rows if row["largest_temporal_patch"] >= args.min_temporal
                  or row["largest_enclosed_hole"] >= args.min_enclosed]
    candidates.sort(key=lambda row: (row["largest_temporal_patch"],
                                     row["largest_enclosed_hole"]), reverse=True)
    report = {"status": "candidates_require_visual_review",
              "source_mask_root": str(args.mask_root.resolve()),
              "source_video": complete["video_path"],
              "frame_count": count,
              "thresholds": {"min_temporal": args.min_temporal,
                             "min_enclosed": args.min_enclosed},
              "endpoint_metrics": [rows[0], rows[-1]],
              "candidate_count": len(candidates),
              "candidates_ranked": candidates,
              "all_frame_metrics": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"report={args.output} candidates={len(candidates)}")
    for row in candidates[:30]:
        print(f"f{row['frame']} temporal={row['largest_temporal_patch']} "
              f"enclosed={row['largest_enclosed_hole']} "
              f"area_deficit={row['area_deficit_pct']}%")


if __name__ == "__main__":
    main()
