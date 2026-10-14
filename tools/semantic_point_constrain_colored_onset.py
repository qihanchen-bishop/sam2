#!/usr/bin/env python3
"""Trim early colored-target masks to source-supported pixels without changing raw SAM2."""

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
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--min-component", type=int, default=5,
                        help="Minimum source-color component area; use 1 only for a visually confirmed first-pixel sliver")
    parser.add_argument("--mode", choices=("intersect", "replace"), default="intersect",
                        help="Replace is for a source-confirmed onset where SAM2 is spatially disjoint from the first color pixels")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Source-reviewed target ROI; required when SAM2 has no object pixels")
    parser.add_argument("--hsv-low", type=int, nargs=3, metavar=("H", "S", "V"),
                        help="Source-reviewed lower HSV threshold for a dim onset")
    parser.add_argument("--hsv-high", type=int, nargs=3, metavar=("H", "S", "V"),
                        help="Upper HSV threshold; must be paired with --hsv-low")
    args = parser.parse_args()
    if args.min_component < 1:
        raise ValueError("Minimum component area must be positive")
    if (args.hsv_low is None) != (args.hsv_high is None):
        raise ValueError("--hsv-low and --hsv-high must be paired")
    prompt = json.loads(args.prompt.read_text())
    task = prompt.get("task", "")
    if task not in ("o_n", "g_n"):
        raise ValueError("Only orange or green targets have a reliable source-color onset gate")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    visible = next(obj["visible_ranges"][0][0] for obj in prompt["objects"]
                   if obj["id"] == 2)
    start = int(visible)
    if args.end_frame < start or args.end_frame >= prompt["frame_count"]:
        raise ValueError("End frame must fall within the visible source video")
    source_dir = args.mask_root / "mask_frames"
    paths = sorted(source_dir.glob("*.png"))
    if len(paths) != prompt["frame_count"]:
        raise ValueError("Input masks are not frame-aligned")
    output_dir = args.output / "mask_frames"
    output_dir.mkdir(parents=True)
    for source in paths:
        os.link(source, output_dir / source.name)
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise ValueError("Could not decode prompt source video")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * args.dilation + 1, 2 * args.dilation + 1))
    rows = []
    try:
        for index in range(start, args.end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Could not decode frame {index}")
            source = source_dir / f"{index:05d}.png"
            mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != frame.shape[:2]:
                raise ValueError(f"Mask/source mismatch at frame {index}")
            yy, xx = np.where(mask == 2)
            if args.roi is not None:
                x0, y0, x1, y1 = args.roi
                if not (0 <= x0 < x1 <= mask.shape[1] and
                        0 <= y0 < y1 <= mask.shape[0]):
                    raise ValueError(f"Invalid target ROI {args.roi}")
                if len(xx) and (xx.min() < x0 or xx.max() >= x1 or
                                yy.min() < y0 or yy.max() >= y1):
                    raise ValueError(f"Existing object extends outside reviewed ROI at f{index}")
            else:
                if not len(xx):
                    raise ValueError(f"Visible object missing at frame {index}; provide a reviewed ROI")
                pad = 8
                x0, x1 = max(0, int(xx.min()) - pad), min(mask.shape[1], int(xx.max()) + pad + 1)
                y0, y1 = max(0, int(yy.min()) - pad), min(mask.shape[0], int(yy.max()) + pad + 1)
            hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            low, high = ((3, 130, 165), (29, 255, 255)) if task == "o_n" else (
                (32, 65, 50), (98, 255, 255))
            if args.hsv_low is not None:
                low, high = tuple(args.hsv_low), tuple(args.hsv_high)
            color = cv2.inRange(hsv, low, high)
            count, labels, stats, _ = cv2.connectedComponentsWithStats((color > 0).astype(np.uint8), 8)
            candidates = [i for i in range(1, count)
                          if stats[i, cv2.CC_STAT_AREA] >= args.min_component]
            if not candidates:
                raise ValueError(f"No source-color target component at frame {index}")
            winner = max(candidates, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
            source_color = (labels == winner).astype(np.uint8)
            allowed = cv2.dilate(source_color, kernel) > 0
            crop = mask[y0:y1, x0:x1]
            before = crop == 2
            keep = before & allowed if args.mode == "intersect" else allowed
            if not keep.any():
                raise ValueError(f"Color gate removed the complete object at frame {index}")
            removed = before & ~keep
            added = keep & ~before
            near_cloth = cv2.dilate((crop == 1).astype(np.uint8),
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
            changed = mask.copy()
            edit = changed[y0:y1, x0:x1]
            edit[removed] = np.where(near_cloth[removed], 1, 0)
            if args.mode == "replace":
                edit[added] = 2
            target = output_dir / source.name
            temporary = target.with_suffix(".tmp.png")
            if not cv2.imwrite(str(temporary), changed):
                raise IOError(f"Could not write {temporary}")
            os.replace(temporary, target)
            rows.append({"frame": index, "before": int(before.sum()),
                         "source_color": int(source_color.sum()),
                         "after": int(keep.sum()), "removed": int(removed.sum()),
                         "added": int(added.sum())})
    finally:
        cap.release()
    report = {"prompt": str(args.prompt.resolve()), "source_masks": str(args.mask_root.resolve()),
              "output_masks": str(args.output.resolve()), "task": task,
              "dilation": args.dilation, "min_component": args.min_component,
              "roi": args.roi, "hsv_low": args.hsv_low, "hsv_high": args.hsv_high,
              "mode": args.mode,
              "rows": rows,
              "note": "Only early object pixels were trimmed; removed pixels touching the cloth receive occluder, and other removed pixels receive background. Other frames are hard-linked unchanged to the reviewed refinement root."}
    (args.output / "onset_color_constraint.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"onset_color_constraint={args.output} frames={len(rows)} "
          f"removed={sum(row['removed'] for row in rows)}")


if __name__ == "__main__":
    main()
