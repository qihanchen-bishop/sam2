#!/usr/bin/env python3
"""Make a uniform, whole-video visual QA sheet for semantic bootstrap masks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


PALETTE_BGR = np.array([
    [0, 0, 0], [170, 65, 30], [0, 140, 255], [55, 180, 50],
    [240, 80, 50], [235, 55, 225],
], dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--tile-width", type=int, default=300)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    prompt = json.loads(args.prompt.read_text())
    total = int(prompt["frame_count"])
    frames = sorted(set(range(0, total, args.stride)) | {total - 1})
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Could not open source video")
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tile_height = round(original_height * args.tile_width / original_width)
    label_height = 26
    rows = math.ceil(len(frames) / args.columns)
    sheet = np.full((rows * (tile_height + label_height),
                     args.columns * args.tile_width, 3), 248, np.uint8)
    mask_dir = args.mask_root / "mask_frames"
    for index, frame_idx in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        okay, frame = cap.read()
        mask = cv2.imread(str(mask_dir / f"{frame_idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if not okay or mask is None or frame.shape[:2] != mask.shape or int(mask.max()) > 5:
            raise ValueError(f"Missing or invalid source/mask f{frame_idx}")
        colored = PALETTE_BGR[mask]
        overlay = frame.copy()
        foreground = mask > 0
        overlay[foreground] = cv2.addWeighted(
            frame[foreground], 0.45, colored[foreground], 0.55, 0)
        tile = cv2.resize(overlay, (args.tile_width, tile_height),
                          interpolation=cv2.INTER_AREA)
        row, col = divmod(index, args.columns)
        y, x = row * (tile_height + label_height), col * args.tile_width
        sheet[y:y + tile_height, x:x + args.tile_width] = tile
        cv2.putText(sheet, f"ep{prompt['episode_index']} f{frame_idx}",
                    (x + 6, y + tile_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1)
    cap.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError("Could not save timeline")
    print(f"timeline={args.output} frames={len(frames)}", flush=True)


if __name__ == "__main__":
    main()
