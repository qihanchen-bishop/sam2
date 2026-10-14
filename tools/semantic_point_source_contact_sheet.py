#!/usr/bin/env python3
"""Render selected source-video frames for checking model-suggested events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=400)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Crop source coordinates before resizing")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    record = next(row for row in manifest["all_episodes"]
                  if int(row["episode_index"]) == args.episode)
    total = int(record["frame_count"])
    cap = cv2.VideoCapture(record["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {record['video_path']}")
    tiles = []
    try:
        for index in args.frames:
            if not 0 <= index < total:
                raise ValueError(f"Frame {index} outside 0..{total - 1}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            okay, frame = cap.read()
            if not okay:
                raise RuntimeError(f"Could not decode f{index}")
            if args.crop is not None:
                x1, y1, x2, y2 = args.crop
                if not (0 <= x1 < x2 <= frame.shape[1] and
                        0 <= y1 < y2 <= frame.shape[0]):
                    raise ValueError(f"Invalid crop {args.crop} for {frame.shape}")
                frame = frame[y1:y2, x1:x2]
            height = round(frame.shape[0] * args.tile_width / frame.shape[1])
            tile = cv2.resize(frame, (args.tile_width, height),
                              interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 0, 30, 0, 0, cv2.BORDER_CONSTANT,
                                      value=(245, 245, 245))
            cv2.putText(tile, f"ep{args.episode:03d} f{index}", (7, height + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 2)
            tiles.append(tile)
    finally:
        cap.release()
    blank = np.full_like(tiles[0], 245)
    while len(tiles) % args.columns:
        tiles.append(blank)
    sheet = np.vstack([np.hstack(tiles[i:i + args.columns])
                       for i in range(0, len(tiles), args.columns)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet,
                       [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"Could not save {args.output}")
    print(f"source_contact_sheet={args.output} frames={len(args.frames)}")


if __name__ == "__main__":
    main()
