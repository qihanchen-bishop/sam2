#!/usr/bin/env python3
"""Render selected video frames with a pixel coordinate grid for prompt review."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wanted = [int(value) for value in args.frames.split(",")]
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    tiles: list[Image.Image] = []
    for frame_index in wanted:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {frame_index}")
        tile = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(tile)
        for x in range(0, tile.width, 20):
            draw.line((x, 0, x, tile.height), fill=(255, 0, 255), width=1)
            if x % 40 == 0:
                draw.text((x + 2, 2), str(x), fill=(255, 255, 0))
        for y in range(0, tile.height, 20):
            draw.line((0, y, tile.width, y), fill=(255, 0, 255), width=1)
            if y % 40 == 0:
                draw.text((2, y + 2), str(y), fill=(255, 255, 0))
        draw.rectangle((0, tile.height - 24, 90, tile.height), fill=(0, 0, 0))
        draw.text((5, tile.height - 21), f"frame {frame_index}", fill=(255, 255, 255))
        tiles.append(tile)
    capture.release()
    canvas = Image.new("RGB", (640 * 2, 360 * ((len(tiles) + 1) // 2)), (0, 0, 0))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 2) * 640, (index // 2) * 360))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
