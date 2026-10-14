#!/usr/bin/env python3
"""Summarize model-derived prompt previews for human/agent visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=400)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    proposal = json.loads((args.proposal_dir / "proposals.json").read_text())
    tiles = []
    for frame_text, record in sorted(proposal["frames"].items(),
                                     key=lambda item: int(item[0])):
        index = int(frame_text)
        event = record["event"]
        path = args.proposal_dir / f"{event}_f{index:05d}_prompts.jpg"
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)
        height = round(frame.shape[0] * args.tile_width / frame.shape[1])
        tile = cv2.resize(frame, (args.tile_width, height), interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(tile, 0, 32, 0, 0, cv2.BORDER_CONSTANT,
                                  value=(245, 245, 245))
        concerns = len(proposal.get("concerns", {}).get(frame_text, []))
        cv2.putText(tile, f"f{index} {event} flags={concerns}", (6, height + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        tiles.append(tile)
    if not tiles:
        raise ValueError("No prompt previews")
    blank = np.full_like(tiles[0], 245)
    while len(tiles) % args.columns:
        tiles.append(blank)
    sheet = np.vstack([np.hstack(tiles[i:i + args.columns])
                       for i in range(0, len(tiles), args.columns)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet,
                       [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"Could not save {args.output}")
    print(f"contact_sheet={args.output} frames={len(proposal['frames'])}")


if __name__ == "__main__":
    main()
