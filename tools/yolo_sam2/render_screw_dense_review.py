#!/usr/bin/env python3
"""Render dense side-view screw RGB/mask review sheets for temporal auditing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


OBJECT_ID = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--seg-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", default="60-89")
    parser.add_argument("--start", type=int, default=40)
    parser.add_argument("--end", type=int, default=320)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--crop", default="180,135,520,360")
    parser.add_argument("--tile-width", type=int, default=272)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--frames-per-page", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_episodes(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            result.update(range(lo, hi + 1))
        elif part:
            result.add(int(part))
    return result


def main() -> None:
    args = parse_args()
    episodes = parse_episodes(args.episodes)
    crop = tuple(map(int, args.crop.split(",")))
    if len(crop) != 4:
        raise ValueError("--crop must be x1,y1,x2,y2")
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for prompt_path in sorted(args.prompt_root.resolve().rglob("sam2_prompts.json")):
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        episode = int(prompt["file_index"])
        if episode not in episodes or "screw" not in prompt.get("tasks", []):
            continue
        frame_count = int(prompt["frame_count"])
        mask_root = (
            args.seg_root.resolve()
            / prompt["video_key"]
            / f"chunk-{int(prompt['chunk_index']):03d}"
            / f"file-{episode:03d}"
            / f"file-{episode:03d}_sam2_masks"
            / "mask_frames"
        )
        frames = list(range(max(0, args.start), min(frame_count - 1, args.end) + 1, args.stride))
        capture = cv2.VideoCapture(prompt["video_path"])
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open {prompt['video_path']}")
        episode_records = []
        for page_index in range(math.ceil(len(frames) / args.frames_per_page)):
            page_frames = frames[page_index * args.frames_per_page : (page_index + 1) * args.frames_per_page]
            tiles = []
            for frame_index in page_frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed frame {frame_index}: {prompt['video_path']}")
                mask = cv2.imread(str(mask_root / f"{frame_index:05d}.png"), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Missing mask frame {frame_index}: {mask_root}")
                x1, y1, x2, y2 = crop
                rgb = frame[y1:y2, x1:x2].copy()
                object_mask = (mask[y1:y2, x1:x2] == OBJECT_ID)
                overlay = rgb.copy()
                overlay[object_mask] = (
                    0.30 * overlay[object_mask] + 0.70 * np.array([180, 40, 255])
                ).astype(np.uint8)
                contours, _ = cv2.findContours(object_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
                scale = args.tile_width / rgb.shape[1]
                size = (args.tile_width, round(rgb.shape[0] * scale))
                rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
                area = int(object_mask.sum())
                tile = np.vstack([rgb, overlay])
                cv2.rectangle(tile, (0, 0), (tile.shape[1], 23), (0, 0, 0), -1)
                cv2.putText(tile, f"f{frame_index:04d} obj={area}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            blank = np.zeros_like(tiles[0])
            while len(tiles) % args.columns:
                tiles.append(blank.copy())
            rows = [np.hstack(tiles[i:i + args.columns]) for i in range(0, len(tiles), args.columns)]
            sheet = np.vstack(rows)
            header = np.zeros((42, sheet.shape[1], 3), dtype=np.uint8)
            cv2.putText(header, f"episode {episode:03d} screw dense review page {page_index + 1}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 255), 1, cv2.LINE_AA)
            output = output_root / f"episode-{episode:03d}_page-{page_index + 1:02d}.jpg"
            cv2.imwrite(str(output), np.vstack([header, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 95])
            episode_records.append({"page": page_index + 1, "frames": page_frames, "path": str(output)})
        capture.release()
        records.append({"episode": episode, "pages": episode_records})
        print(f"episode {episode:03d}: {len(episode_records)} pages", flush=True)

    (output_root / "review_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done: {len(records)} screw episodes -> {output_root}")


if __name__ == "__main__":
    main()
