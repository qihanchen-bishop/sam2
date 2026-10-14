#!/usr/bin/env python3
"""Build an episode-split U-Net cache from explicitly accepted front masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from semantic_point_seed_prompts import CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--cache-width", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    acceptance = json.loads(args.acceptance.read_text())
    episodes = acceptance["episodes"]
    accepted = {int(key): value for key, value in episodes.items()
                if value["status"] == "accepted"}
    tasks = {record["task"] for record in episodes.values()}
    if any(sum(record["task"] == task for record in accepted.values()) < 2
           for task in tasks):
        raise ValueError("Training requires at least two accepted episodes per task")
    if not set(args.val_episodes) <= set(accepted):
        raise ValueError("Validation episodes must be accepted")
    if {accepted[episode]["task"] for episode in args.val_episodes} != tasks:
        raise ValueError("Validation episodes must cover every task")
    output = {"labels": ["background", *CLASSES],
              "splits": {"train": 0, "val": 0},
              "episodes": {"train": [], "val": []},
              "source_acceptance": str(args.acceptance.resolve()),
              "images": "images/{split}/observation.images.front/file-{episode:03d}/{frame:05d}.jpg",
              "masks": "masks/{split}/observation.images.front/file-{episode:03d}/{frame:05d}.png"}
    for episode, record in sorted(accepted.items()):
        split = "val" if episode in args.val_episodes else "train"
        prompt = json.loads(Path(record["prompt"]).read_text())
        source = Path(prompt["video_path"])
        mask_root = Path(record["mask_root"])
        total = int(prompt["frame_count"])
        mask_dir = mask_root / "mask_frames"
        if len(list(mask_dir.glob("*.png"))) != total:
            raise ValueError(f"Incomplete accepted masks for ep{episode}")
        selected = set(range(0, total, args.frame_stride)) | {total - 1}
        for anchor in prompt["prompt_frames"]:
            selected.update(range(max(0, anchor - 15),
                                  min(total, anchor + 16), 2))
            selected.add(anchor)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {source}")
        written = 0
        for frame_idx in range(total):
            okay, frame = cap.read()
            if not okay:
                raise RuntimeError(f"Source video ended early ep{episode} f{frame_idx}")
            if frame_idx not in selected:
                continue
            mask = cv2.imread(str(mask_dir / f"{frame_idx:05d}.png"),
                              cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != frame.shape[:2] or int(mask.max()) > len(CLASSES):
                raise ValueError(f"Invalid accepted mask ep{episode} f{frame_idx}")
            height = round(frame.shape[0] * args.cache_width / frame.shape[1])
            dimensions = (args.cache_width, height)
            image_small = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(mask, dimensions, interpolation=cv2.INTER_NEAREST)
            subpath = Path("observation.images.front") / f"file-{episode:03d}"
            image_path = args.output / "images" / split / subpath / f"{frame_idx:05d}.jpg"
            mask_path = args.output / "masks" / split / subpath / f"{frame_idx:05d}.png"
            if args.overwrite or not image_path.is_file() or not mask_path.is_file():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), image_small,
                                   [cv2.IMWRITE_JPEG_QUALITY, 93]):
                    raise RuntimeError(f"Could not save {image_path}")
                if not cv2.imwrite(str(mask_path), mask_small):
                    raise RuntimeError(f"Could not save {mask_path}")
            written += 1
        cap.release()
        output["splits"][split] += written
        output["episodes"][split].append(episode)
        print(f"cache ep={episode} split={split} samples={written}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"training_cache={args.output} samples={output['splits']}")


if __name__ == "__main__":
    main()
