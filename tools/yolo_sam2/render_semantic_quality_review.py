#!/usr/bin/env python3
"""Render final mutually-exclusive semantic masks on uniform and worst frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


CANONICAL_CLASS_COLORS = {
    "occluder": (64, 160, 255),
    "tube": (255, 92, 138),
    "object": (255, 105, 97),
    "region": (119, 221, 119),
    "tool": (255, 209, 102),
    "lefttool": (255, 209, 102),
    "righttool": (101, 214, 173),
    "leftarm": (234, 146, 199),
    "rightarm": (173, 214, 101),
}


def semantic_palette(prompt: dict) -> np.ndarray:
    """Build the RGB palette from class names, independent of label order."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for obj in prompt.get("objects", []):
        object_id = int(obj["id"])
        if not 0 < object_id < len(palette):
            continue
        palette[object_id] = CANONICAL_CLASS_COLORS.get(
            str(obj.get("name", "")).strip().lower(), (255, 255, 255)
        )
    return palette


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode", action="append", type=int, default=[])
    parser.add_argument(
        "--extra-frame",
        action="append",
        default=[],
        metavar="EPISODE:FRAME",
        help="Always include a known-risk frame; repeat as needed.",
    )
    parser.add_argument("--uniform-count", type=int, default=12)
    parser.add_argument("--worst-per-class", type=int, default=3)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=320)
    return parser.parse_args()


def selected_frames(quality_root: Path, episode: int, frame_count: int, uniform: int, worst: int) -> list[int]:
    frames = set(np.linspace(0, frame_count - 1, min(uniform, frame_count)).round().astype(int).tolist())
    for path in quality_root.glob("*.npz"):
        data = np.load(path)
        chosen = data["episode_index"] == episode
        if not np.any(chosen):
            continue
        episode_frames = data["frame_index"][chosen]
        scores = data["quality_score"][chosen]
        frames.update(episode_frames[np.argsort(scores)[:worst]].astype(int).tolist())
    return sorted(frames)


def make_sheet(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    tiles = tiles + [blank.copy() for _ in range(rows * columns - len(tiles))]
    return np.vstack([np.hstack(tiles[row * columns:(row + 1) * columns]) for row in range(rows)])


def main() -> None:
    args = parse_args()
    wanted = set(args.episode)
    extra_frames: dict[int, set[int]] = {}
    for value in args.extra_frame:
        try:
            episode_text, frame_text = value.split(":", 1)
            episode, frame = int(episode_text), int(frame_text)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid --extra-frame {value!r}; expected EPISODE:FRAME"
            ) from exc
        extra_frames.setdefault(episode, set()).add(frame)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for complete_path in sorted(args.seg_root.glob("**/complete.json")):
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        prompt = json.loads((complete_path.parent / "sam2_prompts.json").read_text(encoding="utf-8"))
        palette = semantic_palette(prompt)
        # Reviewed seed masks may retain the earlier prompt schema, where the
        # episode identity is stored as ``episode_index`` rather than
        # ``file_index``.  Both refer to the same LeRobot episode for this
        # renderer, so accept either form when assembling mixed seed/full runs.
        episode_value = prompt.get("file_index", prompt.get("episode_index"))
        if episode_value is None:
            raise RuntimeError(
                f"Prompt has no file_index or episode_index: "
                f"{complete_path.parent / 'sam2_prompts.json'}"
            )
        episode = int(episode_value)
        if wanted and episode not in wanted:
            continue
        frame_count = int(complete["frame_count"])
        indices = selected_frames(args.quality_root, episode, frame_count, args.uniform_count, args.worst_per_class)
        indices = sorted(
            set(indices)
            | {
                frame
                for frame in extra_frames.get(episode, set())
                if 0 <= frame < frame_count
            }
        )
        capture = cv2.VideoCapture(complete["video_path"])
        tiles = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read episode {episode} frame {frame_index}")
            mask = cv2.imread(str(complete_path.parent / "mask_frames" / f"{frame_index:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Missing mask episode {episode} frame {frame_index}")
            overlay = palette[mask][:, :, ::-1]
            visible = mask > 0
            blended = frame.copy()
            # OpenCV returns ``None`` for addWeighted on an empty indexed
            # array.  Empty masks are legitimate outside an object's visible
            # range, so leave those frames unchanged.
            if np.any(visible):
                blended[visible] = cv2.addWeighted(
                    frame[visible], 0.50, overlay[visible], 0.50, 0
                )
            cv2.putText(blended, f"ep{episode:03d} f{frame_index}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(blended, f"ep{episode:03d} f{frame_index}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
            height = round(blended.shape[0] * args.tile_width / blended.shape[1])
            tiles.append(cv2.resize(blended, (args.tile_width, height), interpolation=cv2.INTER_AREA))
        capture.release()
        output = args.output_root / f"episode-{episode:03d}_semantic_review.jpg"
        cv2.imwrite(str(output), make_sheet(tiles, args.columns), [cv2.IMWRITE_JPEG_QUALITY, 96])
        results.append({"episode": episode, "frames": indices, "output": str(output.resolve())})
    manifest = args.output_root / "review_manifest.json"
    manifest.write_text(json.dumps({"items": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(results), "manifest": str(manifest.resolve())}))


if __name__ == "__main__":
    main()
