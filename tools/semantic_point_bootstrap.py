#!/usr/bin/env python3
"""Inventory LeRobot front videos and render independent event-selection timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def inventory(dataset: Path, output: Path, per_task: int) -> None:
    task_rows = pq.read_table(dataset / "meta/tasks.parquet").to_pylist()
    task_names = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    by_task: dict[str, list[dict]] = {name: [] for name in task_names.values()}
    episode_files = sorted((dataset / "meta/episodes").rglob("file-*.parquet"))
    for metadata_path in episode_files:
        episode = pq.read_table(
            metadata_path, columns=["episode_index", "tasks", "length"]
        ).to_pylist()[0]
        episode_index = int(episode["episode_index"])
        task_list = episode["tasks"]
        if len(task_list) != 1 or task_list[0] not in by_task:
            raise ValueError(f"Unexpected task mapping in {metadata_path}: {task_list}")
        chunk = episode_index // 1000
        video = (
            dataset
            / "videos/observation.images.front"
            / f"chunk-{chunk:03d}"
            / f"file-{episode_index:03d}.mp4"
        )
        if not video.is_file():
            raise FileNotFoundError(video)
        by_task[task_list[0]].append(
            {
                "episode_index": episode_index,
                "task": task_list[0],
                "video_path": str(video),
                "frame_count": int(episode["length"]),
            }
        )
    selected = []
    ordered_by_task = []
    for task in task_names.values():
        episodes = sorted(by_task[task], key=lambda row: row["episode_index"])
        if len(episodes) < per_task:
            raise ValueError(f"Task {task} has only {len(episodes)} episodes")
        selected.extend(episodes[:per_task])
        ordered_by_task.append(episodes)
    planned_rounds = []
    cursor, batch = 0, per_task
    while any(cursor < len(task_episodes) for task_episodes in ordered_by_task):
        planned_rounds.append([row["episode_index"]
                               for task_episodes in ordered_by_task
                               for row in task_episodes[cursor:cursor + batch]])
        cursor += batch
        batch = 2 * cursor
    data = {
        "dataset_root": str(dataset.resolve()),
        "view": "observation.images.front",
        "task_counts": {task: len(rows) for task, rows in by_task.items()},
        "all_episodes": sorted(
            (row for rows in by_task.values() for row in rows),
            key=lambda row: row["episode_index"],
        ),
        "seed_episodes": selected,
        "growth_rounds": [[row["episode_index"] for row in selected]],
        "planned_growth_rounds": planned_rounds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"inventory={output}")
    print(f"tasks={data['task_counts']} seed={[row['episode_index'] for row in selected]}")


def storyboard(manifest: Path, output_dir: Path, samples: int, columns: int) -> None:
    data = json.loads(manifest.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in data["seed_episodes"]:
        episode = int(record["episode_index"])
        total = int(record["frame_count"])
        indices = sorted(set(np.linspace(0, total - 1, samples).round().astype(int)))
        cap = cv2.VideoCapture(record["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {record['video_path']}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = 320, 250
        tile_h = height + 33
        rows = (len(indices) + columns - 1) // columns
        canvas = np.full((rows * tile_h, columns * width, 3), 245, dtype=np.uint8)
        for index, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_idx} from episode {episode}")
            tile = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            x, y = (index % columns) * width, (index // columns) * tile_h
            canvas[y : y + height, x : x + width] = tile
            cv2.putText(
                canvas,
                f"f{frame_idx}  {frame_idx / fps:.1f}s",
                (x + 6, y + height + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 0),
                2,
            )
        cap.release()
        path = output_dir / f"ep{episode:06d}_timeline.jpg"
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Could not write {path}")
        print(f"timeline={path}", flush=True)


def focus(
    manifest: Path,
    episode: int,
    start: int,
    stop: int,
    stride: int,
    output: Path,
    crop: tuple[int, int, int, int] | None,
    columns: int,
) -> None:
    data = json.loads(manifest.read_text())
    matches = [row for row in data["all_episodes"] if row["episode_index"] == episode]
    if len(matches) != 1:
        raise ValueError(f"Episode {episode} not in manifest")
    record = matches[0]
    if stride < 1 or start < 0 or stop >= record["frame_count"] or start > stop:
        raise ValueError("Invalid frame range")
    cap = cv2.VideoCapture(record["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {record['video_path']}")
    indices = list(range(start, stop + 1, stride))
    tiles = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx}")
        if crop:
            x1, y1, x2, y2 = crop
            frame = frame[y1:y2, x1:x2]
            if frame.size == 0:
                raise ValueError(f"Empty crop: {crop}")
        width = 360
        height = round(frame.shape[0] * width / frame.shape[1])
        tile = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(tile, 0, 32, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(tile, f"ep{episode} f{frame_idx}", (8, height + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        tiles.append(tile)
    cap.release()
    empty = np.full_like(tiles[0], 245)
    while len(tiles) % columns:
        tiles.append(empty)
    canvas = np.vstack([
        np.hstack(tiles[index : index + columns])
        for index in range(0, len(tiles), columns)
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write {output}")
    print(f"focus={output}")


def event_frames(manifest: Path, event_file: Path, output_dir: Path) -> None:
    data = json.loads(manifest.read_text())
    event_data = json.loads(event_file.read_text())
    records = {int(row["episode_index"]): row for row in data["all_episodes"]}
    names = ("first", "occluder_contact", "exposed", "object_contact", "region_contact", "last")
    for episode_text, chosen in event_data["episodes"].items():
        episode = int(episode_text)
        record = records[episode]
        cap = cv2.VideoCapture(record["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {record['video_path']}")
        video_dir = output_dir / f"ep{episode:06d}"
        video_dir.mkdir(parents=True, exist_ok=True)
        tiles = []
        for name in names:
            frame_idx = int(chosen[name])
            if not 0 <= frame_idx < record["frame_count"]:
                raise ValueError(f"Out-of-range {name} frame {frame_idx} for {episode}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read episode {episode} frame {frame_idx}")
            frame_path = video_dir / f"{name}_f{frame_idx:05d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not write {frame_path}")
            tile = cv2.resize(frame, (600, 456), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 0, 35, 0, 0, cv2.BORDER_CONSTANT, value=(245,245,245))
            cv2.putText(tile, f"{name}  f{frame_idx}", (8, 481),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
            tiles.append(tile)
        cap.release()
        sheet = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
        sheet_path = output_dir / f"ep{episode:06d}_events.jpg"
        if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"Could not write {sheet_path}")
        print(f"events={sheet_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p_inv = sub.add_parser("inventory")
    p_inv.add_argument("--dataset", type=Path, required=True)
    p_inv.add_argument("--output", type=Path, required=True)
    p_inv.add_argument("--per-task", type=int, default=2)
    p_story = sub.add_parser("storyboard")
    p_story.add_argument("--manifest", type=Path, required=True)
    p_story.add_argument("--output-dir", type=Path, required=True)
    p_story.add_argument("--samples", type=int, default=36)
    p_story.add_argument("--columns", type=int, default=6)
    p_focus = sub.add_parser("focus")
    p_focus.add_argument("--manifest", type=Path, required=True)
    p_focus.add_argument("--episode", type=int, required=True)
    p_focus.add_argument("--start", type=int, required=True)
    p_focus.add_argument("--stop", type=int, required=True)
    p_focus.add_argument("--stride", type=int, default=5)
    p_focus.add_argument("--output", type=Path, required=True)
    p_focus.add_argument("--crop", type=int, nargs=4)
    p_focus.add_argument("--columns", type=int, default=6)
    p_event = sub.add_parser("events")
    p_event.add_argument("--manifest", type=Path, required=True)
    p_event.add_argument("--event-file", type=Path, required=True)
    p_event.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "inventory":
        inventory(args.dataset, args.output, args.per_task)
    elif args.action == "storyboard":
        storyboard(args.manifest, args.output_dir, args.samples, args.columns)
    elif args.action == "focus":
        focus(args.manifest, args.episode, args.start, args.stop, args.stride,
              args.output, tuple(args.crop) if args.crop else None, args.columns)
    else:
        event_frames(args.manifest, args.event_file, args.output_dir)


if __name__ == "__main__":
    main()
