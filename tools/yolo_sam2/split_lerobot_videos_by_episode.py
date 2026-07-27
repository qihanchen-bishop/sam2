#!/usr/bin/env python3
"""Split LeRobot chunk videos into per-episode videos using episode metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split LeRobot v3 chunk videos into one video per episode."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--video-key",
        default=None,
        help="Video feature key to split, e.g. observation.images.left_front. Defaults to all video keys.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing episode videos.",
    )
    parser.add_argument(
        "--crf",
        default="18",
        help="x264 CRF for frame-accurate re-encoding. Lower is higher quality.",
    )
    return parser.parse_args()


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "This script needs pandas with pyarrow/fastparquet support. "
            "Run it in the lerobot environment, e.g. "
            "`conda run -n lerobot python tools/yolo_sam2/split_lerobot_videos_by_episode.py ...`."
        ) from exc
    return pd


def load_info(dataset: Path) -> dict:
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(f"Missing {info_path}")
    return json.loads(info_path.read_text())


def video_keys(info: dict, requested: str | None) -> list[str]:
    keys = [
        name
        for name, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    if requested is not None:
        if requested not in keys:
            raise RuntimeError(f"--video-key {requested!r} not found. Available: {keys}")
        return [requested]
    if not keys:
        raise RuntimeError("No video features found in meta/info.json")
    return keys


def read_episode_metadata(dataset: Path):
    pd = require_pandas()
    paths = sorted((dataset / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise RuntimeError(f"No episode metadata parquet files found under {dataset / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def run_ffmpeg(src: Path, dst: Path, start: float, end: float, fps: int, crf: str, force: bool) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    if duration <= 0:
        raise RuntimeError(f"Invalid segment duration for {dst}: start={start}, end={end}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if force else "-n",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        crf,
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    info = load_info(dataset)
    fps = int(info.get("fps", 30))
    keys = video_keys(info, args.video_key)
    episodes = read_episode_metadata(dataset)
    manifest = []

    for _, row in episodes.sort_values("episode_index").iterrows():
        episode_idx = int(row["episode_index"])
        for key in keys:
            chunk_idx = int(row[f"videos/{key}/chunk_index"])
            file_idx = int(row[f"videos/{key}/file_index"])
            start = float(row[f"videos/{key}/from_timestamp"])
            end = float(row[f"videos/{key}/to_timestamp"])
            src = dataset / "videos" / key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            if not src.is_file():
                raise RuntimeError(f"Missing source video: {src}")
            dst = output / key / f"episode-{episode_idx:06d}.mp4"
            run_ffmpeg(src, dst, start, end, fps, args.crf, args.force)
            manifest.append(
                {
                    "episode_index": episode_idx,
                    "video_key": key,
                    "source": str(src),
                    "output": str(dst),
                    "from_timestamp": start,
                    "to_timestamp": end,
                    "length": int(row["length"]),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest)} episode videos under {output}")


if __name__ == "__main__":
    main()
