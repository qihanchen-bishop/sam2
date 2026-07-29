#!/usr/bin/env python3
"""Add multi-view SAM2 mask videos to a LeRobot dataset copy."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--seg-root",
        "--seg-task-dir",
        dest="seg_root",
        type=Path,
        required=True,
        help="Root recursively containing file-*_sam2_masks outputs.",
    )
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--video-key",
        action="append",
        default=[],
        help="Source RGB video key to export; repeat for multiple views.",
    )
    parser.add_argument("--crf", default="18")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def read_labels(path: Path) -> list[str]:
    labels = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        name = re.split(r"[:：]", line, maxsplit=1)[0].strip()
        if name:
            labels.append(name)
    if not labels:
        raise RuntimeError(f"No labels found in {path}")
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"Duplicate labels in {path}: {labels}")
    return labels


def normalized_label(label: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", label.strip().lower()).strip("_")


def find_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable:
        return executable
    environment_executable = Path(sys.executable).resolve().parent / name
    if environment_executable.is_file():
        return str(environment_executable)
    raise RuntimeError(f"{name} was not found in PATH or the active environment")


def resolve_labels_path(
    requested: Path | None, seg_root: Path, prompt_paths: list[Path]
) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.append(seg_root / "labels.txt")
    for prompt_path in prompt_paths:
        source = read_json(prompt_path).get("labels_source")
        if source:
            candidates.append(Path(source))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError("Could not find labels.txt; pass --labels explicitly")


def copy_dataset(source_root: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(
                f"{output_root} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {"seg", "yolo", "outputs", "runs", "temp", "final"}
        }

    shutil.copytree(source_root, output_root, ignore=ignore)


def infer_video_key(video_path: Path, source_root: Path) -> str:
    try:
        relative = video_path.resolve().relative_to((source_root / "videos").resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Video {video_path} is not under {source_root / 'videos'}"
        ) from exc
    return relative.parts[0]


def discover_sam2_outputs(
    seg_root: Path, source_root: Path, requested_keys: list[str]
) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for prompt_path in sorted(seg_root.rglob("file-*_sam2_masks/sam2_prompts.json")):
        prompt = read_json(prompt_path)
        video_path = Path(prompt["video_path"]).resolve()
        video_key = prompt.get("video_key") or infer_video_key(video_path, source_root)
        if requested_keys and video_key not in requested_keys:
            continue
        chunk_index = int(
            prompt.get(
                "chunk_index", video_path.parent.name.removeprefix("chunk-")
            )
        )
        file_index = int(
            prompt.get("file_index", video_path.stem.removeprefix("file-"))
        )
        identity = (video_key, chunk_index, file_index)
        if identity in seen:
            raise RuntimeError(f"Duplicate SAM2 output for {identity}: {prompt_path}")
        seen.add(identity)
        mask_dir = prompt_path.parent / "mask_frames"
        if not mask_dir.is_dir():
            raise RuntimeError(f"Missing mask_frames directory: {mask_dir}")
        frame_count = int(prompt.get("frame_count", 0))
        if frame_count <= 0:
            complete_path = prompt_path.parent / "complete.json"
            if complete_path.is_file():
                frame_count = int(read_json(complete_path).get("frame_count", 0))
        if frame_count <= 0:
            raise RuntimeError(f"Missing frame_count in {prompt_path}")
        records.append(
            {
                "video_key": video_key,
                "video_path": video_path,
                "chunk_index": chunk_index,
                "file_index": file_index,
                "frame_count": frame_count,
                "mask_dir": mask_dir,
            }
        )
    if not records:
        raise RuntimeError(f"No SAM2 outputs found under {seg_root}")
    return records


def expected_source_videos(
    source_root: Path, requested_keys: list[str]
) -> set[tuple[str, int, int]]:
    videos_root = source_root / "videos"
    available_keys = sorted(path.name for path in videos_root.iterdir() if path.is_dir())
    keys = requested_keys or available_keys
    missing_keys = sorted(set(keys) - set(available_keys))
    if missing_keys:
        raise RuntimeError(
            f"Unknown video keys {missing_keys}; available keys: {available_keys}"
        )
    expected = set()
    for video_key in keys:
        for path in (videos_root / video_key).glob("chunk-*/*.mp4"):
            expected.add(
                (
                    video_key,
                    int(path.parent.name.removeprefix("chunk-")),
                    int(path.stem.removeprefix("file-")),
                )
            )
    return expected


def output_video_key(source_key: str, label: str) -> str:
    prefix = "observation.images."
    view = source_key.removeprefix(prefix).replace(".", "_")
    return f"{prefix}{view}_{normalized_label(label)}"


def encode_binary_mask_video(
    mask_paths: list[Path],
    class_id: int,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    crf: str,
) -> tuple[int, int]:
    import numpy as np
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        find_executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(output_path),
    ]

    positive_pixels = 0
    total_pixels = len(mask_paths) * width * height
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert("L")
            if mask.size != (width, height):
                mask = mask.resize((width, height), Image.Resampling.NEAREST)
            binary = np.asarray(mask, dtype=np.uint8) == class_id
            positive_pixels += int(binary.sum())
            frame = np.repeat((binary.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}")
    return positive_pixels, total_pixels


def video_feature(width: int, height: int, fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def mask_stats(positive_pixels: int, total_pixels: int) -> dict[str, Any]:
    if total_pixels <= 0:
        raise RuntimeError("Cannot compute stats for zero pixels")
    mean = positive_pixels / total_pixels
    std = math.sqrt(max(mean * (1.0 - mean), 1e-12))
    return {
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 1.0],
        "mean": [mean, mean, mean],
        "std": [std, std, std],
        "count": [total_pixels, total_pixels, total_pixels],
        "q01": [0.0, 0.0, 0.0],
        "q10": [0.0, 0.0, 0.0],
        "q50": [1.0 if mean >= 0.5 else 0.0] * 3,
        "q90": [1.0 if mean >= 0.1 else 0.0] * 3,
        "q99": [1.0 if mean >= 0.01 else 0.0] * 3,
    }


def update_episode_video_metadata(
    output_root: Path, generated_by_source: dict[str, list[str]]
) -> None:
    import pandas as pd

    episode_paths = sorted((output_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_paths:
        raise RuntimeError(
            f"No episode parquet files found under {output_root / 'meta/episodes'}"
        )
    for path in episode_paths:
        frame = pd.read_parquet(path)
        for source_key, generated_keys in generated_by_source.items():
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                source_column = f"videos/{source_key}/{suffix}"
                if source_column not in frame:
                    if suffix in {"chunk_index", "file_index"}:
                        raise RuntimeError(
                            f"{path} does not contain required column {source_column}"
                        )
                    continue
                for generated_key in generated_keys:
                    frame[f"videos/{generated_key}/{suffix}"] = frame[source_column]
        frame.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    seg_root = args.seg_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"Source dataset not found: {source_root}")
    if not seg_root.is_dir():
        raise RuntimeError(f"SAM2 output root not found: {seg_root}")

    prompt_paths = sorted(seg_root.rglob("file-*_sam2_masks/sam2_prompts.json"))
    labels_path = resolve_labels_path(args.labels, seg_root, prompt_paths)
    labels = read_labels(labels_path)
    records = discover_sam2_outputs(seg_root, source_root, args.video_key)

    expected = expected_source_videos(source_root, args.video_key)
    actual = {
        (record["video_key"], record["chunk_index"], record["file_index"])
        for record in records
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if unexpected:
        raise RuntimeError(f"SAM2 outputs do not belong to the source dataset: {unexpected}")
    if missing and not args.allow_missing:
        preview = ", ".join(map(str, missing[:10]))
        raise RuntimeError(
            f"Missing SAM2 outputs for {len(missing)} videos: {preview}. "
            "Finish segmentation or pass --allow-missing."
        )

    copy_dataset(source_root, output_root, args.overwrite)
    info_path = output_root / "meta" / "info.json"
    stats_path = output_root / "meta" / "stats.json"
    info = read_json(info_path)
    stats = read_json(stats_path)
    fps = int(info["fps"])

    generated_by_source: dict[str, list[str]] = {}
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "total": 0}
    )
    dimensions: dict[str, tuple[int, int]] = {}
    for source_key in sorted({record["video_key"] for record in records}):
        feature = info["features"].get(source_key)
        if not feature or feature.get("dtype") != "video":
            raise RuntimeError(f"Missing source video feature {source_key} in info.json")
        height, width = map(int, feature["shape"][:2])
        dimensions[source_key] = (width, height)
        generated_by_source[source_key] = [
            output_video_key(source_key, label) for label in labels
        ]

    for index, record in enumerate(records, start=1):
        width, height = dimensions[record["video_key"]]
        mask_paths = sorted(record["mask_dir"].glob("*.png"))
        if len(mask_paths) != record["frame_count"]:
            raise RuntimeError(
                f"{record['mask_dir']} has {len(mask_paths)} masks, "
                f"expected {record['frame_count']}"
            )
        print(
            f"[{index}/{len(records)}] export {record['video_key']} "
            f"file-{record['file_index']:03d}",
            flush=True,
        )
        for class_id, label in enumerate(labels, start=1):
            generated_key = output_video_key(record["video_key"], label)
            output_path = (
                output_root
                / "videos"
                / generated_key
                / f"chunk-{record['chunk_index']:03d}"
                / f"file-{record['file_index']:03d}.mp4"
            )
            positive, total = encode_binary_mask_video(
                mask_paths,
                class_id,
                output_path,
                width,
                height,
                fps,
                args.crf,
            )
            totals[generated_key]["positive"] += positive
            totals[generated_key]["total"] += total

    for source_key, generated_keys in generated_by_source.items():
        width, height = dimensions[source_key]
        for generated_key in generated_keys:
            info["features"][generated_key] = video_feature(width, height, fps)
            stats[generated_key] = mask_stats(
                totals[generated_key]["positive"],
                totals[generated_key]["total"],
            )

    update_episode_video_metadata(output_root, generated_by_source)
    write_json(info_path, info)
    write_json(stats_path, stats)
    manifest = {
        "source_root": str(source_root),
        "seg_root": str(seg_root),
        "labels_source": str(labels_path),
        "labels": labels,
        "records": len(records),
        "missing_records": [
            {"video_key": key, "chunk_index": chunk, "file_index": file_index}
            for key, chunk, file_index in missing
        ],
        "generated_features": generated_by_source,
    }
    write_json(output_root / "segmentation_manifest.json", manifest)
    print(f"Done: {output_root}")
    for source_key, generated_keys in generated_by_source.items():
        print(f"{source_key}:")
        for generated_key in generated_keys:
            print(f"  {generated_key}")


if __name__ == "__main__":
    main()
