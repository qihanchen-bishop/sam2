#!/usr/bin/env python3
"""Add SAM2 mask videos to a LeRobot dataset copy."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


DEFAULT_CLASS_KEY_MAP = {
    "occluder": "observation.images.occluder",
    "region": "observation.images.region",
    "leftarm": "observation.images.left_arm",
    "left_arm": "observation.images.left_arm",
    "rightarm": "observation.images.right_arm",
    "right_arm": "observation.images.right_arm",
    "object": "observation.images.object",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seg-task-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rgb-key", default="observation.images.left_front")
    parser.add_argument("--crf", default="18")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4) + "\n")


def read_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not labels:
        raise RuntimeError(f"No labels found in {path}")
    return labels


def normalized_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def copy_dataset(source_root: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {"seg", "yolo", "outputs", "runs"}}

    shutil.copytree(source_root, output_root, ignore=ignore)


def discover_sam2_outputs(seg_task_dir: Path) -> list[dict]:
    records = []
    for prompt_path in sorted(seg_task_dir.glob("file-*_sam2_masks/sam2_prompts.json")):
        prompt = read_json(prompt_path)
        stem = prompt_path.parent.name.removesuffix("_sam2_masks")
        file_index = int(stem.removeprefix("file-"))
        mask_dir = prompt_path.parent / "mask_frames"
        if not mask_dir.is_dir():
            raise RuntimeError(f"Missing mask_frames directory: {mask_dir}")
        records.append(
            {
                "stem": stem,
                "file_index": file_index,
                "chunk_index": 0,
                "frame_count": int(prompt["frame_count"]),
                "mask_dir": mask_dir,
            }
        )
    if not records:
        raise RuntimeError(f"No file-*_sam2_masks/sam2_prompts.json found under {seg_task_dir}")
    return records


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
    cmd = [
        "ffmpeg",
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
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert("L")
            if mask.size != (width, height):
                mask = mask.resize((width, height), Image.Resampling.NEAREST)
            binary = (np.asarray(mask, dtype=np.uint8) == class_id)
            positive_pixels += int(binary.sum())
            frame = np.repeat((binary.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}")
    return positive_pixels, total_pixels


def video_feature(width: int, height: int, fps: int) -> dict:
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


def mask_stats(positive_pixels: int, total_pixels: int) -> dict:
    if total_pixels <= 0:
        raise RuntimeError("Cannot compute stats for zero pixels")
    mean = positive_pixels / total_pixels
    std = math.sqrt(max(mean * (1.0 - mean), 1e-12))
    values = [mean, mean, mean]
    stds = [std, std, std]
    count = [total_pixels, total_pixels, total_pixels]
    return {
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 1.0],
        "mean": values,
        "std": stds,
        "count": count,
        "q01": [0.0, 0.0, 0.0],
        "q10": [0.0, 0.0, 0.0],
        "q50": [1.0 if mean >= 0.5 else 0.0] * 3,
        "q90": [1.0 if mean >= 0.1 else 0.0] * 3,
        "q99": [1.0 if mean >= 0.01 else 0.0] * 3,
    }


def update_episode_video_metadata(output_root: Path, rgb_key: str, video_keys: list[str]) -> None:
    import pandas as pd

    episodes_dir = output_root / "meta" / "episodes"
    paths = sorted(episodes_dir.glob("chunk-*/*.parquet"))
    if not paths:
        raise RuntimeError(f"No episode parquet files found under {episodes_dir}")

    rgb_chunk_col = f"videos/{rgb_key}/chunk_index"
    rgb_file_col = f"videos/{rgb_key}/file_index"
    rgb_from_col = f"videos/{rgb_key}/from_timestamp"
    rgb_to_col = f"videos/{rgb_key}/to_timestamp"

    for path in paths:
        df = pd.read_parquet(path)
        if rgb_chunk_col not in df or rgb_file_col not in df:
            raise RuntimeError(f"{path} does not contain video metadata for {rgb_key}")
        for video_key in video_keys:
            df[f"videos/{video_key}/chunk_index"] = df[rgb_chunk_col]
            df[f"videos/{video_key}/file_index"] = df[rgb_file_col]
            if rgb_from_col in df:
                df[f"videos/{video_key}/from_timestamp"] = df[rgb_from_col]
            if rgb_to_col in df:
                df[f"videos/{video_key}/to_timestamp"] = df[rgb_to_col]
        df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    seg_task_dir = args.seg_task_dir.resolve()

    labels = read_labels(seg_task_dir / "labels.txt")
    label_to_key = {}
    for label in labels:
        key = DEFAULT_CLASS_KEY_MAP.get(normalized_label(label))
        if key is None:
            raise RuntimeError(
                f"Label {label!r} is not mapped. Add it to DEFAULT_CLASS_KEY_MAP in this script."
            )
        label_to_key[label] = key

    copy_dataset(source_root, output_root, args.overwrite)

    info_path = output_root / "meta" / "info.json"
    stats_path = output_root / "meta" / "stats.json"
    info = read_json(info_path)
    stats = read_json(stats_path)
    rgb_feature = info["features"][args.rgb_key]
    height, width = rgb_feature["shape"][:2]
    fps = int(info.get("fps") or rgb_feature["info"]["video.fps"])

    records = discover_sam2_outputs(seg_task_dir)
    totals = {label_to_key[label]: {"positive": 0, "total": 0} for label in labels}

    for record in records:
        mask_paths = sorted(record["mask_dir"].glob("*.png"))
        if len(mask_paths) != record["frame_count"]:
            raise RuntimeError(
                f"{record['mask_dir']} has {len(mask_paths)} masks, expected {record['frame_count']}"
            )
        for class_id, label in enumerate(labels, start=1):
            video_key = label_to_key[label]
            output_path = (
                output_root
                / "videos"
                / video_key
                / f"chunk-{record['chunk_index']:03d}"
                / f"file-{record['file_index']:03d}.mp4"
            )
            positive, total = encode_binary_mask_video(
                mask_paths=mask_paths,
                class_id=class_id,
                output_path=output_path,
                width=width,
                height=height,
                fps=fps,
                crf=args.crf,
            )
            totals[video_key]["positive"] += positive
            totals[video_key]["total"] += total
            print(f"wrote {output_path}")

    for video_key, total in totals.items():
        info["features"][video_key] = video_feature(width=width, height=height, fps=fps)
        stats[video_key] = mask_stats(total["positive"], total["total"])

    update_episode_video_metadata(output_root, args.rgb_key, list(totals))
    write_json(info_path, info)
    write_json(stats_path, stats)
    print(f"Done: {output_root}")
    print("Mask keys:")
    for label in labels:
        print(f"  {label}: {label_to_key[label]}")


if __name__ == "__main__":
    main()
