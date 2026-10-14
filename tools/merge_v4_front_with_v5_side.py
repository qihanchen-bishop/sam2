#!/usr/bin/env python3
"""Merge byte-identical v4 front data with reviewed v5 side SAM2 masks."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SIDE_KEY = "observation.images.side"
FRONT_PREFIX = "observation.images.front"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-root", type=Path, required=True)
    parser.add_argument("--origin-root", type=Path, required=True)
    parser.add_argument("--seg-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--crf", default="18")
    parser.add_argument("--include-mask-visual", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def link_or_copy(source: str, destination: str) -> str:
    """Hard-link immutable videos, while copying metadata that will be edited."""
    if "/videos/" not in Path(source).as_posix():
        return shutil.copy2(source, destination)
    try:
        os.link(source, destination)
        return destination
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
        return shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inherited_front_files(root: Path) -> list[Path]:
    paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if (
            f"videos/{FRONT_PREFIX}/" in relative
            or f"videos/{FRONT_PREFIX}_" in relative
        ):
            paths.append(path)
    return sorted(paths)


def merge_side_episode_metadata(origin_root: Path, output_root: Path) -> None:
    import pandas as pd

    source_paths = sorted((origin_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    destination_paths = sorted((output_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    source_by_relative = {path.relative_to(origin_root / "meta" / "episodes"): path for path in source_paths}
    destination_by_relative = {
        path.relative_to(output_root / "meta" / "episodes"): path
        for path in destination_paths
    }
    if source_by_relative.keys() != destination_by_relative.keys():
        raise RuntimeError("Origin and front datasets have different episode parquet layouts")

    side_prefixes = (f"videos/{SIDE_KEY}/", f"stats/{SIDE_KEY}/")
    for relative in sorted(source_by_relative):
        source = pd.read_parquet(source_by_relative[relative])
        destination_path = destination_by_relative[relative]
        destination = pd.read_parquet(destination_path)
        source = source.set_index("episode_index", drop=False)
        destination = destination.set_index("episode_index", drop=False)
        if list(source.index) != list(destination.index):
            raise RuntimeError(f"Episode rows differ in {relative}")
        for column in source.columns:
            if column.startswith(side_prefixes):
                destination[column] = source[column]
        destination.reset_index(drop=True).to_parquet(destination_path, index=False)


def merge_quality_reports(front_root: Path, side_root: Path, output_root: Path) -> None:
    destination = output_root / "segmentation_quality"
    front_summary = read_json(front_root / "segmentation_quality" / "summary.json")
    side_summary = read_json(side_root / "summary.json")
    for path in sorted(side_root.glob("observation_images_side_*.npz")):
        shutil.copy2(path, destination / path.name)

    combined = dict(front_summary)
    combined["streams"] = dict(front_summary.get("streams", {}))
    combined["streams"].update(side_summary.get("streams", {}))
    combined["processed_videos"] = int(front_summary.get("processed_videos", 0)) + int(
        side_summary.get("processed_videos", 0)
    )
    if (
        front_summary.get("processed_frames_per_stream")
        != side_summary.get("processed_frames_per_stream")
    ):
        raise RuntimeError("Front and side quality reports cover different frame counts")
    combined["processed_frames_per_stream"] = front_summary.get(
        "processed_frames_per_stream"
    )
    combined["completeness_failures"] = list(
        front_summary.get("completeness_failures", [])
    ) + list(side_summary.get("completeness_failures", []))
    for stream, metrics in combined["streams"].items():
        metrics["output"] = str(
            (destination / (stream.replace("/", "__").replace(".", "_") + ".npz")).resolve()
        )
    write_json(destination / "summary.json", combined)
    write_json(destination / "side_summary.json", side_summary)
    if (side_root / "video_summary.json").is_file():
        shutil.copy2(side_root / "video_summary.json", destination / "side_video_summary.json")
    if (side_root / "completeness_review.json").is_file():
        shutil.copy2(
            side_root / "completeness_review.json",
            destination / "side_completeness_review.json",
        )


def main() -> None:
    args = parse_args()
    front_root = args.front_root.resolve()
    origin_root = args.origin_root.resolve()
    seg_root = args.seg_root.resolve()
    labels_path = args.labels.resolve()
    dataset_labels_path = args.dataset_labels.resolve()
    output_root = args.output_root.resolve()
    quality_root = args.quality_root.resolve() if args.quality_root else None

    for path, name in (
        (front_root, "front dataset"),
        (origin_root, "origin dataset"),
        (seg_root, "side segmentation"),
    ):
        if not path.is_dir():
            raise RuntimeError(f"Missing {name}: {path}")
    if not labels_path.is_file():
        raise RuntimeError(f"Missing labels file: {labels_path}")
    if not dataset_labels_path.is_file():
        raise RuntimeError(f"Missing dataset labels file: {dataset_labels_path}")
    if output_root.exists():
        raise RuntimeError(f"Output already exists: {output_root}")

    front_info = read_json(front_root / "meta" / "info.json")
    origin_info = read_json(origin_root / "meta" / "info.json")
    for key in ("total_episodes", "total_frames", "fps"):
        if front_info.get(key) != origin_info.get(key):
            raise RuntimeError(f"Dataset mismatch for {key}")
    if SIDE_KEY not in origin_info["features"]:
        raise RuntimeError(f"Origin dataset has no {SIDE_KEY} feature")

    converter_dir = Path(__file__).resolve().parent / "yolo_sam2"
    sys.path.insert(0, str(converter_dir))
    from convert_sam2_masks_to_lerobot import (  # noqa: PLC0415
        discover_sam2_outputs,
        encode_binary_mask_video,
        expected_source_videos,
        mask_stats,
        output_video_key,
        read_labels,
        update_episode_video_metadata,
        video_feature,
    )

    labels = read_labels(labels_path)
    expected_labels = ["occluder", "object", "region", "tool"]
    if labels != expected_labels:
        raise RuntimeError(f"Expected side labels {expected_labels}, got {labels}")

    records = discover_sam2_outputs(seg_root, origin_root, [SIDE_KEY])
    expected = expected_source_videos(origin_root, [SIDE_KEY])
    actual = {
        (record["video_key"], record["chunk_index"], record["file_index"])
        for record in records
    }
    if actual != expected:
        raise RuntimeError(
            f"Side SAM2 completeness mismatch: actual={len(actual)}, expected={len(expected)}"
        )

    shutil.copytree(front_root, output_root, copy_function=link_or_copy)
    shutil.copy2(dataset_labels_path, output_root / "labels.txt")
    source_side = origin_root / "videos" / SIDE_KEY
    destination_side = output_root / "videos" / SIDE_KEY
    shutil.copytree(source_side, destination_side, copy_function=link_or_copy)

    info_path = output_root / "meta" / "info.json"
    stats_path = output_root / "meta" / "stats.json"
    info = read_json(info_path)
    stats = read_json(stats_path)
    origin_stats = read_json(origin_root / "meta" / "stats.json")
    info["features"][SIDE_KEY] = origin_info["features"][SIDE_KEY]
    stats[SIDE_KEY] = origin_stats[SIDE_KEY]
    write_json(info_path, info)
    write_json(stats_path, stats)
    merge_side_episode_metadata(origin_root, output_root)

    height, width = map(int, info["features"][SIDE_KEY]["shape"][:2])
    fps = int(info["fps"])
    generated_keys = [output_video_key(SIDE_KEY, label) for label in labels]
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "total": 0})
    copied_visuals = 0
    for index, record in enumerate(records, start=1):
        mask_paths = sorted(record["mask_dir"].glob("*.png"))
        if len(mask_paths) != int(record["frame_count"]):
            raise RuntimeError(
                f"{record['mask_dir']} has {len(mask_paths)} masks, "
                f"expected {record['frame_count']}"
            )
        print(f"[{index}/{len(records)}] export side file-{record['file_index']:03d}", flush=True)
        for class_id, label in enumerate(labels, start=1):
            generated_key = output_video_key(SIDE_KEY, label)
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
        if args.include_mask_visual:
            visual_destination = (
                output_root
                / "review"
                / "mask_visual"
                / SIDE_KEY
                / f"chunk-{record['chunk_index']:03d}"
                / f"file-{record['file_index']:03d}.mp4"
            )
            visual_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["mask_visual"], visual_destination)
            copied_visuals += 1

    info = read_json(info_path)
    stats = read_json(stats_path)
    for generated_key in generated_keys:
        info["features"][generated_key] = video_feature(width, height, fps)
        stats[generated_key] = mask_stats(
            totals[generated_key]["positive"], totals[generated_key]["total"]
        )
    write_json(info_path, info)
    write_json(stats_path, stats)
    update_episode_video_metadata(output_root, {SIDE_KEY: generated_keys}, set())

    if quality_root:
        if not quality_root.is_dir():
            raise RuntimeError(f"Missing quality root: {quality_root}")
        merge_quality_reports(front_root, quality_root, output_root)

    inherited = inherited_front_files(front_root)
    front_hashes = {
        path.relative_to(front_root).as_posix(): sha256(path)
        for path in inherited
    }
    for relative, expected_hash in front_hashes.items():
        actual_hash = sha256(output_root / relative)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Inherited front file changed: {relative}")

    front_manifest_path = front_root / "segmentation_manifest.json"
    front_manifest = read_json(front_manifest_path) if front_manifest_path.is_file() else {}
    generated_features = dict(front_manifest.get("generated_features", {}))
    generated_features[SIDE_KEY] = generated_keys
    manifest = {
        "version": 1,
        "front_source": str(front_root),
        "origin_source": str(origin_root),
        "side_segmentation_source": str(seg_root),
        "side_labels_source": str(labels_path),
        "dataset_labels_source": str(dataset_labels_path),
        "labels": front_manifest.get(
            "labels", ["occluder", "object", "region", "tool", "leftarm", "rightarm"]
        ),
        "view_labels": {
            FRONT_PREFIX: front_manifest.get(
                "labels", ["occluder", "object", "region", "tool", "leftarm", "rightarm"]
            ),
            SIDE_KEY: labels,
        },
        "side_labels": labels,
        "side_arm_policy": {
            "included": False,
            "reason": (
                "Side-view arms were omitted after pilot review because disjoint visible "
                "arm components and heavy mutual occlusion produced unstable identities and "
                "cross-class leakage. Front leftarm/rightarm labels are inherited unchanged."
            ),
        },
        "records": len(records),
        "generated_features": generated_features,
        "front_byte_identity": {
            "verified": True,
            "file_count": len(front_hashes),
            "sha256": front_hashes,
        },
        "review_artifacts": {
            "mask_visual_included": bool(args.include_mask_visual),
            "mask_visual_count": copied_visuals,
            "quality_root": str(quality_root) if quality_root else None,
        },
    }
    write_json(output_root / "segmentation_manifest.json", manifest)
    write_json(output_root / "segmentation_manifest_v5.json", manifest)
    print(f"Done: {output_root}")


if __name__ == "__main__":
    main()
