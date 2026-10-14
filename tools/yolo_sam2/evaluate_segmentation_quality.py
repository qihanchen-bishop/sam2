#!/usr/bin/env python3
"""Evaluate SAM2 mask videos and export frame-aligned LeRobot quality metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from label_metadata import read_label_names


BASE_THRESHOLDS = {
    "temporal_iou_min": 0.6,
    "area_log_change_max": 0.45,
    "centroid_jump_max": 0.08,
    "max_components": 3,
    "bidirectional_iou_min": 0.6,
    "quality_score_min": 0.6,
    "min_component_area_ratio": 0.0001,
    "max_missing_visible_fraction": 0.15,
    "max_missing_visible_run": 60,
}

CLASS_OVERRIDES = {
    "object": {
        "max_components": 2,
        "max_missing_visible_fraction": 0.08,
        "max_missing_visible_run": 45,
    },
    "occluder": {"temporal_iou_min": 0.5, "max_components": 8},
    "region": {"max_components": 3},
    "tool": {
        "temporal_iou_min": 0.4,
        "centroid_jump_max": 0.12,
        "max_components": 6,
        "bidirectional_iou_min": 0.5,
    },
}

METRIC_DTYPES = {
    "area_ratio": np.float32,
    "centroid_x": np.float32,
    "centroid_y": np.float32,
    "components": np.int16,
    "iou_prev": np.float32,
    "iou_next": np.float32,
    "temporal_iou": np.float32,
    "area_log_change": np.float32,
    "centroid_displacement": np.float32,
    "bidirectional_iou": np.float32,
    "bidirectional_available": np.bool_,
    "quality_score": np.float32,
    "uncertain": np.bool_,
    "low_temporal_iou": np.bool_,
    "area_jump": np.bool_,
    "centroid_jump": np.bool_,
    "component_anomaly": np.bool_,
    "low_bidirectional_iou": np.bool_,
    "visible_expected": np.bool_,
    "missing_when_expected": np.bool_,
    "episode_index": np.int32,
    "frame_index": np.int32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seg-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--review-count", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Evaluate only the completed SAM2 records found under --seg-root. "
            "Intended for reviewed seed videos; full-dataset validation remains "
            "strict by default."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def json_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def stream_name(video_key: str, label: str) -> str:
    return f"{video_key}_{label}"


def file_name(stream: str) -> str:
    return stream.replace(".", "_") + ".npz"


def thresholds_for(label: str) -> dict[str, float | int]:
    values = dict(BASE_THRESHOLDS)
    values.update(CLASS_OVERRIDES.get(label, {}))
    return values


def expected_visibility(prompt_object: dict[str, Any] | None, frame_count: int) -> np.ndarray:
    visible = np.zeros(frame_count, dtype=np.bool_)
    if prompt_object is None:
        return visible
    for raw_range in prompt_object.get("visible_ranges", []):
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            continue
        start = max(0, int(raw_range[0]))
        end = min(frame_count - 1, int(raw_range[1]))
        if start <= end:
            visible[start : end + 1] = True
    return visible


def longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def episode_layout(dataset_root: Path) -> tuple[dict[tuple[int, int], dict[str, int]], int]:
    import pyarrow.parquet as pq

    layout: dict[tuple[int, int], dict[str, int]] = {}
    maximum = 0
    for path in sorted((dataset_root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        data = table.to_pydict()
        for row in range(table.num_rows):
            chunk = int(data["data/chunk_index"][row])
            file_index = int(data["data/file_index"][row])
            start = int(data["dataset_from_index"][row])
            end = int(data["dataset_to_index"][row])
            layout[(chunk, file_index)] = {
                "episode_index": int(data["episode_index"][row]),
                "start": start,
                "end": end,
                "length": int(data["length"][row]),
            }
            maximum = max(maximum, end)
    if not layout:
        raise RuntimeError(f"No episode metadata found under {dataset_root}")
    return layout, maximum


def discover_outputs(seg_root: Path) -> list[dict[str, Any]]:
    records = []
    for prompt_path in sorted(seg_root.rglob("file-*_sam2_masks/sam2_prompts.json")):
        prompt = read_json(prompt_path)
        completion_path = prompt_path.parent / "complete.json"
        if not completion_path.is_file():
            raise RuntimeError(f"Incomplete SAM2 output: {prompt_path.parent}")
        completion = read_json(completion_path)
        video_path = Path(prompt["video_path"])
        chunk_match = re.fullmatch(r"chunk-(\d+)", video_path.parent.name)
        file_match = re.fullmatch(r"file-(\d+)", video_path.stem)
        if "chunk_index" in prompt:
            chunk_index = int(prompt["chunk_index"])
        elif chunk_match:
            chunk_index = int(chunk_match.group(1))
        else:
            raise RuntimeError(f"Cannot infer chunk index from {video_path}")
        if "file_index" in prompt:
            file_index = int(prompt["file_index"])
        elif file_match:
            file_index = int(file_match.group(1))
        else:
            raise RuntimeError(f"Cannot infer file index from {video_path}")
        records.append(
            {
                "prompt": prompt,
                "completion": completion,
                "root": prompt_path.parent,
                "video_key": str(prompt["video_key"]),
                "chunk_index": chunk_index,
                "file_index": file_index,
                "frame_count": int(prompt["frame_count"]),
                "video_path": video_path,
            }
        )
    if not records:
        raise RuntimeError(f"No completed SAM2 outputs found under {seg_root}")
    return records


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(first & second) / union)


def mask_geometry(
    binary: np.ndarray, min_component_area: int
) -> tuple[int, float, float, int]:
    area = int(np.count_nonzero(binary))
    if area == 0:
        return 0, math.nan, math.nan, 0
    moments = cv2.moments(binary.astype(np.uint8), binaryImage=True)
    centroid_x = float(moments["m10"] / moments["m00"])
    centroid_y = float(moments["m01"] / moments["m00"])
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    components = int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_component_area))
    return area, centroid_x, centroid_y, components


def neighbor_metric(values: np.ndarray, operation: str) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float32)
    if len(values) < 2:
        return result
    if operation == "area":
        pair = np.abs(np.log((values[1:] + 1.0) / (values[:-1] + 1.0)))
    else:
        raise ValueError(operation)
    result[1:] = np.maximum(result[1:], pair)
    result[:-1] = np.maximum(result[:-1], pair)
    return result


def temporal_metrics(masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(masks)
    previous = np.ones(count, dtype=np.float32)
    following = np.ones(count, dtype=np.float32)
    for index in range(1, count):
        value = mask_iou(masks[index - 1], masks[index])
        previous[index] = value
        following[index - 1] = value
    return previous, following, np.minimum(previous, following)


def centroid_displacement(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    count = len(xs)
    result = np.zeros(count, dtype=np.float32)
    for index in range(1, count):
        if not (
            math.isfinite(float(xs[index - 1]))
            and math.isfinite(float(ys[index - 1]))
            and math.isfinite(float(xs[index]))
            and math.isfinite(float(ys[index]))
        ):
            continue
        distance = math.hypot(
            float(xs[index] - xs[index - 1]),
            float(ys[index] - ys[index - 1]),
        )
        result[index] = max(result[index], distance)
        result[index - 1] = max(result[index - 1], distance)
    return result


def load_directional_iou(
    record: dict[str, Any], obj_id: int, frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(frame_count, np.nan, dtype=np.float32)
    available = np.zeros(frame_count, dtype=np.bool_)
    metadata = record["completion"].get("bidirectional")
    if not metadata:
        return values, available
    start = max(0, int(metadata["overlap_start_frame"]))
    end = min(frame_count - 1, int(metadata["overlap_end_frame"]))
    root = record["root"] / str(metadata["output"])
    for frame_idx in range(start, end + 1):
        compact_name = f"{frame_idx:05d}.png"
        forward_path = root / "forward" / compact_name
        reverse_path = root / "reverse" / compact_name
        forward = (
            cv2.imread(str(forward_path), cv2.IMREAD_GRAYSCALE)
            if forward_path.is_file()
            else None
        )
        reverse = (
            cv2.imread(str(reverse_path), cv2.IMREAD_GRAYSCALE)
            if reverse_path.is_file()
            else None
        )
        if forward is not None and reverse is not None:
            forward = forward == obj_id
            reverse = reverse == obj_id
        else:
            legacy_name = f"{frame_idx:05d}_obj{obj_id:03d}.png"
            forward_path = root / "forward" / legacy_name
            reverse_path = root / "reverse" / legacy_name
            forward = (
                cv2.imread(str(forward_path), cv2.IMREAD_GRAYSCALE)
                if forward_path.is_file()
                else None
            )
            reverse = (
                cv2.imread(str(reverse_path), cv2.IMREAD_GRAYSCALE)
                if reverse_path.is_file()
                else None
            )
            if forward is not None:
                forward = forward > 0
            if reverse is not None:
                reverse = reverse > 0
        if forward is None or reverse is None:
            continue
        values[frame_idx] = mask_iou(forward, reverse)
        available[frame_idx] = True
    return values, available


def evaluate_class(
    record: dict[str, Any],
    class_id: int,
    thresholds: dict[str, Any],
    prompt_object: dict[str, Any] | None,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    frame_count = record["frame_count"]
    mask_paths = [record["root"] / "mask_frames" / f"{i:05d}.png" for i in range(frame_count)]
    masks: list[np.ndarray] = []
    area = np.zeros(frame_count, dtype=np.float32)
    centroid_x = np.full(frame_count, np.nan, dtype=np.float32)
    centroid_y = np.full(frame_count, np.nan, dtype=np.float32)
    components = np.zeros(frame_count, dtype=np.int16)
    width = height = 0
    for frame_idx, path in enumerate(mask_paths):
        class_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if class_mask is None:
            raise RuntimeError(f"Missing or unreadable mask: {path}")
        height, width = class_mask.shape
        binary = class_mask == class_id
        masks.append(binary)
        minimum = max(1, int(width * height * thresholds["min_component_area_ratio"]))
        pixel_area, cx, cy, component_count = mask_geometry(binary, minimum)
        area[frame_idx] = pixel_area
        centroid_x[frame_idx] = cx / width if math.isfinite(cx) else math.nan
        centroid_y[frame_idx] = cy / height if math.isfinite(cy) else math.nan
        components[frame_idx] = component_count

    iou_prev, iou_next, temporal_iou = temporal_metrics(masks)
    area_log_change = neighbor_metric(area, "area")
    centroid_delta = centroid_displacement(centroid_x, centroid_y)
    bidirectional_iou, bidirectional_available = load_directional_iou(
        record, class_id, frame_count
    )

    low_temporal = temporal_iou < thresholds["temporal_iou_min"]
    area_jump = area_log_change > thresholds["area_log_change_max"]
    centroid_jump = centroid_delta > thresholds["centroid_jump_max"]
    component_anomaly = components > thresholds["max_components"]
    low_bidirectional = bidirectional_available & (
        bidirectional_iou < thresholds["bidirectional_iou_min"]
    )
    visible_expected = expected_visibility(prompt_object, frame_count)
    missing_when_expected = visible_expected & (area == 0)

    temporal_score = np.clip(
        temporal_iou / thresholds["temporal_iou_min"], 0.0, 1.0
    )
    area_score = np.clip(
        1.0 - area_log_change / thresholds["area_log_change_max"], 0.0, 1.0
    )
    centroid_score = np.clip(
        1.0 - centroid_delta / thresholds["centroid_jump_max"], 0.0, 1.0
    )
    component_score = np.clip(
        thresholds["max_components"] / np.maximum(components, 1), 0.0, 1.0
    )
    bidi_score = np.ones(frame_count, dtype=np.float32)
    bidi_score[bidirectional_available] = np.clip(
        bidirectional_iou[bidirectional_available]
        / thresholds["bidirectional_iou_min"],
        0.0,
        1.0,
    )
    quality_score = np.mean(
        np.stack(
            [temporal_score, area_score, centroid_score, component_score, bidi_score]
        ),
        axis=0,
        dtype=np.float32,
    )
    # Keep the aggregate score for ranking, but never let otherwise-good metrics
    # dilute a hard diagnostic failure.  In particular, a zero forward/reverse
    # IoU used to remain "certain" when the other four scores were high.
    uncertain = (
        (quality_score < thresholds["quality_score_min"])
        | low_temporal
        | area_jump
        | centroid_jump
        | component_anomaly
        | low_bidirectional
        | missing_when_expected
    )

    metrics = {
        "area_ratio": (area / float(width * height)).astype(np.float32),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "components": components,
        "iou_prev": iou_prev,
        "iou_next": iou_next,
        "temporal_iou": temporal_iou,
        "area_log_change": area_log_change,
        "centroid_displacement": centroid_delta,
        "bidirectional_iou": bidirectional_iou,
        "bidirectional_available": bidirectional_available,
        "quality_score": quality_score.astype(np.float32),
        "uncertain": uncertain,
        "low_temporal_iou": low_temporal,
        "area_jump": area_jump,
        "centroid_jump": centroid_jump,
        "component_anomaly": component_anomaly,
        "low_bidirectional_iou": low_bidirectional,
        "visible_expected": visible_expected,
        "missing_when_expected": missing_when_expected,
    }
    return metrics, masks


def allocate_metrics(total_frames: int) -> dict[str, np.ndarray]:
    values = {}
    for name, dtype in METRIC_DTYPES.items():
        fill = np.nan if np.issubdtype(dtype, np.floating) else 0
        values[name] = np.full(total_frames, fill, dtype=dtype)
    return values


def render_review(
    stream: str,
    candidates: list[dict[str, Any]],
    output_path: Path,
    count: int,
) -> None:
    selected = sorted(candidates, key=lambda item: item["score"])[:count]
    if not selected:
        return
    tiles = []
    for item in selected:
        capture = cv2.VideoCapture(str(item["video_path"]))
        capture.set(cv2.CAP_PROP_POS_FRAMES, item["frame_index"])
        ok, frame = capture.read()
        capture.release()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binary = (mask == item["class_id"]).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 50, 50), 2)
        tile = Image.fromarray(frame).resize((480, 270), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        caption = (
            f"ep{item['episode_index']} f{item['frame_index']} "
            f"score={item['score']:.2f} tIoU={item['temporal_iou']:.2f} "
            f"biIoU={item['bidirectional_iou']:.2f}"
        )
        draw.rectangle((0, 246, 480, 270), fill=(255, 255, 255))
        draw.text((4, 250), caption, fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    columns = 4
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * 480, rows * 270), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 480, (index // columns) * 270))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    seg_root = args.seg_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"{output_root} is not empty; pass --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)

    labels = read_label_names(args.labels)
    layout, total_frames = episode_layout(dataset_root)
    records = discover_outputs(seg_root)
    if args.allow_partial:
        selected = {
            (record["chunk_index"], record["file_index"]) for record in records
        }
        missing = sorted(selected.difference(layout))
        if missing:
            raise RuntimeError(f"No episode metadata for partial records: {missing}")
        cursor = 0
        partial_layout = {}
        for identity in sorted(selected, key=lambda item: layout[item]["start"]):
            episode = dict(layout[identity])
            episode["start"] = cursor
            cursor += episode["length"]
            episode["end"] = cursor
            partial_layout[identity] = episode
        layout = partial_layout
        total_frames = cursor
    expected = len(layout) * len({record["video_key"] for record in records})
    if len(records) != expected:
        raise RuntimeError(
            f"Expected {expected} completed view videos, found {len(records)}"
        )

    streams: dict[str, dict[str, np.ndarray]] = {}
    reviews: dict[str, list[dict[str, Any]]] = defaultdict(list)
    video_summaries = []
    completeness_failures = []
    for record_index, record in enumerate(records, start=1):
        identity = (record["chunk_index"], record["file_index"])
        if identity not in layout:
            raise RuntimeError(f"No episode metadata for {identity}")
        episode = layout[identity]
        if episode["length"] != record["frame_count"]:
            raise RuntimeError(
                f"Frame mismatch for {identity}: metadata={episode['length']} "
                f"SAM2={record['frame_count']}"
            )
        print(
            f"[{record_index}/{len(records)}] quality {record['video_key']} "
            f"episode {episode['episode_index']}",
            flush=True,
        )
        prompt_objects = {
            str(item["name"]): item for item in record["prompt"]["objects"]
        }
        prompt_ids = {
            label: int(item["id"]) for label, item in prompt_objects.items()
        }
        per_video = {
            "video_key": record["video_key"],
            "episode_index": episode["episode_index"],
            "frame_count": record["frame_count"],
            "classes": {},
        }
        destination = slice(episode["start"], episode["end"])
        for class_id, label in enumerate(labels, start=1):
            stream = stream_name(record["video_key"], label)
            aggregate = streams.setdefault(stream, allocate_metrics(total_frames))
            thresholds = thresholds_for(label)
            metrics, _ = evaluate_class(
                record, class_id, thresholds, prompt_objects.get(label)
            )
            metrics["episode_index"] = np.full(
                record["frame_count"], episode["episode_index"], dtype=np.int32
            )
            metrics["frame_index"] = np.arange(record["frame_count"], dtype=np.int32)
            for name, values in metrics.items():
                aggregate[name][destination] = values

            available = metrics["bidirectional_available"]
            expected_visible_frames = int(np.count_nonzero(metrics["visible_expected"]))
            missing_visible_frames = int(
                np.count_nonzero(metrics["missing_when_expected"])
            )
            missing_visible_fraction = (
                missing_visible_frames / expected_visible_frames
                if expected_visible_frames
                else 0.0
            )
            longest_missing_visible_run = longest_true_run(
                metrics["missing_when_expected"]
            )
            completeness_failed = (
                missing_visible_fraction
                > float(thresholds["max_missing_visible_fraction"])
                or longest_missing_visible_run
                > int(thresholds["max_missing_visible_run"])
            )
            class_summary = {
                "prompted": label in prompt_ids,
                "quality_score_mean": float(np.mean(metrics["quality_score"])),
                "uncertain_fraction": float(np.mean(metrics["uncertain"])),
                "bidirectional_coverage": float(np.mean(available)),
                "bidirectional_iou_mean": (
                    float(np.mean(metrics["bidirectional_iou"][available]))
                    if np.any(available)
                    else None
                ),
                "expected_visible_frames": expected_visible_frames,
                "missing_visible_frames": missing_visible_frames,
                "missing_visible_fraction": missing_visible_fraction,
                "longest_missing_visible_run": longest_missing_visible_run,
                "completeness_failed": completeness_failed,
            }
            per_video["classes"][label] = class_summary
            if completeness_failed:
                completeness_failures.append(
                    {
                        "video_key": record["video_key"],
                        "episode_index": episode["episode_index"],
                        "label": label,
                        "missing_visible_fraction": missing_visible_fraction,
                        "longest_missing_visible_run": longest_missing_visible_run,
                    }
                )
            uncertain_indices = np.flatnonzero(metrics["uncertain"])
            for frame_idx in uncertain_indices:
                bidi = metrics["bidirectional_iou"][frame_idx]
                reviews[stream].append(
                    {
                        "score": float(metrics["quality_score"][frame_idx]),
                        "temporal_iou": float(metrics["temporal_iou"][frame_idx]),
                        "bidirectional_iou": float(bidi) if math.isfinite(float(bidi)) else 1.0,
                        "episode_index": episode["episode_index"],
                        "frame_index": int(frame_idx),
                        "video_path": record["video_path"],
                        "mask_path": record["root"] / "mask_frames" / f"{frame_idx:05d}.png",
                        "class_id": class_id,
                    }
                )
        video_summaries.append(per_video)

    summary = {
        "dataset_root": str(dataset_root),
        "seg_root": str(seg_root),
        "base_thresholds": BASE_THRESHOLDS,
        "mask_keys": sorted(streams),
        "processed_videos": len(records),
        "processed_frames_per_stream": total_frames,
        "streams": {},
        "completeness_failures": completeness_failures,
        "warning": (
            "Quality thresholds are diagnostic. Bidirectional IoU is evaluated only "
            "where independent forward and reverse propagation ranges overlap."
        ),
    }
    for stream, metrics in sorted(streams.items()):
        label = stream.rsplit("_", 1)[-1]
        thresholds = thresholds_for(label)
        output_path = output_root / file_name(stream)
        np.savez_compressed(output_path, **metrics)
        available = metrics["bidirectional_available"]
        failure_names = [
            "low_temporal_iou",
            "area_jump",
            "centroid_jump",
            "component_anomaly",
            "low_bidirectional_iou",
            "missing_when_expected",
        ]
        expected_visible_frames = int(np.count_nonzero(metrics["visible_expected"]))
        missing_visible_frames = int(np.count_nonzero(metrics["missing_when_expected"]))
        summary["streams"][stream] = {
            "output": str(output_path),
            "thresholds": thresholds,
            "processed_frames": total_frames,
            "quality_score_mean": float(np.mean(metrics["quality_score"])),
            "quality_score_p01": float(np.quantile(metrics["quality_score"], 0.01)),
            "uncertain_fraction": float(np.mean(metrics["uncertain"])),
            "bidirectional_coverage": float(np.mean(available)),
            "bidirectional_iou_mean": (
                float(np.mean(metrics["bidirectional_iou"][available]))
                if np.any(available)
                else None
            ),
            "failure_fractions": {
                name: float(np.mean(metrics[name])) for name in failure_names
            },
            "expected_visible_frames": expected_visible_frames,
            "missing_visible_frames": missing_visible_frames,
            "missing_visible_fraction": (
                missing_visible_frames / expected_visible_frames
                if expected_visible_frames
                else 0.0
            ),
        }
        render_review(
            stream,
            reviews[stream],
            output_root / "review" / f"{stream.replace('.', '_')}.jpg",
            args.review_count,
        )

    write_json(output_root / "summary.json", summary)
    write_json(output_root / "video_summary.json", video_summaries)
    print(f"Wrote quality report for {len(streams)} streams to {output_root}")


if __name__ == "__main__":
    main()
