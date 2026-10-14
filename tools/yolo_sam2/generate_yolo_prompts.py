#!/usr/bin/env python3
"""Generate reviewed-style SAM2 prompt JSON files from YOLO detections."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from label_metadata import read_label_names


DEFAULT_COLORS = [
    "#40A0FF",  # occluder
    "#FF6961",  # object
    "#77DD77",  # region
    "#FFD166",  # tool
    "#EA92C7",  # leftarm
    "#ADD665",  # rightarm
]
CANONICAL_CLASS_COLORS = {
    "occluder": "#40A0FF",
    "tube": "#FF5C8A",
    "object": "#FF6961",
    "region": "#77DD77",
    "tool": "#FFD166",
    "lefttool": "#FFD166",
    "righttool": "#65D6AD",
    "leftarm": "#EA92C7",
    "rightarm": "#ADD665",
}

# Small, fast classes keep their selected detection frame when possible. Static
# classes move first when several classes selected the same SAM2 conditioning
# frame. Shared box-prompt frames can otherwise produce a one-frame blocky
# competition artifact in SAM2's low-resolution conditioning output.
STAGGER_CLASS_PRIORITY = [
    "object",
    "tube",
    "lefttool",
    "righttool",
    "tool",
    "rightarm",
    "leftarm",
    "occluder",
    "region",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--video-key",
        action="append",
        default=[],
        help="Video feature to process; repeat for multiple views. Defaults to all views.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--side-tool-conf",
        type=float,
        default=0.50,
        help="Higher tool threshold for side-view black-background false positives.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--prompts-per-class", type=int, default=3)
    parser.add_argument(
        "--prompts-per-class-override",
        action="append",
        default=[],
        metavar="LABEL=COUNT",
        help=(
            "Override the ordinary timeline-spanning prompt count for one "
            "label; repeat for multiple labels. Contact prompts are added "
            "separately."
        ),
    )
    parser.add_argument(
        "--stagger-prompt-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Move same-frame prompts for different classes to the nearest "
            "unoccupied sampled detection frame. This avoids occasional "
            "blocky SAM2 artifacts on multi-class conditioning frames."
        ),
    )
    parser.add_argument(
        "--support-window",
        type=int,
        default=15,
        help="Maximum frame distance for temporal support.",
    )
    parser.add_argument("--support-iou", type=float, default=0.10)
    parser.add_argument(
        "--visibility-gap",
        type=int,
        default=60,
        help="Join class detections separated by at most this many frames.",
    )
    parser.add_argument(
        "--task-aware-object-filter",
        action="store_true",
        help=(
            "For known cube/paperball tasks, reject object detections whose box "
            "does not contain the task-specific object color."
        ),
    )
    parser.add_argument(
        "--object-start-padding",
        action="append",
        default=[],
        metavar="TASK=FRAMES",
        help=(
            "Extend only the first object visible range backwards for the "
            "selected task (for example cube=45). Repeat for multiple tasks. "
            "This lets SAM2 recover a partially exposed object before the "
            "detector becomes stable without relaxing other task classes."
        ),
    )
    parser.add_argument(
        "--compact-object-task",
        action="append",
        default=[],
        help=(
            "Task code whose object detections must be compact. Repeat for multiple "
            "tasks. This rejects elongated tool/arm confusers before prompt selection."
        ),
    )
    parser.add_argument(
        "--compact-object-max-elongation",
        type=float,
        default=2.5,
        help="Maximum max(width/height, height/width) for compact-object tasks.",
    )
    parser.add_argument(
        "--compact-object-max-area-fraction",
        type=float,
        default=0.02,
        help="Maximum object box area divided by image area for compact-object tasks.",
    )
    parser.add_argument(
        "--add-cross-class-negative-points",
        action="store_true",
        help=(
            "At each selected box frame, add negative center points from other "
            "detected classes. This helps keep arms, cloth, region, object, and "
            "tool mutually exclusive during SAM2 propagation."
        ),
    )
    parser.add_argument(
        "--no-cross-class-negative-label",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "Do not add detector-center negative points to this target class; "
            "repeat for multiple labels. Useful for large supporting surfaces "
            "whose box contains smaller foreground classes handled by semantic "
            "composition priority."
        ),
    )
    parser.add_argument(
        "--contact-window",
        type=int,
        default=0,
        help=(
            "Add dense detector-backed prompts before and after contact runs "
            "between object/tube and grippers or region. Zero disables it."
        ),
    )
    parser.add_argument(
        "--contact-step",
        type=int,
        default=6,
        help="Frame spacing for dense prompts inside each contact window.",
    )
    parser.add_argument(
        "--contact-distance-ratio",
        type=float,
        default=0.5,
        help=(
            "Maximum edge-to-edge box distance, normalized by the smaller "
            "box scale, that is considered contact."
        ),
    )
    parser.add_argument(
        "--max-contact-prompts-per-class",
        type=int,
        default=48,
        help="Uniform cap on additional contact prompts for each class/video.",
    )
    parser.add_argument(
        "--contact-include-occluder",
        action="store_true",
        help=(
            "Also densify occluder/object and occluder/gripper/arm contact "
            "transitions. This is useful when objects are revealed from under "
            "cloth and masks are most fragile immediately before and after contact."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--file-index",
        action="append",
        type=int,
        default=[],
        help=(
            "Only process the selected LeRobot video file index; repeat for "
            "multiple episodes. Intended for a reviewed pilot batch."
        ),
    )
    return parser.parse_args()


def read_labels(path: Path) -> list[str]:
    return read_label_names(path)


def parse_object_start_padding(values: list[str]) -> dict[str, int]:
    padding: dict[str, int] = {}
    for value in values:
        task, separator, frames_text = value.partition("=")
        task = task.strip()
        if not separator or not task:
            raise RuntimeError(
                f"Invalid --object-start-padding {value!r}; expected TASK=FRAMES"
            )
        try:
            frames = int(frames_text)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid --object-start-padding {value!r}; FRAMES must be an integer"
            ) from exc
        if frames < 0:
            raise RuntimeError("Object start padding must be non-negative")
        padding[task] = frames
    return padding


def parse_prompt_count_overrides(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        label, separator, count_text = value.partition("=")
        label = label.strip()
        if not separator or not label:
            raise RuntimeError(
                f"Invalid --prompts-per-class-override {value!r}; expected LABEL=COUNT"
            )
        try:
            count = int(count_text)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid --prompts-per-class-override {value!r}; COUNT must be an integer"
            ) from exc
        if count < 1:
            raise RuntimeError("Prompt-count overrides must be positive")
        counts[label] = count
    return counts


def pad_first_visible_range(
    ranges: list[list[int]], tasks: list[str], padding_by_task: dict[str, int]
) -> list[list[int]]:
    if not ranges:
        return ranges
    padding = max((padding_by_task.get(task, 0) for task in tasks), default=0)
    if padding <= 0:
        return ranges
    padded = [list(item) for item in ranges]
    padded[0][0] = max(0, int(padded[0][0]) - padding)
    return padded


def video_tasks(dataset_root: Path, video_keys: list[str]) -> dict[tuple[str, int, int], list[str]]:
    import pandas as pd

    mapping: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    paths = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        return {}
    columns = ["tasks"]
    for video_key in video_keys:
        columns.extend(
            [
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
            ]
        )
    for path in paths:
        table = pd.read_parquet(path, columns=columns)
        for record in table.to_dict(orient="records"):
            tasks = record["tasks"]
            if isinstance(tasks, str):
                tasks = [tasks]
            for video_key in video_keys:
                key = (
                    video_key,
                    int(record[f"videos/{video_key}/chunk_index"]),
                    int(record[f"videos/{video_key}/file_index"]),
                )
                mapping[key].update(str(task) for task in tasks)
    return {key: sorted(tasks) for key, tasks in mapping.items()}


def discover_videos(dataset_root: Path, requested_keys: list[str]) -> list[dict[str, Any]]:
    videos_root = dataset_root / "videos"
    if not videos_root.is_dir():
        raise RuntimeError(f"Missing videos directory: {videos_root}")
    available = sorted(path.name for path in videos_root.iterdir() if path.is_dir())
    keys = requested_keys or available
    missing = sorted(set(keys) - set(available))
    if missing:
        raise RuntimeError(f"Unknown video keys {missing}; available keys: {available}")
    tasks_by_video = video_tasks(dataset_root, keys)

    records = []
    for video_key in keys:
        for video_path in sorted((videos_root / video_key).glob("chunk-*/*.mp4")):
            chunk_text = video_path.parent.name.removeprefix("chunk-")
            file_text = video_path.stem.removeprefix("file-")
            records.append(
                {
                    "video_key": video_key,
                    "video_path": video_path.resolve(),
                    "chunk_index": int(chunk_text),
                    "file_index": int(file_text),
                    "tasks": tasks_by_video.get(
                        (video_key, int(chunk_text), int(file_text)), []
                    ),
                }
            )
    if not records:
        raise RuntimeError(f"No videos found under {videos_root}")
    return records


def video_metadata(video_path: Path) -> tuple[int, int, int, float]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}")
    return frame_count, width, height, fps


def sampled_frames(
    video_path: Path, frame_count: int, stride: int
) -> list[tuple[int, Any]]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    wanted = set(range(0, frame_count, stride))
    wanted.add(frame_count - 1)
    frames = []
    try:
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx in wanted:
                frames.append((frame_idx, frame))
            frame_idx += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Failed to decode frames from {video_path}")
    return frames


def box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def normalized_box_gap(first: list[float], second: list[float]) -> float:
    """Return edge distance normalized by the smaller box's linear scale."""
    dx = max(float(first[0]) - float(second[2]), float(second[0]) - float(first[2]), 0.0)
    dy = max(float(first[1]) - float(second[3]), float(second[1]) - float(first[3]), 0.0)
    first_area = max(1.0, (float(first[2]) - float(first[0])) * (float(first[3]) - float(first[1])))
    second_area = max(1.0, (float(second[2]) - float(second[0])) * (float(second[3]) - float(second[1])))
    scale = math.sqrt(min(first_area, second_area))
    return math.hypot(dx, dy) / scale


def _nearest_candidate(
    candidates: list[dict[str, Any]], target: int, maximum_offset: int
) -> dict[str, Any] | None:
    nearby = [
        item for item in candidates
        if abs(int(item["frame"]) - target) <= maximum_offset
    ]
    if not nearby:
        return None
    return min(
        nearby,
        key=lambda item: (
            abs(int(item["frame"]) - target),
            -float(item["confidence"]),
        ),
    )


def contact_prompt_boxes(
    candidates: dict[int, list[dict[str, Any]]],
    labels: list[str],
    frame_count: int,
    sample_stride: int,
    window: int,
    step: int,
    distance_ratio: float,
    maximum_per_class: int,
    include_occluder: bool = False,
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Select dense prompts around gripper/object and object/region contact runs."""
    if window <= 0:
        return {}, []
    name_to_id = {name: index for index, name in enumerate(labels)}
    pairs = []
    for subject in ("object", "tube"):
        if subject not in name_to_id:
            continue
        for other in ("lefttool", "righttool", "tool", "region"):
            if other in name_to_id:
                pairs.append((name_to_id[subject], name_to_id[other]))
        if include_occluder and "occluder" in name_to_id:
            pairs.append((name_to_id[subject], name_to_id["occluder"]))
    if include_occluder and "occluder" in name_to_id:
        for other in ("lefttool", "righttool", "tool"):
            if other in name_to_id:
                pairs.append((name_to_id["occluder"], name_to_id[other]))

    selected: dict[int, list[dict[str, Any]]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    for first_id, second_id in pairs:
        first_by_frame = {int(item["frame"]): item for item in candidates.get(first_id, [])}
        second_by_frame = {int(item["frame"]): item for item in candidates.get(second_id, [])}
        shared = sorted(set(first_by_frame) & set(second_by_frame))
        close = [
            frame for frame in shared
            if normalized_box_gap(first_by_frame[frame]["box"], second_by_frame[frame]["box"])
            <= distance_ratio
        ]
        if not close:
            continue
        runs = [[close[0], close[0]]]
        for frame in close[1:]:
            # If two proximity runs are closer than the two boundary windows,
            # their dense context already overlaps. Treat them as one contact
            # episode instead of spending the per-class cap on detector
            # flicker inside a stable grasp.
            if frame - runs[-1][1] <= max(sample_stride * 2, step, window * 2):
                runs[-1][1] = frame
            else:
                runs.append([frame, frame])
        for start, end in runs:
            # Densify around *changes* in contact state. For a long grasp or
            # placement, filling every interior frame wastes conditioning
            # slots and can crowd out the moment where masks tend to swap.
            start_targets = range(
                max(0, start - window),
                min(frame_count - 1, start + window) + 1,
                step,
            )
            end_targets = range(
                max(0, end - window),
                min(frame_count - 1, end + window) + 1,
                step,
            )
            dense_targets = sorted(set(start_targets) | set(end_targets))
            for class_id in (first_id, second_id):
                for target in dense_targets:
                    item = _nearest_candidate(
                        candidates.get(class_id, []), target, max(window, sample_stride * 2)
                    )
                    if item is not None:
                        selected[class_id].append(item)
            events.append({
                "classes": [labels[first_id], labels[second_id]],
                "contact_start": start,
                "contact_end": end,
                "window_start": max(0, start - window),
                "window_end": min(frame_count - 1, end + window),
            })

    for class_id, items in list(selected.items()):
        by_frame = {int(item["frame"]): item for item in items}
        ordered = [by_frame[frame] for frame in sorted(by_frame)]
        if maximum_per_class > 0 and len(ordered) > maximum_per_class:
            import numpy as np

            indices = np.linspace(0, len(ordered) - 1, maximum_per_class, dtype=np.int64)
            ordered = [ordered[int(index)] for index in indices]
        selected[class_id] = ordered
    return dict(selected), events


def has_temporal_support(
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    window: int,
    minimum_iou: float,
) -> bool:
    return any(
        other is not candidate
        and abs(other["frame"] - candidate["frame"]) <= window
        and box_iou(other["box"], candidate["box"]) >= minimum_iou
        for other in candidates
    )


def temporally_supported_candidates(
    candidates: list[dict[str, Any]],
    window: int,
    minimum_iou: float,
) -> list[dict[str, Any]]:
    return [
        item
        for item in candidates
        if has_temporal_support(item, candidates, window, minimum_iou)
    ]


def select_prompt_boxes(
    candidates: list[dict[str, Any]],
    count: int,
    frame_count: int,
    support_window: int,
    support_iou: float,
) -> list[dict[str, Any]]:
    supported = temporally_supported_candidates(
        candidates, support_window, support_iou
    )
    pool = supported or candidates
    ordered = sorted(pool, key=lambda item: item["frame"])
    if len(ordered) <= count:
        return ordered

    first_frame = ordered[0]["frame"]
    last_frame = ordered[-1]["frame"]
    if count == 1:
        targets = [(first_frame + last_frame) / 2.0]
    else:
        targets = [
            first_frame + (last_frame - first_frame) * index / (count - 1)
            for index in range(count)
        ]

    selected = []
    for target in targets:
        remaining = [item for item in ordered if item not in selected]
        nearby = [
            item
            for item in remaining
            if abs(item["frame"] - target) <= max(1, support_window)
        ]
        candidates_for_target = nearby or remaining
        selected.append(
            max(
                candidates_for_target,
                key=lambda item: (
                    -abs(item["frame"] - target),
                    item["confidence"],
                ),
            )
        )
    return sorted(selected, key=lambda item: item["frame"])


def stagger_colliding_prompt_frames(
    objects: list[dict[str, Any]],
    candidates: dict[int, list[dict[str, Any]]],
    maximum_offset: int,
) -> list[dict[str, Any]]:
    """Give every class box prompt a distinct, detector-observed frame."""

    rank = {name: index for index, name in enumerate(STAGGER_CLASS_PRIORITY)}
    ordered_objects = sorted(
        objects,
        key=lambda item: (rank.get(str(item["name"]), len(rank)), int(item["id"])),
    )
    occupied: set[int] = set()
    moves: list[dict[str, Any]] = []
    for raw_obj in ordered_objects:
        class_id = int(raw_obj["id"]) - 1
        original_boxes = sorted(raw_obj["boxes"], key=lambda item: int(item["frame"]))
        original_frames = {int(item["frame"]) for item in original_boxes}
        used_by_class: set[int] = set()
        adjusted = []
        for box_item in original_boxes:
            original_frame = int(box_item["frame"])
            replacement = box_item
            if original_frame in occupied:
                alternatives = [
                    item
                    for item in candidates.get(class_id, [])
                    if int(item["frame"]) not in occupied
                    and int(item["frame"]) not in used_by_class
                    and int(item["frame"]) not in original_frames
                    and abs(int(item["frame"]) - original_frame) <= maximum_offset
                ]
                if alternatives:
                    replacement = min(
                        alternatives,
                        key=lambda item: (
                            abs(int(item["frame"]) - original_frame),
                            -float(item["confidence"]),
                            int(item["frame"]),
                        ),
                    )
            replacement = dict(replacement)
            adjusted.append(replacement)
            replacement_frame = int(replacement["frame"])
            used_by_class.add(replacement_frame)
            occupied.add(replacement_frame)
            if replacement_frame != original_frame:
                moves.append(
                    {
                        "class": str(raw_obj["name"]),
                        "from_frame": original_frame,
                        "to_frame": replacement_frame,
                    }
                )
        raw_obj["boxes"] = sorted(adjusted, key=lambda item: int(item["frame"]))
    return moves


def visibility_ranges(
    candidates: list[dict[str, Any]],
    frame_count: int,
    sample_stride: int,
    maximum_gap: int,
) -> list[list[int]]:
    frames = sorted({int(item["frame"]) for item in candidates})
    if not frames:
        return []
    groups = [[frames[0], frames[0]]]
    for frame_idx in frames[1:]:
        if frame_idx - groups[-1][1] <= maximum_gap:
            groups[-1][1] = frame_idx
        else:
            groups.append([frame_idx, frame_idx])
    expanded = [
        [max(0, start - sample_stride), min(frame_count - 1, end + sample_stride)]
        for start, end in groups
    ]
    merged: list[list[int]] = []
    for start, end in expanded:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def class_threshold(
    video_key: str, label: str, base_conf: float, side_tool_conf: float
) -> float:
    if video_key.endswith(".side") and label == "tool":
        return max(base_conf, side_tool_conf)
    return base_conf


def task_object_color_supported(
    frame: Any,
    box: list[float],
    tasks: list[str],
) -> bool:
    if len(tasks) != 1 or tasks[0] not in {"cube", "paperball"}:
        return True

    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(math.floor(x1))))
    y1 = max(0, min(height - 1, int(math.floor(y1))))
    x2 = max(x1 + 1, min(width, int(math.ceil(x2))))
    y2 = max(y1 + 1, min(height, int(math.ceil(y2))))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if tasks[0] == "cube":
        target = (
            ((hue < 12) | (hue > 165))
            & (saturation > 90)
            & (value > 70)
        )
        return float(np.mean(target)) >= 0.15

    target = (
        (hue >= 8)
        & (hue <= 45)
        & (saturation > 20)
        & (value > 60)
    )
    return float(np.mean(target)) >= 0.20


def compact_object_shape_supported(
    box: list[float],
    frame_width: int,
    frame_height: int,
    maximum_elongation: float,
    maximum_area_fraction: float,
) -> bool:
    """Reject slender tools/arms and large cloth regions mislabeled as a small object."""
    x1, y1, x2, y2 = box
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    if width <= 0.0 or height <= 0.0:
        return False
    elongation = max(width / height, height / width)
    area_fraction = (width * height) / max(1.0, float(frame_width * frame_height))
    return (
        elongation <= maximum_elongation
        and area_fraction <= maximum_area_fraction
    )


def detect_candidates(
    model: Any,
    sampled: list[tuple[int, Any]],
    labels: list[str],
    video_key: str,
    tasks: list[str],
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for offset in range(0, len(sampled), args.batch):
        batch = sampled[offset : offset + args.batch]
        results = model.predict(
            source=[frame for _, frame in batch],
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        for (frame_idx, frame), result in zip(batch, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            best_in_frame: dict[int, dict[str, Any]] = {}
            for box, class_id, confidence in zip(
                boxes.xyxy.detach().cpu().tolist(),
                boxes.cls.detach().cpu().tolist(),
                boxes.conf.detach().cpu().tolist(),
            ):
                class_id = int(class_id)
                if not 0 <= class_id < len(labels):
                    continue
                threshold = class_threshold(
                    video_key,
                    labels[class_id],
                    args.conf,
                    args.side_tool_conf,
                )
                if confidence < threshold:
                    continue
                item = {
                    "frame": frame_idx,
                    "box": [round(float(value), 3) for value in box],
                    "confidence": round(float(confidence), 6),
                }
                if (
                    args.task_aware_object_filter
                    and labels[class_id] == "object"
                    and not task_object_color_supported(frame, item["box"], tasks)
                ):
                    continue
                if (
                    labels[class_id] == "object"
                    and len(tasks) == 1
                    and tasks[0] in set(args.compact_object_task)
                    and not compact_object_shape_supported(
                        item["box"],
                        frame.shape[1],
                        frame.shape[0],
                        args.compact_object_max_elongation,
                        args.compact_object_max_area_fraction,
                    )
                ):
                    continue
                if (
                    class_id not in best_in_frame
                    or item["confidence"] > best_in_frame[class_id]["confidence"]
                ):
                    best_in_frame[class_id] = item
            for class_id, item in best_in_frame.items():
                candidates[class_id].append(item)
    return candidates


def prompt_path_for(record: dict[str, Any], output_root: Path) -> Path:
    return (
        output_root
        / record["video_key"]
        / f"chunk-{record['chunk_index']:03d}"
        / f"file-{record['file_index']:03d}"
        / "sam2_prompts.json"
    )


def cross_class_negative_points(
    class_id: int,
    boxes: list[dict[str, Any]],
    candidates: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Use other-class detections on prompt frames as explicit negatives."""
    by_class_and_frame = {
        (other_id, int(item["frame"])): item
        for other_id, items in candidates.items()
        if other_id != class_id
        for item in items
    }
    points = []
    seen = set()
    for box_item in boxes:
        frame = int(box_item["frame"])
        for other_id in sorted(candidates):
            if other_id == class_id:
                continue
            other = by_class_and_frame.get((other_id, frame))
            if other is None:
                continue
            x1, y1, x2, y2 = other["box"]
            x = round((float(x1) + float(x2)) / 2.0, 3)
            y = round((float(y1) + float(y2)) / 2.0, 3)
            key = (frame, x, y)
            if key in seen:
                continue
            seen.add(key)
            points.append({"frame": frame, "x": x, "y": y, "label": 0})
    return points


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.batch < 1 or args.prompts_per_class < 1:
        raise RuntimeError("stride, batch, and prompts-per-class must be positive")
    if args.contact_window < 0 or args.contact_distance_ratio < 0:
        raise RuntimeError("contact window and distance ratio must be non-negative")
    if args.contact_step < 1 or args.max_contact_prompts_per_class < 1:
        raise RuntimeError("contact step and maximum prompt count must be positive")
    if not 0 < args.conf <= 1 or not 0 < args.side_tool_conf <= 1:
        raise RuntimeError("confidence thresholds must be in (0, 1]")
    if not args.yolo_weights.is_file():
        raise RuntimeError(f"YOLO weights not found: {args.yolo_weights}")
    object_start_padding = parse_object_start_padding(args.object_start_padding)
    prompt_count_overrides = parse_prompt_count_overrides(
        args.prompts_per_class_override
    )

    labels = read_labels(args.labels)
    unknown_prompt_labels = sorted(set(prompt_count_overrides) - set(labels))
    if unknown_prompt_labels:
        raise RuntimeError(
            f"Unknown --prompts-per-class-override labels: {unknown_prompt_labels}"
        )
    unknown_negative_labels = sorted(set(args.no_cross_class_negative_label) - set(labels))
    if unknown_negative_labels:
        raise RuntimeError(
            f"Unknown --no-cross-class-negative-label values: {unknown_negative_labels}"
        )
    records = discover_videos(args.dataset_root.resolve(), args.video_key)
    if args.file_index:
        selected = set(args.file_index)
        records = [record for record in records if record["file_index"] in selected]
        found = {record["file_index"] for record in records}
        missing = sorted(selected - found)
        if missing:
            raise RuntimeError(f"Selected file indices not found: {missing}")
    if args.limit > 0:
        records = records[: args.limit]

    from ultralytics import YOLO

    model = YOLO(str(args.yolo_weights.resolve()))
    model_names = [str(model.names[index]) for index in sorted(model.names)]
    if model_names != labels:
        raise RuntimeError(
            f"YOLO classes {model_names} do not match labels.txt classes {labels}"
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = []
    for index, record in enumerate(records, start=1):
        prompt_path = prompt_path_for(record, output_root)
        if prompt_path.is_file() and not args.overwrite:
            print(f"[{index}/{len(records)}] SKIP {record['video_path']}")
            manifest.append(json.loads(prompt_path.read_text(encoding="utf-8")))
            continue
        print(f"[{index}/{len(records)}] YOLO {record['video_path']}", flush=True)
        try:
            frame_count, width, height, fps = video_metadata(record["video_path"])
            sampled = sampled_frames(
                record["video_path"], frame_count, args.sample_stride
            )
            candidates = detect_candidates(
                model,
                sampled,
                labels,
                record["video_key"],
                record["tasks"],
                args,
            )
            contact_boxes, contact_events = contact_prompt_boxes(
                candidates,
                labels,
                frame_count,
                args.sample_stride,
                args.contact_window,
                args.contact_step,
                args.contact_distance_ratio,
                args.max_contact_prompts_per_class,
                args.contact_include_occluder,
            )
            objects = []
            for class_id, label in enumerate(labels):
                class_candidates = candidates.get(class_id, [])
                if not class_candidates:
                    continue
                supported_candidates = temporally_supported_candidates(
                    class_candidates,
                    args.support_window,
                    args.support_iou,
                )
                tracking_candidates = supported_candidates or class_candidates
                boxes = select_prompt_boxes(
                    tracking_candidates,
                    prompt_count_overrides.get(label, args.prompts_per_class),
                    frame_count,
                    args.support_window,
                    args.support_iou,
                )
                # Preserve the ordinary timeline-spanning prompts and add
                # detector-backed frames densely around contact transitions.
                # One box per frame keeps SAM2's conditioning unambiguous.
                boxes_by_frame = {int(item["frame"]): item for item in boxes}
                for item in contact_boxes.get(class_id, []):
                    frame = int(item["frame"])
                    previous = boxes_by_frame.get(frame)
                    if previous is None or float(item["confidence"]) > float(
                        previous["confidence"]
                    ):
                        boxes_by_frame[frame] = item
                boxes = [boxes_by_frame[frame] for frame in sorted(boxes_by_frame)]
                visible_ranges = visibility_ranges(
                    tracking_candidates,
                    frame_count,
                    args.sample_stride,
                    args.visibility_gap,
                )
                if label == "object":
                    visible_ranges = pad_first_visible_range(
                        visible_ranges,
                        record["tasks"],
                        object_start_padding,
                    )
                objects.append(
                    {
                        "id": class_id + 1,
                        "name": label,
                        "color": CANONICAL_CLASS_COLORS.get(
                            label, DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]
                        ),
                        "boxes": boxes,
                        "points": [],
                        "visible_ranges": visible_ranges,
                    }
                )
            if not objects:
                raise RuntimeError("YOLO found no supported classes")
            prompt_frame_moves = (
                stagger_colliding_prompt_frames(
                    objects,
                    candidates,
                    args.support_window,
                )
                if args.stagger_prompt_frames
                else []
            )
            if args.add_cross_class_negative_points:
                for raw_obj in objects:
                    if raw_obj["name"] not in args.no_cross_class_negative_label:
                        raw_obj["points"] = cross_class_negative_points(
                            int(raw_obj["id"]) - 1,
                            raw_obj["boxes"],
                            candidates,
                        )
            data = {
                "version": 2,
                "video_path": str(record["video_path"]),
                "video_key": record["video_key"],
                "chunk_index": record["chunk_index"],
                "file_index": record["file_index"],
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "fps": fps,
                "labels_source": str(args.labels.resolve()),
                "yolo_weights": str(args.yolo_weights.resolve()),
                "yolo_conf": args.conf,
                "side_tool_conf": args.side_tool_conf,
                "sample_stride": args.sample_stride,
                "prompts_per_class": args.prompts_per_class,
                "prompt_count_overrides": prompt_count_overrides,
                "tasks": record["tasks"],
                "task_aware_object_filter": args.task_aware_object_filter,
                "object_start_padding": object_start_padding,
                "compact_object_filter": {
                    "tasks": sorted(set(args.compact_object_task)),
                    "max_elongation": args.compact_object_max_elongation,
                    "max_area_fraction": args.compact_object_max_area_fraction,
                },
                "cross_class_negative_points": args.add_cross_class_negative_points,
                "cross_class_negative_excluded_labels": sorted(
                    set(args.no_cross_class_negative_label)
                ),
                "contact_prompting": {
                    "enabled": args.contact_window > 0,
                    "window": args.contact_window,
                    "step": args.contact_step,
                    "distance_ratio": args.contact_distance_ratio,
                    "max_prompts_per_class": args.max_contact_prompts_per_class,
                    "include_occluder": args.contact_include_occluder,
                    "events": contact_events,
                    "additional_boxes_by_class": {
                        labels[class_id]: len(items)
                        for class_id, items in sorted(contact_boxes.items())
                    },
                },
                "staggered_prompt_frames": args.stagger_prompt_frames,
                "prompt_frame_moves": prompt_frame_moves,
                "objects": objects,
            }
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest.append(data)
            found = ", ".join(item["name"] for item in objects)
            print(f"  wrote {prompt_path} ({found})", flush=True)
        except Exception as exc:
            message = f"{record['video_path']}: {exc}"
            failures.append(message)
            print(f"  FAILED {message}", flush=True)

    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "labels": labels,
        "yolo_weights": str(args.yolo_weights.resolve()),
        "videos": len(manifest),
        "failures": failures,
        "records": [
            {
                "video_path": item["video_path"],
                "video_key": item["video_key"],
                "chunk_index": item["chunk_index"],
                "file_index": item["file_index"],
                "classes": [obj["name"] for obj in item["objects"]],
                "contact_events": len(
                    item.get("contact_prompting", {}).get("events", [])
                ),
            }
            for item in manifest
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(records)} videos failed:\n" + "\n".join(failures)
        )
    print(f"Generated prompts for {len(manifest)} videos under {output_root}")


if __name__ == "__main__":
    main()
