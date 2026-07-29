#!/usr/bin/env python3
"""Generate reviewed-style SAM2 prompt JSON files from YOLO detections."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_COLORS = ["#40A0FF", "#FF6961", "#77DD77", "#FFD166"]


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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


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


def discover_videos(dataset_root: Path, requested_keys: list[str]) -> list[dict[str, Any]]:
    videos_root = dataset_root / "videos"
    if not videos_root.is_dir():
        raise RuntimeError(f"Missing videos directory: {videos_root}")
    available = sorted(path.name for path in videos_root.iterdir() if path.is_dir())
    keys = requested_keys or available
    missing = sorted(set(keys) - set(available))
    if missing:
        raise RuntimeError(f"Unknown video keys {missing}; available keys: {available}")

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


def select_prompt_boxes(
    candidates: list[dict[str, Any]],
    count: int,
    frame_count: int,
    support_window: int,
    support_iou: float,
) -> list[dict[str, Any]]:
    supported = [
        item
        for item in candidates
        if has_temporal_support(item, candidates, support_window, support_iou)
    ]
    pool = supported or candidates
    ranked = sorted(pool, key=lambda item: item["confidence"], reverse=True)
    minimum_gap = max(1, frame_count // max(count * 4, 1))
    selected = []
    for item in ranked:
        if all(abs(item["frame"] - old["frame"]) >= minimum_gap for old in selected):
            selected.append(item)
            if len(selected) == count:
                break
    if len(selected) < count:
        for item in ranked:
            if item not in selected:
                selected.append(item)
                if len(selected) == count:
                    break
    return sorted(selected, key=lambda item: item["frame"])


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


def detect_candidates(
    model: Any,
    sampled: list[tuple[int, Any]],
    labels: list[str],
    video_key: str,
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
        for (frame_idx, _), result in zip(batch, results):
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


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.batch < 1 or args.prompts_per_class < 1:
        raise RuntimeError("stride, batch, and prompts-per-class must be positive")
    if not 0 < args.conf <= 1 or not 0 < args.side_tool_conf <= 1:
        raise RuntimeError("confidence thresholds must be in (0, 1]")
    if not args.yolo_weights.is_file():
        raise RuntimeError(f"YOLO weights not found: {args.yolo_weights}")

    labels = read_labels(args.labels)
    records = discover_videos(args.dataset_root.resolve(), args.video_key)
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
                model, sampled, labels, record["video_key"], args
            )
            objects = []
            for class_id, label in enumerate(labels):
                class_candidates = candidates.get(class_id, [])
                if not class_candidates:
                    continue
                boxes = select_prompt_boxes(
                    class_candidates,
                    args.prompts_per_class,
                    frame_count,
                    args.support_window,
                    args.support_iou,
                )
                objects.append(
                    {
                        "id": class_id + 1,
                        "name": label,
                        "color": DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)],
                        "boxes": boxes,
                        "points": [],
                        "visible_ranges": visibility_ranges(
                            class_candidates,
                            frame_count,
                            args.sample_stride,
                            args.visibility_gap,
                        ),
                    }
                )
            if not objects:
                raise RuntimeError("YOLO found no supported classes")
            data = {
                "version": 1,
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
