#!/usr/bin/env python3
"""Convert SAM2 per-object mask frames into a YOLO dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO detect/segment labels from SAM2 object masks."
    )
    parser.add_argument("--input", type=Path, default=Path("outputs/task1"))
    parser.add_argument("--output", type=Path, default=Path("datasets/task1_yolo_detect"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="labels.txt path; defaults to input/labels.txt or labels_source in prompt JSON.",
    )
    parser.add_argument("--task", choices=["detect", "segment"], default="detect")
    parser.add_argument("--split-mode", choices=["random", "by_video"], default="random")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-video", default=None, help="Video stem for by_video split.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth frame before random splitting.",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=20)
    parser.add_argument("--min-box-size", type=int, default=2)
    parser.add_argument(
        "--quality-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject masks with implausible area or fragmentation.",
    )
    parser.add_argument(
        "--max-mask-fraction",
        type=float,
        default=0.65,
        help="Reject a mask covering more than this fraction of the image.",
    )
    parser.add_argument(
        "--min-largest-component-fraction",
        type=float,
        default=0.5,
        help="Reject a mask when its largest connected component is below this area fraction.",
    )
    parser.add_argument(
        "--max-median-area-ratio",
        type=float,
        default=6.0,
        help="Reject masks larger than this multiple of the video's median mask area.",
    )
    parser.add_argument(
        "--max-temporal-area-ratio",
        type=float,
        default=3.0,
        help="Reject masks larger than this multiple of the local temporal median.",
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
        help="Frames on each side used for the local area median.",
    )
    parser.add_argument(
        "--drop-frame-on-quality-reject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the whole frame when any non-empty mask fails a quality rule.",
    )
    parser.add_argument(
        "--keep-extracted-frames",
        action="store_true",
        help="Keep intermediate clean video frames under output/source_frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output dataset directory if it already exists.",
    )
    return parser.parse_args()


def read_labels(labels_path: Path) -> list[str]:
    labels = []
    for raw_line in labels_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        name = re.split(r"[:：]", line, maxsplit=1)[0].strip()
        if name:
            labels.append(name)
    if not labels:
        raise RuntimeError(f"No labels found in {labels_path}")
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"Duplicate label names in {labels_path}: {labels}")
    return labels


def require_fast_image_deps() -> None:
    missing = []
    for module in ("cv2", "numpy"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            f"Missing fast image dependencies: {', '.join(missing)}. "
            "Install them into the sam2 conda environment, e.g. "
            "`conda run -n sam2 python -m pip install opencv-python-headless numpy`."
        )


def find_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable:
        return executable
    environment_executable = Path(sys.executable).resolve().parent / name
    if environment_executable.is_file():
        return str(environment_executable)
    raise RuntimeError(
        f"{name} was not found; activate the configured conda environment"
    )


def mask_bbox(mask_path: Path, min_mask_pixels: int, min_box_size: int) -> tuple[int, int, int, int, int, int, int] | None:
    import cv2

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask {mask_path}")
    height, width = mask.shape[:2]
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    area = int(cv2.countNonZero(mask))
    if area < min_mask_pixels:
        return None
    x1, y1, box_w, box_h = cv2.boundingRect(points)
    if box_w < min_box_size or box_h < min_box_size:
        return None
    x2 = x1 + box_w - 1
    y2 = y1 + box_h - 1
    return width, height, x1, y1, x2, y2, area


def mask_metrics(mask_path: Path) -> dict | None:
    import cv2
    import numpy as np

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask {mask_path}")
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    area = int(cv2.countNonZero(binary))
    if area == 0:
        return None
    points = cv2.findNonZero(binary)
    x1, y1, box_w, box_h = cv2.boundingRect(points)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    largest_component = (
        int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0
    )
    return {
        "width": width,
        "height": height,
        "area": area,
        "bbox": (width, height, x1, y1, x1 + box_w - 1, y1 + box_h - 1, area),
        "box_width": box_w,
        "box_height": box_h,
        "mask_fraction": area / (width * height),
        "largest_component_fraction": largest_component / area,
    }


def mask_polygon_label(mask_path: Path, class_id: int, min_mask_pixels: int, min_box_size: int) -> str | None:
    import cv2

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask {mask_path}")
    if int((mask > 0).sum()) < min_mask_pixels:
        return None
    contours, _ = cv2.findContours((mask > 0).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w < min_box_size or h < min_box_size:
        return None
    epsilon = 0.002 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(contour) < 3:
        return None
    height, width = mask.shape[:2]
    coords = []
    for px, py in contour:
        coords.extend([f"{px / width:.6f}", f"{py / height:.6f}"])
    return f"{class_id} " + " ".join(coords)


def yolo_box_line(class_id: int, bbox: tuple[int, int, int, int, int, int, int]) -> str:
    width, height, x1, y1, x2, y2, _ = bbox
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    x_center = x1 + box_w / 2
    y_center = y1 + box_h / 2
    return (
        f"{class_id} "
        f"{x_center / width:.6f} {y_center / height:.6f} "
        f"{box_w / width:.6f} {box_h / height:.6f}"
    )


def extract_video_frames(video_path: Path, frame_dir: Path, expected_count: int) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("*.jpg"))
    if len(existing) == expected_count:
        return
    for old in existing:
        old.unlink()
    cmd = [
        find_executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-q:v",
        "2",
        str(frame_dir / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    extracted = sorted(frame_dir.glob("*.jpg"))
    if len(extracted) != expected_count:
        raise RuntimeError(
            f"Extracted {len(extracted)} frames from {video_path}, expected {expected_count}"
        )


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def discover_videos(input_dir: Path) -> list[dict]:
    videos = []
    for prompt_path in sorted(
        input_dir.rglob("file-*_sam2_masks/sam2_prompts.json")
    ):
        data = json.loads(prompt_path.read_text(encoding="utf-8"))
        video_path = Path(data["video_path"])
        if not video_path.is_file():
            raise RuntimeError(f"Source video does not exist: {video_path}")
        obj_to_name = {int(obj["id"]): obj["name"] for obj in data["objects"]}
        relative_output = prompt_path.parent.relative_to(input_dir)
        video_name = slug("__".join(relative_output.parts).replace("_sam2_masks", ""))
        videos.append(
            {
                "name": video_name,
                "prompt_path": prompt_path,
                "mask_dir": prompt_path.parent / "object_mask_frames",
                "video_path": video_path,
                "frame_count": int(data["frame_count"]),
                "obj_to_name": obj_to_name,
                "task": data.get("task", prompt_path.parent.parent.name),
                "video_key": data.get("video_key", ""),
                "labels_source": data.get("labels_source"),
            }
        )
    if not videos:
        raise RuntimeError(f"No sam2_prompts.json files found under {input_dir}")
    return videos


def resolve_labels_path(
    requested_path: Path | None, input_dir: Path, videos: list[dict]
) -> Path:
    candidates = []
    if requested_path is not None:
        candidates.append(requested_path)
    candidates.append(input_dir / "labels.txt")
    candidates.extend(
        Path(value)
        for value in (video.get("labels_source") for video in videos)
        if value
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError(
        "Could not find labels.txt; pass --labels or include labels_source in prompt JSON."
    )


def choose_split(records: list[dict], args: argparse.Namespace) -> None:
    if args.split_mode == "random":
        rng = random.Random(args.seed)
        shuffled = records[:]
        rng.shuffle(shuffled)
        train_count = int(round(len(shuffled) * args.train_ratio))
        train_keys = {record["key"] for record in shuffled[:train_count]}
        for record in records:
            record["split"] = "train" if record["key"] in train_keys else "val"
        return

    video_names = sorted({record["video"] for record in records})
    val_video = args.val_video or video_names[-1]
    if val_video not in video_names:
        raise RuntimeError(f"--val-video {val_video!r} not found in {video_names}")
    for record in records:
        record["split"] = "val" if record["video"] == val_video else "train"


def write_data_yaml(output_dir: Path, labels: list[str], task: str) -> None:
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(labels)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(labels))
    if task == "segment":
        lines.append("task: segment")
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def quality_rejection_reason(
    metrics: dict,
    global_median: float,
    local_median: float | None,
    args: argparse.Namespace,
) -> str | None:
    if not args.quality_filter:
        return None
    if metrics["mask_fraction"] > args.max_mask_fraction:
        return "mask_fraction"
    if (
        metrics["largest_component_fraction"]
        < args.min_largest_component_fraction
    ):
        return "fragmented_mask"
    if (
        global_median > 0
        and metrics["area"] > global_median * args.max_median_area_ratio
    ):
        return "global_area_spike"
    if (
        local_median is not None
        and local_median > 0
        and metrics["area"] > local_median * args.max_temporal_area_ratio
    ):
        return "temporal_area_spike"
    return None


def build_video_annotations(
    video: dict,
    class_to_id: dict[str, int],
    args: argparse.Namespace,
    skipped_masks: Counter,
) -> tuple[dict[int, list[str]], dict[int, Counter], set[int], list[dict]]:
    mask_records = []
    areas_by_object: dict[int, dict[int, int]] = defaultdict(dict)
    for mask_path in sorted(video["mask_dir"].glob("*_obj*.png")):
        frame_text, obj_text = mask_path.stem.split("_obj")
        frame_idx = int(frame_text)
        obj_id = int(obj_text)
        obj_name = video["obj_to_name"].get(obj_id)
        if obj_name not in class_to_id:
            skipped_masks["unknown_class"] += 1
            continue
        metrics = mask_metrics(mask_path)
        if metrics is None:
            skipped_masks["empty"] += 1
            continue
        if (
            metrics["area"] < args.min_mask_pixels
            or metrics["box_width"] < args.min_box_size
            or metrics["box_height"] < args.min_box_size
        ):
            skipped_masks["tiny"] += 1
            continue
        record = {
            "path": mask_path,
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "obj_name": obj_name,
            "metrics": metrics,
        }
        mask_records.append(record)
        areas_by_object[obj_id][frame_idx] = metrics["area"]

    medians = {
        obj_id: statistics.median(frame_areas.values())
        for obj_id, frame_areas in areas_by_object.items()
    }
    labels_by_frame: dict[int, list[str]] = defaultdict(list)
    counts_by_frame: dict[int, Counter] = defaultdict(Counter)
    rejected_frames: set[int] = set()
    rejected_records = []

    for record in mask_records:
        frame_idx = record["frame_idx"]
        obj_id = record["obj_id"]
        frame_areas = areas_by_object[obj_id]
        neighbor_areas = [
            area
            for neighbor_idx, area in frame_areas.items()
            if abs(neighbor_idx - frame_idx) <= args.temporal_window
        ]
        local_median = (
            statistics.median(neighbor_areas) if len(neighbor_areas) >= 3 else None
        )
        reason = quality_rejection_reason(
            record["metrics"], medians[obj_id], local_median, args
        )
        if reason:
            skipped_masks[reason] += 1
            rejected_records.append(
                {
                    "frame_idx": frame_idx,
                    "object": record["obj_name"],
                    "reason": reason,
                    "area": record["metrics"]["area"],
                    "mask_fraction": round(record["metrics"]["mask_fraction"], 6),
                    "largest_component_fraction": round(
                        record["metrics"]["largest_component_fraction"], 6
                    ),
                }
            )
            if args.drop_frame_on_quality_reject:
                rejected_frames.add(frame_idx)
            continue

        class_id = class_to_id[record["obj_name"]]
        if args.task == "detect":
            label_line = yolo_box_line(class_id, record["metrics"]["bbox"])
        else:
            label_line = mask_polygon_label(
                record["path"],
                class_id,
                args.min_mask_pixels,
                args.min_box_size,
            )
            if label_line is None:
                skipped_masks["invalid_polygon"] += 1
                if args.drop_frame_on_quality_reject:
                    rejected_frames.add(frame_idx)
                continue
        labels_by_frame[frame_idx].append(label_line)
        counts_by_frame[frame_idx][record["obj_name"]] += 1

    return labels_by_frame, counts_by_frame, rejected_frames, rejected_records


def main() -> None:
    args = parse_args()
    require_fast_image_deps()
    if not 0 < args.train_ratio < 1:
        raise RuntimeError("--train-ratio must be between 0 and 1")
    if args.frame_stride < 1:
        raise RuntimeError("--frame-stride must be at least 1")
    if args.temporal_window < 1:
        raise RuntimeError("--temporal-window must be at least 1")
    for name in (
        "max_mask_fraction",
        "min_largest_component_fraction",
    ):
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise RuntimeError(f"--{name.replace('_', '-')} must be in (0, 1]")
    for name in ("max_median_area_ratio", "max_temporal_area_ratio"):
        if getattr(args, name) <= 1:
            raise RuntimeError(f"--{name.replace('_', '-')} must be greater than 1")
    if args.output.exists():
        if not args.overwrite:
            raise RuntimeError(f"{args.output} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    videos = discover_videos(args.input)
    labels_path = resolve_labels_path(args.labels, args.input, videos)
    labels = read_labels(labels_path)
    class_to_id = {name: idx for idx, name in enumerate(labels)}

    manifest = {
        "task": args.task,
        "input": str(args.input.resolve()),
        "labels_source": str(labels_path),
        "labels": labels,
        "split_mode": args.split_mode,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "frame_stride": args.frame_stride,
        "quality_filter": args.quality_filter,
        "quality": {
            "max_mask_fraction": args.max_mask_fraction,
            "min_largest_component_fraction": args.min_largest_component_fraction,
            "max_median_area_ratio": args.max_median_area_ratio,
            "max_temporal_area_ratio": args.max_temporal_area_ratio,
            "temporal_window": args.temporal_window,
            "drop_frame_on_quality_reject": args.drop_frame_on_quality_reject,
        },
        "min_mask_pixels": args.min_mask_pixels,
        "min_box_size": args.min_box_size,
        "frames": [],
        "class_counts": Counter(),
        "skipped_masks": Counter(),
        "dropped_frames": Counter(),
        "videos": [],
    }

    records = []
    for video in videos:
        (
            video["labels_by_frame"],
            video["counts_by_frame"],
            video["rejected_frames"],
            video["quality_rejections"],
        ) = build_video_annotations(
            video, class_to_id, args, manifest["skipped_masks"]
        )
        for frame_idx in range(video["frame_count"]):
            if frame_idx % args.frame_stride:
                manifest["dropped_frames"]["stride"] += 1
                continue
            if frame_idx in video["rejected_frames"]:
                manifest["dropped_frames"]["quality"] += 1
                continue
            key = f"{video['name']}_{frame_idx:05d}"
            records.append({"key": key, "video": video["name"], "frame_idx": frame_idx})
    if len(records) < 2:
        raise RuntimeError("Fewer than two frames remain after filtering")
    choose_split(records, args)
    split_by_key = {record["key"]: record["split"] for record in records}

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    source_frames_root = args.output / "source_frames"
    temp_context = tempfile.TemporaryDirectory(prefix="sam2_yolo_frames_") if not args.keep_extracted_frames else None
    frame_root = source_frames_root if args.keep_extracted_frames else Path(temp_context.name)

    try:
        for video in videos:
            video_frame_dir = frame_root / video["name"]
            extract_video_frames(video["video_path"], video_frame_dir, video["frame_count"])

            for frame_idx in range(video["frame_count"]):
                key = f"{video['name']}_{frame_idx:05d}"
                if key not in split_by_key:
                    continue
                split = split_by_key[key]
                src = video_frame_dir / f"{frame_idx + 1:05d}.jpg"
                dst_image = args.output / "images" / split / f"{key}.jpg"
                dst_label = args.output / "labels" / split / f"{key}.txt"
                shutil.copy2(src, dst_image)
                frame_labels = video["labels_by_frame"].get(frame_idx, [])
                frame_counts = video["counts_by_frame"].get(frame_idx, Counter())
                dst_label.write_text(
                    "\n".join(frame_labels) + ("\n" if frame_labels else ""),
                    encoding="utf-8",
                )
                manifest["class_counts"].update(frame_counts)
                manifest["frames"].append(
                    {
                        "key": key,
                        "split": split,
                        "video": video["name"],
                        "source_video": str(video["video_path"]),
                        "frame_idx": frame_idx,
                        "image": str(dst_image.relative_to(args.output)),
                        "label": str(dst_label.relative_to(args.output)),
                        "objects": dict(frame_counts),
                    }
                )

            manifest["videos"].append(
                {
                    "name": video["name"],
                    "video_path": str(video["video_path"]),
                    "frame_count": video["frame_count"],
                    "task": video["task"],
                    "video_key": video["video_key"],
                    "objects": video["obj_to_name"],
                    "quality_rejections": video["quality_rejections"],
                }
            )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    manifest["class_counts"] = dict(manifest["class_counts"])
    manifest["skipped_masks"] = dict(manifest["skipped_masks"])
    manifest["dropped_frames"] = dict(manifest["dropped_frames"])
    write_data_yaml(args.output, labels, args.task)
    (args.output / "labels.txt").write_text(
        "\n".join(labels) + "\n", encoding="utf-8"
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    split_counts = Counter(frame["split"] for frame in manifest["frames"])
    print(f"Wrote {args.task} dataset to {args.output}")
    print(f"Frames: train={split_counts['train']} val={split_counts['val']} total={len(manifest['frames'])}")
    print(f"Class counts: {manifest['class_counts']}")
    print(f"Skipped masks: {manifest['skipped_masks']}")
    print(f"Dropped frames: {manifest['dropped_frames']}")


if __name__ == "__main__":
    main()
