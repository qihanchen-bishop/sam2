#!/usr/bin/env python3
"""Convert SAM2 per-object mask frames into a YOLO dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO detect/segment labels from SAM2 object masks."
    )
    parser.add_argument("--input", type=Path, default=Path("outputs/task1"))
    parser.add_argument("--output", type=Path, default=Path("datasets/task1_yolo_detect"))
    parser.add_argument("--task", choices=["detect", "segment"], default="detect")
    parser.add_argument("--split-mode", choices=["random", "by_video"], default="random")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-video", default=None, help="Video stem for by_video split.")
    parser.add_argument("--min-mask-pixels", type=int, default=20)
    parser.add_argument("--min-box-size", type=int, default=2)
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
    labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    if not labels:
        raise RuntimeError(f"No labels found in {labels_path}")
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
        "ffmpeg",
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


def discover_videos(input_dir: Path) -> list[dict]:
    videos = []
    for prompt_path in sorted(input_dir.glob("file-*_sam2_masks/sam2_prompts.json")):
        data = json.loads(prompt_path.read_text())
        video_path = Path(data["video_path"])
        if not video_path.is_file():
            raise RuntimeError(f"Source video does not exist: {video_path}")
        obj_to_name = {int(obj["id"]): obj["name"] for obj in data["objects"]}
        videos.append(
            {
                "name": prompt_path.parent.name.replace("_sam2_masks", ""),
                "prompt_path": prompt_path,
                "mask_dir": prompt_path.parent / "object_mask_frames",
                "video_path": video_path,
                "frame_count": int(data["frame_count"]),
                "obj_to_name": obj_to_name,
            }
        )
    if not videos:
        raise RuntimeError(f"No sam2_prompts.json files found under {input_dir}")
    return videos


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


def main() -> None:
    args = parse_args()
    require_fast_image_deps()
    if not 0 < args.train_ratio < 1:
        raise RuntimeError("--train-ratio must be between 0 and 1")
    if args.output.exists():
        if not args.overwrite:
            raise RuntimeError(f"{args.output} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    labels = read_labels(args.input / "labels.txt")
    class_to_id = {name: idx for idx, name in enumerate(labels)}
    videos = discover_videos(args.input)

    records = []
    for video in videos:
        for frame_idx in range(video["frame_count"]):
            key = f"{video['name']}_{frame_idx:05d}"
            records.append({"key": key, "video": video["name"], "frame_idx": frame_idx})
    choose_split(records, args)
    split_by_key = {record["key"]: record["split"] for record in records}

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    source_frames_root = args.output / "source_frames"
    temp_context = tempfile.TemporaryDirectory(prefix="sam2_yolo_frames_") if not args.keep_extracted_frames else None
    frame_root = source_frames_root if args.keep_extracted_frames else Path(temp_context.name)

    manifest = {
        "task": args.task,
        "input": str(args.input.resolve()),
        "labels": labels,
        "split_mode": args.split_mode,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "min_mask_pixels": args.min_mask_pixels,
        "min_box_size": args.min_box_size,
        "frames": [],
        "class_counts": Counter(),
        "skipped_masks": Counter(),
        "videos": [],
    }

    try:
        for video in videos:
            video_frame_dir = frame_root / video["name"]
            extract_video_frames(video["video_path"], video_frame_dir, video["frame_count"])
            labels_by_frame: dict[int, list[str]] = defaultdict(list)
            counts_by_frame: dict[int, Counter] = defaultdict(Counter)

            for mask_path in sorted(video["mask_dir"].glob("*_obj*.png")):
                stem = mask_path.stem
                frame_text, obj_text = stem.split("_obj")
                frame_idx = int(frame_text)
                obj_id = int(obj_text)
                obj_name = video["obj_to_name"].get(obj_id)
                if obj_name not in class_to_id:
                    manifest["skipped_masks"]["unknown_class"] += 1
                    continue
                class_id = class_to_id[obj_name]
                if args.task == "detect":
                    bbox = mask_bbox(mask_path, args.min_mask_pixels, args.min_box_size)
                    if bbox is None:
                        manifest["skipped_masks"]["empty_or_tiny"] += 1
                        continue
                    label_line = yolo_box_line(class_id, bbox)
                else:
                    label_line = mask_polygon_label(
                        mask_path, class_id, args.min_mask_pixels, args.min_box_size
                    )
                    if label_line is None:
                        manifest["skipped_masks"]["empty_or_tiny"] += 1
                        continue
                labels_by_frame[frame_idx].append(label_line)
                counts_by_frame[frame_idx][obj_name] += 1
                manifest["class_counts"][obj_name] += 1

            for frame_idx in range(video["frame_count"]):
                key = f"{video['name']}_{frame_idx:05d}"
                split = split_by_key[key]
                src = video_frame_dir / f"{frame_idx + 1:05d}.jpg"
                dst_image = args.output / "images" / split / f"{key}.jpg"
                dst_label = args.output / "labels" / split / f"{key}.txt"
                shutil.copy2(src, dst_image)
                dst_label.write_text("\n".join(labels_by_frame.get(frame_idx, [])) + ("\n" if labels_by_frame.get(frame_idx) else ""))
                manifest["frames"].append(
                    {
                        "key": key,
                        "split": split,
                        "video": video["name"],
                        "source_video": str(video["video_path"]),
                        "frame_idx": frame_idx,
                        "image": str(dst_image.relative_to(args.output)),
                        "label": str(dst_label.relative_to(args.output)),
                        "objects": dict(counts_by_frame.get(frame_idx, {})),
                    }
                )

            manifest["videos"].append(
                {
                    "name": video["name"],
                    "video_path": str(video["video_path"]),
                    "frame_count": video["frame_count"],
                    "objects": video["obj_to_name"],
                }
            )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    manifest["class_counts"] = dict(manifest["class_counts"])
    manifest["skipped_masks"] = dict(manifest["skipped_masks"])
    write_data_yaml(args.output, labels, args.task)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    split_counts = Counter(frame["split"] for frame in manifest["frames"])
    print(f"Wrote {args.task} dataset to {args.output}")
    print(f"Frames: train={split_counts['train']} val={split_counts['val']} total={len(manifest['frames'])}")
    print(f"Class counts: {manifest['class_counts']}")
    print(f"Skipped masks: {manifest['skipped_masks']}")


if __name__ == "__main__":
    main()
