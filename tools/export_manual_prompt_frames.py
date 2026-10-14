#!/usr/bin/env python3
"""Export original frames and rendered box/point previews from manual SAM2 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--video-key", default="observation.images.front")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--mask-preview-dir", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_image(path: Path, image: np.ndarray, parameters: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image, parameters or []):
        raise RuntimeError(f"Cannot write image: {path}")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = str(value).strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid class color: {value!r}")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def frame_annotations(prompt: dict) -> dict[int, list[dict]]:
    width = int(prompt.get("width", 0))
    height = int(prompt.get("height", 0))
    result: dict[int, list[dict]] = {int(frame): [] for frame in prompt["prompt_frames"]}
    for obj in prompt["objects"]:
        color = rgb_hex_to_bgr(obj["color"])
        by_frame: dict[int, dict] = {}
        for item in obj.get("boxes", []):
            frame = int(item["frame"])
            box = [int(value) for value in item["box"]]
            if len(box) != 4:
                raise ValueError(f"Invalid box for {obj['name']} at frame {frame}")
            by_frame.setdefault(frame, {"box": None, "points": []})["box"] = box
        for item in obj.get("points", []):
            frame = int(item["frame"])
            point = {"x": int(item["x"]), "y": int(item["y"]), "label": int(item["label"])}
            if point["label"] not in (0, 1):
                raise ValueError(f"Invalid point label for {obj['name']} at frame {frame}")
            by_frame.setdefault(frame, {"box": None, "points": []})["points"].append(point)
        for frame, values in by_frame.items():
            if frame not in result:
                raise ValueError(f"Annotation frame {frame} missing from prompt_frames")
            result[frame].append({
                "id": int(obj["id"]), "name": str(obj["name"]), "color": color,
                "box": values["box"], "points": values["points"],
            })
    annotated = {frame for frame, entries in result.items() if entries}
    if annotated != set(result):
        raise ValueError(f"Empty prompt frames: {sorted(set(result) - annotated)}")
    if width and height:
        validate_coordinates(result, width, height)
    return result


def validate_coordinates(annotations: dict[int, list[dict]], width: int, height: int) -> None:
    for frame, entries in annotations.items():
        for entry in entries:
            if entry["box"] is not None:
                x1, y1, x2, y2 = entry["box"]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Out-of-range box at frame {frame}: {entry}")
            for point in entry["points"]:
                if not (0 <= point["x"] < width and 0 <= point["y"] < height):
                    raise ValueError(f"Out-of-range point at frame {frame}: {entry}")


def decode_selected_frames(video_path: Path, targets: set[int], expected_count: int) -> dict[int, np.ndarray]:
    if not targets:
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if declared and declared != expected_count:
        raise ValueError(f"Frame-count mismatch for {video_path}: prompt={expected_count}, video={declared}")
    selected: dict[int, np.ndarray] = {}
    final_target = max(targets)
    try:
        for frame_index in range(final_target + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video ended at frame {frame_index}: {video_path}")
            if frame_index in targets:
                selected[frame_index] = frame.copy()
    finally:
        capture.release()
    missing = targets - set(selected)
    if missing:
        raise ValueError(f"Frames were not decoded from {video_path}: {sorted(missing)}")
    return selected


def render_preview(frame: np.ndarray, episode: int, frame_index: int, entries: list[dict]) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    positive_count = sum(point["label"] == 1 for entry in entries for point in entry["points"])
    negative_count = sum(point["label"] == 0 for entry in entries for point in entry["points"])
    for entry in sorted(entries, key=lambda value: value["id"]):
        color = entry["color"]
        box = entry["box"]
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, 3, cv2.LINE_AA)
            (text_width, text_height), baseline = cv2.getTextSize(
                entry["name"], cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            top = max(0, y1 - text_height - baseline - 8)
            right = min(width - 1, x1 + text_width + 10)
            cv2.rectangle(canvas, (x1, top), (right, min(height - 1, top + text_height + baseline + 8)), color, -1)
            cv2.putText(canvas, entry["name"], (x1 + 5, top + text_height + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 22, 29), 2, cv2.LINE_AA)
        for point in entry["points"]:
            center = (point["x"], point["y"])
            cv2.circle(canvas, center, 11, color, -1, cv2.LINE_AA)
            inner = (117, 211, 35) if point["label"] == 1 else (93, 72, 255)
            cv2.circle(canvas, center, 7, inner, -1, cv2.LINE_AA)
            cv2.line(canvas, (center[0] - 4, center[1]), (center[0] + 4, center[1]),
                     (16, 22, 29), 2, cv2.LINE_AA)
            if point["label"] == 1:
                cv2.line(canvas, (center[0], center[1] - 4), (center[0], center[1] + 4),
                         (16, 22, 29), 2, cv2.LINE_AA)
    header = f"ep{episode:06d}  frame {frame_index:05d}  green +=positive  red -=negative  +{positive_count} -{negative_count}"
    (header_width, header_height), baseline = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(canvas, (0, 0), (min(width - 1, header_width + 16), header_height + baseline + 12), (8, 12, 17), -1)
    cv2.putText(canvas, header, (8, header_height + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (245, 248, 252), 2, cv2.LINE_AA)
    return canvas


def make_contact_sheet(previews: list[tuple[int, np.ndarray]], episode: int, title: str) -> np.ndarray:
    columns = 3
    tile_width = 400
    source_height, source_width = previews[0][1].shape[:2]
    tile_height = round(source_height * tile_width / source_width)
    title_height = 34
    rows = (len(previews) + columns - 1) // columns
    sheet = np.full((title_height + rows * tile_height, columns * tile_width, 3), 22, np.uint8)
    cv2.putText(sheet, f"ep{episode:06d} {title}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 248, 252), 2, cv2.LINE_AA)
    for index, (frame_index, preview) in enumerate(previews):
        row, column = divmod(index, columns)
        tile = cv2.resize(preview, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        y0, x0 = title_height + row * tile_height, column * tile_width
        sheet[y0:y0 + tile_height, x0:x0 + tile_width] = tile
        cv2.rectangle(sheet, (x0, y0), (x0 + tile_width - 1, y0 + tile_height - 1), (230, 230, 230), 1)
        cv2.rectangle(sheet, (x0, y0), (x0 + 118, y0 + 25), (0, 0, 0), -1)
        cv2.putText(sheet, f"frame {frame_index:05d}", (x0 + 6, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def final_mask_root(work_dir: Path, episode: int, video_key: str) -> Path:
    item_id = f"ep{episode:06d}__{video_key}"
    preview_root = work_dir / "sam2_preview_integrated" / item_id
    latest_path = preview_root / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    run_id = str(latest["run_id"])
    candidate_root = preview_root / run_id / "output" / f"file-{episode:03d}_sam2_masks"
    review_path = candidate_root / "direction_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "finalized" or not review.get("final_subdir"):
        raise ValueError(f"Episode {episode} does not have a finalized human review")
    result = candidate_root / str(review["final_subdir"])
    if not (result / "mask_frames").is_dir() or not (result / "complete.json").is_file():
        raise ValueError(f"Final mask result is incomplete: {result}")
    return result.resolve()


def render_mask_preview(mask: np.ndarray, objects: list[dict]) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Semantic mask must be single-channel, got {mask.shape}")
    allowed = {0}
    canvas = np.zeros((*mask.shape, 3), np.uint8)
    for obj in objects:
        object_id = int(obj["id"])
        allowed.add(object_id)
        canvas[mask == object_id] = rgb_hex_to_bgr(obj["color"])
    unexpected = sorted(set(int(value) for value in np.unique(mask)) - allowed)
    if unexpected:
        raise ValueError(f"Mask contains unexpected ids: {unexpected}")
    return canvas


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    image_dir = (args.image_dir or work_dir / "prompts/image").resolve()
    preview_dir = (args.preview_dir or work_dir / "prompts/preview").resolve()
    mask_preview_dir = (args.mask_preview_dir or work_dir / "prompts/maskpreview").resolve()
    records = []
    total_frames = 0
    for episode in args.episodes:
        prompt_path = work_dir / "prompts/json" / args.video_key / f"ep{episode:06d}" / "sam2_prompts.json"
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        if int(prompt["episode_index"]) != episode or prompt["video_key"] != args.video_key:
            raise ValueError(f"Prompt identity mismatch: {prompt_path}")
        annotations = frame_annotations(prompt)
        video_path = Path(prompt["video_path"]).resolve()
        decoded = decode_selected_frames(video_path, set(annotations), int(prompt["frame_count"]))
        mask_root = final_mask_root(work_dir, episode, args.video_key)
        episode_name = f"ep{episode:06d}"
        episode_previews: list[tuple[int, np.ndarray]] = []
        episode_mask_previews: list[tuple[int, np.ndarray]] = []
        output_frames = []
        for frame_index in sorted(annotations):
            original = decoded[frame_index]
            height, width = original.shape[:2]
            validate_coordinates({frame_index: annotations[frame_index]}, width, height)
            preview = render_preview(original, episode, frame_index, annotations[frame_index])
            mask_path = mask_root / "mask_frames" / f"{frame_index:05d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != original.shape[:2]:
                raise ValueError(f"Missing or invalid final mask: {mask_path}")
            mask_preview = render_mask_preview(mask, prompt["objects"])
            filename = f"frame-{frame_index:05d}.png"
            image_path = image_dir / episode_name / filename
            preview_path = preview_dir / episode_name / filename
            mask_preview_path = mask_preview_dir / episode_name / filename
            atomic_image(image_path, original, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            atomic_image(preview_path, preview, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            atomic_image(mask_preview_path, mask_preview, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            episode_previews.append((frame_index, preview))
            episode_mask_previews.append((frame_index, mask_preview))
            entries = annotations[frame_index]
            output_frames.append({
                "frame": frame_index,
                "image": str(image_path), "preview": str(preview_path),
                "mask_preview": str(mask_preview_path),
                "image_sha256": sha256(image_path), "preview_sha256": sha256(preview_path),
                "mask_preview_sha256": sha256(mask_preview_path),
                "classes": [entry["name"] for entry in entries],
                "boxes": sum(entry["box"] is not None for entry in entries),
                "positive_points": sum(point["label"] == 1 for entry in entries for point in entry["points"]),
                "negative_points": sum(point["label"] == 0 for entry in entries for point in entry["points"]),
            })
        contact_path = preview_dir / "contact_sheets" / f"{episode_name}.jpg"
        atomic_image(contact_path, make_contact_sheet(episode_previews, episode, "manual prompt frames"),
                     [cv2.IMWRITE_JPEG_QUALITY, 94])
        mask_contact_path = mask_preview_dir / "contact_sheets" / f"{episode_name}.png"
        atomic_image(mask_contact_path, make_contact_sheet(
            episode_mask_previews, episode, "final reviewed mask previews"),
            [cv2.IMWRITE_PNG_COMPRESSION, 3])
        records.append({
            "episode_index": episode, "prompt_json": str(prompt_path),
            "prompt_sha256": sha256(prompt_path), "video_path": str(video_path),
            "frame_count": int(prompt["frame_count"]), "prompt_frames": sorted(annotations),
            "final_mask_root": str(mask_root),
            "contact_sheet": str(contact_path), "mask_contact_sheet": str(mask_contact_path),
            "outputs": output_frames,
        })
        total_frames += len(annotations)
        print(f"EXPORTED {episode_name}: {len(annotations)} prompt frames", flush=True)
    manifest = {
        "workflow": "manual_sam2_prompt_frame_export_v1",
        "source": "current manually reviewed sam2_prompts.json",
        "video_key": args.video_key, "episodes": list(args.episodes),
        "image_dir": str(image_dir), "preview_dir": str(preview_dir),
        "mask_preview_dir": str(mask_preview_dir),
        "total_prompt_frames": total_frames, "records": records,
    }
    atomic_json(work_dir / "prompts/manual_prompt_export.json", manifest)
    print(json.dumps({"episodes": len(records), "frames": total_frames,
                      "image_dir": str(image_dir), "preview_dir": str(preview_dir),
                      "mask_preview_dir": str(mask_preview_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    cv2.setNumThreads(1)
    main()
