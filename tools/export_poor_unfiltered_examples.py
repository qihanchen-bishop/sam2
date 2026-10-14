#!/usr/bin/env python3
"""Export curated, unfiltered semantic-mask failures for visual comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {
    0: "background",
    1: "occluder",
    2: "object",
    3: "region",
    4: "lefttool",
    5: "righttool",
}

# OpenCV BGR. These are the canonical class colors used by the annotation UI.
PALETTE = np.asarray(
    [
        [0, 0, 0],
        [255, 160, 64],
        [97, 105, 255],
        [119, 221, 119],
        [102, 209, 255],
        [173, 214, 101],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--example",
        action="append",
        required=True,
        help="Curated example as EPISODE:FRAME:SHORT_DESCRIPTION.",
    )
    return parser.parse_args()


def parse_examples(values: list[str]) -> list[dict]:
    examples = []
    for value in values:
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Bad --example {value!r}; expected episode:frame:description")
        episode, frame, description = parts
        examples.append(
            {
                "episode": int(episode),
                "frame": int(frame),
                "description": description,
            }
        )
    return examples


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return frame


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if int(mask.max()) >= len(PALETTE):
        raise ValueError(f"Unexpected class id {int(mask.max())} in {path}")
    return mask


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    colored = colorize(mask)
    mixed = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)
    result = frame.copy()
    result[mask > 0] = mixed[mask > 0]
    return result


def add_header(image: np.ndarray, title: str, subtitle: str | None = None) -> np.ndarray:
    header_height = 70 if subtitle else 45
    canvas = np.full((image.shape[0] + header_height, image.shape[1], 3), 246, np.uint8)
    canvas[header_height:] = image
    cv2.putText(
        canvas,
        title,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            canvas,
            subtitle,
            (14, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.51,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
    return canvas


def find_record(manifest: dict, episode: int) -> dict:
    for record in manifest["records"]:
        item = record.get("item", record)
        record_episode = item.get("episode_index", record.get("episode"))
        if record_episode is not None and int(record_episode) == episode:
            return item
    raise KeyError(f"Episode {episode} is absent from manifest")


def find_quality_record(path: Path, frame_index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_frame = row.get("frame_index", row.get("frame"))
            if row_frame is not None and int(row_frame) == frame_index:
                return row
    raise KeyError(f"Frame {frame_index} is absent from {path}")


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write {path}")


def main() -> None:
    args = parse_args()
    examples = parse_examples(args.example)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model = manifest["models"]["generation1_pseudo8"]

    output_dirs = {
        name: args.output / name
        for name in ("maskvisual_raw", "original", "overlay_raw", "comparison")
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    exported = []
    contact_tiles = []
    for ordinal, example in enumerate(examples, start=1):
        episode = example["episode"]
        frame_index = example["frame"]
        record = find_record(manifest, episode)
        episode_dir = args.prediction_root / f"ep{episode:06d}"
        raw_path = episode_dir / "unet" / "raw_mask_frames" / f"{frame_index:05d}.png"
        filtered_path = episode_dir / "unet" / "mask_frames" / f"{frame_index:05d}.png"
        quality_path = (
            args.prediction_root
            / "video_quality_v2"
            / f"ep{episode:06d}"
            / "frame_scores.jsonl"
        )

        frame = read_frame(Path(record["video_path"]), frame_index)
        raw_mask = read_mask(raw_path)
        filtered_mask = read_mask(filtered_path)
        if raw_mask.shape != frame.shape[:2] or filtered_mask.shape != frame.shape[:2]:
            raise ValueError(f"Shape mismatch for episode {episode}, frame {frame_index}")

        quality = find_quality_record(quality_path, frame_index)
        stem = f"{ordinal:02d}_ep{episode:03d}_frame{frame_index:05d}"
        raw_visual = colorize(raw_mask)
        filtered_visual = colorize(filtered_mask)
        raw_overlay = overlay(frame, raw_mask)

        write_image(output_dirs["maskvisual_raw"] / f"{stem}.png", raw_visual)
        write_image(output_dirs["original"] / f"{stem}.png", frame)
        write_image(output_dirs["overlay_raw"] / f"{stem}.png", raw_overlay)

        panel_size = (600, 456)
        panels = [
            add_header(cv2.resize(frame, panel_size, interpolation=cv2.INTER_AREA), "Original RGB"),
            add_header(
                cv2.resize(raw_visual, panel_size, interpolation=cv2.INTER_NEAREST),
                "Generation 1 RAW mask (unfiltered)",
            ),
            add_header(
                cv2.resize(filtered_visual, panel_size, interpolation=cv2.INTER_NEAREST),
                "Component-filtered mask (reference)",
            ),
        ]
        comparison = np.concatenate(panels, axis=1)
        changed_pixels = int(np.count_nonzero(raw_mask != filtered_mask))
        subtitle = (
            f"ep{episode:03d} frame {frame_index} | {example['description']} | "
            f"raw-to-filter changed pixels: {changed_pixels}"
        )
        comparison = add_header(comparison, subtitle)
        write_image(output_dirs["comparison"] / f"{stem}.jpg", comparison)

        tile_width = 320
        tile_height = round(frame.shape[0] * tile_width / frame.shape[1])
        tile_pair = np.concatenate(
            [
                cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA),
                cv2.resize(raw_visual, (tile_width, tile_height), interpolation=cv2.INTER_NEAREST),
            ],
            axis=1,
        )
        contact_tiles.append(
            add_header(
                tile_pair,
                f"{ordinal:02d}  ep{episode:03d} frame {frame_index}",
                f"RGB | generation-1 raw mask   -   {example['description']}",
            )
        )

        raw_classes = {}
        for class_id, class_name in CLASS_NAMES.items():
            if class_id == 0:
                continue
            metrics = quality.get("classes", {}).get(class_name, {})
            raw_metrics = metrics.get("raw", {})
            filtered_metrics = metrics.get("filtered", {})
            if int(raw_metrics.get("area_px", 0)) or int(filtered_metrics.get("area_px", 0)):
                raw_classes[class_name] = {
                    "raw_area": raw_metrics.get("area_px"),
                    "raw_components": raw_metrics.get("component_count"),
                    "filtered_area": filtered_metrics.get("area_px"),
                    "filtered_components": filtered_metrics.get("component_count"),
                }

        raw_quality_scores = [
            metrics.get("raw", {}).get("quality_score")
            for metrics in quality.get("classes", {}).values()
        ]
        raw_quality_scores = [score for score in raw_quality_scores if score is not None]

        exported.append(
            {
                **example,
                "stem": stem,
                "source_video": record["video_path"],
                "raw_mask_source": str(raw_path),
                "filtered_mask_source": str(filtered_path),
                "raw_min_class_quality_score": min(raw_quality_scores) if raw_quality_scores else None,
                "raw_hard_anomaly_classes": quality.get("raw_anomaly_classes", []),
                "raw_hard_anomaly": bool(quality.get("raw_anomaly_classes", [])),
                "filter_changed_pixels": changed_pixels,
                "filter_changed_fraction": changed_pixels / raw_mask.size,
                "class_metrics": raw_classes,
            }
        )

    max_height = max(tile.shape[0] for tile in contact_tiles)
    normalized_tiles = []
    for tile in contact_tiles:
        if tile.shape[0] < max_height:
            tile = cv2.copyMakeBorder(
                tile, 0, max_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(246, 246, 246)
            )
        normalized_tiles.append(tile)
    if len(normalized_tiles) % 2:
        normalized_tiles.append(np.full_like(normalized_tiles[0], 246))
    rows = [
        np.concatenate(normalized_tiles[index : index + 2], axis=1)
        for index in range(0, len(normalized_tiles), 2)
    ]
    contact_sheet = np.concatenate(rows, axis=0)
    contact_sheet = add_header(
        contact_sheet,
        "Generation 1 pseudo8: curated unfiltered failure examples",
        "Each tile: original RGB (left) | raw maskvisual before component filtering (right)",
    )
    write_image(args.output / "contact_sheet.jpg", contact_sheet)

    export_manifest = {
        "purpose": "Presentation examples of poor masks before filtering and iterative improvement",
        "prediction_kind": "generation1_pseudo8 raw/unfiltered semantic prediction",
        "model": model,
        "source_manifest": str(args.manifest),
        "canonical_palette_rgb": {
            CLASS_NAMES[index]: PALETTE[index][::-1].tolist() for index in CLASS_NAMES
        },
        "selection": "Hard-anomaly/low-quality candidates ranked automatically, then visually reviewed",
        "examples": exported,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# 第一代模型未过滤的较差 Mask Visual

这里保存的是第一代 `generation1_pseudo8` 模型的原始语义预测，模型由 8 个视频训练，未经过连通分量过滤，也不是后续迭代模型的结果。

- `maskvisual_raw/`：可直接用于展示的纯 Mask Visual（核心输出）
- `original/`：对应原始 RGB 帧
- `overlay_raw/`：原始预测叠加在 RGB 上，便于定位问题
- `comparison/`：原图、未过滤结果、过滤后参考结果三联图
- `contact_sheet.jpg`：全部样例总览
- `manifest.json`：模型哈希、源视频、帧号、质量指标与类别统计

共导出 {len(exported)} 张人工目视确认的代表帧。纯 Mask Visual 直接读取每个视频的 `raw_mask_frames`，没有调用过滤流程；过滤后结果只作为 comparison 中的参考。
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
