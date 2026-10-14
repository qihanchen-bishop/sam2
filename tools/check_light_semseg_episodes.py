#!/usr/bin/env python3
"""Score selected cached episodes and render their worst semantic-segmentation frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from infer_light_semseg import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--image-key", default="observation.images.front")
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--critical-labels", nargs="+", default=["occluder", "object", "tool"])
    parser.add_argument("--worst-per-episode", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def class_iou(target: np.ndarray, prediction: np.ndarray, class_id: int) -> float | None:
    target_fg = target == class_id
    prediction_fg = prediction == class_id
    union = np.logical_or(target_fg, prediction_fg).sum()
    if union == 0:
        return None
    return float(np.logical_and(target_fg, prediction_fg).sum() / union)


def main() -> None:
    args = parse_args()
    predictor = Predictor(args.model_dir, args.device)
    labels = predictor.config["labels"]
    critical_ids = [labels.index(label) for label in args.critical_labels]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "note": "Targeted cached-frame QA; these episodes may belong to the training split and are not independent validation metrics.",
        "critical_labels": args.critical_labels,
        "episodes": {},
    }

    for episode in args.episodes:
        image_dir = None
        mask_dir = None
        split = None
        for candidate_split in ("train", "val"):
            candidate_image_dir = args.cache_root / "images" / candidate_split / args.image_key / f"file-{episode:03d}"
            candidate_mask_dir = args.cache_root / "masks" / candidate_split / args.image_key / f"file-{episode:03d}"
            if candidate_image_dir.exists() and candidate_mask_dir.exists():
                image_dir, mask_dir, split = candidate_image_dir, candidate_mask_dir, candidate_split
                break
        if image_dir is None or mask_dir is None or split is None:
            raise SystemExit(f"Episode {episode:03d} not found in cache")

        frames: list[dict[str, object]] = []
        per_class_values: dict[str, list[float]] = {label: [] for label in labels}
        image_paths = sorted(image_dir.glob("*.jpg"))
        for image_path in image_paths:
            mask_path = mask_dir / f"{image_path.stem}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or target is None:
                raise RuntimeError(f"Could not load {image_path} or {mask_path}")
            prediction = predictor.predict(image)
            ious = {label: class_iou(target, prediction, class_id) for class_id, label in enumerate(labels)}
            for label, value in ious.items():
                if value is not None:
                    per_class_values[label].append(value)
            critical_present = [ious[labels[class_id]] for class_id in critical_ids if ious[labels[class_id]] is not None]
            critical_score = min(critical_present) if critical_present else 1.0
            frames.append(
                {
                    "frame": int(image_path.stem),
                    "image_path": str(image_path),
                    "target_path": str(mask_path),
                    "critical_min_iou": critical_score,
                    "ious": ious,
                    "image": image,
                    "target": target,
                    "prediction": prediction,
                }
            )

        worst_frames = sorted(frames, key=lambda item: float(item["critical_min_iou"]))[: args.worst_per_episode]
        rendered: list[str] = []
        for rank, frame in enumerate(worst_frames):
            image = frame.pop("image")
            target = frame.pop("target")
            prediction = frame.pop("prediction")
            target_color = cv2.cvtColor(predictor.palette[target], cv2.COLOR_RGB2BGR)
            prediction_color = predictor.colorize_bgr(prediction)
            target_overlay = cv2.addWeighted(image, 0.55, target_color, 0.45, 0)
            prediction_overlay = cv2.addWeighted(image, 0.55, prediction_color, 0.45, 0)
            sheet = np.concatenate([image, target_overlay, prediction_overlay], axis=1)
            for column, title in enumerate(("RGB", "GT", "PRED")):
                x = column * image.shape[1] + 8
                cv2.putText(sheet, title, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(sheet, title, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
            filename = f"ep{episode:03d}_worst{rank + 1}_f{int(frame['frame']):05d}.png"
            cv2.imwrite(str(args.output_dir / filename), sheet)
            rendered.append(filename)

        for frame in frames:
            frame.pop("image", None)
            frame.pop("target", None)
            frame.pop("prediction", None)
        report["episodes"][f"{episode:03d}"] = {
            "split": split,
            "sampled_frames": len(frames),
            "mean_per_class_iou_when_present": {
                label: (float(np.mean(values)) if values else None)
                for label, values in per_class_values.items()
            },
            "minimum_per_class_iou_when_present": {
                label: (float(np.min(values)) if values else None)
                for label, values in per_class_values.items()
            },
            "worst_frames": worst_frames,
            "rendered_previews": rendered,
        }
        print(f"episode={episode:03d} split={split} frames={len(frames)}", flush=True)

    (args.output_dir / "targeted_quality.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"report={args.output_dir / 'targeted_quality.json'}", flush=True)


if __name__ == "__main__":
    main()
