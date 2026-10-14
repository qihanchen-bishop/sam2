#!/usr/bin/env python3
"""Independently evaluate, preview, and package a trained light U-Net model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import train_light_semseg as training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preview-count", type=int, default=28)
    parser.add_argument("--torchscript-name", default="unet_front_v4_torchscript.pt")
    return parser.parse_args()


@torch.no_grad()
def confusion_matrix(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    model.eval()
    hist = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        encoded = targets.reshape(-1) * num_classes + predictions.reshape(-1)
        hist += torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)
    return hist.cpu()


def metrics_from_histogram(hist: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    true_positive = hist.diag()
    ground_truth = hist.sum(dim=1)
    predicted = hist.sum(dim=0)
    union = ground_truth + predicted - true_positive
    iou = true_positive / union.clamp_min(1)
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / ground_truth.clamp_min(1)
    dice = 2 * true_positive / (ground_truth + predicted).clamp_min(1)
    return {
        "miou": float(iou.mean()),
        "foreground_miou": float(iou[1:].mean()),
        "pixel_accuracy": float(true_positive.sum() / hist.sum().clamp_min(1)),
        "per_class": {
            label: {
                "iou": float(iou[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "dice": float(dice[index]),
                "ground_truth_pixels": int(ground_truth[index]),
                "predicted_pixels": int(predicted[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix_rows_gt_columns_pred": hist.to(torch.int64).tolist(),
    }


def preview_indices(samples: list[training.ImageSample], count: int) -> list[int]:
    """Spread previews across episodes, then across time inside each episode."""
    by_episode: dict[tuple[str, int], list[int]] = {}
    for index, sample in enumerate(samples):
        by_episode.setdefault((sample.video_key, sample.file_index), []).append(index)
    episodes = sorted(by_episode)
    selected: list[int] = []
    rounds = max(1, int(np.ceil(count / max(len(episodes), 1))))
    for round_index in range(rounds):
        fraction = (round_index + 1) / (rounds + 1)
        for episode in episodes:
            indices = by_episode[episode]
            selected.append(indices[min(int(fraction * len(indices)), len(indices) - 1)])
            if len(selected) >= count:
                return selected
    return selected


@torch.no_grad()
def save_previews(
    model: torch.nn.Module,
    dataset: training.ImageSemSegDataset,
    samples: list[training.ImageSample],
    device: torch.device,
    output_dir: Path,
    count: int,
) -> list[dict[str, Any]]:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    manifest: list[dict[str, Any]] = []
    for preview_index, dataset_index in enumerate(preview_indices(samples, count)):
        image, target = dataset[dataset_index]
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        target_np = target.numpy().astype(np.uint8)
        rgb = training.tensor_to_rgb(image)
        gt = training.overlay_prediction(image, target_np, training.PREVIEW_PALETTE)
        pred = training.overlay_prediction(image, prediction, training.PREVIEW_PALETTE)
        sheet = np.concatenate([rgb, gt, pred], axis=1)
        for column, title in enumerate(("RGB", "GT", "PRED")):
            x = column * prediction.shape[1] + 8
            cv2.putText(sheet, title, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(sheet, title, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        sample = samples[dataset_index]
        filename = f"preview_{preview_index:03d}_ep{sample.file_index:03d}_f{sample.frame_index:05d}.png"
        cv2.imwrite(str(preview_dir / filename), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        manifest.append(
            {
                "preview": filename,
                "validation_sample_index": dataset_index,
                "episode": sample.file_index,
                "frame": sample.frame_index,
                "image_path": sample.image_path,
                "mask_path": sample.mask_path,
            }
        )
    return manifest


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["args"]
    labels = list(checkpoint["labels"])
    foreground_labels = labels[1:]
    dataset_root = Path(checkpoint_args["dataset_root"])
    image_keys = list(checkpoint_args["image_keys"])
    image_size = tuple(checkpoint["image_size"])
    seed = int(checkpoint_args["seed"])
    val_ratio = float(checkpoint_args["val_ratio"])
    frame_stride = int(checkpoint_args["frame_stride"])
    width = int(checkpoint_args["width"])

    if checkpoint_args.get("source") == "prebuilt-cache":
        print("loading explicit episode-split validation cache...", flush=True)
        cache_manifest = json.loads((args.cache_root / "manifest.json").read_text())
        if cache_manifest["labels"] != labels:
            raise ValueError("Validation cache labels differ from checkpoint labels")
        train_samples = training.prebuilt_image_samples(args.cache_root, "train")
        val_samples = training.prebuilt_image_samples(args.cache_root, "val")
    else:
        print("rebuilding deterministic episode-disjoint validation split...", flush=True)
        all_samples = training.build_samples(dataset_root, foreground_labels, image_keys, frame_stride, None)
        train_video_samples, val_video_samples = training.split_samples(all_samples, val_ratio, seed)
        val_samples = training.to_image_samples(args.cache_root, "val", val_video_samples)
        train_samples = training.to_image_samples(args.cache_root, "train", train_video_samples)
    val_dataset = training.ImageSemSegDataset(val_samples, image_size, augment=False, horizontal_flip_prob=0.0)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = training.TinyUNet(num_classes=len(labels), width=width).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"evaluating {len(val_samples)} validation frames on {device}...", flush=True)
    histogram = confusion_matrix(model, val_loader, device, len(labels))
    evaluation = metrics_from_histogram(histogram, labels)
    evaluation.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "dataset_root": str(dataset_root.resolve()),
            "image_key": image_keys,
            "image_size_wh": list(image_size),
            "frame_stride": frame_stride,
            "training_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "validation_episodes": sorted({sample.file_index for sample in val_samples}),
            "split": ("explicit accepted-episode split" if checkpoint_args.get("source") == "prebuilt-cache"
                      else "episode-disjoint deterministic split"),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, args.output_dir / "best.pt")
    source_work_dir = args.checkpoint.parent
    for filename in ("history.json", "train_manifest.json", "labels.json"):
        source = source_work_dir / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)

    print("creating cross-episode previews...", flush=True)
    preview_manifest = save_previews(model, val_dataset, val_samples, device, args.output_dir, args.preview_count)
    training.write_json(args.output_dir / "preview_manifest.json", preview_manifest)

    print("exporting CPU TorchScript model...", flush=True)
    cpu_model = training.TinyUNet(num_classes=len(labels), width=width).eval()
    cpu_model.load_state_dict(checkpoint["model"])
    example = torch.zeros((1, 3, image_size[1], image_size[0]), dtype=torch.float32)
    traced = torch.jit.trace(cpu_model, example)
    if Path(args.torchscript_name).name != args.torchscript_name:
        raise SystemExit("--torchscript-name must be a filename, not a path")
    torchscript_path = args.output_dir / args.torchscript_name
    torch.jit.save(traced, torchscript_path)
    with torch.no_grad():
        eager_logits = cpu_model(example)
        traced_logits = traced(example)
    export_test = {
        "max_abs_logit_difference": float((eager_logits - traced_logits).abs().max()),
        "argmax_pixel_agreement": float((eager_logits.argmax(1) == traced_logits.argmax(1)).float().mean()),
        "passed": bool(torch.allclose(eager_logits, traced_logits, atol=1e-5, rtol=1e-4)),
    }
    evaluation["torchscript_export_test"] = export_test
    training.write_json(args.output_dir / "evaluation.json", evaluation)
    training.write_json(
        args.output_dir / "model_config.json",
        {
            "architecture": "TinyUNet depthwise-separable",
            "width": width,
            "labels": labels,
            "class_ids": {label: index for index, label in enumerate(labels)},
            "palette_rgb": training.PREVIEW_PALETTE[: len(labels)].tolist(),
            "input_size_wh": list(image_size),
            "input_color": "RGB",
            "input_scale": "uint8 / 255",
            "normalization_mean": training.MEAN.tolist(),
            "normalization_std": training.STD.tolist(),
            "output": "NCHW logits; argmax over channel gives uint8 class id",
            "source_checkpoint": "best.pt",
            "torchscript": torchscript_path.name,
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "miou": evaluation["miou"], "export_test": export_test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
