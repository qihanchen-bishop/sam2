#!/usr/bin/env python3
"""Add benchmark, provenance, README, and checksums to a light-semseg package."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

import train_light_semseg as training


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    evaluation = json.loads((model_dir / "evaluation.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    labels = list(checkpoint["labels"])
    width = int(checkpoint["args"]["width"])
    input_w, input_h = map(int, checkpoint["image_size"])
    device = torch.device(args.device)
    model = training.TinyUNet(num_classes=len(labels), width=width).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    example = torch.zeros((1, 3, input_h, input_w), device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    benchmark = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "input_shape": list(example.shape),
        "warmup_iterations": args.warmup,
        "iterations": args.iterations,
        "mean_latency_ms": elapsed * 1000.0 / args.iterations,
        "throughput_fps_batch1": args.iterations / elapsed,
        "parameter_count": parameter_count,
    }
    write_json(model_dir / "benchmark.json", benchmark)

    provenance = {
        "dataset_root": str(checkpoint["args"]["dataset_root"]),
        "dataset_revision": args.dataset_revision,
        "image_keys": checkpoint["args"]["image_keys"],
        "labels": labels,
        "best_epoch": int(checkpoint["epoch"]),
        "validation_episodes": evaluation["validation_episodes"],
        "validation_samples": evaluation["validation_samples"],
        "split": evaluation["split"],
        "parameter_count": parameter_count,
    }
    write_json(model_dir / "provenance.json", provenance)

    per_class = evaluation["per_class"]
    metric_lines = "\n".join(
        f"- {label}: IoU {values['iou']:.4f}, precision {values['precision']:.4f}, recall {values['recall']:.4f}"
        for label, values in per_class.items()
    )
    readme = f"""# unet_side

Small side-view semantic segmentation model trained on `bettersetup_v5` revision `{args.dataset_revision}`.

- Architecture: TinyUNet with depthwise-separable convolutions, width {width}
- Parameters: {parameter_count:,}
- Input: RGB uint8, resized to {input_w}x{input_h}, ImageNet normalization
- Output classes: {', '.join(labels)}
- Validation split: deterministic episode-disjoint, {evaluation['validation_samples']} frames
- Validation mIoU: {evaluation['miou']:.4f}
- Foreground mIoU: {evaluation['foreground_miou']:.4f}
- Pixel accuracy: {evaluation['pixel_accuracy']:.4f}
- TorchScript parity: {evaluation['torchscript_export_test']['argmax_pixel_agreement']:.4f}
- Batch-1 latency on {benchmark['device']}: {benchmark['mean_latency_ms']:.3f} ms

Per-class validation metrics:

{metric_lines}

Use `unet_side_torchscript.pt` for deployment or `best.pt` with the TinyUNet Python definition.
Class ids, colors, resizing, and normalization are recorded in `model_config.json`.
"""
    (model_dir / "README.md").write_text(readme, encoding="utf-8")

    checksum_lines = []
    for path in sorted(p for p in model_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.relative_to(model_dir)}")
    (model_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"model_dir": str(model_dir), "benchmark": benchmark}, indent=2))


if __name__ == "__main__":
    main()
