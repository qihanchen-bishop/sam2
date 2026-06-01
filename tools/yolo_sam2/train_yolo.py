#!/usr/bin/env python3
"""Train an Ultralytics YOLO model on a generated SAM2-derived dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on a generated dataset.")
    parser.add_argument("--data", type=Path, default=Path("datasets/task1_yolo_detect/data.yaml"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", default="auto")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=Path("runs/yolo"))
    parser.add_argument("--name", default="task1_detect")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def parse_batch(value: str) -> int | float:
    if value == "auto":
        return -1
    try:
        return int(value)
    except ValueError:
        return float(value)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        print(
            "Warning: torch is not installed; Ultralytics will decide the device after dependencies are installed.",
            file=sys.stderr,
        )
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    print("Warning: CUDA is not available; training will run on CPU.", file=sys.stderr)
    return "cpu"


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if not args.data.is_file():
        raise RuntimeError(f"Dataset YAML not found: {args.data}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'ultralytics'. Install the YOLO dependencies first, e.g. "
            "`python -m pip install ultralytics opencv-python-headless pyyaml`."
        ) from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=parse_batch(args.batch),
        patience=args.patience,
        device=resolve_device(args.device),
        project=str(args.project.resolve()),
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        exist_ok=args.exist_ok,
    )
    print(results)


if __name__ == "__main__":
    main()
