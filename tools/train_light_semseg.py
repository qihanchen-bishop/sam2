#!/usr/bin/env python3
"""Train a small semantic segmentation model from LeRobot video masks.

The default paths match the local newdata_3object dataset:

  conda run -n sam2 python tools/train_light_semseg.py --epochs 100 --frame-stride 10

Outputs are written under segdata/temp/<dataset>/semantic_seg_light by default.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


LABELS = ["occluder", "object", "region", "tool"]
IMAGE_KEYS = ["observation.images.front", "observation.images.side"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    image_video: str
    mask_videos: list[str]
    video_key: str
    chunk_index: int
    file_index: int
    frame_index: int


@dataclass(frozen=True)
class ImageSample:
    image_path: str
    mask_path: str
    split: str
    video_key: str
    chunk_index: int
    file_index: int
    frame_index: int


def require_runtime_deps() -> tuple[Any, Any, Any, Any, Any]:
    missing = []
    try:
        import cv2
    except Exception:
        missing.append("opencv-python-headless")
        cv2 = None
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception:
        missing.append("torch torchvision")
        torch = nn = F = DataLoader = Dataset = None
    if missing:
        raise SystemExit(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + "\nUse the repo environment, for example: conda run -n sam2 python ..."
        )
    return cv2, torch, nn, F, DataLoader, Dataset


cv2, torch, nn, F, DataLoader, TorchDataset = require_runtime_deps()


class SeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, num_classes: int, width: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 6
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            SeparableConv(c1, c1),
        )
        self.down1 = SeparableConv(c1, c2, stride=2)
        self.down2 = SeparableConv(c2, c3, stride=2)
        self.down3 = SeparableConv(c3, c4, stride=2)
        self.fuse2 = SeparableConv(c4 + c3, c3)
        self.fuse1 = SeparableConv(c3 + c2, c2)
        self.fuse0 = SeparableConv(c2 + c1, c1)
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        x = self.down3(s2)
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, s2], dim=1))
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse1(torch.cat([x, s1], dim=1))
        x = F.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse0(torch.cat([x, s0], dim=1))
        return self.head(x)


class VideoSemSegDataset(TorchDataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: tuple[int, int],
        mask_threshold: int,
        augment: bool,
        video_cache_size: int,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.augment = augment
        self.video_cache_size = video_cache_size
        self._caps: OrderedDict[str, Any] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frame(self, path: str, frame_index: int) -> np.ndarray:
        cap = self._caps.get(path)
        if cap is None:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            self._caps[path] = cap
            while len(self._caps) > self.video_cache_size:
                _, old_cap = self._caps.popitem(last=False)
                old_cap.release()
        else:
            self._caps.move_to_end(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {path}")
        return frame

    def __del__(self) -> None:
        for cap in getattr(self, "_caps", {}).values():
            cap.release()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = self._read_frame(sample.image_video, sample.frame_index)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for class_id, mask_video in enumerate(sample.mask_videos, start=1):
            mask_frame = self._read_frame(mask_video, sample.frame_index)
            fg = mask_frame.max(axis=2) > self.mask_threshold
            mask[fg] = class_id

        out_w, out_h = self.image_size
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.8:
                image = image.astype(np.float32)
                image *= random.uniform(0.85, 1.15)
                image += random.uniform(-12.0, 12.0)
                image = np.clip(image, 0, 255).astype(np.uint8)

        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor


class ImageSemSegDataset(TorchDataset):
    def __init__(
        self,
        samples: list[ImageSample],
        image_size: tuple[int, int],
        augment: bool,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {sample.mask_path}")

        out_w, out_h = self.image_size
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.8:
                image = image.astype(np.float32)
                image *= random.uniform(0.85, 1.15)
                image += random.uniform(-12.0, 12.0)
                image = np.clip(image, 0, 255).astype(np.uint8)

        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor


def parse_image_size(value: str) -> tuple[int, int]:
    if "x" in value:
        w, h = value.lower().split("x", 1)
        return int(w), int(h)
    size = int(value)
    return size, size


def default_work_dir(dataset_root: Path) -> Path:
    parts = list(dataset_root.parts)
    if "final" in parts:
        parts[parts.index("final")] = "temp"
        return Path(*parts) / "semantic_seg_light"
    return dataset_root.parent / "temp" / dataset_root.name / "semantic_seg_light"


def default_image_cache_root(dataset_root: Path) -> Path:
    parts = list(dataset_root.parts)
    if "final" in parts:
        parts[parts.index("final")] = "temp"
        return Path(*parts) / "semantic_seg_light_cache"
    return dataset_root.parent / "temp" / dataset_root.name / "semantic_seg_light_cache"


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames


def build_samples(
    dataset_root: Path,
    labels: list[str],
    image_keys: list[str],
    frame_stride: int,
    max_samples: int | None,
) -> list[Sample]:
    samples: list[Sample] = []
    for video_key in image_keys:
        image_dir = dataset_root / "videos" / video_key / "chunk-000"
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image video directory: {image_dir}")
        for image_video in sorted(image_dir.glob("file-*.mp4")):
            chunk_index = int(image_video.parent.name.split("-")[-1])
            file_index = int(image_video.stem.split("-")[-1])
            mask_videos = [
                str(dataset_root / "videos" / f"{video_key}_{label}" / image_video.parent.name / image_video.name)
                for label in labels
            ]
            missing = [p for p in mask_videos if not Path(p).exists()]
            if missing:
                raise FileNotFoundError(f"Missing mask videos for {image_video}: {missing[:2]}")
            frames = video_frame_count(image_video)
            for frame_index in range(0, frames, frame_stride):
                samples.append(
                    Sample(
                        image_video=str(image_video),
                        mask_videos=mask_videos,
                        video_key=video_key,
                        chunk_index=chunk_index,
                        file_index=file_index,
                        frame_index=frame_index,
                    )
                )
    if max_samples is not None and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
    return samples


def cache_paths(cache_root: Path, split: str, sample: Sample) -> tuple[Path, Path]:
    rel_dir = Path(split) / sample.video_key / f"file-{sample.file_index:03d}"
    image_path = cache_root / "images" / rel_dir / f"{sample.frame_index:05d}.jpg"
    mask_path = cache_root / "masks" / rel_dir / f"{sample.frame_index:05d}.png"
    return image_path, mask_path


def to_image_samples(cache_root: Path, split: str, samples: list[Sample]) -> list[ImageSample]:
    image_samples: list[ImageSample] = []
    for sample in samples:
        image_path, mask_path = cache_paths(cache_root, split, sample)
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing cached image/mask: {image_path}, {mask_path}")
        image_samples.append(
            ImageSample(
                image_path=str(image_path),
                mask_path=str(mask_path),
                split=split,
                video_key=sample.video_key,
                chunk_index=sample.chunk_index,
                file_index=sample.file_index,
                frame_index=sample.frame_index,
            )
        )
    return image_samples


def read_next_frame(cap: Any, path: str, frame_index: int) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return frame


def ensure_image_cache(
    cache_root: Path,
    split_to_samples: dict[str, list[Sample]],
    labels: list[str],
    mask_threshold: int,
    overwrite: bool,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "labels": ["background", *labels],
        "splits": {split: len(samples) for split, samples in split_to_samples.items()},
        "images": "images/{split}/{video_key}/file-{file_index:03d}/{frame_index:05d}.jpg",
        "masks": "masks/{split}/{video_key}/file-{file_index:03d}/{frame_index:05d}.png",
    }
    write_json(cache_root / "manifest.json", manifest)

    for split, samples in split_to_samples.items():
        pending = [
            sample
            for sample in samples
            if overwrite
            or not cache_paths(cache_root, split, sample)[0].exists()
            or not cache_paths(cache_root, split, sample)[1].exists()
        ]
        print(
            f"cache split={split} samples={len(samples)} missing_or_overwrite={len(pending)}",
            flush=True,
        )
        grouped: dict[str, list[Sample]] = {}
        for sample in pending:
            grouped.setdefault(sample.image_video, []).append(sample)

        for video_idx, (image_video, group) in enumerate(grouped.items(), start=1):
            group.sort(key=lambda s: s.frame_index)
            by_frame = {sample.frame_index: sample for sample in group}
            target_frames = set(by_frame)
            max_frame = group[-1].frame_index
            image_cap = cv2.VideoCapture(image_video)
            if not image_cap.isOpened():
                raise RuntimeError(f"Could not open video: {image_video}")
            mask_caps = []
            for mask_video in group[0].mask_videos:
                cap = cv2.VideoCapture(mask_video)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open mask video: {mask_video}")
                mask_caps.append(cap)

            written = 0
            for frame_index in range(max_frame + 1):
                image_ok, image = image_cap.read()
                if not image_ok:
                    break
                mask_frames = []
                for cap, mask_video in zip(mask_caps, group[0].mask_videos):
                    mask_frames.append(read_next_frame(cap, mask_video, frame_index))
                sample = by_frame.get(frame_index)
                if sample is None:
                    continue
                image_path, mask_path = cache_paths(cache_root, split, sample)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                semantic_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for class_id, mask_frame in enumerate(mask_frames, start=1):
                    semantic_mask[mask_frame.max(axis=2) > mask_threshold] = class_id
                cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cv2.imwrite(str(mask_path), semantic_mask)
                written += 1
                if written == len(target_frames):
                    break

            image_cap.release()
            for cap in mask_caps:
                cap.release()
            if video_idx % 20 == 0 or video_idx == len(grouped):
                print(
                    f"cache split={split} videos_done={video_idx}/{len(grouped)}",
                    flush=True,
                )


def split_samples(
    samples: list[Sample], val_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    episodes = sorted({(s.video_key, s.file_index) for s in samples})
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_ratio)))
    val_episodes = set(episodes[:val_count])
    train = [s for s in samples if (s.video_key, s.file_index) not in val_episodes]
    val = [s for s in samples if (s.video_key, s.file_index) in val_episodes]
    return train, val


def class_weights(samples: list[Sample | ImageSample], num_classes: int, max_scan: int, mask_threshold: int) -> torch.Tensor:
    counts = np.ones(num_classes, dtype=np.float64)
    scan = samples if len(samples) <= max_scan else random.sample(samples, max_scan)
    print(f"scanning {len(scan)} samples for class weights...", flush=True)
    for sample in scan:
        if isinstance(sample, ImageSample):
            mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
        else:
            mask = np.zeros((360, 640), dtype=np.uint8)
            for class_id, mask_video in enumerate(sample.mask_videos, start=1):
                cap = cv2.VideoCapture(mask_video)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
                ok, frame = cap.read()
                cap.release()
                if ok:
                    mask[frame.max(axis=2) > mask_threshold] = class_id
        bincount = np.bincount(mask.reshape(-1), minlength=num_classes)
        counts += bincount[:num_classes]
    weights = 1.0 / np.sqrt(counts / counts.sum())
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: Any, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    hist = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        keep = (targets >= 0) & (targets < num_classes)
        encoded = targets[keep] * num_classes + preds[keep]
        hist += torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)
        total_loss += float(loss.item()) * images.size(0)
        total += images.size(0)
    iou = hist.diag() / (hist.sum(1) + hist.sum(0) - hist.diag()).clamp_min(1.0)
    acc = hist.diag().sum() / hist.sum().clamp_min(1.0)
    return {
        "loss": total_loss / max(total, 1),
        "miou": float(iou.mean().item()),
        "pixel_acc": float(acc.item()),
        **{f"iou_{i}": float(iou[i].item()) for i in range(num_classes)},
    }


def overlay_prediction(image_tensor: torch.Tensor, pred: np.ndarray, palette: np.ndarray) -> np.ndarray:
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = np.clip((image * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)
    color = palette[pred]
    return (image * 0.55 + color * 0.45).astype(np.uint8)


@torch.no_grad()
def save_previews(model: nn.Module, dataset: VideoSemSegDataset, device: torch.device, out_dir: Path, count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = np.array(
        [
            [0, 0, 0],
            [50, 160, 255],
            [255, 80, 80],
            [60, 220, 120],
            [210, 210, 80],
        ],
        dtype=np.uint8,
    )
    model.eval()
    for i in range(min(count, len(dataset))):
        image, _ = dataset[i]
        logits = model(image.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        overlay = overlay_prediction(image, pred, palette)
        cv2.imwrite(str(out_dir / f"preview_{i:03d}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "labels": ["background", *args.labels],
            "image_size": args.image_size,
            "args": vars(args),
        },
        path,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/qihan/data/sam2/segdata/final/newdata_3object"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--source", choices=["image-cache", "video"], default="image-cache")
    parser.add_argument("--image-cache-root", type=Path, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--image-keys", nargs="+", default=IMAGE_KEYS)
    parser.add_argument("--image-size", type=parse_image_size, default=parse_image_size("320x180"))
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-threshold", type=int, default=8)
    parser.add_argument("--class-weight-scan", type=int, default=256)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--video-cache-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir = args.work_dir or default_work_dir(args.dataset_root)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset_root={args.dataset_root}", flush=True)
    print(f"work_dir={args.work_dir}", flush=True)
    print(f"device={args.device}", flush=True)
    if str(args.device).startswith("cuda"):
        print(f"cuda_available={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"cuda_device={torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.source == "image-cache":
        image_cache_root = args.image_cache_root or default_image_cache_root(args.dataset_root)
        print("building video sample index for image cache...", flush=True)
        video_samples = build_samples(
            args.dataset_root,
            args.labels,
            args.image_keys,
            args.frame_stride,
            args.max_samples,
        )
        if not video_samples:
            raise SystemExit("No training samples found.")
        train_video_samples, val_video_samples = split_samples(video_samples, args.val_ratio, args.seed)
        ensure_image_cache(
            image_cache_root,
            {"train": train_video_samples, "val": val_video_samples},
            args.labels,
            args.mask_threshold,
            args.overwrite_cache,
        )
        if args.prepare_cache_only:
            print(f"cache_ready={image_cache_root}", flush=True)
            return
        train_samples = to_image_samples(image_cache_root, "train", train_video_samples)
        val_samples = to_image_samples(image_cache_root, "val", val_video_samples)
        samples = train_samples + val_samples
        source_kind = "image-cache"
    else:
        image_cache_root = None
        print("building video sample index...", flush=True)
        samples = build_samples(
            args.dataset_root,
            args.labels,
            args.image_keys,
            args.frame_stride,
            args.max_samples,
        )
        train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
        source_kind = "video"
    if not samples:
        raise SystemExit("No training samples found.")
    if not train_samples or not val_samples:
        raise SystemExit(f"Bad split: train={len(train_samples)} val={len(val_samples)}")

    write_json(
        args.work_dir / "train_manifest.json",
        {
            "dataset_root": str(args.dataset_root),
            "source": source_kind,
            "image_cache_root": str(image_cache_root) if image_cache_root is not None else None,
            "labels": ["background", *args.labels],
            "image_keys": args.image_keys,
            "frame_stride": args.frame_stride,
            "image_size": list(args.image_size),
            "samples": len(samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "train": [asdict(s) for s in train_samples[:2000]],
            "val": [asdict(s) for s in val_samples[:2000]],
            "note": "Sample lists are truncated to 2000 entries each to keep this manifest small.",
        },
    )
    write_json(args.work_dir / "labels.json", {"labels": ["background", *args.labels]})

    num_classes = len(args.labels) + 1
    if source_kind == "image-cache":
        train_dataset = ImageSemSegDataset(train_samples, args.image_size, augment=True)
        val_dataset = ImageSemSegDataset(val_samples, args.image_size, augment=False)
    else:
        train_dataset = VideoSemSegDataset(
            train_samples,
            args.image_size,
            args.mask_threshold,
            augment=True,
            video_cache_size=args.video_cache_size,
        )
        val_dataset = VideoSemSegDataset(
            val_samples,
            args.image_size,
            args.mask_threshold,
            augment=False,
            video_cache_size=args.video_cache_size,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = TinyUNet(num_classes=num_classes, width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weights(train_samples, num_classes, args.class_weight_scan, args.mask_threshold).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    best_miou = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    print(f"samples train={len(train_samples)} val={len(val_samples)} classes={num_classes}", flush=True)
    print(f"class_weights={weights.detach().cpu().numpy().round(3).tolist()}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        seen = 0
        total_batches = len(train_loader)
        print(f"epoch {epoch:03d}/{args.epochs} train_batches={total_batches}", flush=True)
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and not args.no_amp)):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
            if args.log_interval > 0 and (step == 1 or step % args.log_interval == 0):
                print(
                    f"epoch {epoch:03d}/{args.epochs} "
                    f"step {step:04d}/{total_batches} "
                    f"train_loss={running_loss / max(seen, 1):.4f}",
                    flush=True,
                )

        print(f"epoch {epoch:03d}/{args.epochs} evaluating...", flush=True)
        metrics = evaluate(model, val_loader, device, num_classes)
        metrics["train_loss"] = running_loss / max(seen, 1)
        metrics["epoch"] = epoch
        metrics["seconds"] = time.time() - start
        save_checkpoint(args.work_dir / "latest.pt", model, optimizer, epoch, metrics, args)

        improved = metrics["miou"] > best_miou + args.min_delta
        if improved:
            best_miou = metrics["miou"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(args.work_dir / "best.pt", model, optimizer, epoch, metrics, args)
            save_previews(model, val_dataset, device, args.work_dir / "previews", args.preview_count)
        else:
            stale_epochs += 1

        metrics["best_miou"] = best_miou
        metrics["best_epoch"] = best_epoch
        metrics["stale_epochs"] = stale_epochs
        history.append(metrics)
        write_json(args.work_dir / "history.json", history)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={metrics['train_loss']:.4f} val_loss={metrics['loss']:.4f} "
            f"mIoU={metrics['miou']:.4f} best={best_miou:.4f}@{best_epoch:03d} "
            f"stale={stale_epochs}/{args.patience} pixel_acc={metrics['pixel_acc']:.4f} "
            f"time={metrics['seconds']:.1f}s",
            flush=True,
        )

        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(
                "early_stop="
                f"no validation mIoU improvement greater than {args.min_delta} "
                f"for {args.patience} epochs after epoch {best_epoch}",
                flush=True,
            )
            break

    print(f"best_mIoU={best_miou:.4f} best_epoch={best_epoch}", flush=True)
    print(f"best_checkpoint={args.work_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
