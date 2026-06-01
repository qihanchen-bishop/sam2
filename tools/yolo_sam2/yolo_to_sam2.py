#!/usr/bin/env python3
"""Use YOLO detections as automatic box prompts for SAM2 video propagation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_COLORS = {
    1: (64, 160, 255),
    2: (255, 105, 97),
    3: (119, 221, 119),
    4: (255, 179, 71),
    5: (177, 156, 217),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO -> SAM2 prompt propagation.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("checkpoints/sam2.1_hiera_base_plus.pt"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--labels", type=Path, default=Path("outputs/task1/labels.txt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--add-center-point", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_modules() -> None:
    missing = []
    for module in ("torch", "numpy", "PIL", "cv2", "ultralytics"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        package_hint = "torch ultralytics opencv-python-headless pillow numpy"
        raise RuntimeError(
            f"Missing runtime modules: {', '.join(missing)}. Install dependencies, e.g. "
            f"`python -m pip install {package_hint}` and install this SAM2 repo."
        )


def read_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not labels:
        raise RuntimeError(f"No labels found in {path}")
    return labels


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_frames(video_path: Path, frame_dir: Path) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(frame_dir / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames


def select_prompts(frame_dir: Path, weights: Path, labels: list[str], conf: float, imgsz: int, device: str) -> dict[int, dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    prompts: dict[int, dict] = {}
    results = model.predict(
        source=str(frame_dir),
        stream=True,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    for result in results:
        frame_idx = int(Path(result.path).stem)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        scores = boxes.conf.detach().cpu().numpy()
        best_in_frame: dict[int, tuple[float, list[float]]] = {}
        for box, class_id, score in zip(xyxy, cls, scores):
            if class_id < 0 or class_id >= len(labels) or class_id in prompts:
                continue
            current = best_in_frame.get(class_id)
            if current is None or score > current[0]:
                best_in_frame[class_id] = (float(score), [float(v) for v in box.tolist()])
        for class_id, (score, box) in best_in_frame.items():
            if class_id not in prompts:
                prompts[class_id] = {
                    "frame_idx": frame_idx,
                    "obj_id": class_id + 1,
                    "name": labels[class_id],
                    "box": box,
                    "confidence": score,
                }
        if len(prompts) == len(labels):
            break
    return prompts


def save_mask_png(path: Path, mask) -> None:
    from PIL import Image

    Image.fromarray(mask).save(path)


def write_outputs(output_dir: Path, frames: list[Path], video_segments: dict[int, dict[int, object]], labels: list[str], mask_threshold: float) -> None:
    import cv2
    import numpy as np

    mask_dir = output_dir / "mask_frames"
    obj_mask_dir = output_dir / "object_mask_frames"
    overlay_dir = output_dir / "overlay_frames"
    mask_dir.mkdir(parents=True, exist_ok=True)
    obj_mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame {frames[0]}")
    height, width = first_frame.shape[:2]
    video_writer = cv2.VideoWriter(
        str(output_dir / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height),
    )

    for frame_idx, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Failed to read frame {frame_path}")
        combined = np.zeros((height, width), dtype=np.uint8)
        overlay = frame.copy()
        per_obj = video_segments.get(frame_idx, {})
        for obj_id, logits in sorted(per_obj.items()):
            mask = (logits > mask_threshold).astype(np.uint8)
            mask = np.squeeze(mask)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            combined[mask > 0] = obj_id
            binary = (mask * 255).astype(np.uint8)
            save_mask_png(obj_mask_dir / f"{frame_idx:05d}_obj{obj_id:03d}.png", binary)
            color = DEFAULT_COLORS.get(obj_id, (255, 255, 255))
            color_layer = np.zeros_like(frame)
            color_layer[:, :] = color[::-1]
            overlay = np.where(mask[..., None] > 0, (0.55 * overlay + 0.45 * color_layer).astype(np.uint8), overlay)
        save_mask_png(mask_dir / f"{frame_idx:05d}.png", combined)
        cv2.imwrite(str(overlay_dir / f"{frame_idx:05d}.jpg"), overlay)
        video_writer.write(overlay)
    video_writer.release()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    require_modules()
    if not args.video.is_file():
        raise RuntimeError(f"Video not found: {args.video}")
    if not args.yolo_weights.is_file():
        raise RuntimeError(f"YOLO weights not found: {args.yolo_weights}")
    if not args.sam2_checkpoint.is_file():
        raise RuntimeError(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")
    if args.output.exists():
        if not args.overwrite:
            raise RuntimeError(f"{args.output} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    labels = read_labels(args.labels)
    device = resolve_device(args.device)
    temp_context = tempfile.TemporaryDirectory(prefix="yolo_sam2_frames_") if not args.keep_frames else None
    frame_dir = args.output / "frames" if args.keep_frames else Path(temp_context.name)

    try:
        frames = extract_frames(args.video, frame_dir)
        prompts = select_prompts(frame_dir, args.yolo_weights, labels, args.conf, args.imgsz, device)
        skipped = [name for idx, name in enumerate(labels) if idx not in prompts]
        if not prompts:
            raise RuntimeError(f"YOLO found no prompt boxes at conf >= {args.conf}")

        predictor = build_sam2_video_predictor(
            args.sam2_config,
            str(args.sam2_checkpoint),
            device=device,
        )
        inference_state = predictor.init_state(video_path=str(frame_dir), async_loading_frames=False)

        for class_id in sorted(prompts):
            prompt = prompts[class_id]
            box = np.array(prompt["box"], dtype=np.float32)
            points = labels_array = None
            if args.add_center_point:
                x1, y1, x2, y2 = box
                points = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
                labels_array = np.array([1], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt["frame_idx"],
                obj_id=prompt["obj_id"],
                box=box,
                points=points,
                labels=labels_array,
            )

        video_segments: dict[int, dict[int, object]] = {}
        autocast_enabled = device.startswith("cuda")
        with torch.inference_mode():
            context = torch.autocast("cuda", dtype=torch.bfloat16) if autocast_enabled else torch.autocast("cpu", enabled=False)
            with context:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        int(obj_id): out_mask_logits[i].detach().cpu().numpy()
                        for i, obj_id in enumerate(out_obj_ids)
                    }

        write_outputs(args.output, frames, video_segments, labels, args.mask_threshold)
        prompt_json = {
            "version": 1,
            "video_path": str(args.video),
            "yolo_weights": str(args.yolo_weights),
            "conf": args.conf,
            "labels": labels,
            "prompts": [prompts[class_id] for class_id in sorted(prompts)],
            "skipped_classes": skipped,
        }
        (args.output / "sam2_prompts.json").write_text(json.dumps(prompt_json, indent=2) + "\n")
        print(f"Wrote SAM2 outputs to {args.output}")
        if skipped:
            print(f"Skipped classes with no YOLO prompt: {', '.join(skipped)}")
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    main()
