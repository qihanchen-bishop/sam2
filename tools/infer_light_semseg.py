#!/usr/bin/env python3
"""Run the packaged front-view U-Net on one image or video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    return parser.parse_args()


class Predictor:
    def __init__(self, model_dir: Path, device: str) -> None:
        self.config = json.loads((model_dir / "model_config.json").read_text())
        self.device = torch.device(device)
        self.model = torch.jit.load(str(model_dir / self.config["torchscript"]), map_location=self.device).eval()
        self.width, self.height = self.config["input_size_wh"]
        self.mean = np.asarray(self.config["normalization_mean"], dtype=np.float32)
        self.std = np.asarray(self.config["normalization_std"], dtype=np.float32)
        self.palette = np.asarray(self.config["palette_rgb"], dtype=np.uint8)

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        class_ids = self.model(tensor).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(class_ids, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    def colorize_bgr(self, class_ids: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(self.palette[class_ids], cv2.COLOR_RGB2BGR)


def write_image_outputs(
    predictor: Predictor,
    bgr: np.ndarray,
    stem: str,
    output_dir: Path,
    alpha: float,
) -> None:
    class_ids = predictor.predict(bgr)
    color = predictor.colorize_bgr(class_ids)
    overlay = cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0)
    cv2.imwrite(str(output_dir / f"{stem}_class_ids.png"), class_ids)
    cv2.imwrite(str(output_dir / f"{stem}_color.png"), color)
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)


def run_video(predictor: Predictor, input_path: Path, output_dir: Path, alpha: float) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    class_path = output_dir / f"{input_path.stem}_class_ids.mkv"
    overlay_path = output_dir / f"{input_path.stem}_overlay.mp4"
    class_writer = cv2.VideoWriter(str(class_path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height), False)
    overlay_writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
    if not class_writer.isOpened() or not overlay_writer.isOpened():
        capture.release()
        raise SystemExit("Could not open output video writer (requires FFV1 and mp4v codecs).")
    frames = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        class_ids = predictor.predict(bgr)
        color = predictor.colorize_bgr(class_ids)
        overlay = cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0)
        class_writer.write(class_ids)
        overlay_writer.write(overlay)
        frames += 1
        if frames % 100 == 0:
            print(f"processed_frames={frames}", flush=True)
    capture.release()
    class_writer.release()
    overlay_writer.release()
    print(f"class_id_video={class_path}", flush=True)
    print(f"overlay_video={overlay_path}", flush=True)
    print(f"frames={frames}", flush=True)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise SystemExit("--overlay-alpha must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(args.model_dir, args.device)
    if args.input.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {args.input}")
        write_image_outputs(predictor, image, args.input.stem, args.output_dir, args.overlay_alpha)
        print(f"output_dir={args.output_dir}", flush=True)
    else:
        run_video(predictor, args.input, args.output_dir, args.overlay_alpha)


if __name__ == "__main__":
    main()
