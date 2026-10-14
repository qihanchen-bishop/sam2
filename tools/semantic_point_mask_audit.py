#!/usr/bin/env python3
"""Render contact-window and endpoint mask evidence for one bootstrap episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = np.array([(0, 0, 0), (170, 100, 20), (0, 145, 250),
                   (45, 160, 40), (255, 70, 65), (220, 75, 220)], dtype=np.uint8)
NAMES = ("background", "occluder", "object", "region", "lefttool", "righttool")


def get_selected_frames(cap: cv2.VideoCapture, indices: list[int]) -> dict[int, np.ndarray]:
    """Decode once in frame order; repeated random H.264 seeks are very slow."""
    wanted = set(indices)
    result: dict[int, np.ndarray] = {}
    for index in range(max(indices) + 1):
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Cannot read video frame {index}")
        if index in wanted:
            result[index] = frame
    return result


def get_mask(root: Path, index: int) -> np.ndarray:
    path = root / f"{index:05d}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.max() >= len(COLORS):
        raise ValueError(f"Unexpected class id in {path}: {mask.max()}")
    return mask


def overlay(frame: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    if frame.shape[:2] != mask.shape:
        raise ValueError(f"Frame/mask dimensions differ at {label}")
    colored = COLORS[mask]
    canvas = cv2.addWeighted(frame, 0.58, colored, 0.42, 0)
    canvas[mask == 0] = frame[mask == 0]
    return cv2.putText(canvas, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--extra-frame", type=int, nargs="*", default=[],
                        help="Additional source frames to inspect without changing prompts")
    args = parser.parse_args()
    data = json.loads(args.prompt.read_text())
    expected = int(data["frame_count"])
    mask_dir = args.mask_root / "mask_frames"
    if len(list(mask_dir.glob("*.png"))) != expected:
        raise ValueError(f"Incomplete mask frames: expected {expected}")
    cap = cv2.VideoCapture(data["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Could not open source video")
    events = sorted(set(map(int, data["prompt_frames"])) | set(args.extra_frame))
    if any(not 0 <= event < expected for event in events):
        raise ValueError("Audit frame outside source video")
    ordered = sorted(set(min(expected - 1, max(0, event + offset))
                         for event in events for offset in (-args.radius, 0, args.radius)))
    frames = get_selected_frames(cap, ordered)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for event in events:
        tiles = []
        for offset in (-args.radius, 0, args.radius):
            index = min(expected - 1, max(0, event + offset))
            frame = frames[index]
            mask = get_mask(mask_dir, index)
            tile = overlay(frame, mask, f"ep{data['episode_index']} f{index}")
            tile = cv2.resize(tile, (600, 456), interpolation=cv2.INTER_AREA)
            tiles.append(tile)
        sheet = np.hstack(tiles)
        path = args.output / f"event_f{event:05d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"event_sheet={path}", flush=True)
    for index in ordered:
        mask = get_mask(mask_dir, index)
        counts = np.bincount(mask.reshape(-1), minlength=len(NAMES))
        row = {"frame": index, "pixels": dict(zip(NAMES, map(int, counts)))}
        reverse_path = args.mask_root / "directional_mask_frames/reverse" / f"{index:05d}.png"
        forward_path = args.mask_root / "directional_mask_frames/forward" / f"{index:05d}.png"
        if reverse_path.exists() and forward_path.exists():
            forward = cv2.imread(str(forward_path), cv2.IMREAD_GRAYSCALE)
            reverse = cv2.imread(str(reverse_path), cv2.IMREAD_GRAYSCALE)
            row["forward_reverse_iou"] = {}
            for class_id, name in enumerate(NAMES[1:], start=1):
                union = (forward == class_id) | (reverse == class_id)
                if union.any():
                    inter = (forward == class_id) & (reverse == class_id)
                    row["forward_reverse_iou"][name] = round(float(inter.sum() / union.sum()), 4)
        rows.append(row)
    cap.release()
    report = {"episode_index": data["episode_index"], "status": "audit_not_accepted",
              "frame_count": expected, "checked_frames": rows}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"audit={args.output / 'audit.json'}")


if __name__ == "__main__":
    main()
