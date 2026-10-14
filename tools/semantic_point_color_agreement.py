#!/usr/bin/env python3
"""Audit visible colored target pixels against a provisional mask sequence.

This is an appearance cross-check, not an acceptance decision or a substitute
for original-resolution review at contact and endpoint frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from semantic_point_seed_prompts import best_component


def object_color_roi(frame: np.ndarray, task: str) -> np.ndarray:
    """Same target-color gate as clustered_masks, cropped to its valid ROI."""
    hsv = cv2.cvtColor(frame[310:, 600:], cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if task == "o_n":
        candidate = ((hue >= 3) & (hue <= 29) & (saturation >= 130)
                     & (value >= 165)).astype(np.uint8)
    else:
        candidate = ((hue >= 32) & (hue <= 98) & (saturation >= 65)
                     & (value >= 50)).astype(np.uint8)
    opened = cv2.morphologyEx(candidate, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
    return best_component(opened, 70) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompt = json.loads(args.prompt.read_text())
    if prompt["task"] not in ("o_n", "g_n"):
        raise ValueError("Color agreement only supports orange/green targets")
    object_record = next(item for item in prompt["objects"] if item["name"] == "object")
    first_visible = int(object_record["visible_ranges"][0][0])
    total = int(prompt["frame_count"])
    mask_dir = args.mask_root / "mask_frames"
    if len(list(mask_dir.glob("*.png"))) != total:
        raise ValueError("Incomplete mask sequence")
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Could not open source video")
    rows = []
    for frame_idx in range(total):
        okay, frame = cap.read()
        mask = cv2.imread(str(mask_dir / f"{frame_idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if not okay or mask is None or mask.shape != frame.shape[:2]:
            raise ValueError(f"Bad video/mask pair at f{frame_idx}")
        if frame_idx < first_visible:
            continue
        color = object_color_roi(frame, prompt["task"])
        labeled = mask[310:, 600:] == 2
        color_area = int(color.sum())
        label_area = int((mask == 2).sum())
        intersection = int((color & labeled).sum())
        guard = cv2.dilate(color.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        supported = int((guard & labeled).sum())
        rows.append({"frame": frame_idx, "color_pixels": color_area,
                     "label_pixels": label_area,
                     "visible_color_recall": round(intersection / max(color_area, 1), 4),
                     "label_near_color_fraction": round(supported / max(label_area, 1), 4)})
        if frame_idx % 100 == 0:
            print(f"color_agreement {frame_idx + 1}/{total}", flush=True)
    cap.release()
    bad_recall = [row["frame"] for row in rows
                  if row["color_pixels"] >= 70 and row["visible_color_recall"] < 0.55]
    off_color = [row["frame"] for row in rows
                 if row["label_pixels"] >= 70
                 and row["label_near_color_fraction"] < 0.5]
    report = {"status": "review_only_not_accepted", "episode_index": prompt["episode_index"],
              "task": prompt["task"], "prompt": str(args.prompt.resolve()),
              "mask_root": str(args.mask_root.resolve()), "first_visible": first_visible,
              "frames_checked": len(rows), "bad_recall_frames": bad_recall,
              "off_color_frames": off_color, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"color_agreement={args.output} bad_recall={len(bad_recall)} "
          f"off_color={len(off_color)}", flush=True)


if __name__ == "__main__":
    main()
