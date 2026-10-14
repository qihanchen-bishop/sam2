#!/usr/bin/env python3
"""Propose source-color/semantic-model recovery when SAM2 loses a colored target.

This writes a new provisional mask root. It never accepts a video or edits the
SAM2 output in place. Black targets are intentionally excluded: dark tools are
not sufficiently distinguishable by the source-color gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from infer_light_semseg import Predictor
from semantic_point_model_prompts import model_probabilities
from semantic_point_seed_prompts import clustered_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--semantic-threshold", type=float, default=0.53)
    args = parser.parse_args()
    prompt = json.loads(args.prompt.read_text())
    if prompt["task"] not in ("o_n", "g_n"):
        raise ValueError("Only orange/green targets can use source-color recovery")
    obj = next(item for item in prompt["objects"] if item["name"] == "object")
    visible_start = int(obj["visible_ranges"][0][0])
    total = int(prompt["frame_count"])
    predictor = Predictor(args.model_dir, args.device)
    expected = ["background", "occluder", "object", "region", "lefttool", "righttool"]
    if predictor.config["labels"] != expected:
        raise ValueError(f"Model labels mismatch: {predictor.config['labels']}")
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Could not open source video")
    source = args.mask_root / "mask_frames"
    if len(list(source.glob("*.png"))) != total:
        raise ValueError("Source masks are incomplete")
    target = args.output / "mask_frames"
    target.mkdir(parents=True, exist_ok=True)
    report = {"status": "provisional_not_accepted", "episode_index": prompt["episode_index"],
              "source_mask_root": str(args.mask_root.resolve()),
              "source_prompt": str(args.prompt.resolve()),
              "model_dir": str(args.model_dir.resolve()), "frame_count": total,
              "visible_start": visible_start, "recovered_frames": 0,
              "recovered_pixels": 0, "low_object_frames": [],
              "color_missing_frames": [], "ambiguous_candidates": [],
              "large_centroid_jumps": [], "frame_summary": []}
    previous_center = None
    for frame_idx in range(total):
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Cannot decode source frame {frame_idx}")
        path = source / f"{frame_idx:05d}.png"
        old = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if old is None or old.shape != frame.shape[:2]:
            raise ValueError(f"Invalid source mask {path}")
        new = old.copy()
        source_area = int((old == 2).sum())
        color_area = 0
        semantic_support = 0.0
        if frame_idx >= visible_start:
            colors = clustered_masks(frame, prompt["task"])
            anchor = colors["object"] > 0
            color_area = int(anchor.sum())
            if color_area >= 70:
                ys, xs = np.nonzero(anchor)
                center = np.array([float(np.median(xs)), float(np.median(ys))])
                jump = (float(np.linalg.norm(center - previous_center))
                        if previous_center is not None else 0.0)
                if jump > 85:
                    report["large_centroid_jumps"].append([frame_idx, round(jump, 1)])
                probabilities = model_probabilities(predictor, frame)
                semantic_support = float(np.median(probabilities[2][anchor]))
                if semantic_support < 0.12:
                    report["ambiguous_candidates"].append(
                        [frame_idx, round(semantic_support, 3)])
                if jump <= 85 and semantic_support >= 0.12:
                    guard = cv2.dilate(anchor.astype(np.uint8),
                                       np.ones((13, 13), np.uint8)) > 0
                    candidate = anchor | ((probabilities[2] >= args.semantic_threshold)
                                          & guard)
                    candidate = cv2.morphologyEx(candidate.astype(np.uint8),
                                                  cv2.MORPH_CLOSE,
                                                  np.ones((5, 5), np.uint8)) > 0
                    # Object has highest pixel priority. Restore any tray area
                    # that an off-color SAM2 object mask had previously covered.
                    old_obj = new == 2
                    new[old_obj] = np.where(colors["region"][old_obj] > 0, 3, 0)
                    new[candidate] = 2
                    report["recovered_frames"] += 1
                    report["recovered_pixels"] += int(candidate.sum()) - source_area
                    previous_center = center
            else:
                report["color_missing_frames"].append(frame_idx)
            if int((new == 2).sum()) < 70:
                report["low_object_frames"].append(frame_idx)
        report["frame_summary"].append({"frame": frame_idx,
                                        "source_object_pixels": source_area,
                                        "color_object_pixels": color_area,
                                        "final_object_pixels": int((new == 2).sum()),
                                        "semantic_support": round(semantic_support, 3)})
        if not cv2.imwrite(str(target / f"{frame_idx:05d}.png"), new):
            raise RuntimeError(f"Could not write mask f{frame_idx}")
        if frame_idx % 100 == 0:
            print(f"object_recovery {frame_idx + 1}/{total}", flush=True)
    cap.release()
    (args.output / "object_recovery.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"provisional_object_recovery={args.output} "
          f"recovered_frames={report['recovered_frames']} "
          f"low_object_frames={len(report['low_object_frames'])}")


if __name__ == "__main__":
    main()
