#!/usr/bin/env python3
"""Verify frame-level integrity of explicitly accepted semantic masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*",
                        help="Check only these accepted episodes; default checks all")
    args = parser.parse_args()
    acceptance = json.loads(args.acceptance.read_text())
    class_count = len(acceptance["class_order"])
    results: dict[str, dict] = {}
    errors: list[str] = []
    for episode_text, record in acceptance["episodes"].items():
        if record["status"] != "accepted":
            continue
        episode = int(episode_text)
        if args.episodes is not None and episode not in args.episodes:
            continue
        prompt = json.loads(Path(record["prompt"]).read_text())
        total = int(prompt["frame_count"])
        mask_dir = Path(record["mask_root"]) / "mask_frames"
        paths = sorted(mask_dir.glob("*.png"))
        cap = cv2.VideoCapture(prompt["video_path"])
        if not cap.isOpened():
            errors.append(f"ep{episode}: source video unreadable")
            continue
        video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if video_total != total or len(paths) != total:
            errors.append(f"ep{episode}: metadata/video/mask counts {total}/{video_total}/{len(paths)}")
        names = [f"{index:05d}.png" for index in range(total)]
        if [path.name for path in paths] != names:
            errors.append(f"ep{episode}: missing or extra mask filenames")
            continue
        visibility = {}
        for obj in prompt["objects"]:
            if "visible_ranges" in obj:
                visibility[int(obj["id"])] = set(
                    index for start, stop in obj["visible_ranges"]
                    for index in range(int(start), int(stop) + 1))
        counts = np.zeros(class_count, dtype=np.int64)
        visible_frames = np.zeros(class_count, dtype=np.int64)
        for index, path in enumerate(paths):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (height, width):
                errors.append(f"ep{episode} f{index}: unreadable mask or wrong dimensions")
                continue
            if int(mask.max()) >= class_count:
                errors.append(f"ep{episode} f{index}: out-of-range class ID {int(mask.max())}")
                continue
            area = np.bincount(mask.reshape(-1), minlength=class_count)
            for class_id, allowed in visibility.items():
                if index not in allowed and area[class_id]:
                    errors.append(f"ep{episode} f{index}: class {class_id} outside visible range")
                if class_id == 2 and index in allowed and not area[class_id]:
                    errors.append(f"ep{episode} f{index}: visible object has no pixels")
            counts += area
            visible_frames += (area > 0)
        result = {"frames": total, "dimensions_wh": [width, height], "fps": fps,
                  "pixel_counts": dict(zip(acceptance["class_order"], map(int, counts))),
                  "frames_with_class": dict(zip(acceptance["class_order"],
                                                map(int, visible_frames)))}
        results[episode_text] = result
        print(f"ep{episode} checked={total} classes={result['frames_with_class']}", flush=True)
    report = {"status": "passed" if not errors else "failed", "episodes": results,
              "errors": errors, "acceptance": str(args.acceptance.resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"integrity={args.output} status={report['status']}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
