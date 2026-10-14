#!/usr/bin/env python3
"""Render dense YOLO-generated SAM2 prompts and summarize suspicious boxes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    "occluder": (255, 160, 64),
    "tube": (138, 92, 255),
    "object": (97, 105, 255),
    "region": (119, 221, 119),
    "tool": (102, 209, 255),
    "lefttool": (102, 209, 255),
    "righttool": (173, 214, 101),
    "leftarm": (199, 146, 234),
    "rightarm": (101, 214, 173),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=240)
    parser.add_argument(
        "--columns",
        type=int,
        default=12,
        help="Wrap dense prompt previews after this many tiles.",
    )
    parser.add_argument("--low-confidence", type=float, default=0.40)
    return parser.parse_args()


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index}")
    return frame


def tile_for(frame: np.ndarray, item: dict, label: str, width: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(value)) for value in item["box"]]
    color = COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    title = f"{label} f{int(item['frame'])} c{float(item['confidence']):.2f}"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 350), 30), (0, 0, 0), -1)
    cv2.putText(frame, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def make_sheet(rows: list[list[np.ndarray]], width: int) -> np.ndarray:
    columns = max(len(row) for row in rows)
    tile_height = rows[0][0].shape[0]
    blank = np.zeros((tile_height, width, 3), dtype=np.uint8)
    return np.vstack([np.hstack(row + [blank.copy()] * (columns - len(row))) for row in rows])


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def main() -> None:
    args = parse_args()
    if args.columns < 1:
        raise RuntimeError("--columns must be positive")
    paths = sorted(args.prompt_root.glob("**/sam2_prompts.json"))
    if not paths:
        raise RuntimeError(f"No prompts under {args.prompt_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    episodes = []
    flags = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        capture = cv2.VideoCapture(data["video_path"])
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {data['video_path']}")
        rows = []
        classes = {}
        for obj in data["objects"]:
            label = obj["name"]
            boxes = sorted(obj["boxes"], key=lambda item: int(item["frame"]))
            row = [tile_for(read_frame(capture, int(item["frame"])), item, label, args.tile_width) for item in boxes]
            if row:
                rows.extend(
                    row[start : start + args.columns]
                    for start in range(0, len(row), args.columns)
                )
            confidences = [float(item["confidence"]) for item in boxes]
            areas = [box_area(item["box"]) for item in boxes]
            ratios = [max(a, b) / max(1.0, min(a, b)) for a, b in zip(areas, areas[1:])]
            classes[label] = {
                "box_count": len(boxes),
                "frames": [int(item["frame"]) for item in boxes],
                "confidence_min": min(confidences) if confidences else None,
                "confidence_mean": float(np.mean(confidences)) if confidences else None,
                "maximum_adjacent_area_ratio": max(ratios, default=1.0),
                "visible_ranges": obj.get("visible_ranges", []),
            }
            if confidences and min(confidences) < args.low_confidence:
                flags.append({"file_index": data["file_index"], "label": label, "kind": "low_confidence", "value": min(confidences)})
            if ratios and max(ratios) > 3.0:
                flags.append({"file_index": data["file_index"], "label": label, "kind": "area_jump", "value": max(ratios)})
        capture.release()
        if rows:
            output = args.output_root / f"file-{int(data['file_index']):03d}_dense_prompts.jpg"
            cv2.imwrite(str(output), make_sheet(rows, args.tile_width), [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            output = None
        episodes.append({
            "file_index": int(data["file_index"]),
            "tasks": data.get("tasks", []),
            "frame_count": int(data["frame_count"]),
            "classes": classes,
            "sheet": str(output.resolve()) if output else None,
        })
    summary = {"episodes": episodes, "flags": flags}
    summary_path = args.output_root / "prompt_review_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "flags": len(flags), "summary": str(summary_path.resolve())}))


if __name__ == "__main__":
    main()
