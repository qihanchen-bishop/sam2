#!/usr/bin/env python3
"""Build SAM2 prompts for right grippers accidentally absorbed by occluder masks.

The front-view right gripper is a pale/black mechanism moving over the green
cloth.  Earlier detector prompts learned the persistent lower-left arm base,
while the broad occluder box caused SAM2 to absorb the gripper into class 1.
This helper finds low-saturation gripper components *inside the current
occluder mask*, creates temporally stable padded boxes, and emits one-object
rightarm correction prompts for review and propagation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=220)
    parser.add_argument("--strong-area", type=int, default=480)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--range-padding", type=int, default=3)
    parser.add_argument("--box-padding", type=int, default=12)
    parser.add_argument("--max-prompts", type=int, default=20)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_box(
    frame: np.ndarray,
    semantic: np.ndarray,
    min_area: int,
) -> tuple[list[float], int] | None:
    height, width = semantic.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yy, xx = np.indices(semantic.shape)
    # The moving right gripper is in the upper-right part of the front view.
    # Restricting to current occluder pixels excludes the pale right arm, tool,
    # object and the already-correct persistent left-arm base.
    seed = (
        (semantic == 1)
        & (xx > round(width * 0.55))
        & (yy < round(height * 0.62))
        & (hsv[:, :, 1] < 55)
        & (hsv[:, :, 2] > 75)
    )
    binary = seed.astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8)
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    components = [
        row
        for row in stats[1:count]
        if int(row[cv2.CC_STAT_AREA]) >= min_area
        and int(row[cv2.CC_STAT_WIDTH]) >= 10
        and int(row[cv2.CC_STAT_HEIGHT]) >= 8
    ]
    if not components:
        return None
    x, y, w, h, area = max(components, key=lambda row: int(row[cv2.CC_STAT_AREA]))
    return [float(x), float(y), float(x + w), float(y + h)], int(area)


def temporal_runs(rows: list[dict], max_gap: int) -> list[list[dict]]:
    if not rows:
        return []
    runs: list[list[dict]] = [[rows[0]]]
    for row in rows[1:]:
        if int(row["frame"]) <= int(runs[-1][-1]["frame"]) + max_gap:
            runs[-1].append(row)
        else:
            runs.append([row])
    return runs


def choose_prompts(runs: list[list[dict]], max_prompts: int) -> list[dict]:
    chosen: dict[int, dict] = {}
    for run in runs:
        # Always keep the strongest observation from each visibility interval.
        strongest = max(run, key=lambda row: int(row["area"]))
        chosen[int(strongest["frame"])] = strongest
        # Add interval endpoints and quartiles when the gripper remains visible.
        count = min(5, len(run))
        for index in np.linspace(0, len(run) - 1, count).round().astype(int):
            row = run[int(index)]
            chosen[int(row["frame"])] = row
    rows = list(chosen.values())
    if len(rows) > max_prompts:
        # Preserve temporal coverage first, then prefer stronger detections.
        indices = np.linspace(0, len(rows) - 1, max_prompts).round().astype(int)
        rows = [sorted(rows, key=lambda row: int(row["frame"]))[int(i)] for i in indices]
    return sorted(rows, key=lambda row: int(row["frame"]))


def pad_box(box: list[float], width: int, height: int, padding: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        float(max(0, x1 - padding)),
        float(max(0, y1 - padding)),
        float(min(width, x2 + padding)),
        float(min(height, y2 + padding)),
    ]


def render_review(video_path: Path, rows: list[dict], output: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    tiles = []
    for row in rows:
        frame_index = int(row["frame"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read {video_path} frame {frame_index}")
        x1, y1, x2, y2 = (round(value) for value in row["box"])
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (199, 146, 234), 2)
        cv2.putText(
            frame,
            f"f{frame_index} area={row['area']}",
            (max(4, x1), max(20, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (199, 146, 234),
            2,
            cv2.LINE_AA,
        )
        tiles.append(frame)
    capture.release()
    columns = 4
    rows_count = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank.copy() for _ in range(rows_count * columns - len(tiles))])
    sheet = np.vstack(
        [
            np.hstack(tiles[row * columns : (row + 1) * columns])
            for row in range(rows_count)
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96])


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for complete_path in sorted(args.seg_root.glob("**/complete.json")):
        completion = read_json(complete_path)
        stored_prompt = read_json(complete_path.parent / "sam2_prompts.json")
        video_path = Path(completion["video_path"])
        frame_count = int(completion["frame_count"])
        mask_dir = complete_path.parent / "mask_frames"
        capture = cv2.VideoCapture(str(video_path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detections = []
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read {video_path} frame {frame_index}")
            semantic = cv2.imread(
                str(mask_dir / f"{frame_index:05d}.png"), cv2.IMREAD_GRAYSCALE
            )
            if semantic is None:
                raise RuntimeError(f"Missing semantic mask for {video_path} frame {frame_index}")
            candidate = candidate_box(frame, semantic, args.min_area)
            if candidate is None:
                continue
            box, area = candidate
            detections.append({"frame": frame_index, "box": box, "area": area})
        capture.release()

        runs = temporal_runs(detections, args.max_gap)
        runs = [
            run
            for run in runs
            if max(int(row["area"]) for row in run) >= args.strong_area
            or (len(run) >= 3 and max(int(row["area"]) for row in run) >= args.min_area)
        ]
        if not runs:
            continue
        selected = choose_prompts(runs, args.max_prompts)
        for row in selected:
            row["box"] = pad_box(row["box"], width, height, args.box_padding)
        ranges = []
        for run in runs:
            start = max(0, int(run[0]["frame"]) - args.range_padding)
            end = min(frame_count - 1, int(run[-1]["frame"]) + args.range_padding)
            if ranges and start <= ranges[-1][1] + 1:
                ranges[-1][1] = max(ranges[-1][1], end)
            else:
                ranges.append([start, end])

        episode = int(
            stored_prompt.get(
                "file_index",
                stored_prompt.get("episode_index", int(video_path.stem.removeprefix("file-"))),
            )
        )
        prompt = {
            "version": 1,
            "video_path": str(video_path.resolve()),
            "video_key": stored_prompt.get("video_key", "observation.images.front"),
            "chunk_index": int(stored_prompt.get("chunk_index", 0)),
            "file_index": episode,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "labels_source": stored_prompt.get("labels_source"),
            "objects": [
                {
                    "id": 6,
                    "name": "rightarm",
                    "color": "#ADD665",
                    "boxes": [
                        {"frame": int(row["frame"]), "box": row["box"]}
                        for row in selected
                    ],
                    "visible_ranges": ranges,
                }
            ],
        }
        destination = (
            args.output_root
            / prompt["video_key"]
            / f"chunk-{prompt['chunk_index']:03d}"
            / f"file-{episode:03d}"
            / "sam2_prompts.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        review_path = args.output_root / "review" / f"episode-{episode:03d}_right_gripper.jpg"
        render_review(video_path, selected, review_path)
        manifest.append(
            {
                "episode_index": episode,
                "prompt": str(destination.resolve()),
                "review": str(review_path.resolve()),
                "detections": len(detections),
                "prompts": len(selected),
                "visible_ranges": ranges,
            }
        )

    manifest.sort(key=lambda item: int(item["episode_index"]))
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"videos": len(manifest), "items": manifest}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"videos": len(manifest), "manifest": str(manifest_path.resolve())}))


if __name__ == "__main__":
    main()
