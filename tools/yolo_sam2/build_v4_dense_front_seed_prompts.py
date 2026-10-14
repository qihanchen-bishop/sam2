#!/usr/bin/env python3
"""Build and preview dense front-view seed prompts for bettersetup_v4.

Existing masks are used only to draft boxes. Every generated contact sheet must be
visually reviewed before the prompts are passed to SAM2.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np


FRAME_SELECTIONS = {
    "cube": [40, 140, 250, 364, 437, 510, 650, 765],
    "paperball": [40, 110, 165, 220, 275, 330, 440, 522, 605],
    "screw": [40, 97, 146, 170, 219, 250, 292, 341, 390, 438, 487, 536],
}

OMIT = {
    "cube": {"object": {40, 140}},
    "paperball": {"object": {40, 110}},
    "screw": {
        "object": {40, 97, 292},
        "tool": {292},
        "region": {292},
    },
}

# Frames where the previous candidate mask omitted a still-visible target.  These
# boxes were redrawn against the source RGB frames; they do not come from the old
# mask.  f292 remains omitted because the screw is fully/ambiguously occluded.
MANUAL_BOXES = {
    "screw": {
        "object": {
            146: [215.0, 72.0, 260.0, 101.0],
            170: [214.0, 70.0, 260.0, 99.0],
            219: [185.0, 153.0, 237.0, 184.0],
        },
        "region": {
            341: [190.0, 316.0, 432.0, 360.0],
        },
    },
}

# A tool-only pixel for the object prompt.  This is especially important while
# the screw touches the orange tool: it tells SAM2 that adjacent tool pixels are
# not part of the screw.
OBJECT_TOOL_NEGATIVE_POINTS = {
    "screw": {
        146: (150.0, 64.0),
        170: (190.0, 49.0),
        219: (165.0, 160.0),
        250: (180.0, 200.0),
    },
}

VISIBLE_RANGES_OVERRIDES = {
    # Frame-by-frame RGB review: the screw first becomes visible at f143,
    # disappears behind the arm after f265, and reappears on the region at f329.
    # The previous [75, 265]/[325, 560] ranges caused arm pixels to hallucinate as
    # object before f143 and treated fully occluded frames as missing detections.
    "screw": {
        "object": [[143, 265], [329, 560]],
    },
}

PADDING = {
    "occluder": 3,
    "object": 4,
    "region": 0,
    "tool": 4,
    "leftarm": 4,
    "rightarm": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-prompt-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def bbox_for(mask: np.ndarray, class_id: int, pad: int) -> list[float] | None:
    ys, xs = np.where(mask == class_id)
    if not len(xs):
        return None
    height, width = mask.shape
    return [
        float(max(0, int(xs.min()) - pad)),
        float(max(0, int(ys.min()) - pad)),
        float(min(width, int(xs.max()) + 1 + pad)),
        float(min(height, int(ys.max()) + 1 + pad)),
    ]


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def frame_from_video(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def render_sheet(data: dict, output_path: Path) -> None:
    colors = {
        int(obj["id"]): tuple(int(obj["color"].lstrip("#")[i : i + 2], 16) for i in (4, 2, 0))
        for obj in data["objects"]
    }
    by_frame: dict[int, list[tuple[dict, dict]]] = {}
    for obj in data["objects"]:
        for item in obj.get("boxes", []):
            by_frame.setdefault(int(item["frame"]), []).append((obj, item))
    point_by_frame: dict[int, list[tuple[dict, dict]]] = {}
    for obj in data["objects"]:
        for item in obj.get("points", []):
            point_by_frame.setdefault(int(item["frame"]), []).append((obj, item))

    tiles = []
    for frame_index in data["prompt_frames"]:
        tile = frame_from_video(Path(data["video_path"]), int(frame_index))
        for obj, item in by_frame.get(int(frame_index), []):
            x1, y1, x2, y2 = (round(v) for v in item["box"])
            color = colors[int(obj["id"])]
            cv2.rectangle(tile, (x1, y1), (x2 - 1, y2 - 1), color, 2)
            cv2.putText(tile, obj["name"], (x1 + 2, max(15, y1 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        for obj, item in point_by_frame.get(int(frame_index), []):
            x, y = round(item["x"]), round(item["y"])
            color = (0, 0, 255) if int(item["label"]) == 0 else colors[int(obj["id"])]
            cv2.drawMarker(tile, (x, y), color, cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.rectangle(tile, (0, 0), (92, 22), (0, 0, 0), -1)
        cv2.putText(tile, f"f{frame_index}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    columns = 4
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (rows * columns - len(tiles)))
    sheet = np.vstack([np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 96])


def main() -> None:
    args = parse_args()
    prompt_paths = sorted(args.base_prompt_root.rglob("sam2_prompts.json"))
    if len(prompt_paths) != 3:
        raise RuntimeError(f"Expected 3 front seed prompts, found {len(prompt_paths)}")

    manifest = []
    for source_path in prompt_paths:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        task = source["task"]
        if task not in FRAME_SELECTIONS:
            raise RuntimeError(f"Unexpected task {task}")
        data = copy.deepcopy(source)
        frames = FRAME_SELECTIONS[task]
        data["version"] = 7
        data["prompt_frames"] = frames

        episode_index = int(data["episode_index"])
        mask_dir = (
            args.mask_root
            / "observation.images.front"
            / task
            / f"file-{episode_index:03d}_sam2_masks"
            / "mask_frames"
        )
        masks = {
            frame: cv2.imread(str(mask_dir / f"{frame:05d}.png"), cv2.IMREAD_GRAYSCALE)
            for frame in frames
        }
        if any(mask is None for mask in masks.values()):
            raise RuntimeError(f"Missing draft masks for {task}")

        boxes_by_name: dict[str, dict[int, list[float]]] = {}
        for obj in data["objects"]:
            name = obj["name"]
            class_id = int(obj["id"])
            omitted = OMIT.get(task, {}).get(name, set())
            boxes = []
            boxes_by_name[name] = {}
            for frame in frames:
                if frame in omitted:
                    continue
                box = MANUAL_BOXES.get(task, {}).get(name, {}).get(frame)
                if box is None:
                    box = bbox_for(masks[frame], class_id, PADDING[name])
                if box is None:
                    continue
                if name == "region" and frame not in MANUAL_BOXES.get(task, {}).get(name, {}):
                    box = [190.0, 276.0, 432.0, 360.0]
                boxes.append({"frame": frame, "box": box})
                boxes_by_name[name][frame] = box
            obj["boxes"] = boxes
            obj["points"] = []
            override_ranges = VISIBLE_RANGES_OVERRIDES.get(task, {}).get(name)
            if override_ranges is not None:
                obj["visible_ranges"] = override_ranges

        objects = {obj["name"]: obj for obj in data["objects"]}
        for frame, tool_box in boxes_by_name.get("tool", {}).items():
            tx, ty = center(tool_box)
            if frame in boxes_by_name.get("rightarm", {}):
                objects["rightarm"]["points"].append(
                    {"frame": frame, "x": tx, "y": ty, "label": 0}
                )
            object_box = boxes_by_name.get("object", {}).get(frame)
            if object_box is not None:
                ox, oy = center(object_box)
                objects["tool"]["points"].append(
                    {"frame": frame, "x": ox, "y": oy, "label": 0}
                )

        for frame, (tx, ty) in OBJECT_TOOL_NEGATIVE_POINTS.get(task, {}).items():
            if frame in boxes_by_name.get("object", {}) and frame in boxes_by_name.get("tool", {}):
                objects["object"]["points"].append(
                    {"frame": frame, "x": tx, "y": ty, "label": 0}
                )

        destination = args.output_root / "json" / data["video_key"] / task / "sam2_prompts.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        sheet_path = args.output_root / "review" / f"{task}_front_dense_prompts.jpg"
        render_sheet(data, sheet_path)
        manifest.append({"task": task, "prompt_frames": frames, "json": str(destination), "review": str(sheet_path)})

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps({"version": 3, "items": manifest}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"prompts": len(manifest), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
