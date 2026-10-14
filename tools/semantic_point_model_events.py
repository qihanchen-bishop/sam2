#!/usr/bin/env python3
"""Find candidate transition frames from semantic masks, for visual correction.

Unlike fixed-fraction sampling, events are selected from visible class changes.
The output is provisional until a human/agent checks the source frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_light_semseg import Predictor
from semantic_point_seed_prompts import CLASSES, clustered_masks, find_tools


@torch.inference_mode()
def infer_small(predictor: Predictor, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (predictor.width, predictor.height),
                         interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 255.0 - predictor.mean) / predictor.std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(predictor.device)
    probabilities = torch.softmax(predictor.model(tensor), dim=1)[0]
    return (probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8),
            probabilities.max(dim=0).values.cpu().numpy())


def distance(a: np.ndarray, b: np.ndarray) -> float | None:
    if not a.any() or not b.any():
        return None
    field = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)
    return float(field[a].min())


def analyze_episode(predictor: Predictor, record: dict, stride: int,
                    confidence: float) -> dict:
    total = int(record["frame_count"])
    indices = sorted(set(range(0, total, stride)) | {total - 1})
    cap = cv2.VideoCapture(record["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {record['video_path']}")
    timeline = []
    initial_occluder = None
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Could not read f{frame_idx}")
        class_ids, max_probability = infer_small(predictor, frame)
        valid = max_probability >= confidence
        occluder = (class_ids == 1) & valid
        obj = (class_ids == 2) & valid
        region = (class_ids == 3) & valid
        righttool = (class_ids == 5) & valid
        if initial_occluder is None:
            initial_occluder = occluder
        overlap = int((occluder & initial_occluder).sum())
        union = int((occluder | initial_occluder).sum())
        occluder_iou = overlap / union if union else 0.0
        colors = clustered_masks(frame, record["task"])
        color_obj = colors["object"]
        color_small = cv2.resize(color_obj, (predictor.width, predictor.height),
                                 interpolation=cv2.INTER_NEAREST) > 0
        source_tool_distance = None
        if color_obj.any():
            tools = find_tools(frame, color_obj, "extra")
            if "righttool" in tools:
                line = np.asarray(tools["righttool"]["line"], float)
                direction = line[1] - line[0]
                direction /= max(np.linalg.norm(direction), 1e-6)
                tip = np.rint(line[1] + direction * 25).astype(int)
                x = int(np.clip(tip[0], 0, color_obj.shape[1] - 1))
                y = int(np.clip(tip[1], 0, color_obj.shape[0] - 1))
                source_tool_distance = float(cv2.distanceTransform(
                    (~(color_obj > 0)).astype(np.uint8), cv2.DIST_L2, 5)[y, x])
        source_region_distance = distance(color_obj > 0, colors["region"] > 0)
        timeline.append({"frame": frame_idx, "object_area": int(obj.sum()),
                         "object_color_overlap": int((obj & color_small).sum()),
                         "occluder_iou_to_first": round(occluder_iou, 4),
                         "righttool_object_distance": distance(righttool, obj),
                         "righttool_object_source_distance": source_tool_distance,
                         "object_region_distance": distance(obj, region),
                         "object_region_source_distance": source_region_distance})
    cap.release()

    def first_where(predicate, after: int = -1) -> int | None:
        return next((row["frame"] for row in timeline
                     if row["frame"] > after and predicate(row)), None)

    exposed = first_where(lambda row: row["object_area"] >= 30 and
                          row["object_color_overlap"] >= 12)
    if exposed is None:
        exposed = first_where(lambda row: row["object_area"] >= 80 and
                              row["object_color_overlap"] >= 8)
    contact = first_where(lambda row: row["righttool_object_distance"] is not None and
                          row["righttool_object_distance"] <= 8 and
                          row["righttool_object_source_distance"] is not None and
                          row["righttool_object_source_distance"] <= 22,
                          after=exposed if exposed is not None else -1)
    placed = first_where(lambda row: row["object_region_distance"] is not None and
                         row["object_region_distance"] <= 6 and
                         row["object_region_source_distance"] is not None and
                         row["object_region_source_distance"] <= 18,
                         after=contact if contact is not None else -1)
    lift = first_where(lambda row: row["occluder_iou_to_first"] < 0.92 and
                       (exposed is None or row["frame"] <= exposed))
    events = {"task": record["task"], "first": 0,
              "occluder_contact": lift, "exposed": exposed,
              "object_contact": contact, "region_contact": placed,
              "last": total - 1}
    missing = [name for name, value in events.items()
               if name != "task" and value is None]
    occupied = {value for name, value in events.items()
                if name != "task" and value is not None}
    # Dense conditioning bracketing the appearance/contact/placement changes.
    # These are still candidates, not accepted prompts.
    for name, offsets in (("occluder_contact", (-8, 8)),
                          ("exposed", (-8, 8)),
                          ("object_contact", (-18, -6, 6, 18)),
                          ("region_contact", (-18, -6, 6, 18))):
        center = events[name]
        if center is None:
            continue
        for offset in offsets:
            candidate = min(total - 1, max(0, center + offset))
            if candidate not in occupied:
                events[f"{name}_{offset:+d}"] = candidate
                occupied.add(candidate)
    for fraction in (0.25, 0.5, 0.75):
        candidate = int(round((total - 1) * fraction))
        if candidate not in occupied:
            events[f"extra_{int(fraction * 100)}pct"] = candidate
            occupied.add(candidate)
    return {"events": events, "missing": missing, "timeline": timeline,
            "status": "model_event_candidates_not_reviewed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.62)
    args = parser.parse_args()
    predictor = Predictor(args.model_dir, args.device)
    expected = ["background", *CLASSES]
    if predictor.config["labels"] != expected:
        raise ValueError(f"Model labels {predictor.config['labels']} != {expected}")
    manifest = json.loads(args.manifest.read_text())
    records = {int(row["episode_index"]): row for row in manifest["all_episodes"]}
    result = {"status": "candidates_not_reviewed", "model_dir": str(args.model_dir.resolve()),
              "episodes": {}, "analyses": {}}
    for episode in args.episodes:
        analysis = analyze_episode(predictor, records[episode], args.stride,
                                   args.confidence)
        result["analyses"][str(episode)] = analysis
        result["episodes"][str(episode)] = analysis["events"]
        print(f"episode={episode} events={analysis['events']} missing={analysis['missing']}",
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"event_candidates={args.output}")


if __name__ == "__main__":
    main()
