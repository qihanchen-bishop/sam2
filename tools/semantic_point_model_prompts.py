#!/usr/bin/env python3
"""Propose SAM2 boxes and component-aware points from a trained semantic model.

This creates reviewable proposals only. It never accepts or propagates them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_light_semseg import Predictor
from semantic_point_seed_prompts import CLASSES, clustered_masks, extent, render


def interior_point(mask: np.ndarray, confidence: np.ndarray,
                   wanted_y: float | None = None) -> list[int] | None:
    if not mask.any():
        return None
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    score = distance * np.maximum(confidence, 0.01)
    if wanted_y is not None:
        yy = np.arange(mask.shape[0])[:, None]
        score *= np.exp(-((yy - wanted_y) / 55) ** 2)
    score[~mask] = -1
    _, peak, _, (x, y) = cv2.minMaxLoc(score)
    return [int(x), int(y)] if peak > 0 else None


def components(mask: np.ndarray, minimum: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    found = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum]
    found.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    return [labels == i for i in found]


@torch.inference_mode()
def model_probabilities(predictor: Predictor, frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (predictor.width, predictor.height),
                         interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 255.0 - predictor.mean) / predictor.std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(predictor.device)
    logits = predictor.model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return np.stack([cv2.resize(channel, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
                     for channel in probabilities])


def make_proposal(frame: np.ndarray, task: str,
                  probabilities: np.ndarray, threshold: float,
                  object_visible: bool = True) -> tuple[dict, list[str]]:
    class_ids = probabilities.argmax(axis=0)
    pixel_confidence = probabilities.max(axis=0)
    confident = pixel_confidence >= threshold
    colors = clustered_masks(frame, task)
    result: dict[str, dict] = {}
    concerns = []
    for class_id, name in enumerate(CLASSES, 1):
        if name == "object" and not object_visible:
            continue
        mask = (class_ids == class_id) & confident
        minimum = 40 if name == "object" else (500 if name == "region" else 80)
        parts = components(mask, minimum)
        if not parts:
            if name != "object":
                concerns.append(f"{name}: no confident component")
            continue
        if name in ("object", "region"):
            parts = parts[:1]
        elif name in ("lefttool", "righttool"):
            parts = parts[:2]
        else:
            parts = parts[:8]
        core = np.logical_or.reduce(parts)
        if name == "righttool":
            # This fixed front camera sees the right robot's stationary silver
            # base in the upper-right corner. The semantic model sometimes
            # calls its dark cap and silver body a tool; those pixels are not
            # the moving shaft and must never become SAM2 positive points.
            # Crop only the base overlap; the shaft/tips below remain eligible.
            base_bottom = int(round(frame.shape[0] * 0.29))
            base_left = int(round(frame.shape[1] * 0.79))
            removed = int(core[:base_bottom, base_left:].sum())
            if removed:
                core[:base_bottom, base_left:] = False
                concerns.append(f"righttool: excluded {removed} stationary-base pixels")
            if not core.any():
                concerns.append("righttool: only stationary-base prediction")
                continue
        if name == "occluder":
            # The tan target tray can be joined to the cloth by a thin false
            # semantic bridge. Its color is similar, but unlike the cloth it
            # never spans a broad image row. Bound the proposal by the last
            # broad source-image cloth row before collecting box/points.
            cloth_rows = (colors["cloth"] > 0).sum(axis=1)
            broad_rows = np.flatnonzero(
                cloth_rows >= max(150, int(0.45 * cloth_rows.max())))
            if len(broad_rows):
                lower_edge = min(frame.shape[0] - 1, int(broad_rows[-1]) + 22)
                spill = int(core[lower_edge + 1:].sum())
                if spill > 250:
                    concerns.append(f"occluder: trimmed {spill} pixels below source cloth")
                core[lower_edge + 1:] = False
        if name == "object":
            color_overlap = int((core & (colors["object"] > 0)).sum())
            if color_overlap < 25:
                concerns.append("object: semantic prediction lacks task-color support")
                continue
        positives: list[list[int]] = []
        constituent_counts = {}
        if name == "occluder":
            for part_name, part_mask, count in (
                ("skin", core & (colors["cloth"] > 0), 3),
                ("red", core & (colors["red"] > 0), 2),
            ):
                candidates = components(part_mask, 20)
                chosen = 0
                for component in candidates[:count]:
                    point = interior_point(component, probabilities[class_id])
                    if point is not None:
                        positives.append(point)
                        chosen += 1
                        if int(component.sum()) >= 15000:
                            remainder = component.astype(np.uint8)
                            cv2.circle(remainder, tuple(point), 60, 0, -1)
                            second = interior_point(remainder > 0, probabilities[class_id])
                            if second is not None:
                                positives.append(second)
                                chosen += 1
                constituent_counts[part_name] = chosen
            # One connected cloth component may still contain a newly lifted
            # right-hand flap. Spatially cover it even when the semantic core
            # includes the flap, since SAM2 may ignore an unprompted lobe.
            skin_core = core & (colors["cloth"] > 0) & ~(colors["red"] > 0)
            skin_y, skin_x = np.nonzero(skin_core)
            if len(skin_x) >= 1000:
                right_band = skin_core & (np.arange(core.shape[1])[None, :]
                                          >= np.quantile(skin_x, 0.72))
                right_point = interior_point(right_band, probabilities[class_id])
                if (right_point is not None and all(
                        np.linalg.norm(np.array(right_point) - np.array(old)) >= 55
                        for old in positives)):
                    positives.append(right_point)
                    constituent_counts["skin"] += 1
            # A lifted, skin-colored flap can be a new visible part of the
            # same occluder while the semantic model still predicts only its
            # flat half. Place a *review-required* source-color point in each
            # substantial missing lobe rather than silently dropping it.
            source_skin = (colors["cloth"] > 0) & (colors["region"] == 0)
            source_skin[650:] = False
            source_skin[:, 920:] = False
            model_near = cv2.dilate(core.astype(np.uint8),
                                    np.ones((41, 41), np.uint8)) > 0
            missing_skin = source_skin & ~model_near
            extra_skin = 0
            for component in components(missing_skin, 2500)[:2]:
                point = interior_point(component,
                                       np.ones(core.shape, dtype=np.float32))
                if point is not None:
                    positives.append(point)
                    core |= component
                    extra_skin += 1
            constituent_counts["source_only_skin"] = extra_skin
            if extra_skin:
                concerns.append(f"occluder: {extra_skin} source-visible skin lobe(s) outside model core; review points")
            dark = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < 100).astype(np.uint8)
            near_cloth = cv2.dilate((colors["cloth"] > 0).astype(np.uint8),
                                    np.ones((41, 41), np.uint8))
            screw_count, _, screw_stats, screw_centers = cv2.connectedComponentsWithStats(
                dark & near_cloth, 8)
            screws = []
            for index in range(1, screw_count):
                x, y, width, height, area = screw_stats[index]
                if 30 <= area <= 850 and width <= 35 and height <= 35 and x < 880 and y > 300:
                    point = [int(round(screw_centers[index, 0])),
                             int(round(screw_centers[index, 1]))]
                    if core[point[1], point[0]] and all(np.linalg.norm(np.array(point) - np.array(old)) > 10
                           for old in positives):
                        positives.append(point)
                        screws.append(point)
                    elif not core[point[1], point[0]]:
                        concerns.append("occluder: visible screw lacks confident semantic support")
            constituent_counts["screws"] = len(screws)
            if not positives:
                point = interior_point(core, probabilities[class_id])
                if point is not None:
                    positives.append(point)
                    constituent_counts["skin"] = 1
            if constituent_counts.get("red", 0) == 0 and (colors["red"] > 0).sum() > 100:
                concerns.append("occluder: visible red part has no confident semantic point")
        elif name in ("lefttool", "righttool"):
            source_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            source_sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
            shaft_y, shaft_x = np.nonzero(core & (source_gray < 95)
                                          & ~(colors["cloth"] > 0)
                                          & ~(colors["red"] > 0))
            source_cloth_tool = np.zeros(core.shape, dtype=bool)
            axis_near = np.ones(core.shape, dtype=bool)
            if len(shaft_y) >= 100 and np.ptp(shaft_y) >= 80:
                upper = shaft_y <= np.quantile(shaft_y, 0.8)
                slope, intercept = np.polyfit(shaft_y[upper], shaft_x[upper], 1)
                yy, xx = np.indices(core.shape)
                axis_distance = np.abs(xx - (slope * yy + intercept))
                axis_near = axis_distance <= 38
                source_cloth_tool = ((axis_distance <= 27)
                                     & (yy >= np.quantile(shaft_y, 0.65)))
            # The semantic model often extends a gripper into the light-gray
            # tabletop. Such pixels are a plausible-looking fourth positive
            # point but make SAM2 propagate background as tool. Require
            # source-visible dark/metal support, and audit any missing jaw.
            tool_core = (core & (source_gray < 178) & (source_sat < 100)
                         & (~(colors["cloth"] > 0) | source_cloth_tool)
                         & ~(colors["red"] > 0)
                         & ~(colors["region"] > 0) & axis_near)
            object_entry = result.get("object")
            if object_entry is not None:
                near_object = np.zeros(core.shape, dtype=np.uint8)
                for obj_point in object_entry["positive"]:
                    cv2.circle(near_object, tuple(obj_point), 35, 1, -1)
                tool_core &= ~((colors["object"] > 0) & (near_object > 0))
                ox1, oy1, ox2, oy2 = object_entry["box"]
                tool_core[max(0, oy1 - 14):oy2 + 15,
                          max(0, ox1 - 14):ox2 + 15] = False
            if int(tool_core.sum()) < 40:
                concerns.append(f"{name}: source-image tool support is sparse")
            if not tool_core.any():
                concerns.append(f"{name}: no source-image tool support")
                continue
            ys = np.nonzero(tool_core)[0]
            lower_end = float(ys.max())
            yy = np.arange(tool_core.shape[0])[:, None]
            if name == "righttool" and len(shaft_y) >= 100 and np.ptp(shaft_y) >= 80:
                shaft_core = tool_core & (source_gray < 105)
                shaft_ceiling = float(np.quantile(shaft_y, 0.72))
                for wanted in (float(np.quantile(shaft_y, 0.35)),
                               float(np.quantile(shaft_y, 0.65))):
                    band = shaft_core & (yy <= shaft_ceiling) & (np.abs(yy - wanted) <= 34)
                    point = interior_point(band, probabilities[class_id], wanted)
                    if point is not None and point not in positives:
                        positives.append(point)
                # The two jaws straddle the shaft axis. Sample each side
                # independently so a distal pair cannot both land on the
                # same shaft or on gray background below the gripper.
                tip_floor = max(float(np.quantile(shaft_y, 0.72)), lower_end - 95)
                tip_core = tool_core & (yy >= tip_floor)
                axis_x = slope * yy + intercept
                xx = np.arange(tool_core.shape[1])[None, :]
                for side in (-1, 1):
                    jaw = tip_core & ((xx - axis_x) * side >= 6)
                    point = interior_point(jaw, probabilities[class_id], lower_end - 28)
                    if point is not None and point not in positives:
                        positives.append(point)
                    else:
                        concerns.append(f"{name}: source-supported jaw side {side} missing")
            else:
                # The left gripper stays approximately coaxial with its rod.
                targets = [(float(np.quantile(ys, 0.22)), 45),
                           (float(np.quantile(ys, 0.55)), 45),
                           (max(float(np.quantile(ys, 0.7)), lower_end - 65), 24),
                           (lower_end - 10, 24)]
                for wanted, half_band in targets:
                    band = tool_core & (np.abs(yy - wanted) <= half_band)
                    point = interior_point(band, probabilities[class_id], wanted)
                    if point is None:
                        point = interior_point(tool_core, probabilities[class_id], wanted)
                    if point is not None and point not in positives:
                        positives.append(point)
            if len(positives) < 4:
                remaining = tool_core.astype(np.uint8)
                for point in positives:
                    cv2.circle(remaining, tuple(point), 12, 0, -1)
                while len(positives) < 4:
                    point = interior_point(remaining > 0, probabilities[class_id])
                    if point is None or point in positives:
                        break
                    positives.append(point)
                    cv2.circle(remaining, tuple(point), 12, 0, -1)
            if len(positives) < 4:
                concerns.append(f"{name}: fewer than four shaft/jaw points")
        else:
            work = core.astype(np.uint8)
            for _ in range(2 if name == "object" else 3):
                point = interior_point(work, probabilities[class_id])
                if point is None:
                    break
                positives.append(point)
                # A small target still needs two interior seeds when its
                # visible core permits them; 35 px often erased the entire
                # orange/green component after the first seed.
                cv2.circle(work, tuple(point), 12 if name == "object" else 35,
                           0, -1)
        if not positives:
            concerns.append(f"{name}: no usable interior point")
            continue
        result[name] = {"box": extent(core, 8), "positive": positives,
                        "confidence": round(float(np.median(probabilities[class_id][core])), 4),
                        "components": constituent_counts}
        if name in ("lefttool", "righttool"):
            ys, xs = np.nonzero(core)
            line = []
            for quantile in (0.05, 0.75):
                wanted_y = int(round(float(np.quantile(ys, quantile))))
                nearby_x = xs[np.abs(ys - wanted_y) <= 6]
                line.append([int(round(float(np.median(nearby_x)))) if len(nearby_x)
                             else int(round(float(np.median(xs)))), wanted_y])
            result[name]["line"] = line
    for name, entry in result.items():
        negatives = []
        for other_name, other in result.items():
            if other_name == name:
                continue
            if other_name == "occluder":
                # One negative per *constituent*, not one per cloth sampling
                # point. Repeating many skin negatives can extinguish a small
                # target's single positive during SAM2 propagation.
                categories: dict[str, list[list[int]]] = {"skin": [], "red": [],
                                                          "screw": []}
                for point in other["positive"]:
                    x, y = point
                    category = ("red" if colors["red"][y, x] else
                                "skin" if colors["cloth"][y, x] else "screw")
                    categories[category].append(point)
                ex = (entry["box"][0] + entry["box"][2]) / 2
                ey = (entry["box"][1] + entry["box"][3]) / 2
                for category in ("skin", "red"):
                    if categories[category]:
                        negatives.append(min(categories[category], key=lambda point:
                                             (point[0] - ex) ** 2 + (point[1] - ey) ** 2))
                # Separate mounting screws remain separate physical parts.
                negatives.extend(categories["screw"])
            elif other_name in ("lefttool", "righttool"):
                parts = other["positive"]
                negatives.append(parts[min(1, len(parts) - 1)])
                if len(parts) >= 3:
                    negatives.append(parts[-1])
            else:
                negatives.append(other["positive"][0])
        x1, y1, x2, y2 = entry["box"]
        background = (min(frame.shape[1] - 1, x2 + 18),
                      min(frame.shape[0] - 1, y2 + 18))
        if class_ids[background[1], background[0]] == 0:
            negatives.append(list(background))
        if name == "righttool":
            # A point on the stationary base remains a useful explicit
            # confuser even when the tighter tool box excludes it.
            negatives.append([int(round(frame.shape[1] * 0.87)),
                              int(round(frame.shape[0] * 0.17))])
        entry["negative"] = list(dict.fromkeys(tuple(point) for point in negatives))
        entry["negative"] = [list(point) for point in entry["negative"]]
    return result, concerns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--point-confidence", type=float, default=0.72)
    args = parser.parse_args()
    predictor = Predictor(args.model_dir, args.device)
    expected = ["background", *CLASSES]
    if predictor.config["labels"] != expected:
        raise ValueError(f"Model labels {predictor.config['labels']} != {expected}")
    manifest = json.loads(args.manifest.read_text())
    records = {int(row["episode_index"]): row for row in manifest["all_episodes"]}
    record = records[args.episode]
    chosen = json.loads(args.events.read_text())["episodes"][str(args.episode)]
    cap = cv2.VideoCapture(record["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Could not open episode video")
    output = {"status": "model_proposal_not_reviewed", "episode_index": args.episode,
              "task": record["task"], "video_path": record["video_path"],
              "frame_count": record["frame_count"], "model_dir": str(args.model_dir.resolve()),
              "frames": {}, "concerns": {},
              "absent_events": [name for name, frame in chosen.items()
                                if name != "task" and frame is None]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for event, raw_frame in chosen.items():
        if event == "task":
            continue
        if raw_frame is None:
            continue
        frame_idx = int(raw_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Could not read frame {frame_idx}")
        probabilities = model_probabilities(predictor, frame)
        proposal, concerns = make_proposal(frame, record["task"], probabilities,
                                           args.point_confidence,
                                           object_visible=frame_idx >= int(chosen["exposed"]))
        output["frames"][str(frame_idx)] = {"event": event, "classes": proposal}
        if concerns:
            output["concerns"][str(frame_idx)] = concerns
        preview = args.output_dir / f"{event}_f{frame_idx:05d}_prompts.jpg"
        cv2.imwrite(str(preview), render(frame, frame_idx, proposal),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    cap.release()
    path = args.output_dir / "proposals.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"model_proposals={path} concerns={sum(map(len, output['concerns'].values()))}")


if __name__ == "__main__":
    main()
