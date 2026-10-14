#!/usr/bin/env python3
"""Create inspectable, component-aware seed prompt *proposals* for newur front videos.

This is independent of the legacy prompt generation/review pipeline.  Nothing
from this file is accepted for SAM2 until its preview has been checked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("occluder", "object", "region", "lefttool", "righttool")
IDS = {name: i + 1 for i, name in enumerate(CLASSES)}
COLORS = {"occluder": (160, 20, 170), "object": (0, 150, 255),
          "region": (30, 190, 30), "lefttool": (255, 80, 40),
          "righttool": (240, 70, 230)}
# Full-resolution visual corrections for the pilot.  Preserve these as
# provenance instead of silently changing the generic geometry heuristic.
MANUAL_OVERRIDES = {
    (0, 700): {"righttool": {"box": [720, 0, 950, 460],
                             "positive": [(880, 130), (834, 246),
                                          (747, 411), (801, 430)]}},
    (0, 1150): {"righttool": {"positive": [(895, 191), (851, 300),
                                           (795, 401), (837, 420)]}},
    (1, 1190): {"righttool": {"positive": [(874, 248), (830, 352),
                                           (777, 425), (807, 442)]}},
    (31, 900): {"righttool": {"positive": [(932, 231), (871, 368),
                                           (805, 466), (857, 457)]}},
}
MANUAL_DROP_CLASSES = {(30, 80): ("object",)}  # Black gripper tip, not exposed block.


def interior_points(mask: np.ndarray, count: int, min_spacing: int = 35) -> list[tuple[int, int]]:
    work = (mask > 0).astype(np.uint8)
    if not work.any():
        return []
    dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
    found = []
    for _ in range(count):
        _, peak, _, pos = cv2.minMaxLoc(dist)
        if peak < 2:
            break
        found.append(pos)
        cv2.circle(dist, pos, min_spacing, 0, -1)
    return [(int(x), int(y)) for x, y in found]


def best_component(mask: np.ndarray, min_area: int = 25,
                   max_area: int | None = None) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area
                  and (max_area is None or stats[i, cv2.CC_STAT_AREA] <= max_area)]
    if not candidates:
        return np.zeros_like(mask, dtype=np.uint8)
    winner = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return (labels == winner).astype(np.uint8)


def extent(mask: np.ndarray, padding: int = 8) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
            min(mask.shape[1] - 1, int(xs.max()) + padding),
            min(mask.shape[0] - 1, int(ys.max()) + padding)]


def clustered_masks(frame: np.ndarray, task: str) -> dict[str, np.ndarray]:
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    yy, xx = np.indices((h, w))
    cloth = ((hh >= 5) & (hh <= 28) & (ss >= 25) & (ss <= 130)
             & (vv >= 105) & (vv <= 245) & (xx >= 60) & (xx <= 1080)
             & (yy >= 155) & (yy <= 730)).astype(np.uint8)
    cloth = cv2.morphologyEx(cloth, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8))
    cloth = best_component(cloth, 1500)
    red = (((hh <= 12) | (hh >= 167)) & (ss >= 72) & (vv >= 65)
           & (xx >= 90) & (xx <= 1080) & (yy >= 160) & (yy <= 730)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # The target tray is below/right of the cloth.  Keep the largest warm disk.
    # The tray can move horizontally between episodes; color/shape and its
    # lower-frame position are stable, but a right-side-only ROI is not.
    region = ((hh >= 9) & (hh <= 34) & (ss >= 48) & (ss <= 170)
              & (vv >= 55) & (vv <= 195) & (xx >= 100) & (yy >= 650)).astype(np.uint8)
    region = best_component(cv2.morphologyEx(region, cv2.MORPH_CLOSE,
                                             np.ones((9, 9), np.uint8)), 2000)
    if task == "o_n":
        obj = ((hh >= 3) & (hh <= 29) & (ss >= 130) & (vv >= 165)
               & (xx >= 600) & (yy >= 310)).astype(np.uint8)
    elif task == "g_n":
        # The green block starts under the cloth on the right but can finish
        # in a tray around x500. A right-only ROI silently discarded correct
        # terminal object prompts and hid the failure behind the region mask.
        obj = ((hh >= 32) & (hh <= 98) & (ss >= 65) & (vv >= 50)
               & (xx >= 400) & (yy >= 310)).astype(np.uint8)
    else:
        # Black targets: compact dark components, unlike the long, thin tools.
        # Some black-target episodes expose the block left of x660 before
        # grasping; an x660 crop discarded the entire source component and
        # suppressed otherwise valid semantic object proposals. Keep the ROI
        # left of the right-hand gripper but still exclude the mounting screws
        # and left base through shape/area checks below.
        dark = ((vv < 95) & (xx >= 480) & (yy >= 355) & (yy <= 850)).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(dark, 8)
        obj = np.zeros((h, w), dtype=np.uint8)
        candidates = []
        for i in range(1, count):
            x, y, cw, ch, area = stats[i]
            if 80 <= area <= 13000 and 0.32 <= cw / max(ch, 1) <= 4.5 and cw <= 175 and ch <= 150:
                candidates.append((i, area, centers[i, 0]))
        if candidates:
            # Prefer the central small block to stray dark table marks.
            winner = max(candidates, key=lambda c: c[1] - abs(c[2] - 820) * 2)[0]
            obj = (labels == winner).astype(np.uint8)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    obj = best_component(obj, 70)
    return {"cloth": cloth, "red": red, "region": region, "object": obj}


def find_tools(frame: np.ndarray, object_mask: np.ndarray,
               event_name: str) -> dict[str, dict]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 55, 155)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, minLineLength=130, maxLineGap=12)
    candidates = []
    if lines is not None:
        for raw in lines[:, 0]:
            x1, y1, x2, y2 = map(int, raw)
            if y1 > y2:
                x1, x2, y1, y2 = x2, x1, y2, y1
            dy, dx = y2 - y1, x2 - x1
            length = np.hypot(dx, dy)
            if dy < 120 or abs(dx) > dy * 0.85 or y1 > 480 or y2 > 720:
                continue
            mx = (x1 + x2) / 2
            if not 310 <= mx <= 1180:
                continue
            candidates.append((length, np.array([x1, y1], float),
                               np.array([x2, y2], float), mx))
    candidates.sort(reverse=True, key=lambda x: x[0])
    chosen = []
    for candidate in candidates:
        if all(abs(candidate[3] - other[3]) > 110 for other in chosen):
            chosen.append(candidate)
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        return {}
    chosen.sort(key=lambda x: x[3])
    result = {}
    for name, (_, top, bottom, _) in zip(("lefttool", "righttool"), chosen):
        direction = (bottom - top) / np.linalg.norm(bottom - top)
        # Two shaft sites plus two jaw sites.  The left jaw is almost coaxial
        # with its shaft; the right jaw forks around the object at contact.
        shaft = [top * 0.62 + bottom * 0.38, top * 0.25 + bottom * 0.75]
        if name == "lefttool":
            offsets = (5, 25) if event_name == "occluder_contact" else (-30, -6)
            tip = [bottom + direction * offset for offset in offsets]
        else:
            normal = np.array([direction[1], -direction[0]])
            object_box = extent(object_mask)
            fork = 20 if object_box is None else max(20, min(38, (object_box[2] - object_box[0]) / 2 + 8))
            center = bottom + direction * 20
            tip = [center - normal * fork, center + normal * fork]
        points = [(int(round(p[0])), int(round(p[1]))) for p in shaft + tip]
        corners = np.stack((top, bottom + direction * 55))
        box = [max(0, int(corners[:, 0].min()) - 24),
               max(0, int(corners[:, 1].min()) - 24),
               min(frame.shape[1] - 1, int(corners[:, 0].max()) + 24),
               min(frame.shape[0] - 1, int(corners[:, 1].max()) + 24)]
        result[name] = {"box": box, "points": points,
                        "line": [top.astype(int).tolist(), bottom.astype(int).tolist()]}
    return result


def proposal(frame: np.ndarray, task: str, event_name: str,
             episode: int, frame_idx: int,
             *, object_hidden: bool = False) -> dict[str, dict]:
    masks = clustered_masks(frame, task)
    if object_hidden:
        masks["object"][:] = 0
    tools = find_tools(frame, masks["object"], event_name)
    result = {}
    cloth = masks["cloth"]
    if cloth.any():
        positives = interior_points(cloth, 8, 85)
        positives += interior_points(masks["red"], 4, 75)
        # Small dark mounting screws are occluder constituents too.
        dark = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < 100).astype(np.uint8)
        vicinity = cv2.dilate(cloth, np.ones((41, 41), np.uint8))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(dark & vicinity, 8)
        for i in range(1, count):
            x, y, cw, ch, area = stats[i]
            if 30 <= area <= 850 and cw <= 35 and ch <= 35 and x < 880 and y > 300:
                positives.append(tuple(map(int, centers[i].astype(int))))
        result["occluder"] = {"box": extent(cloth, 16), "positive": positives,
                               "components": {"skin": len(interior_points(cloth, 8, 85)),
                                              "red": len(interior_points(masks["red"], 4, 75)),
                                              "screws": len(positives) - len(interior_points(cloth, 8, 85)) - len(interior_points(masks["red"], 4, 75))}}
    for name in ("object", "region"):
        mask = masks[name]
        if mask.any():
            points = interior_points(mask, 2 if name == "object" else 3,
                                     16 if name == "object" else 42)
            result[name] = {"box": extent(mask, 10), "positive": points}
    for name, data in tools.items():
        result[name] = {"box": data["box"], "positive": data["points"],
                        "line": data["line"]}
    for name, changed in MANUAL_OVERRIDES.get((episode, frame_idx), {}).items():
        result[name].update(changed)
    for name in MANUAL_DROP_CLASSES.get((episode, frame_idx), ()):
        result.pop(name, None)
    if "occluder" in result:
        # A skin-color pixel on a metallic tip is not an occluder prompt.
        # Preserve constituent ordering so representative negatives remain valid.
        occluder = result["occluder"]
        components = occluder["components"]
        old_points = occluder["positive"]
        occupied_points = [point for other_name in ("lefttool", "righttool", "object")
                           for point in result.get(other_name, {}).get("positive", [])]
        kept = []
        start = 0
        for part_name in ("skin", "red", "screws"):
            segment = old_points[start:start + components[part_name]]
            filtered = [point for point in segment if all(
                np.linalg.norm(np.array(point) - np.array(tool_point)) >= 12
                for tool_point in occupied_points)]
            kept.extend(filtered)
            components[part_name] = len(filtered)
            start += len(segment)
        occluder["positive"] = kept
    # A negative at each competing visible constituent, plus one adjacent
    # background point.  Do not place negatives on target-class pixels.
    for name, data in result.items():
        neg = []
        for other_name, other in result.items():
            if other_name != name:
                if other_name == "occluder":
                    parts = other.get("components", {})
                    skin = parts.get("skin", 0)
                    red = parts.get("red", 0)
                    # One representative each for skin, red tubing, screw.
                    # Too many negatives suppress small SAM2 targets.
                    for segment in (other["positive"][:skin],
                                    other["positive"][skin:skin + red],
                                    other["positive"][skin + red:]):
                        if segment:
                            neg.append(segment[0])
                elif other_name in ("lefttool", "righttool"):
                    neg.extend((other["positive"][0], other["positive"][2]))
                else:
                    neg.append(other["positive"][0])
        box = data["box"]
        bg = (min(frame.shape[1] - 1, box[2] + 15), min(frame.shape[0] - 1, box[3] + 15))
        neg.append(bg)
        if name == "righttool":
            # Fixed gray robot base sits inside many right-shaft boxes.
            neg.append((1010, 145))
        if name == "lefttool":
            # The loose power cable must not become part of the left shaft.
            neg.append((230, 55))
        data["negative"] = list(dict.fromkeys(neg))
    return result


def render(frame: np.ndarray, frame_idx: int, proposal_data: dict[str, dict]) -> np.ndarray:
    canvas = frame.copy()
    for name, data in proposal_data.items():
        color = COLORS[name]
        x1, y1, x2, y2 = data["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, name, (x1 + 4, max(18, y1 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        for x, y in data["positive"]:
            cv2.circle(canvas, (x, y), 7, color, -1)
            cv2.circle(canvas, (x, y), 8, (255, 255, 255), 1)
        for x, y in data["negative"]:
            cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_TILTED_CROSS, 10, 2)
    cv2.putText(canvas, f"f{frame_idx} PROPOSAL - NOT REVIEWED", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-filter", type=int)
    parser.add_argument("--extra", type=int, nargs="*", default=[])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    event_data = json.loads(args.events.read_text())
    records = {str(row["episode_index"]): row for row in manifest["seed_episodes"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for episode_text, chosen in event_data["episodes"].items():
        record = records[str(int(episode_text))]
        episode = int(episode_text)
        if args.episode_filter is not None and episode != args.episode_filter:
            continue
        ep_dir = args.output_dir / f"ep{episode:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        out = {"status": "proposal_not_reviewed", "episode_index": episode,
               "task": record["task"], "video_path": record["video_path"],
               "frame_count": record["frame_count"], "frames": {}}
        selections = [(name, value) for name, value in chosen.items() if name != "task"]
        selections += [("extra", value) for value in args.extra]
        for event_name, raw_frame_idx in selections:
            frame_idx = int(raw_frame_idx)
            image_path = (args.frame_dir / f"ep{episode:06d}" /
                          f"{event_name}_f{frame_idx:05d}.png")
            frame = cv2.imread(str(image_path)) if image_path.is_file() else None
            if frame is None:
                if event_name != "extra":
                    raise FileNotFoundError(image_path)
                cap = cv2.VideoCapture(record["video_path"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                okay, frame = cap.read()
                cap.release()
                if not okay:
                    raise RuntimeError(f"Could not read extra frame {frame_idx}")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), frame)
            # At the opening frame the block is fully hidden under silicone.
            # A black shaft is an especially dangerous false block proposal.
            data = proposal(frame, record["task"], event_name, episode, frame_idx,
                            object_hidden=event_name == "first")
            out["frames"][str(frame_idx)] = {"event": event_name, "classes": data}
            preview = ep_dir / f"{event_name}_f{frame_idx:05d}_prompts.jpg"
            if not cv2.imwrite(str(preview), render(frame, frame_idx, data),
                               [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError(f"Could not write {preview}")
        (ep_dir / "proposals.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
        print(f"proposals={ep_dir}", flush=True)


if __name__ == "__main__":
    main()
