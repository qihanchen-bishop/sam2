#!/usr/bin/env python3
"""Geometry/color repair of provisional SAM2 masks; writes a separate audit copy.

Only removes tool pixels far from a visually detectable shaft/gripper and adds
the red silicone constituent back to occluder.  Never edits SAM2 output in place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from semantic_point_seed_prompts import clustered_masks, find_tools


def interpolated_line(anchors: dict[int, np.ndarray], frame_idx: int) -> np.ndarray:
    keys = sorted(anchors)
    before = max((key for key in keys if key <= frame_idx), default=keys[0])
    after = min((key for key in keys if key >= frame_idx), default=keys[-1])
    if before == after:
        return anchors[before]
    weight = (frame_idx - before) / (after - before)
    return anchors[before] * (1 - weight) + anchors[after] * weight


def source_aligned_line(expected: np.ndarray,
                        detected: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """Keep the prompt's upper shaft and accept a source-visible longer tip.

    The model core can end at the colored shaft and omit the distal metal tip.
    A Hough line is useful only when it has the same axis at the prompt's
    distal anchor; comparing endpoint distance alone rejects valid extensions.
    """
    if detected is None:
        return expected, False
    # A nearby detected endpoint tracks the current frame more faithfully
    # than interpolation between prompt frames; retain that original case.
    if float(np.linalg.norm(detected[1] - expected[1])) <= 115:
        return detected, False
    e_dy = float(expected[1, 1] - expected[0, 1])
    d_dy = float(detected[1, 1] - detected[0, 1])
    if e_dy < 60 or d_dy < 60:
        return expected, False
    e_slope = float((expected[1, 0] - expected[0, 0]) / e_dy)
    d_slope = float((detected[1, 0] - detected[0, 0]) / d_dy)
    d_x_at_anchor = float(detected[0, 0] + d_slope *
                          (expected[1, 1] - detected[0, 1]))
    aligned = (abs(d_x_at_anchor - expected[1, 0]) <= 27
               and abs(d_slope - e_slope) <= 0.16)
    if not aligned:
        return expected, True
    # Do not shorten a reviewed prompt line to a partial Hough segment.
    if detected[1, 1] <= expected[1, 1] + 10:
        return expected, False
    return np.stack((expected[0], detected[1])), False


def geometry_mask(shape: tuple[int, int], line: np.ndarray,
                  name: str, hsv: np.ndarray, red: np.ndarray,
                  left_extension: int = 20) -> np.ndarray:
    top, bottom = line
    direction = bottom - top
    direction /= max(np.linalg.norm(direction), 1e-6)
    extend = left_extension if name == "lefttool" else 100
    endpoint = bottom + direction * extend
    core = np.zeros(shape, dtype=np.uint8)
    cv2.line(core, tuple(top.astype(int)), tuple(endpoint.astype(int)), 255, 30)
    fork = np.zeros(shape, dtype=np.uint8)
    center = bottom + direction * (25 if name == "lefttool" else 60)
    axes = (48, 60) if name == "lefttool" else (75, 85)
    cv2.ellipse(fork, tuple(center.astype(int)), axes, 0, 0, 360, 255, -1)
    hh, ss, vv = cv2.split(hsv)
    neutral = (((ss < 60) & (vv < 220)) | (vv < 100)) & ~(red > 0)
    return (core > 0) | ((fork > 0) & neutral)


def recovery_mask(shape: tuple[int, int], line: np.ndarray, name: str,
                  hsv: np.ndarray, red: np.ndarray, object_mask: np.ndarray,
                  left_extension: int) -> np.ndarray:
    top, bottom = line
    direction = bottom - top
    direction /= max(np.linalg.norm(direction), 1e-6)
    extend = left_extension if name == "lefttool" else 85
    centerline = np.zeros(shape, dtype=np.uint8)
    cv2.line(centerline, tuple(top.astype(int)),
             tuple((bottom + direction * extend).astype(int)), 255, 19)
    _, saturation, value = cv2.split(hsv)
    tool_appearance = ((value < 110) | ((saturation < 30) &
                                        (value >= 50) & (value < 210)))
    return (centerline > 0) & tool_appearance & ~(red > 0) & ~(object_mask > 0)


def fill_enclosed_holes(mask: np.ndarray, class_id: int, max_area: int) -> int:
    fg = (mask == class_id).astype(np.uint8)
    contours, hierarchy = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    added = 0
    for index, contour in enumerate(contours):
        if hierarchy[0, index, 3] < 0 or cv2.contourArea(contour) > max_area:
            continue
        candidate = np.zeros_like(fg)
        cv2.drawContours(candidate, contours, index, 255, -1)
        region = (candidate > 0) & (mask == 0)
        added += int(region.sum())
        mask[region] = class_id
    return added


def remove_object_islands(mask: np.ndarray, min_area: int) -> int:
    """Keep the visible object core, not tiny same-colored tray reflections."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (mask == 2).astype(np.uint8), 8)
    if count <= 2:
        return 0
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_x, main_y, main_width, main_height, _ = stats[main]
    removed = 0
    for component in range(1, count):
        if component == main:
            continue
        x, y, width, height, area = stats[component]
        x_gap = max(0, main_x - (x + width), x - (main_x + main_width))
        y_gap = max(0, main_y - (y + height), y - (main_y + main_height))
        if area >= min_area and np.hypot(x_gap, y_gap) <= 50:
            continue
        island = components == component
        removed += int(island.sum())
        mask[island] = 0
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text())
    prompt = json.loads(args.prompt.read_text())
    total = int(prompt["frame_count"])
    anchors = {name: {} for name in ("lefttool", "righttool")}
    lift_frames = []
    region_contact_frames = []
    for frame_text, record in proposal["frames"].items():
        frame_idx = int(frame_text)
        if record["event"] == "occluder_contact":
            lift_frames.append(frame_idx)
        if record["event"] == "region_contact":
            region_contact_frames.append(frame_idx)
        for name in anchors:
            if name in record["classes"]:
                anchors[name][frame_idx] = np.array(record["classes"][name]["line"], float)
    if any(not values for values in anchors.values()):
        raise ValueError("Missing shaft-line anchors")
    cap = cv2.VideoCapture(prompt["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    mask_dir = args.mask_root / "mask_frames"
    if len(list(mask_dir.glob("*.png"))) != total:
        raise ValueError("Incomplete SAM2 mask set")
    out_mask_dir = args.output / "mask_frames"
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    yy, xx = np.indices((height, width))
    stats = {"frames": total, "removed_tool_pixels": {"lefttool": 0, "righttool": 0},
             "recovered_tool_pixels": {"lefttool": 0, "righttool": 0},
             "removed_lefttool_cloth_pixels": 0,
             "added_red_pixels": 0, "added_skin_pixels": 0,
             "reassigned_red_from_object_pixels": 0,
             "removed_object_offcolor_pixels": 0,
             "filled_hole_pixels": 0, "removed_region_rim_pixels": 0,
             "added_region_pixels": 0, "removed_object_island_pixels": 0,
             "line_fallback_frames": {"lefttool": 0, "righttool": 0}}
    for frame_idx in range(total):
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Could not read source frame {frame_idx}")
        old = cv2.imread(str(mask_dir / f"{frame_idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if old is None or old.shape != frame.shape[:2]:
            raise ValueError(f"Invalid SAM2 mask at f{frame_idx}")
        new = old.copy()
        color = clustered_masks(frame, prompt["task"])
        detected = find_tools(frame, color["object"], "extra")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_lines = {}
        frame_extensions = {}
        for class_id, name in ((4, "lefttool"), (5, "righttool")):
            expected = interpolated_line(anchors[name], frame_idx)
            detected_line = (np.array(detected[name]["line"], float)
                             if name in detected else None)
            line, fallback = source_aligned_line(expected, detected_line)
            if fallback:
                stats["line_fallback_frames"][name] += 1
            left_extension = (40 if name == "lefttool" and any(
                abs(frame_idx - lift) <= 35 for lift in lift_frames) else 20)
            frame_lines[name] = line
            frame_extensions[name] = left_extension
            allowed = geometry_mask(old.shape, line, name, hsv, color["red"],
                                    left_extension)
            removed = (new == class_id) & ~allowed
            stats["removed_tool_pixels"][name] += int(removed.sum())
            new[removed] = 0
        for class_id, name in ((4, "lefttool"), (5, "righttool")):
            recovered = recovery_mask(old.shape, frame_lines[name], name, hsv,
                                      color["red"], color["object"],
                                      frame_extensions[name])
            recovered &= (new != 2) & (new != (5 if class_id == 4 else 4))
            stats["recovered_tool_pixels"][name] += int((recovered & (new != class_id)).sum())
            new[recovered] = class_id
        left_top, left_bottom = frame_lines["lefttool"]
        left_direction = left_bottom - left_top
        left_direction /= max(np.linalg.norm(left_direction), 1e-6)
        tight_left = np.zeros(old.shape, dtype=np.uint8)
        cv2.line(tight_left, tuple(left_top.astype(int)),
                 tuple((left_bottom + left_direction * frame_extensions["lefttool"]).astype(int)),
                 255, 20)
        cloth_as_tool = (new == 4) & (color["cloth"] > 0) & (tight_left == 0)
        stats["removed_lefttool_cloth_pixels"] += int(cloth_as_tool.sum())
        new[cloth_as_tool] = 0
        # Red/pink silicone is part of occluder, never a tool or the block.
        red = color["red"] > 0
        object_ys, object_xs = np.nonzero(color["object"])
        red_limit = int(np.median(object_xs)) - 8 if len(object_xs) else 820
        red[:, max(0, red_limit):] = False
        red[:280] = False
        red[680:] = False
        red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8)) > 0
        red_object = red & (new == 2)
        stats["reassigned_red_from_object_pixels"] += int(red_object.sum())
        new[red_object] = 1
        add_red = red & (new == 0)
        stats["added_red_pixels"] += int(add_red.sum())
        new[add_red] = 1
        region_core = color["region"] > 0
        if region_core.any():
            region_guard = cv2.dilate(region_core.astype(np.uint8),
                                      np.ones((11, 11), np.uint8)) > 0
            rim = (new == 3) & ~region_guard
            stats["removed_region_rim_pixels"] += int(rim.sum())
            new[rim] = 0
            add_region = region_core & (new == 0)
            stats["added_region_pixels"] += int(add_region.sum())
            new[add_region] = 3
        near_occluder = cv2.dilate((new == 1).astype(np.uint8),
                                   np.ones((31, 31), np.uint8)) > 0
        skin = (color["cloth"] > 0) & (xx < 920) & (yy < 675)
        add_skin = skin & near_occluder & (new == 0)
        stats["added_skin_pixels"] += int(add_skin.sum())
        new[add_skin] = 1
        for class_id, max_area in ((1, 180), (2, 120), (3, 180)):
            stats["filled_hole_pixels"] += fill_enclosed_holes(new, class_id, max_area)
        if (prompt["task"] == "o_n" and region_contact_frames
                and frame_idx >= min(region_contact_frames)
                and int((color["object"] > 0).sum()) >= 150):
            object_guard = cv2.dilate((color["object"] > 0).astype(np.uint8),
                                      np.ones((25, 25), np.uint8)) > 0
            offcolor = (new == 2) & ~object_guard
            stats["removed_object_offcolor_pixels"] += int(offcolor.sum())
            new[offcolor] = np.where(region_core[offcolor], 3, 0)
        object_min_area = (120 if region_contact_frames and frame_idx >= min(region_contact_frames)
                           else 50)
        stats["removed_object_island_pixels"] += remove_object_islands(new, object_min_area)
        if not cv2.imwrite(str(out_mask_dir / f"{frame_idx:05d}.png"), new):
            raise RuntimeError(f"Could not save refined mask f{frame_idx}")
        if frame_idx % 100 == 0:
            print(f"refined {frame_idx + 1}/{total}", flush=True)
    cap.release()
    stats["status"] = "provisional_not_accepted"
    stats["source_prompt"] = str(args.prompt.resolve())
    stats["source_mask_root"] = str(args.mask_root.resolve())
    (args.output / "refinement.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"refined_masks={out_mask_dir}")


if __name__ == "__main__":
    main()
