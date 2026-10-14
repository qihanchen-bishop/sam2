#!/usr/bin/env python3
"""Reference-light, class-wise quality assessment for semantic video masks.

The evaluator keeps two branches:

* ``raw``: direct model predictions (model quality)
* ``filtered``: predictions after the fixed connected-component refinement
  (usable pseudo-label quality)

For every class it measures geometry (area, centroid and bounding box), topology
(Betti-0/component count and Betti-1/hole count/area), and bidirectional
flow-validated temporal IoU.  Whole-video robust trajectories turn these
descriptors into interpretable anomaly evidence.  This is a reference-light
quality proxy, not a replacement for semantic ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from sam2_direction_review import write_json
from unet_global_temporal_scores import count_states, penalized_curve, rolling_median


BRANCHES = ("raw", "filtered")
FEATURES = (
    "area_px", "area_ratio", "component_count", "centroid_x", "centroid_y",
    "bbox_w", "bbox_h", "hole_count", "hole_area_px", "hole_area_ratio",
)

POLICY = {
    "version": "class_motion_topology_quality_v2",
    "description": "Class-wise raw/refined geometric-topological trajectories plus occlusion-aware flow IoU.",
    "flow_width": 300,
    "flow_forward_backward_threshold_px": 1.5,
    "flow_min_class_valid_fraction": 0.50,
    "trajectory_initial_median_seconds": 0.8,
    "trajectory_smooth_seconds": 0.35,
    "persistent_step_seconds": 0.5,
    "persistent_step_min_relative_change": 0.15,
    "persistent_step_max_relative_mad": 0.05,
    "component_transition_seconds": 0.5,
    "robust_multiplier": 4.0,
    "area_relative_floor": 0.08,
    "centroid_distance_floor": 0.02,
    "bbox_relative_floor": 0.10,
    "hole_area_ratio_floor": 0.005,
    "flow_iou_absolute_floor": 0.12,
    "minimum_hole_area_px": 20,
    "minimum_motion_difference_px_at_flow_scale": 4,
    "evidence_groups": ["area", "spatial_shape", "topology", "motion"],
    "hard_anomaly": "At least two independent evidence groups exceed robust class/video thresholds.",
    "selection_context_frames": 2,
    "maximum_filter_change_fraction_for_training": 0.20,
    "limitations": [
        "Temporal consistency cannot detect a semantically wrong mask that remains stable.",
        "Optical flow is ignored where forward/backward consistency or class coverage is insufficient.",
        "Thresholds are robustly estimated per class and video but require ground-truth calibration for formal guarantees.",
    ],
}


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(valid, bool), False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _robust_curve(values: np.ndarray, fps: float, abs_floor: float, rel_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Robust smooth curve with local noise, preserving input scale."""
    y = np.asarray(values, np.float64)
    n = len(y)
    if n == 0:
        return y.copy(), y.copy()
    if n < 3:
        return y.copy(), np.full(n, abs_floor, np.float64)
    radius = max(1, min(n - 1, round(fps * POLICY["trajectory_initial_median_seconds"])))
    curve = rolling_median(y, radius)
    strength = max(1.0, (fps * POLICY["trajectory_smooth_seconds"]) ** 4)
    weights = np.ones(n, np.float64)
    for _ in range(8):
        residual = y - curve
        local_mad = rolling_median(np.abs(residual), radius) * 1.4826
        scale = np.maximum(local_mad, np.maximum(abs_floor, rel_floor * np.maximum(np.abs(curve), np.abs(y))))
        ratio = np.abs(residual) / np.maximum(4.685 * scale, 1e-8)
        robust = np.where(ratio < 1.0, (1.0 - ratio ** 2) ** 2, 0.01)
        weights = np.maximum(0.01, robust)
        curve = penalized_curve(y, weights, strength)
    noise = rolling_median(np.abs(y - curve), radius) * 1.4826
    return curve, np.maximum(noise, abs_floor / POLICY["robust_multiplier"])


def _persistent_steps(values: np.ndarray, fps: float, relative: bool = True) -> list[int]:
    """Find persistent level changes so real events do not become outliers."""
    y = np.asarray(values, np.float64)
    n = len(y)
    width = max(3, round(fps * POLICY["persistent_step_seconds"]))
    out: list[int] = []
    for t in range(width, n - 2 * width + 1):
        pre, post = y[t - width:t], y[t:t + 2 * width]
        a, b = float(np.median(pre)), float(np.median(post))
        if relative:
            scale = max(abs(a), abs(b), 1e-6)
            change = abs(a - b) / scale
            dispersion = max(np.median(np.abs(pre - a)), np.median(np.abs(post - b))) / scale
            immediate = max(abs(y[t] - b), abs(y[t - 1] - a)) / scale
        else:
            change = abs(a - b)
            dispersion = max(np.median(np.abs(pre - a)), np.median(np.abs(post - b)))
            immediate = max(abs(y[t] - b), abs(y[t - 1] - a))
        if change < POLICY["persistent_step_min_relative_change"]:
            continue
        if dispersion > POLICY["persistent_step_max_relative_mad"] or immediate > 0.10:
            continue
        if out and t - out[-1] < width:
            continue
        out.append(t)
    return out


def _fit_visible(values: np.ndarray, visible: np.ndarray, fps: float, abs_floor: float,
                 rel_floor: float = 0.0, preserve_steps: bool = True) -> tuple[np.ndarray, np.ndarray]:
    expected = np.full(len(values), np.nan, np.float64)
    noise = np.full(len(values), np.nan, np.float64)
    for start, end in _segments(visible & np.isfinite(values)):
        y = np.asarray(values[start:end], np.float64)
        boundaries = [0]
        if preserve_steps and len(y) >= max(9, round(1.5 * fps)):
            boundaries += _persistent_steps(y, fps, relative=True)
        boundaries += [len(y)]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if b <= a:
                continue
            curve, local_noise = _robust_curve(y[a:b], fps, abs_floor, rel_floor)
            expected[start + a:start + b] = curve
            noise[start + a:start + b] = local_noise
    return expected, noise


def _relative_severity(values: np.ndarray, expected: np.ndarray, noise: np.ndarray,
                       floor: float, pixel_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.maximum(np.maximum(np.abs(values), np.abs(expected)), pixel_floor)
    error = np.abs(values - expected) / np.maximum(scale, 1e-8)
    tolerance = np.maximum(floor, POLICY["robust_multiplier"] * noise / np.maximum(scale, 1e-8))
    severity = np.maximum(0.0, error / np.maximum(tolerance, 1e-8) - 1.0)
    severity[~np.isfinite(error)] = 0.0
    return error, tolerance, severity


def _absolute_severity(error: np.ndarray, noise: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    tolerance = np.maximum(floor, POLICY["robust_multiplier"] * noise)
    severity = np.maximum(0.0, error / np.maximum(tolerance, 1e-8) - 1.0)
    severity[~np.isfinite(error)] = 0.0
    return tolerance, severity


def _mask_descriptor(mask: np.ndarray, class_id: int, min_hole_area: int) -> np.ndarray:
    binary = np.asarray(mask == class_id, np.uint8)
    height, width = binary.shape
    area = int(binary.sum())
    if area == 0:
        return np.array([0, 0, 0, np.nan, np.nan, 0, 0, 0, 0, 0], np.float64)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_count = count - 1
    sizes = stats[1:, cv2.CC_STAT_AREA]
    weighted_centroid = (centroids[1:] * sizes[:, None]).sum(axis=0) / max(area, 1)
    ys, xs = np.nonzero(binary)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = binary[max(0, y0 - 1):min(height, y1 + 1), max(0, x0 - 1):min(width, x1 + 1)]
    inverse = np.asarray(crop == 0, np.uint8)
    holes = 0
    hole_area = 0
    if inverse.size:
        h_count, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
        border_labels = set(np.unique(np.r_[h_labels[0], h_labels[-1], h_labels[:, 0], h_labels[:, -1]]).tolist())
        for label in range(1, h_count):
            pixels = int(h_stats[label, cv2.CC_STAT_AREA])
            if label not in border_labels and pixels >= min_hole_area:
                holes += 1
                hole_area += pixels
    return np.array([
        area, area / (height * width), component_count,
        weighted_centroid[0] / width, weighted_centroid[1] / height,
        (x1 - x0) / width, (y1 - y0) / height,
        holes, hole_area, hole_area / max(area + hole_area, 1),
    ], np.float64)


def _make_flow_estimator():
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    estimator.setFinestScale(2)
    estimator.setGradientDescentIterations(12)
    estimator.setVariationalRefinementIterations(0)
    return estimator


def _flow_iou(previous_gray: np.ndarray, current_gray: np.ndarray,
              previous_masks: dict[str, np.ndarray], current_masks: dict[str, np.ndarray],
              class_ids: list[int], estimator, scale_area: float) -> tuple[dict[str, np.ndarray], float]:
    forward = estimator.calc(previous_gray, current_gray, None)
    backward = estimator.calc(current_gray, previous_gray, None)
    height, width = current_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x, map_y = grid_x + backward[..., 0], grid_y + backward[..., 1]
    inside = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    sampled_forward_x = cv2.remap(forward[..., 0], map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    sampled_forward_y = cv2.remap(forward[..., 1], map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    fb_error = np.sqrt((backward[..., 0] + sampled_forward_x) ** 2 +
                       (backward[..., 1] + sampled_forward_y) ** 2)
    valid = inside & np.isfinite(fb_error) & (fb_error <= POLICY["flow_forward_backward_threshold_px"])
    results: dict[str, np.ndarray] = {}
    for branch in BRANCHES:
        previous = previous_masks[branch]
        current = current_masks[branch]
        warped = cv2.remap(previous, map_x, map_y, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        values = np.full(len(class_ids), np.nan, np.float64)
        for index, class_id in enumerate(class_ids):
            target = current == class_id
            source = warped == class_id
            class_pixels = int(target.sum())
            valid_fraction = float((target & valid).sum() / class_pixels) if class_pixels else 1.0
            union = (target | source) & valid
            union_count = int(union.sum())
            if valid_fraction < POLICY["flow_min_class_valid_fraction"]:
                continue
            if union_count == 0:
                continue
            difference = int(((target ^ source) & valid).sum())
            if difference * scale_area <= POLICY["minimum_motion_difference_px_at_flow_scale"]:
                values[index] = 1.0
            else:
                values[index] = float(((target & source) & valid).sum() / union_count)
        results[branch] = values
    return results, float(valid.mean())


def extract_features(video_path: Path, mask_root: Path, output: Path, objects: list[dict], fps: float) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "features.npz"
    metadata_path = output / "features_metadata.json"
    complete = json.loads((mask_root / "complete.json").read_text())
    source = {
        "video_path": str(video_path.resolve()),
        "mask_root": str(mask_root.resolve()),
        "frame_count": complete["frame_count"],
        "model_sha256": complete["model_sha256"],
        "component_filter": complete.get("component_filter"),
        "feature_extractor_version": "class_motion_topology_features_v1",
    }
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        # v1 stored the decision-policy version in the feature provenance.  It
        # does not affect descriptors, so compare required extraction fields
        # and allow cached features to be rescored by later decision policies.
        incompatible = any(
            metadata["source"].get(key) != value
            for key, value in source.items()
            if key != "feature_extractor_version"
        )
        feature_version = metadata["source"].get("feature_extractor_version")
        legacy_compatible = metadata["source"].get("policy_version") == "class_motion_topology_quality_v1"
        if incompatible or (feature_version != source["feature_extractor_version"] and not legacy_compatible):
            raise ValueError(f"Feature cache source changed: {output}")
        arrays = dict(np.load(cache, allow_pickle=False))
        return {"arrays": arrays, "metadata": metadata}

    class_ids = [int(obj["id"]) for obj in objects]
    n = int(complete["frame_count"])
    descriptors = np.full((len(BRANCHES), n, len(class_ids), len(FEATURES)), np.nan, np.float64)
    flow_iou = np.full((len(BRANCHES), n, len(class_ids)), np.nan, np.float64)
    flow_valid = np.full(n, np.nan, np.float64)
    filter_changed_px = np.zeros(n, np.int64)
    foreground_raw_px = np.zeros(n, np.int64)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    flow_width = min(POLICY["flow_width"], video_width)
    flow_height = max(32, round(video_height * flow_width / video_width))
    scale_area = (video_width * video_height) / (flow_width * flow_height)
    estimator = _make_flow_estimator()
    previous_gray = None
    previous_small_masks = None
    started = time.time()
    try:
        for frame in range(n):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Video ended at {frame}/{n}: {video_path}")
            masks = {
                "raw": cv2.imread(str(mask_root / "raw_mask_frames" / f"{frame:05d}.png"), cv2.IMREAD_UNCHANGED),
                "filtered": cv2.imread(str(mask_root / "mask_frames" / f"{frame:05d}.png"), cv2.IMREAD_UNCHANGED),
            }
            if any(value is None or value.shape != bgr.shape[:2] for value in masks.values()):
                raise ValueError(f"Missing or invalid mask at frame {frame}")
            foreground_raw_px[frame] = int(np.count_nonzero(masks["raw"]))
            filter_changed_px[frame] = int(np.count_nonzero(masks["raw"] != masks["filtered"]))
            for branch_index, branch in enumerate(BRANCHES):
                for class_index, class_id in enumerate(class_ids):
                    descriptors[branch_index, frame, class_index] = _mask_descriptor(
                        masks[branch], class_id, POLICY["minimum_hole_area_px"])
            gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (flow_width, flow_height),
                              interpolation=cv2.INTER_AREA)
            small_masks = {branch: cv2.resize(mask, (flow_width, flow_height), interpolation=cv2.INTER_NEAREST)
                           for branch, mask in masks.items()}
            if previous_gray is not None:
                pair, valid_fraction = _flow_iou(previous_gray, gray, previous_small_masks, small_masks,
                                                 class_ids, estimator, scale_area)
                for branch_index, branch in enumerate(BRANCHES):
                    flow_iou[branch_index, frame] = pair[branch]
                flow_valid[frame] = valid_fraction
            previous_gray, previous_small_masks = gray, small_masks
            if frame == 0 or (frame + 1) % 100 == 0:
                elapsed = time.time() - started
                print(f"FEATURES {output.parent.name} {frame + 1}/{n} {elapsed:.1f}s", flush=True)
        if cap.read()[0]:
            raise ValueError(f"Video has more than declared {n} frames")
    finally:
        cap.release()
    np.savez_compressed(cache, descriptors=descriptors, flow_iou=flow_iou,
                        flow_valid=flow_valid, filter_changed_px=filter_changed_px,
                        foreground_raw_px=foreground_raw_px)
    metadata = {
        "source": source, "branches": list(BRANCHES), "features": list(FEATURES),
        "class_ids": class_ids, "objects": objects, "fps": fps,
        "video_size_wh": [video_width, video_height], "flow_size_wh": [flow_width, flow_height],
        "seconds": time.time() - started,
    }
    write_json(metadata_path, metadata)
    return {"arrays": dict(np.load(cache, allow_pickle=False)), "metadata": metadata}


def _branch_assessment(descriptors: np.ndarray, flow_iou: np.ndarray, fps: float,
                       image_area: int) -> dict[str, np.ndarray]:
    n, class_count, _ = descriptors.shape
    index = {name: FEATURES.index(name) for name in FEATURES}
    geometry = np.zeros((n, class_count), np.float64)
    topology = np.zeros((n, class_count), np.float64)
    motion = np.zeros((n, class_count), np.float64)
    details: dict[str, np.ndarray] = {}
    for name in ("area", "centroid", "bbox", "components", "holes", "flow"):
        details[name + "_severity"] = np.zeros((n, class_count), np.float64)
    details["expected_component_count"] = np.zeros((n, class_count), np.float64)
    details["expected_hole_count"] = np.zeros((n, class_count), np.float64)
    details["expected_area_px"] = np.full((n, class_count), np.nan, np.float64)
    details["expected_flow_iou"] = np.full((n, class_count), np.nan, np.float64)

    for class_index in range(class_count):
        area = descriptors[:, class_index, index["area_px"]]
        visible = area > 0
        expected_area, area_noise = _fit_visible(area, visible, fps, 1.0, 0.01, preserve_steps=True)
        _, _, area_severity = _relative_severity(area, expected_area, area_noise,
                                                 POLICY["area_relative_floor"], pixel_floor=20.0)
        details["area_severity"][:, class_index] = area_severity
        details["expected_area_px"][:, class_index] = expected_area

        cx = descriptors[:, class_index, index["centroid_x"]]
        cy = descriptors[:, class_index, index["centroid_y"]]
        expected_cx, noise_cx = _fit_visible(cx, visible, fps, 0.002, 0.0, preserve_steps=True)
        expected_cy, noise_cy = _fit_visible(cy, visible, fps, 0.002, 0.0, preserve_steps=True)
        centroid_error = np.sqrt((cx - expected_cx) ** 2 + (cy - expected_cy) ** 2)
        centroid_noise = np.sqrt(noise_cx ** 2 + noise_cy ** 2)
        _, centroid_severity = _absolute_severity(centroid_error, centroid_noise,
                                                  POLICY["centroid_distance_floor"])
        details["centroid_severity"][:, class_index] = centroid_severity

        bbox_severity = np.zeros(n, np.float64)
        for feature in ("bbox_w", "bbox_h"):
            values = descriptors[:, class_index, index[feature]]
            expected, noise = _fit_visible(values, visible, fps, 0.002, 0.01, preserve_steps=True)
            _, _, severity = _relative_severity(values, expected, noise, POLICY["bbox_relative_floor"], 0.01)
            bbox_severity = np.maximum(bbox_severity, severity)
        details["bbox_severity"][:, class_index] = bbox_severity

        counts = descriptors[:, class_index, index["component_count"]].astype(np.int64)
        expected_counts = count_states(counts, max(1.0, fps * POLICY["component_transition_seconds"]))
        component_severity = (counts != expected_counts).astype(np.float64)
        details["components_severity"][:, class_index] = component_severity
        details["expected_component_count"][:, class_index] = expected_counts

        hole_counts = descriptors[:, class_index, index["hole_count"]].astype(np.int64)
        expected_holes = count_states(hole_counts, max(1.0, fps * POLICY["component_transition_seconds"]))
        hole_count_severity = (hole_counts != expected_holes).astype(np.float64)
        hole_ratio = descriptors[:, class_index, index["hole_area_ratio"]]
        expected_hole_ratio, hole_noise = _fit_visible(hole_ratio, visible, fps, 0.0005, 0.0, preserve_steps=True)
        hole_error = np.abs(hole_ratio - expected_hole_ratio)
        _, hole_area_severity = _absolute_severity(hole_error, hole_noise, POLICY["hole_area_ratio_floor"])
        hole_severity = np.maximum(hole_count_severity, hole_area_severity)
        details["holes_severity"][:, class_index] = hole_severity
        details["expected_hole_count"][:, class_index] = expected_holes

        iou = flow_iou[:, class_index]
        valid = np.isfinite(iou)
        expected_iou, iou_noise = _fit_visible(iou, valid, fps, 0.005, 0.0, preserve_steps=False)
        flow_error = np.maximum(0.0, expected_iou - iou)
        _, edge_severity = _absolute_severity(flow_error, iou_noise, POLICY["flow_iou_absolute_floor"])
        edge_bad = edge_severity > 0
        frame_flow_severity = np.zeros(n, np.float64)
        for frame in range(n):
            incoming = edge_severity[frame] if frame > 0 and valid[frame] else 0.0
            outgoing = edge_severity[frame + 1] if frame + 1 < n and valid[frame + 1] else 0.0
            if incoming > 0 and outgoing > 0:
                frame_flow_severity[frame] = min(incoming, outgoing)
            elif incoming > 0 or outgoing > 0:
                frame_flow_severity[frame] = 0.5 * max(incoming, outgoing)
        details["flow_severity"][:, class_index] = frame_flow_severity
        details["expected_flow_iou"][:, class_index] = expected_iou

        geometry[:, class_index] = np.maximum.reduce([area_severity, centroid_severity, bbox_severity])
        topology[:, class_index] = np.maximum(component_severity, hole_severity)
        motion[:, class_index] = frame_flow_severity

    spatial_shape = np.maximum(details["centroid_severity"], details["bbox_severity"])
    evidence = np.stack([details["area_severity"], spatial_shape, topology, motion], axis=-1)
    evidence_count = (evidence > 0).sum(axis=-1)
    hard = evidence_count >= 2
    # Score is descriptive; the hard decision remains interpretable and explicit.
    score = 100.0 * np.exp(-np.minimum(4.0, evidence.sum(axis=-1)))
    return {
        **details, "geometry_severity": geometry, "spatial_shape_severity": spatial_shape,
        "topology_severity": topology,
        "motion_severity": motion, "evidence_count": evidence_count,
        "hard_anomaly": hard, "quality_score": score,
    }


def assess_video(video_path: Path, mask_root: Path, output: Path, fps: float) -> dict:
    complete = json.loads((mask_root / "complete.json").read_text())
    objects = complete["objects"]
    extracted = extract_features(video_path, mask_root, output, objects, fps)
    arrays, metadata = extracted["arrays"], extracted["metadata"]
    descriptors = arrays["descriptors"]
    branch_results = {
        branch: _branch_assessment(descriptors[index], arrays["flow_iou"][index], fps,
                                   metadata["video_size_wh"][0] * metadata["video_size_wh"][1])
        for index, branch in enumerate(BRANCHES)
    }
    n = descriptors.shape[1]
    class_names = [obj["name"] for obj in objects]
    feature_index = {name: FEATURES.index(name) for name in FEATURES}
    filtered_hard = branch_results["filtered"]["hard_anomaly"].copy()
    # Expand only the final training exclusion, while preserving exact detections.
    expanded = filtered_hard.copy()
    radius = int(POLICY["selection_context_frames"])
    for class_index in range(len(class_names)):
        flagged = np.flatnonzero(filtered_hard[:, class_index])
        for frame in flagged:
            expanded[max(0, frame - radius):min(n, frame + radius + 1), class_index] = True
    changed_fraction = arrays["filter_changed_px"] / np.maximum(arrays["foreground_raw_px"], 1)
    excessive_filter = changed_fraction > POLICY["maximum_filter_change_fraction_for_training"]
    accepted = ~(expanded.any(axis=1) | excessive_filter)
    selection = []
    rows = []
    jsonl = output / "frame_scores.jsonl"
    with jsonl.open("w") as json_handle:
        for frame in range(n):
            classes = {}
            for class_index, name in enumerate(class_names):
                class_row = {}
                for branch_index, branch in enumerate(BRANCHES):
                    values = descriptors[branch_index, frame, class_index]
                    result = branch_results[branch]
                    class_row[branch] = {
                        "area_px": int(values[feature_index["area_px"]]),
                        "component_count": int(values[feature_index["component_count"]]),
                        "centroid_xy": [None if not np.isfinite(values[feature_index["centroid_x"]]) else float(values[feature_index["centroid_x"]]),
                                        None if not np.isfinite(values[feature_index["centroid_y"]]) else float(values[feature_index["centroid_y"]])],
                        "bbox_wh": [float(values[feature_index["bbox_w"]]), float(values[feature_index["bbox_h"]])],
                        "hole_count": int(values[feature_index["hole_count"]]),
                        "hole_area_px": int(values[feature_index["hole_area_px"]]),
                        "flow_iou_previous": None if not np.isfinite(arrays["flow_iou"][branch_index, frame, class_index]) else float(arrays["flow_iou"][branch_index, frame, class_index]),
                        "geometry_severity": float(result["geometry_severity"][frame, class_index]),
                        "area_severity": float(result["area_severity"][frame, class_index]),
                        "spatial_shape_severity": float(result["spatial_shape_severity"][frame, class_index]),
                        "topology_severity": float(result["topology_severity"][frame, class_index]),
                        "motion_severity": float(result["motion_severity"][frame, class_index]),
                        "quality_score": float(result["quality_score"][frame, class_index]),
                        "hard_anomaly": bool(result["hard_anomaly"][frame, class_index]),
                    }
                classes[name] = class_row
            reasons = []
            if expanded[frame].any():
                reasons.append("filtered_temporal_structural_anomaly")
            if excessive_filter[frame]:
                reasons.append("excessive_filter_dependence")
            record = {
                "frame": frame, "time_seconds": frame / fps, "accepted_for_training": bool(accepted[frame]),
                "selection_reasons": reasons,
                "filter_changed_pixels": int(arrays["filter_changed_px"][frame]),
                "filter_change_fraction": float(changed_fraction[frame]),
                "flow_valid_fraction": None if not np.isfinite(arrays["flow_valid"][frame]) else float(arrays["flow_valid"][frame]),
                "raw_anomaly_classes": [name for i, name in enumerate(class_names) if branch_results["raw"]["hard_anomaly"][frame, i]],
                "filtered_anomaly_classes": [name for i, name in enumerate(class_names) if filtered_hard[frame, i]],
                "expanded_rejection_classes": [name for i, name in enumerate(class_names) if expanded[frame, i]],
                "classes": classes,
            }
            json_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            selection.append({"frame": frame, "accepted": bool(accepted[frame]), "reasons": reasons,
                              "quality_score": float(min(branch_results["filtered"]["quality_score"][frame]))})
            rows.append({
                "frame": frame, "time_seconds": frame / fps, "accepted_for_training": int(accepted[frame]),
                "quality_score": selection[-1]["quality_score"],
                "raw_anomaly_classes": ";".join(record["raw_anomaly_classes"]),
                "filtered_anomaly_classes": ";".join(record["filtered_anomaly_classes"]),
                "expanded_rejection_classes": ";".join(record["expanded_rejection_classes"]),
                "filter_changed_pixels": record["filter_changed_pixels"],
                "filter_change_fraction": record["filter_change_fraction"],
                "flow_valid_fraction": record["flow_valid_fraction"],
            })
    with (output / "frame_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output / "selection.json", {"policy": POLICY, "frames": selection})
    per_class = {}
    for class_index, name in enumerate(class_names):
        per_class[name] = {
            branch: {
                "hard_anomaly_frames": int(branch_results[branch]["hard_anomaly"][:, class_index].sum()),
                "median_quality_score": float(np.median(branch_results[branch]["quality_score"][:, class_index])),
                "median_flow_iou": float(np.nanmedian(arrays["flow_iou"][BRANCHES.index(branch), :, class_index]))
                    if np.isfinite(arrays["flow_iou"][BRANCHES.index(branch), :, class_index]).any() else None,
            } for branch in BRANCHES
        }
    summary = {
        "frame_count": n, "accepted_frames": int(accepted.sum()), "excluded_frames": int((~accepted).sum()),
        "acceptance_rate": float(accepted.mean()),
        "raw_any_anomaly_frames": int(branch_results["raw"]["hard_anomaly"].any(axis=1).sum()),
        "filtered_any_anomaly_frames": int(filtered_hard.any(axis=1).sum()),
        "expanded_filtered_anomaly_frames": int(expanded.any(axis=1).sum()),
        "excessive_filter_dependence_frames": int(excessive_filter.sum()),
        "filter_changed_frames": int((arrays["filter_changed_px"] > 0).sum()),
        "filter_removed_or_reassigned_pixels": int(arrays["filter_changed_px"].sum()),
        "per_class": per_class,
        "interpretation": "Reference-light temporal/structural quality proxy; not semantic ground-truth accuracy.",
    }
    write_json(output / "summary.json", summary)
    return summary


def run_round(round_root: Path, output_name: str, feature_source_name: str | None = None) -> dict:
    round_root = round_root.resolve()
    output = round_root / output_name
    if output.parent != round_root:
        raise ValueError("Output must be a direct child of round root")
    output.mkdir(exist_ok=True)
    source = json.loads((round_root / "manifest.json").read_text())
    records = []
    manifest = {
        "workflow": "class_motion_topology_quality_assessment", "policy": POLICY,
        "source_manifest": str((round_root / "manifest.json").resolve()), "records": records,
    }
    for source_record in source["records"]:
        record = {"id": source_record["id"], "item": source_record["item"], "status": "running"}
        records.append(record); write_json(output / "manifest.json", manifest)
        item = source_record["item"]
        if feature_source_name:
            feature_source = round_root / feature_source_name / source_record["id"]
            feature_target = output / source_record["id"]
            feature_target.mkdir(parents=True, exist_ok=True)
            for name in ("features.npz", "features_metadata.json"):
                target = feature_target / name
                if not target.exists():
                    target.hardlink_to(feature_source / name)
        record["summary"] = assess_video(Path(item["video_path"]), round_root / source_record["id"] / "unet",
                                         output / source_record["id"], float(item["fps"]))
        record["status"] = "complete"
        write_json(output / "manifest.json", manifest)
        print("QUALITY COMPLETE", record["id"], json.dumps(record["summary"], ensure_ascii=False), flush=True)
    total_frames = sum(record["summary"]["frame_count"] for record in records)
    total_accepted = sum(record["summary"]["accepted_frames"] for record in records)
    aggregate = {
        "videos": len(records), "frames": total_frames, "accepted_frames": total_accepted,
        "excluded_frames": total_frames - total_accepted,
        "acceptance_rate": total_accepted / total_frames,
        "raw_any_anomaly_frames": sum(r["summary"]["raw_any_anomaly_frames"] for r in records),
        "filtered_any_anomaly_frames": sum(r["summary"]["filtered_any_anomaly_frames"] for r in records),
        "filter_changed_frames": sum(r["summary"]["filter_changed_frames"] for r in records),
        "policy_version": POLICY["version"], "finished_at": time.time(),
    }
    write_json(output / "complete.json", aggregate)
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--output-name", default="video_quality_v2")
    parser.add_argument("--feature-source-name")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    print(json.dumps(run_round(args.round_root, args.output_name, args.feature_source_name), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
