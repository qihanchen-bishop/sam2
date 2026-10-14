#!/usr/bin/env python3
"""Deterministic tests for research video-mask quality descriptors."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from video_segmentation_quality import FEATURES, _branch_assessment, _mask_descriptor


def test_descriptor_components_and_holes():
    mask = np.zeros((80, 100), np.uint8)
    cv2.rectangle(mask, (10, 10), (50, 60), 2, -1)
    cv2.rectangle(mask, (22, 22), (35, 35), 0, -1)
    cv2.rectangle(mask, (70, 15), (75, 20), 2, -1)
    values = _mask_descriptor(mask, 2, min_hole_area=20)
    index = {name: FEATURES.index(name) for name in FEATURES}
    assert values[index["component_count"]] == 2
    assert values[index["hole_count"]] == 1
    assert values[index["hole_area_px"]] == 14 * 14
    assert 0 < values[index["area_ratio"]] < 1


def test_transient_multigroup_event_is_detected():
    n, classes = 150, 1
    descriptors = np.zeros((n, classes, len(FEATURES)), np.float64)
    index = {name: FEATURES.index(name) for name in FEATURES}
    descriptors[:, 0, index["area_px"]] = 1000
    descriptors[:, 0, index["area_ratio"]] = .01
    descriptors[:, 0, index["component_count"]] = 1
    descriptors[:, 0, index["centroid_x"]] = .5
    descriptors[:, 0, index["centroid_y"]] = .5
    descriptors[:, 0, index["bbox_w"]] = .1
    descriptors[:, 0, index["bbox_h"]] = .1
    descriptors[:, 0, index["hole_count"]] = 0
    descriptors[:, 0, index["hole_area_px"]] = 0
    descriptors[:, 0, index["hole_area_ratio"]] = 0
    # A one-frame error changes two independent v2 groups: area and spatial shape.
    descriptors[75, 0, index["area_px"]] = 1600
    descriptors[75, 0, index["area_ratio"]] = .016
    descriptors[75, 0, index["centroid_x"]] = .7
    descriptors[75, 0, index["bbox_w"]] = .2
    flow = np.full((n, classes), .98, np.float64)
    result = _branch_assessment(descriptors, flow, fps=30.0, image_area=100000)
    assert result["hard_anomaly"][75, 0]
    assert result["area_severity"][75, 0] > 0
    assert result["spatial_shape_severity"][75, 0] > 0
    assert result["topology_severity"][75, 0] == 0
    assert not result["hard_anomaly"][30, 0]


def test_persistent_component_state_is_not_transient():
    n, classes = 180, 1
    descriptors = np.zeros((n, classes, len(FEATURES)), np.float64)
    index = {name: FEATURES.index(name) for name in FEATURES}
    descriptors[:, 0, index["area_px"]] = 1000
    descriptors[:, 0, index["area_ratio"]] = .01
    descriptors[:, 0, index["component_count"]] = 1
    descriptors[:, 0, index["centroid_x"]] = .5
    descriptors[:, 0, index["centroid_y"]] = .5
    descriptors[:, 0, index["bbox_w"]] = .1
    descriptors[:, 0, index["bbox_h"]] = .1
    descriptors[:, 0, index["hole_count"]] = 0
    descriptors[:, 0, index["hole_area_px"]] = 0
    descriptors[:, 0, index["hole_area_ratio"]] = 0
    descriptors[90:, 0, index["component_count"]] = 2
    flow = np.full((n, classes), .98, np.float64)
    result = _branch_assessment(descriptors, flow, fps=30.0, image_area=100000)
    # A stable new state is preserved by the sequence model.
    assert not result["hard_anomaly"][120, 0]


def main():
    test_descriptor_components_and_holes()
    test_transient_multigroup_event_is_detected()
    test_persistent_component_state_is_not_transient()
    print("video segmentation quality tests passed")


if __name__ == "__main__":
    main()
