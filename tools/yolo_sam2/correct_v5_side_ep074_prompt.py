#!/usr/bin/env python3
"""Apply the reviewed v5 side episode-074 screw prompt corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORRECTIONS = (
    {
        "frame": 195,
        "box": [201.0, 201.0, 284.0, 235.0],
        "positive_points": [(225.0, 219.0), (267.0, 219.0)],
        "object_negative_points": [(160.0, 170.0), (200.0, 100.0), (520.0, 220.0)],
        "reason": "first stable partial exposure beneath the occluder",
    },
    {
        "frame": 324,
        "box": [427.0, 201.0, 484.0, 234.0],
        "positive_points": [(442.0, 219.0), (472.0, 219.0)],
        "object_negative_points": [(330.0, 170.0), (405.0, 170.0), (545.0, 225.0)],
        "reason": "full visible screw directly below the tool at region approach",
    },
    {
        "frame": 327,
        "box": [427.0, 201.0, 484.0, 234.0],
        "positive_points": [(442.0, 219.0), (472.0, 219.0)],
        "object_negative_points": [(330.0, 170.0), (430.0, 150.0), (545.0, 225.0)],
        "reason": "full screw extent immediately before tool/region contact",
    },
    {
        "frame": 336,
        "box": [427.0, 201.0, 484.0, 234.0],
        "positive_points": [(442.0, 219.0), (472.0, 219.0)],
        "object_negative_points": [(330.0, 170.0), (382.0, 180.0), (545.0, 225.0)],
        "reason": "replace the low-confidence partial box and its negative-only reset",
    },
    {
        "frame": 339,
        "box": [427.0, 201.0, 484.0, 234.0],
        "positive_points": [(442.0, 219.0), (472.0, 219.0)],
        "object_negative_points": [(330.0, 170.0), (382.0, 180.0), (545.0, 225.0)],
        "reason": "full visible screw after the tool turns edge-on over the region",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def point(frame: int, x: float, y: float, label: int) -> dict[str, Any]:
    return {"frame": frame, "x": x, "y": y, "label": label}


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if int(data.get("file_index", -1)) != 74 or data.get("tasks") != ["screw"]:
        raise RuntimeError("This correction is only valid for side screw episode 074")

    objects = {item["name"]: item for item in data["objects"]}
    if set(objects) != {"occluder", "object", "region", "tool"}:
        raise RuntimeError(f"Unexpected labels: {sorted(objects)}")
    screw = objects["object"]
    screw["visible_ranges"][0][0] = 177

    replaced_frames = {336}
    correction_frames = {int(item["frame"]) for item in CORRECTIONS}
    screw["boxes"] = [
        item
        for item in screw["boxes"]
        if int(item["frame"]) not in replaced_frames | correction_frames
    ]
    for item in CORRECTIONS:
        screw["boxes"].append(
            {
                "frame": int(item["frame"]),
                "box": list(item["box"]),
                "confidence": 1.0,
                "source": "manual_review_ep074",
            }
        )
    screw["boxes"].sort(key=lambda item: int(item["frame"]))

    for raw_object in data["objects"]:
        raw_object.setdefault("points", [])
        raw_object["points"] = [
            item
            for item in raw_object["points"]
            if int(item["frame"]) not in correction_frames
        ]

    for item in CORRECTIONS:
        frame = int(item["frame"])
        for x, y in item["positive_points"]:
            screw["points"].append(point(frame, x, y, 1))
        for x, y in item["object_negative_points"]:
            screw["points"].append(point(frame, x, y, 0))

    for raw_object in data["objects"]:
        raw_object["points"].sort(
            key=lambda item: (int(item["frame"]), int(item["label"]), item["x"], item["y"])
        )

    data.setdefault("manual_corrections", []).append(
        {
            "version": "ep074_screw_contact_r4",
            "visible_start": 177,
            "removed_object_box_frames": sorted(replaced_frames),
            "added_object_box_frames": sorted(correction_frames),
            "details": [
                {"frame": int(item["frame"]), "reason": item["reason"]}
                for item in CORRECTIONS
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote corrected episode-074 prompt: {args.output}")


if __name__ == "__main__":
    main()
