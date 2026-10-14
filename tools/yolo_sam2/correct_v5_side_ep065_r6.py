#!/usr/bin/env python3
"""Correct the continuously visible side-view screw in episode 065."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_FRAMES = (152, 154, 162, 177, 189, 198, 210, 225, 240, 252, 260)
BOX = [207.0, 184.0, 250.0, 215.0]
POSITIVE_POINTS = ((216.0, 201.0), (241.0, 201.0))
NEGATIVE_POINTS = (
    (191.0, 199.5),
    (266.0, 199.5),
    (228.5, 164.0),
    (228.5, 233.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if int(data.get("file_index", -1)) != 65 or data.get("tasks") != ["screw"]:
        raise RuntimeError("Input is not the reviewed side-view screw prompt for episode 065")
    screw = next(item for item in data["objects"] if item["name"] == "object")

    # Dense frame review shows one continuous visible interval. The earlier split
    # at 192/264 was the source of the false-empty masks while the tool moved over
    # (but did not fully hide) the screw.
    screw["visible_ranges"] = [[150, 625]]
    replaced = set(PROMPT_FRAMES) | {
        int(item["frame"])
        for item in screw["boxes"]
        if 145 <= int(item["frame"]) <= 263
    }
    screw["boxes"] = [
        item for item in screw["boxes"] if int(item["frame"]) not in replaced
    ]
    screw.setdefault("points", [])
    screw["points"] = [
        item for item in screw["points"] if int(item["frame"]) not in replaced
    ]

    for frame in PROMPT_FRAMES:
        frame_box = [208.0, 190.0, 251.0, 216.0] if frame == 152 else BOX
        screw["boxes"].append(
            {
                "frame": frame,
                "box": frame_box,
                "confidence": 1.0,
                "source": "manual_screw_continuity_r6",
            }
        )
        for x, y in POSITIVE_POINTS:
            screw["points"].append(
                {"frame": frame, "x": x, "y": y, "label": 1}
            )
        for x, y in NEGATIVE_POINTS:
            screw["points"].append(
                {"frame": frame, "x": x, "y": y, "label": 0}
            )
        if frame == 152:
            screw["points"].append(
                {"frame": frame, "x": 228.5, "y": 184.0, "label": 0}
            )

    screw["boxes"].sort(key=lambda item: int(item["frame"]))
    screw["points"].sort(
        key=lambda item: (
            int(item["frame"]),
            int(item["label"]),
            float(item["x"]),
            float(item["y"]),
        )
    )
    data.setdefault("manual_corrections", []).append(
        {
            "version": "side_screw_ep065_continuity_r6",
            "visible_ranges": [[150, 625]],
            "object_box_frames": list(PROMPT_FRAMES),
            "reason": (
                "dense review: screw remains visibly exposed during frames "
                "193-263 while cloth/tool pass nearby"
            ),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
