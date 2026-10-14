#!/usr/bin/env python3
"""Apply reviewed side-view screw prompt corrections for bettersetup v5 r5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Each box encloses the complete visible screw, including both metallic ends.
# Positive points deliberately hit both ends so SAM2 does not follow only one
# disconnected-looking component of the same screw.
CORRECTIONS: dict[int, dict[str, Any]] = {
    60: {"visible_start": 78, "items": [
        (81, [128, 210, 177, 244], [(138, 228), (166, 228)]),
        (105, [128, 210, 177, 244], [(138, 228), (166, 228)]),
        (132, [127, 210, 178, 244], [(138, 228), (167, 228)]),
        (156, [121, 207, 179, 244], [(133, 227), (168, 227)]),
        (183, [127, 209, 177, 245], [(138, 228), (166, 228)]),
    ]},
    61: {"visible_start": 174, "items": [
        (177, [102, 209, 160, 245], [(113, 228), (149, 228)]),
        (189, [102, 209, 160, 245], [(113, 228), (149, 228)]),
        (201, [100, 210, 158, 246], [(111, 229), (147, 229)]),
    ]},
    62: {"visible_start": 135, "items": [
        (138, [120, 210, 176, 246], [(131, 229), (165, 229)]),
        (156, [120, 210, 176, 246], [(131, 229), (165, 229)]),
        (162, [120, 210, 176, 246], [(131, 229), (165, 229)]),
        (177, [120, 207, 175, 245], [(131, 228), (164, 228)]),
        (186, [120, 208, 176, 246], [(131, 229), (165, 229)]),
        (204, [160, 207, 205, 247], [(169, 228), (195, 228)]),
        (207, [190, 207, 228, 247], [(198, 228), (219, 228)]),
    ]},
    63: {"visible_start": 141, "drop_frames": [213, 237, 255], "items": [
        (144, [115, 210, 175, 245], [(127, 229), (164, 229)]),
        (153, [115, 210, 175, 245], [(127, 229), (164, 229)]),
        (165, [115, 210, 175, 245], [(127, 229), (164, 229)]),
        (201, [312, 204, 355, 243], [(322, 225), (345, 225)]),
        (210, [348, 204, 388, 243], [(358, 225), (378, 225)]),
        (225, [314, 204, 388, 244], [(328, 225), (376, 225)]),
        (240, [314, 204, 388, 244], [(328, 225), (376, 225)]),
        (258, [314, 204, 388, 244], [(328, 225), (376, 225)]),
    ]},
    64: {"visible_start": 81, "items": [
        (84, [130, 204, 184, 242], [(141, 224), (173, 224)]),
        (108, [128, 205, 186, 242], [(140, 224), (174, 224)]),
        (135, [122, 206, 182, 245], [(134, 227), (170, 227)]),
        (159, [155, 210, 199, 246], [(164, 229), (190, 229)]),
    ]},
    65: {"visible_start": 159, "items": [
        (162, [207, 185, 250, 215], [(216, 201), (241, 201)]),
        (177, [207, 185, 250, 215], [(216, 201), (241, 201)]),
        (189, [207, 184, 250, 215], [(216, 200), (241, 200)]),
    ]},
    66: {"visible_start": 153, "items": [
        (156, [187, 185, 231, 214], [(196, 201), (222, 201)]),
        (180, [188, 185, 233, 215], [(197, 201), (224, 201)]),
        (201, [188, 185, 233, 215], [(197, 201), (224, 201)]),
        (228, [187, 183, 233, 215], [(196, 201), (224, 201)]),
    ]},
    67: {"visible_start": 135, "items": [
        (138, [157, 185, 204, 216], [(167, 202), (195, 202)]),
        (162, [157, 185, 204, 217], [(167, 203), (195, 203)]),
        (189, [157, 186, 205, 216], [(167, 203), (196, 203)]),
        (213, [157, 185, 204, 217], [(167, 202), (195, 202)]),
    ]},
    68: {"visible_start": 132, "items": [
        (135, [171, 184, 213, 216], [(180, 201), (204, 201)]),
        (153, [171, 184, 213, 216], [(180, 201), (204, 201)]),
        (177, [171, 185, 213, 216], [(180, 202), (204, 202)]),
        (195, [171, 184, 213, 216], [(180, 201), (204, 201)]),
        (216, [171, 184, 213, 216], [(180, 201), (204, 201)]),
    ]},
    69: {"visible_start": 141, "items": [
        (144, [184, 185, 221, 215], [(192, 201), (213, 201)]),
        (168, [184, 185, 221, 216], [(192, 202), (213, 202)]),
        (189, [184, 185, 221, 216], [(192, 202), (213, 202)]),
        (213, [184, 184, 221, 216], [(192, 201), (213, 201)]),
    ]},
    72: {"visible_start": 162, "items": [
        (165, [211, 199, 260, 233], [(221, 217), (250, 217)]),
        (177, [211, 199, 260, 233], [(221, 217), (250, 217)]),
        (180, [211, 199, 260, 233], [(221, 217), (250, 217)]),
    ]},
    75: {"visible_start": 174, "items": [
        (177, [231, 192, 283, 224], [(241, 209), (273, 209)]),
        (192, [231, 192, 283, 224], [(241, 209), (273, 209)]),
        (207, [231, 192, 283, 224], [(241, 209), (273, 209)]),
        (225, [231, 192, 283, 224], [(241, 209), (273, 209)]),
        (234, [231, 192, 283, 224], [(241, 209), (273, 209)]),
        (267, [231, 192, 283, 224], [(241, 209), (273, 209)]),
    ]},
    76: {"visible_start": 150, "items": [
        (153, [226, 191, 281, 225], [(237, 209), (270, 209)]),
        (159, [226, 191, 281, 225], [(237, 209), (270, 209)]),
        (165, [226, 191, 281, 225], [(237, 209), (270, 209)]),
        (177, [226, 191, 281, 225], [(237, 209), (270, 209)]),
        (201, [226, 191, 281, 225], [(237, 209), (270, 209)]),
    ]},
    77: {"visible_start": 150, "items": [
        (153, [225, 191, 276, 224], [(235, 209), (266, 209)]),
        (159, [225, 191, 276, 224], [(235, 209), (266, 209)]),
        (177, [225, 191, 276, 224], [(235, 209), (266, 209)]),
    ]},
    78: {"visible_start": 180, "items": [
        (183, [226, 189, 277, 224], [(236, 207), (267, 207)]),
        (192, [226, 189, 277, 224], [(236, 207), (267, 207)]),
    ]},
    79: {"visible_start": 135, "items": [
        (138, [232, 190, 284, 224], [(242, 208), (274, 208)]),
        (159, [232, 190, 284, 224], [(242, 208), (274, 208)]),
        (180, [232, 187, 284, 224], [(242, 207), (274, 207)]),
        (201, [232, 184, 284, 224], [(242, 205), (274, 205)]),
    ]},
    80: {"visible_start": 72, "items": [(153, [253, 211, 308, 247], [(264, 230), (298, 230)])]},
    81: {"visible_start": 87, "items": [
        (90, [262, 214, 326, 248], [(273, 231), (314, 231)]),
        (108, [262, 214, 326, 248], [(273, 231), (314, 231)]),
        (126, [260, 213, 325, 248], [(272, 231), (313, 231)]),
        (144, [260, 213, 325, 248], [(272, 231), (313, 231)]),
        (156, [260, 212, 327, 248], [(272, 230), (314, 230)]),
    ]},
    82: {"visible_start": 90, "items": [
        (96, [256, 212, 322, 247], [(268, 230), (311, 230)]),
        (120, [256, 211, 322, 246], [(268, 229), (311, 229)]),
        (141, [256, 211, 322, 246], [(268, 229), (311, 229)]),
        (156, [253, 211, 322, 247], [(266, 230), (311, 230)]),
        (180, [252, 210, 322, 246], [(266, 229), (311, 229)]),
    ]},
    83: {"visible_start": 90, "items": [
        (96, [257, 211, 323, 246], [(269, 229), (312, 229)]),
        (111, [257, 211, 323, 246], [(269, 229), (312, 229)]),
        (132, [256, 211, 323, 246], [(268, 229), (312, 229)]),
        (147, [256, 210, 323, 245], [(268, 228), (312, 228)]),
        (171, [256, 209, 323, 245], [(268, 228), (312, 228)]),
        (189, [256, 209, 323, 245], [(268, 228), (312, 228)]),
    ]},
    84: {"visible_start": 90, "items": [
        (96, [260, 211, 324, 245], [(271, 228), (313, 228)]),
        (117, [260, 211, 324, 245], [(271, 228), (313, 228)]),
        (138, [260, 210, 324, 245], [(271, 228), (313, 228)]),
        (159, [260, 210, 324, 245], [(271, 228), (313, 228)]),
        (180, [260, 210, 324, 245], [(271, 228), (313, 228)]),
    ]},
    85: {"visible_start": 165, "items": [
        (170, [288, 186, 335, 214], [(296, 201), (326, 201)]),
        (185, [288, 186, 335, 214], [(296, 201), (326, 201)]),
        (205, [288, 186, 335, 214], [(296, 201), (326, 201)]),
        (220, [288, 186, 335, 214], [(296, 201), (326, 201)]),
        (237, [305, 186, 349, 214], [(315, 201), (339, 201)]),
    ]},
    86: {"visible_start": 99, "items": [
        (105, [314, 186, 342, 216], [(327, 202)]),
        (130, [314, 186, 342, 216], [(327, 202)]),
        (155, [314, 186, 342, 216], [(327, 202)]),
        (180, [314, 186, 342, 216], [(327, 202)]),
        (195, [314, 186, 342, 216], [(327, 202)]),
    ]},
    87: {"visible_start": 143, "items": [
        (145, [282, 185, 322, 215], [(292, 201), (312, 201)]),
        (150, [282, 185, 341, 215], [(292, 201), (330, 201)]),
        (155, [282, 185, 341, 215], [(292, 201), (330, 201)]),
        (170, [282, 185, 341, 215], [(292, 201), (330, 201)]),
        (183, [282, 185, 341, 215], [(292, 201), (330, 201)]),
    ]},
    88: {"visible_start": 115, "items": [
        (120, [289, 185, 339, 215], [(299, 201), (329, 201)]),
        (135, [289, 185, 339, 215], [(299, 201), (329, 201)]),
        (150, [289, 185, 339, 215], [(299, 201), (329, 201)]),
    ]},
    89: {"visible_start": 160, "items": [
        (160, [283, 185, 334, 214], [(293, 201), (324, 201)]),
        (180, [283, 185, 334, 214], [(293, 201), (324, 201)]),
        (200, [283, 185, 334, 214], [(293, 201), (324, 201)]),
        (220, [283, 185, 334, 214], [(293, 201), (324, 201)]),
        (240, [283, 185, 334, 214], [(293, 201), (324, 201)]),
        (255, [286, 184, 335, 215], [(296, 201), (326, 201)]),
    ]},
}


def parse_episodes(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            result.update(range(lo, hi + 1))
        elif part:
            result.add(int(part))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", default="60-69,72,75-89")
    return parser.parse_args()


def point(frame: int, x: float, y: float, label: int) -> dict[str, Any]:
    return {"frame": frame, "x": float(x), "y": float(y), "label": label}


def main() -> None:
    args = parse_args()
    wanted = parse_episodes(args.episodes)
    unknown = wanted - set(CORRECTIONS)
    if unknown:
        raise RuntimeError(f"No reviewed correction spec for episodes: {sorted(unknown)}")

    written = 0
    for episode in sorted(wanted):
        source = next(args.input_root.resolve().rglob(f"file-{episode:03d}/sam2_prompts.json"))
        data = json.loads(source.read_text(encoding="utf-8"))
        if int(data.get("file_index", -1)) != episode or data.get("tasks") != ["screw"]:
            raise RuntimeError(f"Unexpected prompt identity: {source}")
        objects = {item["name"]: item for item in data["objects"]}
        if set(objects) != {"occluder", "object", "region", "tool"}:
            raise RuntimeError(f"Unexpected labels in {source}: {sorted(objects)}")
        screw = objects["object"]
        spec = CORRECTIONS[episode]
        screw["visible_ranges"][0][0] = int(spec["visible_start"])
        correction_frames = {int(item[0]) for item in spec["items"]}
        correction_frames.update(int(frame) for frame in spec.get("drop_frames", []))
        screw["boxes"] = [
            item for item in screw["boxes"] if int(item["frame"]) not in correction_frames
        ]
        screw.setdefault("points", [])
        screw["points"] = [
            item for item in screw["points"] if int(item["frame"]) not in correction_frames
        ]
        for frame, box, positive_points in spec["items"]:
            screw["boxes"].append({
                "frame": int(frame),
                "box": [float(value) for value in box],
                "confidence": 1.0,
                "source": "manual_screw_full_extent_r5",
            })
            for x, y in positive_points:
                screw["points"].append(point(int(frame), x, y, 1))
            x1, y1, x2, y2 = map(float, box)
            for x, y in ((x1 - 16, (y1 + y2) / 2), (x2 + 16, (y1 + y2) / 2), ((x1 + x2) / 2, y1 - 20), ((x1 + x2) / 2, y2 + 18)):
                screw["points"].append(point(int(frame), x, y, 0))
        screw["boxes"].sort(key=lambda item: int(item["frame"]))
        screw["points"].sort(key=lambda item: (int(item["frame"]), int(item["label"]), item["x"], item["y"]))
        data.setdefault("manual_corrections", []).append({
            "version": "side_screw_full_extent_r5",
            "visible_start": int(spec["visible_start"]),
            "added_or_replaced_object_box_frames": sorted(correction_frames),
            "reason": "dense review: label the full visible screw rather than one metallic end",
        })
        destination = (
            args.output_root.resolve()
            / data["video_key"]
            / f"chunk-{int(data['chunk_index']):03d}"
            / f"file-{episode:03d}"
            / "sam2_prompts.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"episode {episode:03d}: {destination}")
        written += 1
    print(f"Wrote {written} corrected screw prompts")


if __name__ == "__main__":
    main()
