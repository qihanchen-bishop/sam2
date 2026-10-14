#!/usr/bin/env python3
"""Flag unsafe seed-prompt pixels before any SAM2 propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from semantic_point_seed_prompts import clustered_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--frames", type=Path,
                        help="Raw extracted frames; if omitted, decode proposal source video")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.proposal.read_text())
    episode = int(data["episode_index"])
    flags = []
    cap = None
    if args.frames is None:
        cap = cv2.VideoCapture(data["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source video {data['video_path']}")
    for frame_text, record in data["frames"].items():
        frame_idx = int(frame_text)
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            okay, frame = cap.read()
            if not okay:
                raise RuntimeError(f"Could not decode source frame {frame_idx}")
        else:
            frame_path = args.frames / f"ep{episode:06d}" / f"{record['event']}_f{frame_idx:05d}.png"
            frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"ep{episode} f{frame_idx}")
        masks = clustered_masks(frame, data["task"])
        for name, part in record["classes"].items():
            box = part["box"]
            positive = part["positive"]
            negative = part["negative"]
            if name in ("lefttool", "righttool") and len(positive) != 4:
                flags.append({"frame": frame_idx, "class": name,
                              "severity": "critical", "reason": "not_four_tool_positive_points"})
            for point in positive:
                x, y = map(int, point)
                if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                    flags.append({"frame": frame_idx, "class": name,
                                  "severity": "critical", "reason": "positive_out_of_image",
                                  "point": point})
                    continue
                if not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
                    flags.append({"frame": frame_idx, "class": name,
                                  "severity": "critical", "reason": "positive_outside_box",
                                  "point": point})
                if any(np.linalg.norm(np.array(point) - np.array(q)) < 5 for q in negative):
                    flags.append({"frame": frame_idx, "class": name,
                                  "severity": "critical", "reason": "positive_negative_collision",
                                  "point": point})
                if name == "object" and not masks["object"][y, x]:
                    flags.append({"frame": frame_idx, "class": name,
                                  "severity": "review", "reason": "object_positive_outside_color_cluster",
                                  "point": point})
                if name == "region" and not masks["region"][y, x]:
                    flags.append({"frame": frame_idx, "class": name,
                                  "severity": "review", "reason": "region_positive_outside_color_cluster",
                                  "point": point})
                object_entry = record["classes"].get("object")
                near_reviewed_object = (object_entry is not None and any(
                    np.linalg.norm(np.array(point) - np.array(obj_point)) < 35
                    for obj_point in object_entry["positive"]))
                if name in ("lefttool", "righttool") and object_entry is not None:
                    ox1, oy1, ox2, oy2 = object_entry["box"]
                    if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                        flags.append({"frame": frame_idx, "class": name,
                                      "severity": "critical", "reason": "tool_positive_inside_object_box",
                                      "point": point})
                if (name in ("lefttool", "righttool") and masks["object"][y, x]
                        and near_reviewed_object):
                    flags.append({"frame": frame_idx, "class": name,
                                  "severity": "critical", "reason": "tool_positive_on_object",
                                  "point": point})
                if name in ("lefttool", "righttool") and masks["red"][y, x]:
                    flags.append({"frame": frame_idx, "class": name,
                                  "severity": "critical", "reason": "tool_positive_on_red_occluder",
                                  "point": point})
                if name in ("lefttool", "righttool"):
                    hsv_pixel = cv2.cvtColor(frame[y:y + 1, x:x + 1],
                                             cv2.COLOR_BGR2HSV)[0, 0]
                    if int(hsv_pixel[2]) >= 205 and int(hsv_pixel[1]) < 35:
                        flags.append({"frame": frame_idx, "class": name,
                                      "severity": "review", "reason": "tool_positive_on_bright_neutral_pixel",
                                      "point": point})
    if cap is not None:
        cap.release()
    counts = {"critical": sum(f["severity"] == "critical" for f in flags),
              "review": sum(f["severity"] == "review" for f in flags)}
    report = {"episode_index": episode, "status": "audit_only_not_accepted",
              "frame_count": len(data["frames"]), "counts": counts, "flags": flags}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"prompt_audit={args.output} counts={counts}")


if __name__ == "__main__":
    main()
