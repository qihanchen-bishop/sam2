#!/usr/bin/env python3
"""Export an explicitly reviewed seed proposal to SAM2's video prompt format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CLASSES = ("occluder", "object", "region", "lefttool", "righttool")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-visible-start", type=int, required=True)
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--pixel-audit", type=Path, required=True)
    parser.add_argument("--video-override", type=Path,
                        help="Diagnostic clip only; never use as the source of accepted labels")
    parser.add_argument("--limit-frames", type=int,
                        help="Diagnostic clip frame count, paired with --video-override")
    args = parser.parse_args()
    source = json.loads(args.proposal.read_text())
    pixel_audit = json.loads(args.pixel_audit.read_text())
    if (int(pixel_audit["episode_index"]) != int(source["episode_index"])
            or int(pixel_audit["frame_count"]) != len(source["frames"])
            or pixel_audit["counts"]["critical"] != 0):
        raise ValueError("Prompt pixel audit does not match or has critical flags")
    if not args.review_note.strip():
        raise ValueError("A substantive review note is required")
    if (args.video_override is None) != (args.limit_frames is None):
        raise ValueError("--video-override and --limit-frames must be used together")
    total = args.limit_frames or int(source["frame_count"])
    if not 0 <= args.object_visible_start < total:
        raise ValueError("object-visible-start out of range")
    data = {
        "version": 1,
        "dataset": "newur",
        "video_key": "observation.images.front",
        "episode_index": int(source["episode_index"]),
        "task": source["task"],
        "video_path": str(args.video_override.resolve()) if args.video_override else source["video_path"],
        "frame_count": total,
        "prompt_frames": sorted(int(frame) for frame in source["frames"]
                                if int(frame) < total),
        "labels_source": "/data/qihan/sam2/segdata/origin/newur/labels.txt",
        "provenance": {"method": ("semantic_point_bootstrap_model_v1"
                                  if "model" in source.get("status", "")
                                  else "semantic_point_bootstrap_seed_v1"),
                       "review_note": args.review_note,
                       "proposal_path": str(args.proposal.resolve()),
                       "pixel_audit_path": str(args.pixel_audit.resolve())},
        "objects": [],
    }
    if args.video_override:
        data["provenance"]["diagnostic_clip_only"] = True
    for class_id, name in enumerate(CLASSES, start=1):
        obj = {"id": class_id, "name": name, "boxes": [], "points": []}
        if name == "object":
            obj["visible_ranges"] = [[args.object_visible_start, total - 1]]
        for frame_text, record in source["frames"].items():
            frame_idx = int(frame_text)
            if frame_idx >= total:
                continue
            part = record["classes"].get(name)
            if part is None:
                continue
            if name == "object" and frame_idx < args.object_visible_start:
                raise ValueError(f"Object prompt at f{frame_idx} before visible start")
            obj["boxes"].append({"frame": frame_idx, "box": part["box"]})
            for label, key in ((1, "positive"), (0, "negative")):
                for x, y in part[key]:
                    obj["points"].append({"frame": frame_idx, "x": x,
                                          "y": y, "label": label})
        if not obj["boxes"]:
            raise ValueError(f"No box for {name}")
        data["objects"].append(obj)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"sam2_prompt={args.output}")


if __name__ == "__main__":
    main()
