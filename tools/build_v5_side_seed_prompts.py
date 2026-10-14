#!/usr/bin/env python3
"""Create the v5 four-class side seed prompt set from reviewed v3 prompts.

The v3 and v4 side source videos are byte-identical. Old masks are not reused;
only the reviewed prompt coordinates and visibility ranges are migrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


KEEP_LABELS = {"occluder", "object", "region", "tool"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--old-dataset", type=Path, required=True)
    parser.add_argument("--new-dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for source in sorted(args.source_root.glob("*/sam2_prompts.json")):
        payload = json.loads(source.read_text())
        old_video = Path(payload["video_path"])
        relative_video = old_video.relative_to(args.old_dataset)
        new_video = args.new_dataset / relative_video
        if not new_video.is_file():
            raise FileNotFoundError(new_video)
        old_digest = sha256(old_video)
        new_digest = sha256(new_video)
        if old_digest != new_digest:
            raise RuntimeError(f"Source video differs for {relative_video}")
        objects = [item for item in payload["objects"] if item["name"] in KEEP_LABELS]
        if {item["name"] for item in objects} != KEEP_LABELS:
            raise RuntimeError(f"Incomplete four-class prompts in {source}")
        for item in objects:
            for box in item.get("boxes", []):
                box.setdefault("confidence", 1.0)
        by_name = {item["name"]: item for item in objects}
        task = str(payload["task"])
        if task == "paperball":
            # At f605 the original box included most of the gray gripper. Keep
            # the prompt tight on the orange plate itself.
            for box in by_name["tool"]["boxes"]:
                if int(box["frame"]) == 605:
                    box["box"] = [277.0, 139.0, 327.0, 178.0]
            by_name["tool"].setdefault("points", []).extend(
                [
                    {"frame": 605, "x": 301.0, "y": 159.0, "label": 1},
                    {"frame": 605, "x": 311.0, "y": 140.0, "label": 0},
                ]
            )
            by_name["tool"].setdefault("points", []).append(
                {"frame": 275, "x": 404.0, "y": 223.0, "label": 0}
            )
            by_name["object"].setdefault("points", []).append(
                {"frame": 275, "x": 368.0, "y": 179.0, "label": 0}
            )
        if task == "screw":
            by_name["tool"].setdefault("points", []).append(
                {"frame": 240, "x": 369.0, "y": 237.0, "label": 0}
            )
            by_name["object"].setdefault("points", []).append(
                {"frame": 240, "x": 330.0, "y": 190.0, "label": 0}
            )
        payload.update(
            {
                "dataset": "bettersetup_v5_side_seed_r1",
                "video_path": str(new_video.resolve()),
                "chunk_index": 0,
                "file_index": int(payload["episode_index"]),
                "labels_source": str(args.labels.resolve()),
                "view_description": "从左往右看；side只分割occluder/object/region/tool，机械臂为背景",
                "objects": objects,
            }
        )
        output = args.output_root / "observation.images.side" / task / "sam2_prompts.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        records.append(
            {
                "task": task,
                "episode_index": payload["episode_index"],
                "prompt_frames": payload["prompt_frames"],
                "source_prompt": str(source.resolve()),
                "source_video_sha256": old_digest,
                "output": str(output.resolve()),
            }
        )
    if len(records) != 3:
        raise RuntimeError(f"Expected three side seed prompts, found {len(records)}")
    manifest = args.output_root / "migration_manifest.json"
    manifest.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"prompts": len(records), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
