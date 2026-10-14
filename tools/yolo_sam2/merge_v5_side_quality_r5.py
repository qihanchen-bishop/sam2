#!/usr/bin/env python3
"""Install an updated side quality report into a copied v5 dataset candidate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SIDE_STREAMS = (
    "observation.images.side_occluder",
    "observation.images.side_object",
    "observation.images.side_region",
    "observation.images.side_tool",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-final", type=Path, required=True)
    parser.add_argument("--side-quality", type=Path, required=True)
    parser.add_argument("--side-seg-root", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    source_final = args.source_final.resolve()
    side_quality = args.side_quality.resolve()
    destination = candidate / "segmentation_quality"

    side_summary = read_json(side_quality / "summary.json")
    side_video_summary = read_json(side_quality / "video_summary.json")
    if set(side_summary["streams"]) != set(SIDE_STREAMS):
        raise RuntimeError(
            f"Unexpected side quality streams: {sorted(side_summary['streams'])}"
        )

    for stream in SIDE_STREAMS:
        filename = stream.replace(".", "_") + ".npz"
        shutil.copy2(side_quality / filename, destination / filename)
    shutil.copy2(side_quality / "summary.json", destination / "side_summary.json")
    shutil.copy2(
        side_quality / "video_summary.json", destination / "side_video_summary.json"
    )

    combined = read_json(source_final / "segmentation_quality" / "summary.json")
    for stream in SIDE_STREAMS:
        combined["streams"][stream] = side_summary["streams"][stream]
    combined["seg_root"] = str(args.side_seg_root.resolve())
    combined["processed_videos"] = side_summary["processed_videos"]
    combined["processed_frames_per_stream"] = side_summary[
        "processed_frames_per_stream"
    ]
    combined["completeness_failures"] = side_summary.get(
        "completeness_failures", []
    )
    write_json(destination / "summary.json", combined)

    if len(side_video_summary) != int(side_summary["processed_videos"]):
        raise RuntimeError("Side video summary count does not match summary")

    manifest = read_json(source_final / "segmentation_manifest.json")
    manifest["version"] = 2
    manifest["revision"] = "side_screw_r5"
    manifest["side_segmentation_source"] = str(args.side_seg_root.resolve())
    manifest["side_quality_source"] = str(side_quality)
    manifest["side_screw_reaudit"] = {
        "episodes": list(range(60, 90)),
        "rerun_episodes": [
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            72,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
        ],
        "retained_reviewed_episodes": [70, 71, 73, 74],
        "class": "object",
        "task": "screw",
        "method": "dense manual prompts plus forward/reverse SAM2 propagation",
    }
    write_json(candidate / "segmentation_manifest.json", manifest)

    print(f"Merged four side quality streams into {destination}")
    print(f"Updated combined manifest: {candidate / 'segmentation_manifest.json'}")


if __name__ == "__main__":
    main()
