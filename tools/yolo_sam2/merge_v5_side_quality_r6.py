#!/usr/bin/env python3
"""Install the episode-065 side correction and refreshed quality into v5."""

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
    parser.add_argument("--cleanup-report", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    source_final = args.source_final.resolve()
    side_quality = args.side_quality.resolve()
    destination = candidate / "segmentation_quality"
    destination.mkdir(parents=True, exist_ok=True)

    side_summary = read_json(side_quality / "summary.json")
    side_video_summary = read_json(side_quality / "video_summary.json")
    if set(side_summary["streams"]) != set(SIDE_STREAMS):
        raise RuntimeError(f"Unexpected streams: {sorted(side_summary['streams'])}")
    if len(side_video_summary) != int(side_summary["processed_videos"]):
        raise RuntimeError("Side video summary count does not match summary")

    for stream in SIDE_STREAMS:
        filename = stream.replace(".", "_") + ".npz"
        shutil.copy2(side_quality / filename, destination / filename)
    shutil.copy2(side_quality / "summary.json", destination / "side_summary.json")
    shutil.copy2(side_quality / "video_summary.json", destination / "side_video_summary.json")

    combined = read_json(source_final / "segmentation_quality" / "summary.json")
    for stream in SIDE_STREAMS:
        combined["streams"][stream] = side_summary["streams"][stream]
    combined["seg_root"] = str(args.side_seg_root.resolve())
    combined["processed_videos"] = side_summary["processed_videos"]
    combined["processed_frames_per_stream"] = side_summary["processed_frames_per_stream"]
    combined["completeness_failures"] = side_summary.get("completeness_failures", [])
    write_json(destination / "summary.json", combined)

    manifest = read_json(source_final / "segmentation_manifest.json")
    manifest["version"] = 2
    manifest["revision"] = "side_screw_ep065_r6"
    manifest["side_segmentation_source"] = str(args.side_seg_root.resolve())
    manifest["side_quality_source"] = str(side_quality)
    manifest.setdefault("review_artifacts", {})["quality_root"] = str(side_quality)
    manifest["side_episode_065_correction"] = {
        "class": "object",
        "task": "screw",
        "prompt_frames": [152, 154, 162, 177, 189, 198, 210, 225, 240, 252, 260],
        "corrected_ranges": [[150, 262]],
        "fully_tool_occluded_range_retained_empty": [370, 390],
        "method": "dense positive/negative prompts plus forward/reverse SAM2 propagation",
        "small_component_cleanup": read_json(args.cleanup_report),
    }
    write_json(candidate / "segmentation_manifest.json", manifest)

    print(f"Merged four refreshed side quality streams into {destination}")
    print(f"Revision: {manifest['revision']}")


if __name__ == "__main__":
    main()
