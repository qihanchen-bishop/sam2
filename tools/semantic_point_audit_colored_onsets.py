#!/usr/bin/env python3
"""Find source-color onset candidates before semantic events, for visual review only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from semantic_point_seed_prompts import clustered_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--episodes", type=int, nargs="+",
                        help="Restrict audit to these episode indices")
    args = parser.parse_args()
    event_data = json.loads(args.candidates.read_text())
    manifest = json.loads(args.manifest.read_text())
    records = {str(row["episode_index"]): row for row in manifest["all_episodes"]}
    results = {}
    for episode, events in event_data["episodes"].items():
        if args.episodes is not None and int(episode) not in args.episodes:
            continue
        record = records[episode]
        if record["task"] not in ("o_n", "g_n"):
            continue
        exposed = events.get("exposed")
        if exposed is None:
            results[episode] = {"status": "model_exposure_missing"}
            continue
        cap = cv2.VideoCapture(record["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open ep{episode} video")
        run = 0
        first = None
        area_at_first = 0
        inspected = 0
        required_run = min(args.consecutive, int(exposed) + 1)
        for frame_idx in range(min(int(exposed) + 1, int(record["frame_count"]))):
            okay, frame = cap.read()
            if not okay:
                raise RuntimeError(f"Cannot read ep{episode} f{frame_idx}")
            area = int((clustered_masks(frame, record["task"])["object"] > 0).sum())
            inspected += 1
            if area >= args.min_area:
                if run == 0:
                    first, area_at_first = frame_idx, area
                run += 1
                if run >= required_run:
                    break
            else:
                run = 0
                first = None
        cap.release()
        detected = first if run >= required_run else None
        results[episode] = {
            "status": "source_color_candidate_not_reviewed",
            "task": record["task"], "model_exposed": int(exposed),
            "source_color_candidate": detected, "area_at_candidate": area_at_first if detected is not None else None,
            "model_lag_frames": int(exposed) - detected if detected is not None else None,
            "inspected_frames": inspected,
        }
        print(f"ep{episode} model={exposed} source_candidate={detected} "
              f"lag={results[episode]['model_lag_frames']}", flush=True)
    report = {"status": "audit_only_not_reviewed", "source_candidates": str(args.candidates.resolve()),
              "min_area": args.min_area, "consecutive": args.consecutive, "episodes": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"source_onset_audit={args.output}")


if __name__ == "__main__":
    main()
