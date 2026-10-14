#!/usr/bin/env python3
"""Apply source-video-reviewed event corrections without losing dense anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVENTS = ("occluder_contact", "exposed", "object_contact", "region_contact")
OFFSETS = {
    "occluder_contact": (-8, 8),
    "exposed": (-8, -2, 2, 4, 8, 16),
    "object_contact": (-18, -6, -2, 2, 6, 18),
    "region_contact": (-18, -6, -2, 2, 6, 18),
}


def rebuild(row: dict) -> dict:
    last = int(row["last"])
    base = {name: row.get(name) for name in ("task", "first", *EVENTS, "last")}
    occupied = {value for key, value in base.items()
                if key != "task" and value is not None}
    for name in EVENTS:
        center = base[name]
        if center is None:
            continue
        for offset in OFFSETS[name]:
            frame = max(0, min(last, center + offset))
            if frame not in occupied:
                base[f"{name}_{offset:+d}"] = frame
                occupied.add(frame)
    for percentage in (25, 50, 75):
        frame = round(last * percentage / 100)
        if frame not in occupied:
            base[f"extra_{percentage}pct"] = frame
            occupied.add(frame)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    review = json.loads(args.review.read_text())
    if not review.get("review_note", "").strip():
        raise ValueError("Source-video review note required")
    applied = []
    for entry in review["episodes"]:
        key = str(int(entry["episode_index"]))
        if key not in data["episodes"] or key not in data["analyses"]:
            raise ValueError(f"Missing episode {key}")
        if not entry.get("evidence") or not entry.get("review_note", "").strip():
            raise ValueError(f"Episode {key} requires evidence and review note")
        current = data["episodes"][key]
        last = int(current["last"])
        for name, value in entry.get("events", {}).items():
            if name not in EVENTS or not 0 <= int(value) <= last:
                raise ValueError(f"Invalid event {key}:{name}={value}")
            current[name] = int(value)
        current = rebuild(current)
        data["episodes"][key] = current
        analysis = data["analyses"][key]
        analysis["events"] = current
        analysis["missing"] = [name for name in EVENTS if current[name] is None]
        analysis["status"] = "source_video_events_reviewed"
        applied.append({"episode_index": int(key), "review_note": entry["review_note"],
                        "evidence": entry["evidence"], "events": entry.get("events", {})})
    data.setdefault("source_video_event_reviews", []).extend(applied)
    data["status"] = "partially_source_video_reviewed"
    data["review_file"] = str(args.review.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"reviewed_events={args.output} reviewed_episodes={len(applied)}")


if __name__ == "__main__":
    main()
