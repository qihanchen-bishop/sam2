#!/usr/bin/env python3
"""Record source-video-reviewed event corrections in a new candidate file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    changes = json.loads(args.overrides.read_text())
    if not changes.get("review_note", "").strip():
        raise ValueError("A source-video review note is required")
    episode = str(int(changes["episode_index"]))
    if episode not in data["episodes"] or episode not in data["analyses"]:
        raise ValueError(f"Episode {episode} is missing from event candidates")
    last = int(data["episodes"][episode]["last"])
    for name, value in changes["events"].items():
        frame = int(value)
        if name in ("first", "last") or not 0 <= frame <= last:
            raise ValueError(f"Invalid event override {name}={value}")
        data["episodes"][episode][name] = frame
        data["analyses"][episode]["events"][name] = frame
    data.setdefault("manual_event_overrides", []).append({
        "episode_index": int(episode), "review_note": changes["review_note"],
        "evidence": changes.get("evidence", []), "events": changes["events"],
        "source": str(args.overrides.resolve()),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"reviewed_event_candidates={args.output} episode={episode} "
          f"changes={len(changes['events'])}")


if __name__ == "__main__":
    main()
