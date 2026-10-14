#!/usr/bin/env python3
"""Prepare model-derived prompts and visual reviews for a balanced growth round."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--event-stride", type=int, default=10)
    parser.add_argument("--event-confidence", type=float, default=0.62)
    parser.add_argument("--point-confidence", type=float, default=0.72)
    parser.add_argument("--event-candidates", type=Path,
                        help="Reuse previously generated event candidates after changing prompt logic")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = args.output_dir / "event_candidates.json"
    if args.event_candidates:
        source = json.loads(args.event_candidates.read_text())
        absent = [episode for episode in args.episodes
                  if str(episode) not in source["episodes"]]
        if absent:
            raise ValueError(f"Missing event candidates for episodes {absent}")
        events.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    else:
        run([sys.executable, str(scripts / "semantic_point_model_events.py"),
             "--model-dir", str(args.model_dir), "--manifest", str(args.manifest),
             "--episodes", *map(str, args.episodes), "--output", str(events),
             "--device", args.device, "--stride", str(args.event_stride),
             "--confidence", str(args.event_confidence)])
    event_data = json.loads(events.read_text())
    statuses = {}
    for episode in args.episodes:
        missing = event_data["analyses"][str(episode)]["missing"]
        if missing:
            statuses[str(episode)] = {"status": "event_review_needed", "missing": missing}
            print(f"ep{episode} needs event review: {missing}", flush=True)
            continue
        episode_dir = args.output_dir / f"ep{episode:06d}"
        run([sys.executable, str(scripts / "semantic_point_model_prompts.py"),
             "--model-dir", str(args.model_dir), "--manifest", str(args.manifest),
             "--events", str(events), "--episode", str(episode),
             "--output-dir", str(episode_dir), "--device", args.device,
             "--point-confidence", str(args.point_confidence)])
        audit = episode_dir / "pixel_audit.json"
        run([sys.executable, str(scripts / "semantic_point_prompt_audit.py"),
             "--proposal", str(episode_dir / "proposals.json"),
             "--output", str(audit)])
        run([sys.executable, str(scripts / "semantic_point_prompt_contact_sheet.py"),
             "--proposal-dir", str(episode_dir),
             "--output", str(episode_dir / "contact_sheet.jpg")])
        proposal = json.loads((episode_dir / "proposals.json").read_text())
        pixel_audit = json.loads(audit.read_text())
        statuses[str(episode)] = {
            "status": "prompt_review_needed",
            "frames": len(proposal["frames"]),
            "pixel_audit": pixel_audit["counts"],
            "proposal_concerns": sum(map(len, proposal["concerns"].values())),
        }
        print(f"ep{episode} candidate prompts ready: {statuses[str(episode)]}", flush=True)
    output = {"status": "candidates_not_reviewed", "model_dir": str(args.model_dir.resolve()),
              "episodes": statuses, "event_candidates": str(events.resolve())}
    (args.output_dir / "round_review_queue.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"round_review_queue={args.output_dir / 'round_review_queue.json'}")


if __name__ == "__main__":
    main()
