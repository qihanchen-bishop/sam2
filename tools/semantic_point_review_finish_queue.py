#!/usr/bin/env python3
"""Prepare rod-complete, audited provisional reviews as episodes finish.

This never changes acceptance.json or publishes training data. Existing
complete outputs are reused so an interrupted queue can resume safely.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_step(name: str, command: list[str], log: Path) -> None:
    print(f"starting {name}; log={log}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {log}")
    print(f"provisional {name} complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--proposal-file", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--wait-timeout-minutes", type=int, default=120)
    args = parser.parse_args()
    if args.proposal_file and len(args.episodes) != 1:
        parser.error("--proposal-file requires a single episode")
    script_dir = Path(__file__).resolve().parent
    review_root = args.work_root / "round12_reviewprep"
    review_root.mkdir(parents=True, exist_ok=True)
    for episode in args.episodes:
        prompt = args.prompt_root / f"ep{episode:06d}" / "sam2_prompts.json"
        proposal = (args.proposal_file or args.proposal_root /
                    f"ep{episode:06d}" / "proposals.json")
        metadata = json.loads(prompt.read_text())
        total = int(metadata["frame_count"])
        prefix = review_root / f"sam2_episode_{episode:03d}"
        repaired = prefix.with_name(prefix.name + "_refined_r1_repaired")
        deadline = time.monotonic() + args.wait_timeout_minutes * 60
        while (not (repaired / "isolated_repair.json").is_file()
               or len(list((repaired / "mask_frames").glob("*.png"))) != total):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ep{episode} repaired root unavailable: {repaired}")
            time.sleep(30)
        rod = prefix.with_name(prefix.name + "_rodcomplete_r1")
        if (not (rod / "rod_completion.json").is_file()
                or len(list((rod / "mask_frames").glob("*.png"))) != total):
            run_step(f"ep{episode} rod completion", [
                sys.executable, str(script_dir / "semantic_point_extend_tool_rods.py"),
                "--proposal", str(proposal), "--prompt", str(prompt),
                "--mask-root", str(repaired), "--output", str(rod),
            ], review_root / f"ep{episode:03d}_rodcomplete.log")
        audit = prefix.with_name(prefix.name + "_rodcomplete_r1_audit")
        if not (audit / "audit.json").is_file():
            run_step(f"ep{episode} final event audit", [
                sys.executable, str(script_dir / "semantic_point_mask_audit.py"),
                "--prompt", str(prompt), "--mask-root", str(rod),
                "--output", str(audit),
            ], review_root / f"ep{episode:03d}_rodcomplete_audit.log")
        timeline = rod / "timeline_stride70.jpg"
        if not timeline.is_file():
            run_step(f"ep{episode} final timeline", [
                sys.executable, str(script_dir / "semantic_point_review_timeline.py"),
                "--prompt", str(prompt), "--mask-root", str(rod),
                "--output", str(timeline), "--stride", "70",
            ], review_root / f"ep{episode:03d}_rodcomplete_timeline.log")
        if metadata["task"] in ("o_n", "g_n"):
            color = review_root / f"ep{episode:03d}_rodcomplete_color_agreement.json"
            if not color.is_file():
                run_step(f"ep{episode} object color agreement", [
                    sys.executable, str(script_dir / "semantic_point_color_agreement.py"),
                    "--prompt", str(prompt), "--mask-root", str(rod),
                    "--output", str(color),
                ], review_root / f"ep{episode:03d}_rodcomplete_color_agreement.log")
        print(f"ep{episode} ready_for_visual_review={rod}", flush=True)
    print("all_provisional_reviews_ready; no_episode_was_accepted", flush=True)


if __name__ == "__main__":
    main()
