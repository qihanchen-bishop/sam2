#!/usr/bin/env python3
"""Prepare provisional seed reviews after full SAM2 completion; never accept masks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def complete(root: Path, total: int) -> bool:
    return ((root / "complete.json").is_file()
            and len(list((root / "mask_frames").glob("*.png"))) == total)


def run_step(name: str, command: list[str], log: Path) -> None:
    print(f"starting {name}; log={log}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {log}")
    print(f"provisional {name} complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path,
                        help="Reviewed prompt directory; defaults to seed reviewed_prompts_r1")
    parser.add_argument("--proposal-root", type=Path,
                        help="Source proposal directory; defaults to seed prompt_proposals")
    parser.add_argument("--proposal-file", type=Path,
                        help="Reviewed proposal JSON for a single episode")
    parser.add_argument("--output-root", type=Path,
                        help="Directory for per-episode revisions; defaults to --base")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--timeline-stride", type=int, default=70)
    parser.add_argument("--wait-timeout-minutes", type=int, default=120)
    args = parser.parse_args()
    if args.proposal_file and len(args.episodes) != 1:
        parser.error("--proposal-file requires exactly one --episodes value")
    script_dir = Path(__file__).resolve().parent
    prompt_root = args.prompt_root or args.base / "reviewed_prompts_r1"
    proposal_root = args.proposal_root or args.base / "prompt_proposals"
    output_root = args.output_root or args.base
    output_root.mkdir(parents=True, exist_ok=True)
    for episode in args.episodes:
        prompt = prompt_root / f"ep{episode:06d}" / "sam2_prompts.json"
        proposal = args.proposal_file or proposal_root / f"ep{episode:06d}" / "proposals.json"
        total = int(json.loads(prompt.read_text())["frame_count"])
        sam2 = args.sam2_root / f"file-{episode:03d}_sam2_masks"
        wait_started = time.monotonic()
        while not complete(sam2, total):
            if time.monotonic() - wait_started > args.wait_timeout_minutes * 60:
                raise TimeoutError(f"ep{episode} SAM2 did not complete: {sam2}")
            partial = sam2.with_name(sam2.name + ".partial")
            if not partial.exists() and not sam2.exists():
                print(f"waiting for ep{episode} propagation", flush=True)
            time.sleep(30)
        prefix = output_root / (f"sam2_episode_{episode:03d}"
                                if args.output_root else f"sam2_seed_ep{episode:03d}")
        refined = prefix.with_name(prefix.name + "_refined_r1")
        repaired = prefix.with_name(prefix.name + "_refined_r1_repaired")
        audit = prefix.with_name(prefix.name + "_refined_r1_repaired_audit")
        if not ((refined / "refinement.json").is_file()
                and len(list((refined / "mask_frames").glob("*.png"))) == total):
            run_step(f"ep{episode} refinement", [
                sys.executable, str(script_dir / "semantic_point_refine_masks.py"),
                "--proposal", str(proposal), "--prompt", str(prompt),
                "--mask-root", str(sam2), "--output", str(refined),
            ], prefix.with_name(prefix.name + "_refined_r1.log"))
        if not ((repaired / "isolated_repair.json").is_file()
                and len(list((repaired / "mask_frames").glob("*.png"))) == total):
            run_step(f"ep{episode} isolated/endpoint repair", [
                sys.executable, str(script_dir / "semantic_point_repair_isolated_frames.py"),
                "--prompt", str(prompt), "--mask-root", str(refined),
                "--output", str(repaired), "--repair-endpoints",
            ], prefix.with_name(prefix.name + "_refined_r1_repaired.log"))
        if not (audit / "audit.json").is_file():
            run_step(f"ep{episode} event audit", [
                sys.executable, str(script_dir / "semantic_point_mask_audit.py"),
                "--prompt", str(prompt), "--mask-root", str(repaired),
                "--output", str(audit),
            ], prefix.with_name(prefix.name + "_refined_r1_repaired_audit.log"))
        timeline = repaired / f"timeline_stride{args.timeline_stride}.jpg"
        if not timeline.is_file():
            run_step(f"ep{episode} timeline", [
                sys.executable, str(script_dir / "semantic_point_review_timeline.py"),
                "--prompt", str(prompt), "--mask-root", str(repaired),
                "--output", str(timeline), "--stride", str(args.timeline_stride),
            ], prefix.with_name(prefix.name + "_timeline.log"))
        print(f"ep{episode} ready_for_visual_review={repaired}", flush=True)
    print("all_provisional_reviews_ready; no_video_was_accepted", flush=True)


if __name__ == "__main__":
    main()
