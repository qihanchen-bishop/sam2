#!/usr/bin/env python3
"""Run reviewed seed SAM2 propagations sequentially on one GPU.

The queue never accepts masks or starts semantic training. A failed episode
stops the queue so later outputs are not mistaken for completed work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def completed(output_root: Path, episode: int, total: int) -> bool:
    root = output_root / f"file-{episode:03d}_sam2_masks"
    return ((root / "complete.json").is_file() and
            len(list((root / "mask_frames").glob("*.png"))) == total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--wait-for-session")
    parser.add_argument("--require-complete", type=Path,
                        help="Completion JSON required after the waited-on session exits")
    args = parser.parse_args()
    if args.wait_for_session:
        while subprocess.run(["tmux", "has-session", "-t", args.wait_for_session],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            time.sleep(30)
    if args.require_complete and not args.require_complete.is_file():
        raise RuntimeError(f"Previous propagation did not complete: {args.require_complete}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for episode in args.episodes:
        prompt_path = args.prompt_root / f"ep{episode:06d}" / "sam2_prompts.json"
        prompt = json.loads(prompt_path.read_text())
        total = int(prompt["frame_count"])
        if completed(args.output_root, episode, total):
            print(f"already_complete ep{episode}", flush=True)
            continue
        partial = args.output_root / f"file-{episode:03d}_sam2_masks.partial"
        if partial.exists():
            raise RuntimeError(f"Stale partial output requires review: {partial}")
        command = [sys.executable, str(Path(__file__).with_name("sam2_prompts_to_masks.py")),
                   "--prompt-json", str(prompt_path), "--output-root", str(args.output_root),
                   "--device", args.device, "--composition-mode", "priority",
                   "--save-bidirectional"]
        log_path = args.output_root / f"ep{episode:06d}.log"
        print(f"starting ep{episode} log={log_path}", flush=True)
        environment = dict(os.environ, TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=environment)
        if result.returncode != 0 or not completed(args.output_root, episode, total):
            raise RuntimeError(f"Propagation failed ep{episode}; inspect {log_path}")
        print(f"propagated_not_accepted ep{episode}", flush=True)
    print("queue_complete_outputs_still_need_review", flush=True)


if __name__ == "__main__":
    main()
