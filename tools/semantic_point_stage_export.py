#!/usr/bin/env python3
"""Stage only accepted masks for the LeRobot binary-mask video exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def link_checked(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source:
            return
        raise FileExistsError(f"Refusing to replace existing stage path {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acceptance = json.loads(args.acceptance.read_text())
    manifest = json.loads(args.manifest.read_text())
    expected = {int(row["episode_index"]) for row in manifest["all_episodes"]}
    accepted = {int(key): record for key, record in acceptance["episodes"].items()
                if record["status"] == "accepted"}
    if set(accepted) != expected:
        missing = sorted(expected - set(accepted))
        extra = sorted(set(accepted) - expected)
        raise ValueError(f"Final export requires every episode accepted: missing={missing} extra={extra}")
    for episode, record in sorted(accepted.items()):
        prompt = Path(record["prompt"])
        mask_root = Path(record["mask_root"])
        total = int(json.loads(prompt.read_text())["frame_count"])
        mask_dir = mask_root / "mask_frames"
        if len(list(mask_dir.glob("*.png"))) != total:
            raise ValueError(f"Incomplete accepted masks ep{episode}")
        staged = args.output / "observation.images.front" / "chunk-000" / f"file-{episode:03d}_sam2_masks"
        staged.mkdir(parents=True, exist_ok=True)
        link_checked(prompt, staged / "sam2_prompts.json")
        link_checked(mask_dir, staged / "mask_frames")
    (args.output / "stage_manifest.json").write_text(json.dumps({
        "status": "all_accepted_export_stage", "episodes": sorted(accepted),
        "source_acceptance": str(args.acceptance.resolve()),
    }, indent=2) + "\n")
    print(f"export_stage={args.output} accepted={len(accepted)}")


if __name__ == "__main__":
    main()
