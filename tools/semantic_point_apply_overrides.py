#!/usr/bin/env python3
"""Apply explicitly reviewed point/box corrections to model prompt proposals."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from semantic_point_seed_prompts import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.proposal.read_text())
    overrides = json.loads(args.overrides.read_text())
    if int(overrides["episode_index"]) != int(data["episode_index"]):
        raise ValueError("Override episode does not match proposal")
    if not overrides.get("review_note", "").strip():
        raise ValueError("A substantive manual review note is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for preview in args.proposal.parent.glob("*_prompts.jpg"):
        shutil.copy2(preview, args.output_dir / preview.name)
    cap = cv2.VideoCapture(data["video_path"])
    if not cap.isOpened():
        raise RuntimeError("Could not open source video")
    for frame_text, changes in overrides["frames"].items():
        record = data["frames"].get(frame_text)
        if record is None:
            raise ValueError(f"Frame {frame_text} is not a proposed prompt frame")
        for name, edits in changes.items():
            current = record["classes"].get(name)
            if current is None:
                if not {"box", "positive", "negative"} <= set(edits):
                    raise ValueError(f"New class {name} at f{frame_text} requires box, positive and negative")
                current = {key: edits[key] for key in ("box", "positive", "negative")}
                record["classes"][name] = current
            old_positive = [list(point) for point in current["positive"]]
            for key in ("positive", "negative", "box"):
                if key in edits:
                    current[key] = edits[key]
            if "positive" in edits:
                old_set = {tuple(point) for point in old_positive}
                for other_name, other in record["classes"].items():
                    if other_name == name:
                        continue
                    kept = [point for point in other["negative"]
                            if tuple(point) not in old_set]
                    kept.extend(current["positive"])
                    other["negative"] = [list(point) for point in
                                         dict.fromkeys(tuple(point) for point in kept)]
        frame_idx = int(frame_text)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        okay, frame = cap.read()
        if not okay:
            raise RuntimeError(f"Could not decode f{frame_idx}")
        preview = args.output_dir / f"{record['event']}_f{frame_idx:05d}_prompts.jpg"
        if not cv2.imwrite(str(preview), render(frame, frame_idx, record["classes"]),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
            raise RuntimeError(f"Could not save {preview}")
    cap.release()
    data["status"] = "manual_revised_model_proposal_not_reviewed_for_propagation"
    data["manual_overrides"] = {"source": str(args.overrides.resolve()),
                                "review_note": overrides["review_note"],
                                "frames": sorted(map(int, overrides["frames"]))}
    output = args.output_dir / "proposals.json"
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"revised_proposal={output}")


if __name__ == "__main__":
    main()
