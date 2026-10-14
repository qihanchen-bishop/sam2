#!/usr/bin/env python3
"""Merge a narrow SAM2 correction into existing semantic masks safely.

Only pixels with the requested source class in the base mask and target class
in the correction mask are changed.  Every modified PNG is copied to a backup
tree before it is replaced, making the operation reversible.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--target-id", type=int, required=True)
    parser.add_argument("--appearance-filter", action="store_true")
    parser.add_argument("--max-saturation", type=int, default=55)
    parser.add_argument("--min-value", type=int, default=75)
    parser.add_argument("--appearance-radius", type=int, default=7)
    parser.add_argument("--min-appearance-seed", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def episode_from_output(path: Path) -> int:
    prompt = json.loads((path / "sam2_prompts.json").read_text(encoding="utf-8"))
    value = prompt.get("file_index", prompt.get("episode_index"))
    if value is None:
        raise RuntimeError(f"No episode identity in {path / 'sam2_prompts.json'}")
    return int(value)


def output_index(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for complete in root.glob("**/complete.json"):
        output = complete.parent
        episode = episode_from_output(output)
        if episode in result:
            raise RuntimeError(f"Duplicate episode {episode} under {root}")
        result[episode] = output
    return result


def merge_directory(
    base_dir: Path,
    correction_dir: Path,
    backup_dir: Path,
    source_id: int,
    target_id: int,
    dry_run: bool,
    capture: cv2.VideoCapture | None = None,
    max_saturation: int = 55,
    min_value: int = 75,
    appearance_radius: int = 7,
    min_appearance_seed: int = 30,
) -> tuple[int, int]:
    changed_frames = 0
    changed_pixels = 0
    if not base_dir.is_dir() or not correction_dir.is_dir():
        return changed_frames, changed_pixels
    for correction_path in sorted(correction_dir.glob("*.png")):
        base_path = base_dir / correction_path.name
        if not base_path.is_file():
            continue
        base = cv2.imread(str(base_path), cv2.IMREAD_GRAYSCALE)
        correction = cv2.imread(str(correction_path), cv2.IMREAD_GRAYSCALE)
        if base is None or correction is None or base.shape != correction.shape:
            raise RuntimeError(f"Invalid mask pair: {base_path}, {correction_path}")
        selected = (base == source_id) & (correction == target_id)
        if capture is not None and np.any(selected):
            frame_index = int(correction_path.stem)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read appearance frame {frame_index}")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            seed = (
                selected
                & (hsv[:, :, 1] < max_saturation)
                & (hsv[:, :, 2] > min_value)
            )
            if int(np.count_nonzero(seed)) < min_appearance_seed:
                selected[:] = False
            else:
                diameter = 2 * appearance_radius + 1
                nearby = cv2.dilate(
                    seed.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)),
                ) > 0
                selected &= nearby
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        changed_frames += 1
        changed_pixels += count
        if dry_run:
            continue
        backup_path = backup_dir / correction_path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(base_path, backup_path)
        base[selected] = target_id
        if not cv2.imwrite(str(base_path), base):
            raise RuntimeError(f"Cannot write {base_path}")
    return changed_frames, changed_pixels


def main() -> None:
    args = parse_args()
    base = output_index(args.base_root)
    correction = output_index(args.correction_root)
    rows = []
    for episode, correction_output in sorted(correction.items()):
        if episode not in base:
            raise RuntimeError(f"Correction episode {episode} missing from base root")
        base_output = base[episode]
        relative = base_output.relative_to(args.base_root)
        episode_backup = args.backup_root / relative
        completion = json.loads((base_output / "complete.json").read_text(encoding="utf-8"))
        capture = None
        if args.appearance_filter:
            capture = cv2.VideoCapture(str(completion["video_path"]))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {completion['video_path']}")
        total_frames = 0
        total_pixels = 0
        streams = [("mask_frames", "mask_frames")]
        for direction in ("forward", "reverse"):
            streams.append(
                (
                    f"directional_mask_frames/{direction}",
                    f"directional_mask_frames/{direction}",
                )
            )
        stream_rows = {}
        for base_name, correction_name in streams:
            frames, pixels = merge_directory(
                base_output / base_name,
                correction_output / correction_name,
                episode_backup / base_name,
                args.source_id,
                args.target_id,
                args.dry_run,
                capture if base_name == "mask_frames" else None,
                args.max_saturation,
                args.min_value,
                args.appearance_radius,
                args.min_appearance_seed,
            )
            stream_rows[base_name] = {"changed_frames": frames, "changed_pixels": pixels}
            total_frames += frames
            total_pixels += pixels
        if capture is not None:
            capture.release()
        if total_pixels:
            rows.append(
                {
                    "episode_index": episode,
                    "base_output": str(base_output.resolve()),
                    "correction_output": str(correction_output.resolve()),
                    "streams": stream_rows,
                    "changed_frames_all_streams": total_frames,
                    "changed_pixels_all_streams": total_pixels,
                }
            )

    report = {
        "dry_run": bool(args.dry_run),
        "source_id": args.source_id,
        "target_id": args.target_id,
        "appearance_filter": bool(args.appearance_filter),
        "correction_episodes": len(correction),
        "changed_episodes": len(rows),
        "changed_frames_all_streams": sum(row["changed_frames_all_streams"] for row in rows),
        "changed_pixels_all_streams": sum(row["changed_pixels_all_streams"] for row in rows),
        "items": rows,
    }
    args.backup_root.mkdir(parents=True, exist_ok=True)
    report_path = args.backup_root / (
        "dry_run_report.json" if args.dry_run else "merge_report.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "items"}))
    print(report_path.resolve())


if __name__ == "__main__":
    main()
