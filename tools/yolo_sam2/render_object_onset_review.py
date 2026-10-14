#!/usr/bin/env python3
"""Render RGB/mask contact sheets around the first non-empty object mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


OBJECT_ID = 2
OFFSETS = (-60, -45, -30, -15, 0, 15, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--seg-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def first_positive(mask_dir: Path) -> tuple[int | None, int]:
    for path in sorted(mask_dir.glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        area = int(np.count_nonzero(mask == OBJECT_ID))
        if area:
            return int(path.stem), area
    return None, 0


def read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Failed to decode frame {frame_index}")
    return frame


def annotate_pair(
    frame: np.ndarray,
    mask: np.ndarray,
    frame_index: int,
    tile_width: int,
) -> np.ndarray:
    scale = tile_width / frame.shape[1]
    tile_height = round(frame.shape[0] * scale)
    frame = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (tile_width, tile_height), interpolation=cv2.INTER_NEAREST)
    object_mask = mask == OBJECT_ID
    overlay = frame.copy()
    overlay[object_mask] = (
        0.35 * overlay[object_mask] + 0.65 * np.array([180, 50, 255])
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        object_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
    area = int(np.count_nonzero(object_mask))
    label = f"f{frame_index:04d} object_px={area}"
    for image in (frame, overlay):
        cv2.rectangle(image, (0, 0), (tile_width, 24), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([frame, overlay])


def main() -> None:
    args = parse_args()
    prompt_root = args.prompt_root.resolve()
    seg_root = args.seg_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    prompt_paths = sorted(prompt_root.rglob("sam2_prompts.json"))
    for index, prompt_path in enumerate(prompt_paths, start=1):
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        file_index = int(prompt["file_index"])
        frame_count = int(prompt["frame_count"])
        mask_root = (
            seg_root
            / prompt["video_key"]
            / f"chunk-{int(prompt['chunk_index']):03d}"
            / f"file-{file_index:03d}"
            / f"file-{file_index:03d}_sam2_masks"
        )
        complete = mask_root / "complete.json"
        if not complete.is_file():
            continue
        mask_paths = sorted((mask_root / "mask_frames").glob("*.png"))
        directional_paths = sorted((mask_root / "directional_mask_frames").rglob("*.png"))
        if len(mask_paths) != frame_count or len(directional_paths) != 2 * frame_count:
            # A propagation worker may be between writing complete.json and the
            # final atomic directory update. Skip it until the next review run.
            continue
        mask_by_frame = {int(path.stem): path for path in mask_paths}
        object_prompt = next(obj for obj in prompt["objects"] if obj["name"] == "object")
        visible_start = int(object_prompt["visible_ranges"][0][0])
        first_box = min(int(box["frame"]) for box in object_prompt["boxes"])
        onset, onset_area = first_positive(mask_root / "mask_frames")
        anchor = visible_start if onset is None else onset
        frames = sorted(
            {
                max(0, min(frame_count - 1, anchor + offset))
                for offset in OFFSETS
            }
            | {visible_start, first_box}
        )

        capture = cv2.VideoCapture(prompt["video_path"])
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {prompt['video_path']}")
        tiles = []
        for frame_index in frames:
            frame = read_frame(capture, frame_index)
            mask_path = mask_by_frame[frame_index]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {mask_path}")
            tiles.append(annotate_pair(frame, mask, frame_index, args.tile_width))
        capture.release()
        sheet = np.hstack(tiles)
        header = np.zeros((44, sheet.shape[1], 3), dtype=np.uint8)
        task = ",".join(prompt.get("tasks", []))
        title = (
            f"episode {file_index:03d} task={task} visible_start={visible_start} "
            f"first_mask={onset} first_box={first_box}"
        )
        cv2.putText(
            header,
            title,
            (8, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        output = output_root / f"episode-{file_index:03d}_object_onset.jpg"
        cv2.imwrite(str(output), np.vstack([header, sheet]))
        records.append(
            {
                "episode": file_index,
                "task": task,
                "visible_start": visible_start,
                "first_object_mask": onset,
                "first_object_area": onset_area,
                "first_object_box": first_box,
                "frames": frames,
                "review": str(output),
            }
        )
        print(f"[{index}/{len(prompt_paths)}] {output.name}", flush=True)

    (output_root / "review_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done: {len(records)} completed episodes -> {output_root}")


if __name__ == "__main__":
    main()
