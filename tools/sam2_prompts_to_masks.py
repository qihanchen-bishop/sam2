#!/usr/bin/env python3
"""Propagate reviewed SAM2 prompt JSON files through their source videos."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_COLORS = [
    (64, 160, 255),
    (255, 105, 97),
    (119, 221, 119),
    (255, 209, 102),
    (177, 156, 217),
    (64, 210, 210),
]
CLASS_PRIORITY = ["occluder", "region", "leftarm", "rightarm", "object", "tool"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM2 video segmentation from reviewed sam2_prompts.json files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-json", type=Path, help="One sam2_prompts.json file.")
    source.add_argument(
        "--prompt-root",
        type=Path,
        help="Root recursively containing sam2_prompts.json files.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path("checkpoints/sam2.1_hiera_base_plus.pt"),
    )
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="Hydra config name relative to sam2/configs.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files; 0 means all.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate prompt JSON and source paths without loading SAM2.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def find_executable(name: str, required: bool = True) -> str | None:
    executable = shutil.which(name)
    if executable:
        return executable
    environment_executable = Path(sys.executable).resolve().parent / name
    if environment_executable.is_file():
        return str(environment_executable)
    if required:
        raise RuntimeError(
            f"{name} was not found; activate the configured conda environment"
        )
    return None


def add_clear_non_cond_memory_compat(predictor: Any) -> None:
    if hasattr(predictor, "_clear_obj_non_cond_mem_around_input"):
        return

    def clear_obj_non_cond_memory(
        self: Any, inference_state: dict, frame_idx: int, obj_idx: int
    ) -> None:
        stride = self.memory_temporal_stride_for_eval
        frame_begin = frame_idx - stride * self.num_maskmem
        frame_end = frame_idx + stride * self.num_maskmem
        outputs = inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
        for index in range(frame_begin, frame_end + 1):
            outputs.pop(index, None)

    predictor._clear_obj_non_cond_mem_around_input = types.MethodType(
        clear_obj_non_cond_memory, predictor
    )


def disable_unavailable_hole_filling(predictor: Any) -> None:
    try:
        import sam2._C  # noqa: F401
    except (ImportError, OSError) as exc:
        if getattr(predictor, "fill_hole_area", 0) > 0:
            predictor.fill_hole_area = 0
            print(
                "SAM2 CUDA post-processing extension is unavailable; "
                f"small-hole filling is disabled ({exc})."
            )


def discover_prompt_files(args: argparse.Namespace) -> tuple[list[Path], Path | None]:
    if args.prompt_json:
        path = args.prompt_json.resolve()
        if not path.is_file():
            raise RuntimeError(f"Prompt JSON not found: {path}")
        return [path], None

    root = args.prompt_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Prompt root not found: {root}")
    paths = sorted(root.rglob("sam2_prompts.json"))
    if not paths:
        raise RuntimeError(f"No sam2_prompts.json files found under {root}")
    if args.limit > 0:
        paths = paths[: args.limit]
    return paths, root


def load_and_validate_prompt(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    video_value = data.get("video_path")
    if not video_value:
        raise RuntimeError(f"{path}: missing video_path")
    video_path = Path(str(video_value)).expanduser().resolve()
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise RuntimeError(f"{path}: video not found or unsupported: {video_path}")

    frame_count = int(data.get("frame_count", 0))
    if frame_count <= 0:
        raise RuntimeError(f"{path}: frame_count must be positive")
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"{path}: objects must be a non-empty list")

    object_ids: set[int] = set()
    prompt_count = 0
    for raw_obj in objects:
        obj_id = int(raw_obj["id"])
        if not 1 <= obj_id <= 255:
            raise RuntimeError(f"{path}: object id must be in [1, 255], got {obj_id}")
        if obj_id in object_ids:
            raise RuntimeError(f"{path}: duplicate object id {obj_id}")
        object_ids.add(obj_id)
        if not str(raw_obj.get("name", "")).strip():
            raise RuntimeError(f"{path}: object {obj_id} has no name")

        seen_box_frames: set[int] = set()
        for item in raw_obj.get("boxes", []):
            frame_idx = int(item["frame"])
            if not 0 <= frame_idx < frame_count:
                raise RuntimeError(f"{path}: box frame {frame_idx} is out of range")
            if frame_idx in seen_box_frames:
                raise RuntimeError(f"{path}: object {obj_id} has two boxes on frame {frame_idx}")
            seen_box_frames.add(frame_idx)
            box = [float(value) for value in item["box"]]
            if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
                raise RuntimeError(f"{path}: invalid box for object {obj_id} on frame {frame_idx}")
            prompt_count += 1

        for item in raw_obj.get("points", []):
            frame_idx = int(item["frame"])
            if not 0 <= frame_idx < frame_count:
                raise RuntimeError(f"{path}: point frame {frame_idx} is out of range")
            if int(item["label"]) not in (0, 1):
                raise RuntimeError(f"{path}: point label must be 0 or 1")
            float(item["x"])
            float(item["y"])
            prompt_count += 1

        previous_end = -1
        for item in raw_obj.get("visible_ranges", []):
            if not isinstance(item, list) or len(item) != 2:
                raise RuntimeError(
                    f"{path}: visible range for object {obj_id} must be [start, end]"
                )
            start, end = (int(value) for value in item)
            if not 0 <= start <= end < frame_count:
                raise RuntimeError(
                    f"{path}: invalid visible range [{start}, {end}] for object {obj_id}"
                )
            if start <= previous_end:
                raise RuntimeError(
                    f"{path}: visible ranges for object {obj_id} must be sorted and disjoint"
                )
            previous_end = end

        if not raw_obj.get("boxes") and not raw_obj.get("points"):
            raise RuntimeError(f"{path}: object {obj_id} has no box or point prompt")

    if prompt_count == 0:
        raise RuntimeError(f"{path}: no prompts found")
    data["_prompt_path"] = path
    data["_video_path"] = video_path
    return data


def output_dir_for_prompt(
    prompt_path: Path, data: dict[str, Any], prompt_root: Path | None, output_root: Path
) -> Path:
    if prompt_root is None:
        relative_parent = Path()
    else:
        relative_parent = prompt_path.parent.relative_to(prompt_root)
    return output_root / relative_parent / f"{data['_video_path'].stem}_sam2_masks"


def extract_frames(video_path: Path, frame_dir: Path) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    command = [
        find_executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(frame_dir / "%05d.jpg"),
    ]
    subprocess.run(command, check=True)
    frames = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames


def prompt_inputs_for_object(raw_obj: dict[str, Any]) -> dict[int, dict[str, Any]]:
    import numpy as np

    inputs: dict[int, dict[str, Any]] = {}
    for item in raw_obj.get("boxes", []):
        frame_idx = int(item["frame"])
        inputs.setdefault(frame_idx, {})["box"] = np.asarray(item["box"], dtype=np.float32)
    for item in raw_obj.get("points", []):
        frame_idx = int(item["frame"])
        frame_input = inputs.setdefault(frame_idx, {})
        frame_input.setdefault("points", []).append((float(item["x"]), float(item["y"])))
        frame_input.setdefault("labels", []).append(int(item["label"]))
    for frame_input in inputs.values():
        if "points" in frame_input:
            frame_input["points"] = np.asarray(frame_input["points"], dtype=np.float32)
            frame_input["labels"] = np.asarray(frame_input["labels"], dtype=np.int32)
    return inputs


def save_propagated_masks(
    predictor: Any,
    inference_state: dict[str, Any],
    object_mask_dir: Path,
    start_frame_idx: int,
    reverse: bool,
    threshold: float,
    total_frames: int,
) -> None:
    import numpy as np
    from PIL import Image

    direction = "reverse" if reverse else "forward"
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame_idx,
        reverse=reverse,
    ):
        for index, obj_id in enumerate(out_obj_ids):
            mask = np.squeeze((out_mask_logits[index] > threshold).cpu().numpy())
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                object_mask_dir / f"{out_frame_idx:05d}_obj{int(obj_id):03d}.png"
            )
        if out_frame_idx % 50 == 0 or out_frame_idx in (0, total_frames - 1):
            print(f"  {direction}: frame {out_frame_idx + 1}/{total_frames}", flush=True)


def apply_visibility_ranges(
    object_mask_dir: Path,
    objects: list[dict[str, Any]],
    total_frames: int,
    frame_size: tuple[int, int],
) -> None:
    from PIL import Image

    empty_mask = Image.new("L", frame_size, 0)
    for raw_obj in objects:
        ranges = raw_obj.get("visible_ranges")
        if not ranges:
            continue
        obj_id = int(raw_obj["id"])
        visible = {
            frame_idx
            for start, end in ranges
            for frame_idx in range(int(start), int(end) + 1)
        }
        for frame_idx in range(total_frames):
            if frame_idx not in visible:
                empty_mask.save(
                    object_mask_dir / f"{frame_idx:05d}_obj{obj_id:03d}.png"
                )


def parse_color(value: Any, obj_id: int) -> tuple[int, int, int]:
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 6:
            try:
                return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))
            except ValueError:
                pass
    return DEFAULT_COLORS[(obj_id - 1) % len(DEFAULT_COLORS)]


def export_visual_outputs(
    output_dir: Path,
    frames: list[Path],
    objects: list[dict[str, Any]],
) -> None:
    import numpy as np
    from PIL import Image

    object_mask_dir = output_dir / "object_mask_frames"
    mask_dir = output_dir / "mask_frames"
    visual_dir = output_dir / "mask_visual_frames"
    for directory in (mask_dir, visual_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def priority(raw_obj: dict[str, Any]) -> tuple[int, int]:
        name = str(raw_obj["name"])
        rank = CLASS_PRIORITY.index(name) if name in CLASS_PRIORITY else len(CLASS_PRIORITY)
        return rank, int(raw_obj["id"])

    ordered_objects = sorted(objects, key=priority)
    for frame_idx, frame_path in enumerate(frames):
        frame = Image.open(frame_path).convert("RGB")
        width, height = frame.size
        class_mask = np.zeros((height, width), dtype=np.uint8)
        visual_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for raw_obj in ordered_objects:
            obj_id = int(raw_obj["id"])
            mask_path = object_mask_dir / f"{frame_idx:05d}_obj{obj_id:03d}.png"
            if not mask_path.exists():
                continue
            binary = np.asarray(Image.open(mask_path).convert("L")) > 0
            color = np.asarray(parse_color(raw_obj.get("color"), obj_id), dtype=np.uint8)
            class_mask[binary] = obj_id
            visual_mask[binary] = color
        Image.fromarray(class_mask, mode="L").save(mask_dir / f"{frame_idx:05d}.png")
        Image.fromarray(visual_mask, mode="RGB").save(visual_dir / f"{frame_idx:05d}.png")


def probe_fps(video_path: Path) -> float:
    ffprobe = find_executable("ffprobe", required=False)
    if ffprobe is None:
        return 24.0
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        value = subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
        numerator, denominator = value.split("/", maxsplit=1)
        fps = float(numerator) / float(denominator)
        return fps if fps > 0 else 24.0
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, ZeroDivisionError):
        return 24.0


def encode_mask_visual_video(output_dir: Path, video_path: Path) -> None:
    command = [
        find_executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{probe_fps(video_path):.6f}",
        "-i",
        str(output_dir / "mask_visual_frames" / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_dir / "mask_visual.mp4"),
    ]
    subprocess.run(command, check=True)


def run_one(
    predictor: Any,
    data: dict[str, Any],
    destination: Path,
    args: argparse.Namespace,
    device: str,
) -> None:
    import numpy as np
    import torch
    from PIL import Image

    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    if destination.exists():
        if (destination / "complete.json").is_file() and not args.overwrite:
            print(f"SKIP completed output: {destination}")
            return
        if not args.overwrite:
            raise RuntimeError(
                f"Output exists without complete.json: {destination}; "
                "inspect it or pass --overwrite"
            )
        shutil.rmtree(destination)
    partial.mkdir(parents=True)
    object_mask_dir = partial / "object_mask_frames"
    object_mask_dir.mkdir()

    temp_context = None
    if args.keep_frames:
        frame_dir = partial / "frames"
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="sam2_prompt_frames_")
        frame_dir = Path(temp_context.name)

    inference_state = None
    try:
        frames = extract_frames(data["_video_path"], frame_dir)
        if len(frames) != int(data["frame_count"]):
            raise RuntimeError(
                f"Frame count mismatch: JSON={data['frame_count']}, decoded={len(frames)}"
            )
        width, height = Image.open(frames[0]).size
        inference_state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
            async_loading_frames=True,
        )

        prompt_frames: list[int] = []
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            for raw_obj in data["objects"]:
                obj_id = int(raw_obj["id"])
                for frame_idx, frame_input in sorted(
                    prompt_inputs_for_object(raw_obj).items()
                ):
                    box = frame_input.get("box")
                    if box is not None:
                        box[0::2] = np.clip(box[0::2], 0, width - 1)
                        box[1::2] = np.clip(box[1::2], 0, height - 1)
                    points = frame_input.get("points")
                    if points is not None:
                        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=frame_input.get("labels"),
                        box=box,
                    )
                    prompt_frames.append(frame_idx)

            save_propagated_masks(
                predictor,
                inference_state,
                object_mask_dir,
                min(prompt_frames),
                False,
                args.mask_threshold,
                len(frames),
            )
            first_prompt_frame = min(prompt_frames)
            if first_prompt_frame > 0:
                save_propagated_masks(
                    predictor,
                    inference_state,
                    object_mask_dir,
                    first_prompt_frame,
                    True,
                    args.mask_threshold,
                    len(frames),
                )

        apply_visibility_ranges(
            object_mask_dir,
            data["objects"],
            len(frames),
            (width, height),
        )
        export_visual_outputs(partial, frames, data["objects"])
        encode_mask_visual_video(partial, data["_video_path"])
        clean_prompt = {key: value for key, value in data.items() if not key.startswith("_")}
        (partial / "sam2_prompts.json").write_text(
            json.dumps(clean_prompt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        completion = {
            "video_path": str(data["_video_path"]),
            "frame_count": len(frames),
            "object_ids": [int(item["id"]) for item in data["objects"]],
            "mask_threshold": args.mask_threshold,
            "sam2_config": args.sam2_config,
            "sam2_checkpoint": str(args.sam2_checkpoint.resolve()),
            "outputs": [
                "mask_frames",
                "mask_visual_frames",
                "mask_visual.mp4",
                "object_mask_frames",
            ],
        }
        (partial / "complete.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        partial.rename(destination)
    finally:
        if inference_state is not None:
            predictor.reset_state(inference_state)
        if temp_context is not None:
            temp_context.cleanup()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    prompt_paths, prompt_root = discover_prompt_files(args)
    records: list[tuple[Path, dict[str, Any], Path]] = []
    for prompt_path in prompt_paths:
        data = load_and_validate_prompt(prompt_path)
        destination = output_dir_for_prompt(
            prompt_path, data, prompt_root, args.output_root.resolve()
        )
        records.append((prompt_path, data, destination))
        print(
            f"OK {prompt_path}: {len(data['objects'])} objects, "
            f"{data['frame_count']} frames -> {destination}"
        )

    if args.validate_only:
        print(f"Validated {len(records)} prompt file(s).")
        return
    if not args.sam2_checkpoint.is_file():
        raise RuntimeError(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = resolve_device(args.device)
    print(f"Loading SAM2 on {device}: {args.sam2_checkpoint}")
    predictor = build_sam2_video_predictor(
        args.sam2_config,
        str(args.sam2_checkpoint.resolve()),
        device=device,
        hydra_overrides_extra=["++model.clear_non_cond_mem_around_input=true"],
    )
    add_clear_non_cond_memory_compat(predictor)
    disable_unavailable_hole_filling(predictor)

    failures: list[str] = []
    for index, (prompt_path, data, destination) in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {data['_video_path'].name}", flush=True)
        try:
            run_one(predictor, data, destination, args, device)
            print(f"DONE {destination}", flush=True)
        except Exception as exc:
            failures.append(f"{prompt_path}: {exc}")
            print(f"FAILED {prompt_path}: {exc}", flush=True)

    del predictor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(records)} videos failed:\n" + "\n".join(failures)
        )
    print(f"Completed {len(records)} video(s).")


if __name__ == "__main__":
    main()
