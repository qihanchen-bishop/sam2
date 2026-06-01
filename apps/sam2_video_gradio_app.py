#!/usr/bin/env python3
"""A small local web UI for interactive SAM 2 video segmentation.

Run from the repository root after installing SAM 2 and the UI dependencies:

    pip install -e ".[notebooks]" gradio decord
    python apps/sam2_video_gradio_app.py

The UI follows the workflow in notebooks/video_predictor_example.ipynb:
pick a video, prompt objects on one or more frames with points and/or boxes,
then propagate the masks through the whole video.
"""

from __future__ import annotations

import hashlib
import gc
import colorsys
import json
import os
import shutil
import subprocess
import tempfile
import types
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def normalize_proxy_env() -> None:
    """Keep Gradio/httpx local health checks away from system proxies."""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    local_hosts = "127.0.0.1,localhost,0.0.0.0"
    os.environ["NO_PROXY"] = local_hosts
    os.environ["no_proxy"] = local_hosts
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


normalize_proxy_env()

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2_video_predictor


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(tempfile.gettempdir()) / "sam2_video_gradio"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FRAME_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}
PREVIEW_MAX_WIDTH = 640
PREVIEW_MAX_HEIGHT = 480
ANNOTATION_MARGIN = 80
PROMPT_FILE_NAME = "sam2_prompts.json"
TASK_LABELS_FILE_NAME = "labels.txt"
DEFAULT_CLASS_PRIORITY = ["occluder", "region", "leftarm", "rightarm", "object"]

MODEL_OPTIONS = {
    "SAM 2.1 Tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoints/sam2.1_hiera_tiny.pt",
    ),
    "SAM 2.1 Small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoints/sam2.1_hiera_small.pt",
    ),
    "SAM 2.1 Base+": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoints/sam2.1_hiera_base_plus.pt",
    ),
    "SAM 2.1 Large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoints/sam2.1_hiera_large.pt",
    ),
}

PALETTE = [
    (64, 160, 255),
    (255, 105, 97),
    (77, 190, 123),
    (255, 183, 77),
    (171, 121, 255),
    (64, 210, 210),
    (240, 98, 146),
    (174, 213, 129),
]


def rgb_to_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def hex_to_rgb(color: str | None) -> tuple[int, int, int]:
    if not color:
        return PALETTE[0]
    value = color.strip().lstrip("#")
    if len(value) != 6:
        return PALETTE[0]
    try:
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except ValueError:
        return PALETTE[0]


@dataclass
class PromptObject:
    name: str
    color: str | None = None
    points: list[tuple[int, float, float, int]] = field(default_factory=list)
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    completed: bool = False


@dataclass
class AppState:
    video_path: str | None = None
    frame_dir: str | None = None
    frame_names: list[str] = field(default_factory=list)
    first_frame: Image.Image | None = None
    current_frame_idx: int = 0
    annotation_zoom: float = 1.0
    preview_size: tuple[int, int] | None = None
    output_root: str | None = None
    task_name: str | None = None
    task_dir: str | None = None
    task_labels: list[str] = field(default_factory=list)
    label_dir: str | None = None
    saved_mask_dir: str | None = None
    show_saved_masks: bool = True
    mask_alpha: int = 80
    display_mode: str = "当前对象"
    active_obj_id: int | None = None
    objects: dict[int, PromptObject] = field(default_factory=dict)
    next_obj_id: int = 1
    pending_box_start: tuple[int, float, float] | None = None


PREDICTOR_CACHE: dict[str, Any] = {}


def select_device() -> torch.device:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = select_device()


def abs_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def list_videos(root: str) -> tuple[gr.Dropdown, str]:
    root_path = abs_path(root)
    if not root_path.exists() or not root_path.is_dir():
        return gr.Dropdown(choices=[], value=None), f"目录不存在: {root_path}"

    choices: list[str] = []
    for item in sorted(root_path.iterdir()):
        if item.is_file() and item.suffix.lower() in VIDEO_EXTS:
            choices.append(str(item))
        elif item.is_dir() and contains_jpeg_frames(item):
            choices.append(str(item))

    msg = f"找到 {len(choices)} 个视频/帧目录。"
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None), msg


def list_subdirectories(root: str | Path) -> list[tuple[str, str]]:
    root_path = abs_path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []
    dirs = [p for p in root_path.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name.lower())
    return [(p.name, str(p)) for p in dirs]


def directory_dropdown(root: str | Path) -> gr.Dropdown:
    return gr.Dropdown(choices=list_subdirectories(root), value=None)


def go_parent_dir(current_dir: str) -> tuple[str, gr.Dropdown, str]:
    parent = abs_path(current_dir).parent
    return str(parent), directory_dropdown(parent), f"当前目录: {parent}"


def enter_selected_dir(current_dir: str, selected_dir: str | None) -> tuple[str, gr.Dropdown, str]:
    target = abs_path(selected_dir) if selected_dir else abs_path(current_dir)
    if not target.exists() or not target.is_dir():
        raise gr.Error(f"目录不存在: {target}")
    return str(target), directory_dropdown(target), f"当前目录: {target}"


def use_current_dir(current_dir: str) -> tuple[str, gr.Dropdown, str]:
    target = str(abs_path(current_dir))
    video_dropdown, msg = list_videos(target)
    return target, video_dropdown, msg


def task_dropdown(output_root: str, value: str | None = None) -> gr.Dropdown:
    tasks = list_tasks(output_root)
    return gr.Dropdown(choices=tasks, value=value if value in tasks else (tasks[0] if tasks else None))


def label_dropdown(labels: list[str], value: str | None = None) -> gr.Dropdown:
    return gr.Dropdown(choices=labels, value=value if value in labels else (labels[0] if labels else None))


def scan_tasks(output_root: str) -> tuple[gr.Dropdown, str]:
    root = abs_path(output_root or (REPO_ROOT / "outputs"))
    root.mkdir(parents=True, exist_ok=True)
    tasks = list_tasks(root)
    msg = f"输出根目录: {root}\n找到 {len(tasks)} 个 task。"
    if not tasks:
        msg += "\n请在“新建/选择 task”中输入 task 名称并点击“使用 task”。"
    return gr.Dropdown(choices=tasks, value=tasks[0] if tasks else None), msg


def use_task(output_root: str, task_name: str, state: AppState | None) -> tuple[gr.Dropdown, gr.Dropdown, str, AppState]:
    state = state or AppState()
    if not task_name:
        raise gr.Error("请输入或选择 task。")
    output_root_path = abs_path(output_root or (REPO_ROOT / "outputs"))
    task_path = task_dir_from_root(output_root_path, task_name)
    task_path.mkdir(parents=True, exist_ok=True)
    labels = load_task_labels(task_path)
    if not task_labels_file(task_path).exists():
        save_task_labels(task_path, labels)

    state.output_root = str(output_root_path)
    state.task_name = task_name.strip()
    state.task_dir = str(task_path)
    state.task_labels = labels
    if state.video_path:
        state.label_dir = label_dir_from_root(task_path, state.video_path)
        state.saved_mask_dir = saved_mask_dir_from_root(task_path, state.video_path)

    msg = f"当前 task: {state.task_name}\nTask 目录: {task_path}\n标签文件: {task_labels_file(task_path)}"
    if labels:
        msg += "\n可用标签: " + ", ".join(labels)
    else:
        msg += "\n当前 task 还没有标签，请先在“新增标签”里维护 labels.txt。"
    tasks = list_tasks(output_root_path)
    return gr.Dropdown(choices=tasks, value=state.task_name), label_dropdown(labels), msg, state


def add_task_label(
    output_root: str, task_name: str, new_label: str, state: AppState | None
) -> tuple[gr.Dropdown, str, AppState]:
    state = state or AppState()
    if not task_name:
        raise gr.Error("请先选择 task。")
    label = (new_label or "").strip()
    if not label:
        raise gr.Error("请输入新增标签名称。")
    task_path = task_dir_from_root(output_root or state.output_root or (REPO_ROOT / "outputs"), task_name)
    task_path.mkdir(parents=True, exist_ok=True)
    labels = load_task_labels(task_path)
    if label not in labels:
        labels.append(label)
        save_task_labels(task_path, labels)
    state.output_root = str(abs_path(output_root or (REPO_ROOT / "outputs")))
    state.task_name = task_name.strip()
    state.task_dir = str(task_path)
    state.task_labels = labels
    return label_dropdown(labels, label), f"已更新 task 标签文件: {task_labels_file(task_path)}", state


def task_dir_from_root(output_root: str | Path | None, task_name: str | None) -> Path:
    root = abs_path(output_root or (REPO_ROOT / "outputs"))
    task = (task_name or "default").strip()
    if not task:
        task = "default"
    return root / task


def task_labels_file(task_dir: str | Path) -> Path:
    return abs_path(task_dir) / TASK_LABELS_FILE_NAME


def load_task_labels(task_dir: str | Path) -> list[str]:
    path = task_labels_file(task_dir)
    if not path.exists():
        return []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label = line.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def save_task_labels(task_dir: str | Path, labels: list[str]) -> Path:
    path = task_labels_file(task_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_labels: list[str] = []
    for label in labels:
        label = label.strip()
        if label and label not in clean_labels:
            clean_labels.append(label)
    path.write_text("\n".join(clean_labels) + ("\n" if clean_labels else ""), encoding="utf-8")
    return path


def list_tasks(output_root: str | Path) -> list[str]:
    root = abs_path(output_root)
    if not root.exists() or not root.is_dir():
        return []
    tasks: list[str] = []
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue
        if path.name.endswith("_sam2_labels") or path.name.endswith("_sam2_masks"):
            continue
        tasks.append(path.name)
    return tasks


def label_dir_from_root(task_root: str | Path | None, video_path: str | None) -> str:
    root = abs_path(task_root or (REPO_ROOT / "outputs" / "default"))
    stem = Path(video_path or "video").stem
    return str(root / f"{stem}_sam2_masks")


def legacy_label_dir_from_root(task_root: str | Path | None, video_path: str | None) -> str:
    root = abs_path(task_root or (REPO_ROOT / "outputs" / "default"))
    stem = Path(video_path or "video").stem
    return str(root / f"{stem}_sam2_labels")


def saved_mask_dir_from_root(task_root: str | Path | None, video_path: str | None) -> str:
    root = abs_path(task_root or (REPO_ROOT / "outputs" / "default"))
    stem = Path(video_path or "video").stem
    return str(root / f"{stem}_sam2_masks" / "mask_frames")


def prompt_file(label_dir: str | Path) -> Path:
    return abs_path(label_dir) / PROMPT_FILE_NAME


def contains_jpeg_frames(path: Path) -> bool:
    return any(p.suffix in FRAME_EXTS for p in path.iterdir() if p.is_file())


def frame_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def get_frame_names(frame_dir: Path) -> list[str]:
    frames = [p.name for p in frame_dir.iterdir() if p.suffix in FRAME_EXTS]
    frames.sort(key=lambda name: frame_sort_key(frame_dir / name))
    if not frames:
        raise RuntimeError(f"没有在 {frame_dir} 里找到 JPEG 帧。")
    return frames


def cache_dir_for_video(video_path: Path) -> Path:
    key = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return CACHE_ROOT / key


def prepare_frame_dir(video_path: str) -> tuple[Path, list[str]]:
    src = abs_path(video_path)
    if src.is_dir():
        return src, get_frame_names(src)

    if src.suffix.lower() not in VIDEO_EXTS:
        raise RuntimeError(f"不支持的视频格式: {src.suffix}")

    out_dir = cache_dir_for_video(src)
    marker = out_dir / ".complete"
    if marker.exists():
        return out_dir, get_frame_names(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in out_dir.glob("*.jpg"):
        old_frame.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(out_dir / "%05d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        try:
            extract_frames_with_decord(src, out_dir)
        except Exception as decord_exc:
            raise RuntimeError(
                "找不到 ffmpeg，并且 decord 抽帧也失败了。请安装 ffmpeg，或安装 decord，"
                "或直接选择已经抽帧好的 JPEG 目录。"
            ) from decord_exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg 抽帧失败:\n{err}") from exc

    marker.write_text("ok\n", encoding="utf-8")
    return out_dir, get_frame_names(out_dir)


def extract_frames_with_decord(video_path: Path, out_dir: Path) -> None:
    import decord

    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(str(video_path))
    if len(reader) == 0:
        raise RuntimeError(f"视频没有可读取帧: {video_path}")

    for idx, frame in enumerate(reader):
        frame_np = frame.asnumpy()
        Image.fromarray(frame_np).save(out_dir / f"{idx:05d}.jpg", quality=95)


def load_video(
    video_path: str, output_root: str, task_name: str, state: AppState | None
) -> tuple[Image.Image, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Slider, gr.Slider, str, gr.Dropdown, gr.ColorPicker, gr.Slider, str | None]:
    if not video_path:
        raise gr.Error("请先选择一个视频或 JPEG 帧目录。")
    if not task_name:
        raise gr.Error("请先选择或创建 task。")

    state = state or AppState()
    task_path = task_dir_from_root(output_root or (REPO_ROOT / "outputs"), task_name)
    task_path.mkdir(parents=True, exist_ok=True)
    task_labels = load_task_labels(task_path)
    frame_dir, frame_names = prepare_frame_dir(video_path)
    first = Image.open(frame_dir / frame_names[0]).convert("RGB")

    state.output_root = str(abs_path(output_root or (REPO_ROOT / "outputs")))
    state.task_name = task_name.strip()
    state.task_dir = str(task_path)
    state.task_labels = task_labels
    state.video_path = str(abs_path(video_path))
    state.frame_dir = str(frame_dir)
    state.frame_names = frame_names
    state.first_frame = first
    state.current_frame_idx = 0
    state.annotation_zoom = 1.0
    state.preview_size = get_preview_size(first, state.annotation_zoom)
    state.objects = {}
    state.next_obj_id = 1
    state.active_obj_id = None
    state.pending_box_start = None
    state.label_dir = label_dir_from_root(task_path, video_path)
    state.saved_mask_dir = saved_mask_dir_from_root(task_path, video_path)

    msg = f"已载入 {Path(video_path).name}: {len(frame_names)} 帧，尺寸 {first.width}x{first.height}。"
    msg += f"\nTask: {state.task_name}"
    label_path = prompt_file(state.label_dir)
    if label_path.exists():
        load_prompts_from_disk(state.label_dir, state)
        sync_active_object_for_current_frame(state)
        msg += f"\n已从标签路径加载已有标注: {label_path}"
    elif prompt_file(legacy_label_dir_from_root(task_path, video_path)).exists():
        legacy_path = prompt_file(legacy_label_dir_from_root(task_path, video_path))
        load_prompts_from_disk(str(legacy_path.parent), state)
        sync_active_object_for_current_frame(state)
        msg += f"\n已从旧标签路径加载已有标注: {legacy_path}"
        msg += f"\n后续会保存到新路径: {label_path}"
    else:
        msg += f"\n标签路径: {state.label_dir}"
    saved_mask_dir = Path(state.saved_mask_dir)
    if saved_mask_dir.exists():
        msg += f"\n已找到已有传播 mask: {saved_mask_dir}"
    overlay_video = saved_mask_dir.parent / "overlay.mp4"
    overlay_video_path = str(overlay_video) if overlay_video.exists() else None
    if overlay_video_path:
        msg += f"\n已找到已有 Overlay 视频: {overlay_video_path}"
    if not task_labels:
        msg += f"\n警告: 当前 task 没有标签，请先维护 {task_labels_file(task_path)}。"
    frame_slider = gr.Slider(minimum=0, maximum=max(len(frame_names) - 1, 1), step=1, value=0)
    zoom_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0)
    return (
        draw_prompts(state),
        msg,
        state,
        object_dropdown(state, state.active_obj_id),
        all_object_dropdown(state, state.active_obj_id),
        annotation_dropdown(state),
        annotated_frame_dropdown(state),
        frame_slider,
        zoom_slider,
        state.output_root,
        label_dropdown(task_labels),
        current_object_color_picker(state),
        gr.Slider(value=state.mask_alpha),
        overlay_video_path,
    )


def load_frame_image(state: AppState, frame_idx: int) -> Image.Image:
    if state.frame_dir is None or not state.frame_names:
        raise gr.Error("请先加载视频。")
    frame_idx = max(0, min(int(frame_idx), len(state.frame_names) - 1))
    return Image.open(Path(state.frame_dir) / state.frame_names[frame_idx]).convert("RGB")


def change_frame(frame_idx: int, state: AppState | None) -> tuple[Image.Image, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.ColorPicker]:
    state = state or AppState()
    if frame_idx is None:
        return draw_prompts(state), objects_summary(state), state, object_dropdown(state, state.active_obj_id), annotation_dropdown(state), annotated_frame_dropdown(state), current_object_color_picker(state)
    state.first_frame = load_frame_image(state, int(frame_idx))
    state.current_frame_idx = int(frame_idx)
    state.preview_size = get_preview_size(state.first_frame, state.annotation_zoom)
    if state.pending_box_start and state.pending_box_start[0] != state.current_frame_idx:
        state.pending_box_start = None
    sync_active_object_for_current_frame(state)
    msg = f"当前标注帧: {state.current_frame_idx}/{len(state.frame_names) - 1}\n{objects_summary(state)}"
    return draw_prompts(state), msg, state, object_dropdown(state, state.active_obj_id), annotation_dropdown(state), annotated_frame_dropdown(state), current_object_color_picker(state)


def jump_to_annotated_frame(frame_idx: int, state: AppState | None):
    image, msg, state, obj_dd, anno_dd, frame_dd, color_picker = change_frame(frame_idx, state)
    value = int(frame_idx) if frame_idx is not None else (state.current_frame_idx if state else 0)
    return image, msg, state, obj_dd, anno_dd, frame_dd, color_picker, gr.Slider(value=value)


def change_zoom(zoom: float, state: AppState | None) -> tuple[Image.Image | None, str, AppState]:
    state = state or AppState()
    state.annotation_zoom = float(zoom)
    if state.first_frame is not None:
        state.preview_size = get_preview_size(state.first_frame, state.annotation_zoom)
    msg = f"标注放大倍数: {state.annotation_zoom:.1f}x\n{objects_summary(state)}"
    return draw_prompts(state), msg, state


def get_preview_size(image: Image.Image, zoom: float = 1.0) -> tuple[int, int]:
    scale = min(PREVIEW_MAX_WIDTH / image.width, PREVIEW_MAX_HEIGHT / image.height, 1.0)
    scale *= max(float(zoom), 0.25)
    return max(1, round(image.width * scale)), max(1, round(image.height * scale))


def display_to_original(state: AppState, x: float, y: float) -> tuple[float, float]:
    if state.first_frame is None or state.preview_size is None:
        return x, y
    x -= ANNOTATION_MARGIN
    y -= ANNOTATION_MARGIN
    preview_w, preview_h = state.preview_size
    return x * state.first_frame.width / preview_w, y * state.first_frame.height / preview_h


def original_to_display(state: AppState, x: float, y: float) -> tuple[float, float]:
    if state.first_frame is None or state.preview_size is None:
        return x, y
    preview_w, preview_h = state.preview_size
    return (
        x * preview_w / state.first_frame.width + ANNOTATION_MARGIN,
        y * preview_h / state.first_frame.height + ANNOTATION_MARGIN,
    )


def clamp_point(state: AppState, x: float, y: float) -> tuple[float, float]:
    if state.first_frame is None:
        return x, y
    return (
        min(max(x, 0.0), float(state.first_frame.width - 1)),
        min(max(y, 0.0), float(state.first_frame.height - 1)),
    )


def clamp_box(state: AppState, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0 = clamp_point(state, box[0], box[1])
    x1, y1 = clamp_point(state, box[2], box[3])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def box_size(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return abs(box[2] - box[0]), abs(box[3] - box[1])


def next_available_obj_id(state: AppState) -> int:
    obj_id = 1
    while obj_id in state.objects:
        obj_id += 1
    return obj_id


def object_has_prompt_on_frame(obj: PromptObject, frame_idx: int) -> bool:
    return frame_idx in obj.boxes or any(point_frame == frame_idx for point_frame, *_ in obj.points)


def generated_color(index: int) -> str:
    if index <= len(PALETTE):
        return rgb_to_hex(PALETTE[index - 1])
    hue = ((index - 1) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.62, 0.95)
    return rgb_to_hex((round(r * 255), round(g * 255), round(b * 255)))


def next_unique_color(state: AppState, obj_id: int) -> str:
    used = {obj.color.lower() for oid, obj in state.objects.items() if oid != obj_id and obj.color}
    candidate_idx = obj_id
    while True:
        color = generated_color(candidate_idx)
        if color.lower() not in used:
            return color
        candidate_idx += 1


def ensure_object_colors(state: AppState) -> None:
    used: set[str] = set()
    for obj_id, obj in sorted(state.objects.items()):
        color = rgb_to_hex(hex_to_rgb(obj.color)) if obj.color else None
        if not color or color.lower() in used:
            obj.color = next_unique_color(state, obj_id)
        else:
            obj.color = color
        used.add(obj.color.lower())


def object_color(state: AppState, obj_id: int) -> tuple[int, int, int]:
    obj = state.objects.get(obj_id)
    if obj is None:
        return hex_to_rgb(generated_color(obj_id))
    if not obj.color:
        obj.color = next_unique_color(state, obj_id)
    return hex_to_rgb(obj.color)


def export_priority_rank(state: AppState, obj_id: int) -> tuple[int, int]:
    obj = state.objects.get(obj_id)
    name = obj.name if obj else ""
    if name in DEFAULT_CLASS_PRIORITY:
        return DEFAULT_CLASS_PRIORITY.index(name), obj_id
    if name in state.task_labels:
        return len(DEFAULT_CLASS_PRIORITY) + state.task_labels.index(name), obj_id
    return len(DEFAULT_CLASS_PRIORITY) + len(state.task_labels), obj_id


def masks_in_export_priority(state: AppState, masks: dict[int, np.ndarray]) -> list[tuple[int, np.ndarray]]:
    return sorted(masks.items(), key=lambda item: export_priority_rank(state, int(item[0])))


def current_object_color_picker(state: AppState) -> gr.ColorPicker:
    ensure_object_colors(state)
    if state.active_obj_id in state.objects:
        return gr.ColorPicker(value=state.objects[state.active_obj_id].color or generated_color(state.active_obj_id))
    return gr.ColorPicker(value=generated_color(1))


def sync_active_object_for_current_frame(state: AppState) -> None:
    if state.active_obj_id in state.objects:
        active_obj = state.objects[state.active_obj_id]
        if not active_obj.completed or object_has_prompt_on_frame(active_obj, state.current_frame_idx):
            return
    for obj_id, obj in sorted(state.objects.items()):
        if object_has_prompt_on_frame(obj, state.current_frame_idx):
            state.active_obj_id = obj_id
            return
    state.active_obj_id = None


def get_or_create_object(state: AppState, obj_id: int, obj_name: str) -> PromptObject:
    if obj_id not in state.objects:
        state.objects[obj_id] = PromptObject(name=obj_name or f"object-{obj_id}", color=next_unique_color(state, obj_id))
        state.next_obj_id = max(state.next_obj_id, obj_id + 1)
    elif obj_name:
        state.objects[obj_id].name = obj_name
    if not state.objects[obj_id].color:
        state.objects[obj_id].color = next_unique_color(state, obj_id)
    return state.objects[obj_id]


def create_new_object(state: AppState, obj_label: str) -> int:
    obj_id = next_available_obj_id(state)
    get_or_create_object(state, obj_id, obj_label)
    state.active_obj_id = obj_id
    state.next_obj_id = next_available_obj_id(state)
    state.pending_box_start = None
    return obj_id


def add_object(obj_label: str, state: AppState | None) -> tuple[gr.Dropdown, gr.Dropdown, gr.ColorPicker, gr.Radio, str, AppState]:
    state = state or AppState()
    if not state.task_labels:
        raise gr.Error("当前 task 没有标签，请先在 task 标签文件中新增标签。")
    if obj_label not in state.task_labels:
        raise gr.Error("对象类别必须从当前 task 的标签列表中选择。")
    unfinished = [
        obj_id
        for obj_id, obj in state.objects.items()
        if not obj.completed
    ]
    if unfinished:
        raise gr.Error(f"请先点击“完成当前对象”结束对象 {unfinished[0]} 的标注，再新增对象。")
    obj_id = create_new_object(state, obj_label)
    msg = objects_summary(state) + maybe_auto_save(state)
    return object_dropdown(state, obj_id), all_object_dropdown(state, obj_id), current_object_color_picker(state), gr.Radio(value="框"), msg, state


def object_dropdown(state: AppState, value: int | None = None) -> gr.Dropdown:
    choices = [
        (f"{obj_id}: {obj.name} {'✓' if obj.completed else '(编辑中)'}", obj_id)
        for obj_id, obj in state.objects.items()
        if object_has_prompt_on_frame(obj, state.current_frame_idx)
        or obj_id == state.active_obj_id
    ]
    valid_values = {choice_value for _, choice_value in choices}
    return gr.Dropdown(choices=choices, value=value if value in valid_values else None)


def all_object_dropdown(state: AppState, value: int | None = None) -> gr.Dropdown:
    choices = []
    for obj_id, obj in sorted(state.objects.items()):
        prompt_frames = sorted({frame_idx for frame_idx, *_ in obj.points} | set(obj.boxes))
        frames = ",".join(str(idx) for idx in prompt_frames) if prompt_frames else "-"
        status = "已完成" if obj.completed else "编辑中"
        choices.append((f"{obj_id}: {obj.name} [{status}] | 标注帧 {frames}", obj_id))
    valid_values = {choice_value for _, choice_value in choices}
    return gr.Dropdown(choices=choices, value=value if value in valid_values else None)


def annotation_dropdown(state: AppState) -> gr.Dropdown:
    choices: list[tuple[str, str]] = []
    for obj_id, obj in sorted(state.objects.items()):
        if state.current_frame_idx in obj.boxes:
            x0, y0, x1, y1 = obj.boxes[state.current_frame_idx]
            choices.append(
                (
                    f"{obj_id}:{obj.name} | 框 ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})",
                    f"box:{obj_id}:{state.current_frame_idx}",
                )
            )
        point_idx = 0
        for frame_idx, x, y, label in obj.points:
            if frame_idx != state.current_frame_idx:
                continue
            point_name = "正点" if label == 1 else "负点"
            choices.append(
                (
                    f"{obj_id}:{obj.name} | {point_name} #{point_idx + 1} ({x:.0f},{y:.0f})",
                    f"point:{obj_id}:{state.current_frame_idx}:{point_idx}",
                )
            )
            point_idx += 1
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def annotated_frame_dropdown(state: AppState) -> gr.Dropdown:
    frames = sorted(
        {frame_idx for obj in state.objects.values() for frame_idx, *_ in obj.points}
        | {frame_idx for obj in state.objects.values() for frame_idx in obj.boxes}
    )
    choices = [(f"frame {idx}", idx) for idx in frames]
    value = state.current_frame_idx if state.current_frame_idx in frames else None
    return gr.Dropdown(choices=choices, value=value)


def serialize_prompts(state: AppState) -> dict[str, Any]:
    return {
        "version": 1,
        "video_path": state.video_path,
        "frame_count": len(state.frame_names),
        "objects": [
            {
                "id": obj_id,
                "name": obj.name,
                "color": obj.color,
                "completed": obj.completed,
                "points": [
                    {"frame": frame_idx, "x": x, "y": y, "label": label}
                    for frame_idx, x, y, label in obj.points
                ],
                "boxes": [
                    {"frame": frame_idx, "box": list(box)}
                    for frame_idx, box in sorted(obj.boxes.items())
                ],
            }
            for obj_id, obj in sorted(state.objects.items())
        ],
    }


def load_prompts_from_disk(label_dir: str, state: AppState) -> None:
    path = prompt_file(label_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    objects: dict[int, PromptObject] = {}
    for raw_obj in data.get("objects", []):
        obj_id = int(raw_obj["id"])
        obj = PromptObject(name=str(raw_obj.get("name") or f"object-{obj_id}"))
        color = raw_obj.get("color")
        obj.color = str(color) if color else None
        obj.completed = bool(raw_obj.get("completed", True))
        for point in raw_obj.get("points", []):
            frame_idx = int(point["frame"])
            if state.frame_names and not 0 <= frame_idx < len(state.frame_names):
                continue
            obj.points.append(
                (
                    frame_idx,
                    float(point["x"]),
                    float(point["y"]),
                    int(point["label"]),
                )
            )
        for box_item in raw_obj.get("boxes", []):
            frame_idx = int(box_item["frame"])
            if state.frame_names and not 0 <= frame_idx < len(state.frame_names):
                continue
            box = box_item["box"]
            obj.boxes[frame_idx] = tuple(float(v) for v in box)
        objects[obj_id] = obj

    state.objects = objects
    ensure_object_colors(state)
    state.next_obj_id = max(objects, default=0) + 1
    state.active_obj_id = min(objects) if objects else None
    state.pending_box_start = None


def save_prompts_to_disk(label_dir: str, state: AppState) -> Path:
    path = prompt_file(label_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_prompts(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def maybe_auto_save(state: AppState) -> str:
    if not state.label_dir:
        return ""
    path = save_prompts_to_disk(state.label_dir, state)
    return f"\n已自动保存标注: {path}"


def set_label_dir(label_dir: str, state: AppState | None) -> tuple[Image.Image | None, str, AppState, gr.Dropdown]:
    state = state or AppState()
    if not label_dir:
        raise gr.Error("请输入或选择标签路径。")
    state.label_dir = str(abs_path(label_dir))
    msg = f"标签路径: {state.label_dir}"
    path = prompt_file(state.label_dir)
    if path.exists():
        load_prompts_from_disk(state.label_dir, state)
        sync_active_object_for_current_frame(state)
        msg += f"\n已加载已有标注: {path}"
    else:
        msg += f"\n未找到已有标注文件，将在保存时创建: {path}"
    return draw_prompts(state), msg, state, object_dropdown(state, state.active_obj_id)


def set_label_root(label_root: str, state: AppState | None) -> tuple[Image.Image | None, str, AppState, gr.Dropdown]:
    state = state or AppState()
    if not label_root:
        raise gr.Error("请输入或选择输出根目录。")
    task_path = task_dir_from_root(label_root, state.task_name)
    state.output_root = str(abs_path(label_root))
    state.task_dir = str(task_path)
    state.task_labels = load_task_labels(task_path)
    state.saved_mask_dir = saved_mask_dir_from_root(task_path, state.video_path)
    label_dir = label_dir_from_root(task_path, state.video_path)
    image, msg, state, dropdown = set_label_dir(label_dir, state)
    if state.saved_mask_dir and Path(state.saved_mask_dir).exists():
        msg += f"\n已找到已有传播 mask: {state.saved_mask_dir}"
    return image, msg, state, dropdown


def save_label_dir(label_root: str, state: AppState | None) -> tuple[str, AppState]:
    state = state or AppState()
    task_path = task_dir_from_root(label_root, state.task_name)
    state.output_root = str(abs_path(label_root or (REPO_ROOT / "outputs")))
    state.task_dir = str(task_path)
    state.saved_mask_dir = saved_mask_dir_from_root(task_path, state.video_path)
    state.label_dir = label_dir_from_root(task_path, state.video_path)
    path = save_prompts_to_disk(state.label_dir, state)
    return f"已保存标注: {path}", state


def use_current_dir_as_label(current_dir: str, state: AppState | None) -> tuple[str, Image.Image | None, str, AppState, gr.Dropdown]:
    label_root = str(abs_path(current_dir))
    image, msg, state, dropdown = set_label_root(label_root, state)
    return label_root, image, msg, state, dropdown


def change_object(obj_id: int | None, state: AppState | None) -> tuple[Image.Image | None, str, AppState, gr.Dropdown, gr.ColorPicker]:
    state = state or AppState()
    if obj_id is None:
        if state.active_obj_id in state.objects and not state.objects[state.active_obj_id].completed:
            obj = state.objects[state.active_obj_id]
            return draw_prompts(state), objects_summary(state), state, label_dropdown(state.task_labels, obj.name), current_object_color_picker(state)
        state.active_obj_id = None
    else:
        state.active_obj_id = int(obj_id)
        state.pending_box_start = None
    obj_name = state.objects[state.active_obj_id].name if state.active_obj_id in state.objects else None
    return draw_prompts(state), objects_summary(state), state, label_dropdown(state.task_labels, obj_name), current_object_color_picker(state)


def select_existing_object(
    obj_id: int | None, state: AppState | None
) -> tuple[Image.Image | None, str, AppState, gr.Dropdown, gr.Dropdown, gr.ColorPicker, gr.Radio]:
    state = state or AppState()
    if obj_id is None or int(obj_id) not in state.objects:
        raise gr.Error("请先在“全部已有对象”中选择一个对象。")
    obj_id = int(obj_id)
    state.active_obj_id = obj_id
    state.pending_box_start = None
    obj = state.objects[obj_id]
    msg = (
        f"已选择对象 {obj_id}: {obj.name}。可以在当前帧继续新增点或框；"
        "首次新增标注后对象会变为编辑中，完成后再点击“完成当前对象”。\n"
        + objects_summary(state)
    )
    return (
        draw_prompts(state),
        msg,
        state,
        object_dropdown(state, obj_id),
        label_dropdown(state.task_labels, obj.name),
        current_object_color_picker(state),
        gr.Radio(value="框"),
    )


def rename_current_object(
    obj_id: int | None, obj_label: str, state: AppState | None
) -> tuple[Image.Image | None, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.ColorPicker]:
    state = state or AppState()
    if obj_id is None and state.active_obj_id is not None:
        obj_id = state.active_obj_id
    if obj_id is None or int(obj_id) not in state.objects:
        raise gr.Error("请先选择要修改类别的对象。")
    if obj_label not in state.task_labels:
        raise gr.Error("对象类别必须从当前 task 的标签列表中选择。")
    obj_id = int(obj_id)
    state.objects[obj_id].name = obj_label
    state.active_obj_id = obj_id
    msg = f"已将对象 {obj_id} 的类别修改为: {obj_label}\n" + objects_summary(state) + maybe_auto_save(state)
    return (
        draw_prompts(state),
        msg,
        state,
        object_dropdown(state, obj_id),
        all_object_dropdown(state, obj_id),
        annotation_dropdown(state),
        annotated_frame_dropdown(state),
        current_object_color_picker(state),
    )


def update_object_color(
    obj_id: int | None, color: str, state: AppState | None
) -> tuple[Image.Image | None, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    state = state or AppState()
    if obj_id is None and state.active_obj_id is not None:
        obj_id = state.active_obj_id
    if obj_id is None or int(obj_id) not in state.objects:
        raise gr.Error("请先选择要修改颜色的对象。")
    clean_color = rgb_to_hex(hex_to_rgb(color))
    obj_id = int(obj_id)
    used = {
        obj.color.lower()
        for other_id, obj in state.objects.items()
        if other_id != obj_id and obj.color
    }
    if clean_color.lower() in used:
        raise gr.Error("这个颜色已经被其他对象使用，请换一个颜色。")
    state.objects[obj_id].color = clean_color
    state.active_obj_id = obj_id
    msg = f"已更新对象 {obj_id}: {state.objects[obj_id].name} 的颜色为 {clean_color}\n" + objects_summary(state) + maybe_auto_save(state)
    return draw_prompts(state), msg, state, object_dropdown(state, obj_id), all_object_dropdown(state, obj_id), annotation_dropdown(state), annotated_frame_dropdown(state)


def change_mask_alpha(alpha: int | float, state: AppState | None) -> tuple[Image.Image | None, str, AppState]:
    state = state or AppState()
    state.mask_alpha = int(max(0, min(255, round(float(alpha)))))
    return draw_prompts(state), f"mask 透明度: {state.mask_alpha}/255\n{objects_summary(state)}", state


def change_display_mode(display_mode: str, state: AppState | None) -> tuple[Image.Image | None, str, AppState]:
    state = state or AppState()
    state.display_mode = display_mode
    return draw_prompts(state), objects_summary(state), state


def change_saved_mask_visibility(show_saved_masks: bool, state: AppState | None) -> tuple[Image.Image | None, str, AppState]:
    state = state or AppState()
    state.show_saved_masks = bool(show_saved_masks)
    msg = objects_summary(state)
    if state.saved_mask_dir:
        msg += f"\n已有 mask 目录: {state.saved_mask_dir}"
    return draw_prompts(state), msg, state


def finish_current_object(
    state: AppState | None, obj_id: int | None
) -> tuple[Image.Image | None, str, AppState, gr.Dropdown, gr.Dropdown]:
    state = state or AppState()
    if obj_id is None and state.active_obj_id is not None:
        obj_id = state.active_obj_id
    if obj_id is None or int(obj_id) not in state.objects:
        raise gr.Error("请先选择一个正在标注的对象。")
    obj_id = int(obj_id)
    obj = state.objects[obj_id]
    if not obj.points and not obj.boxes:
        raise gr.Error("当前对象还没有任何点或框，不能完成。")
    obj.completed = True
    state.pending_box_start = None
    state.active_obj_id = obj_id
    msg = f"对象 {obj_id}: {obj.name} 已完成。\n" + objects_summary(state) + maybe_auto_save(state)
    return draw_prompts(state), msg, state, object_dropdown(state, obj_id), all_object_dropdown(state, obj_id)


def image_click(
    state: AppState | None,
    obj_id: int | None,
    obj_label: str,
    mode: str,
    evt: gr.SelectData,
) -> tuple[Image.Image, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.ColorPicker]:
    state = state or AppState()
    if state.first_frame is None:
        raise gr.Error("请先加载视频。")
    if obj_id is None:
        obj_id = state.active_obj_id
    if obj_label not in state.task_labels:
        raise gr.Error("对象类别必须从当前 task 的标签列表中选择。")

    auto_created = False
    if obj_id is None:
        obj_id = create_new_object(state, obj_label)
        auto_created = True
    else:
        obj_id = int(obj_id)
    state.active_obj_id = obj_id

    if obj_id in state.objects:
        obj = state.objects[obj_id]
        if obj.completed and obj.name != obj_label:
            obj_id = create_new_object(state, obj_label)
            obj = state.objects[obj_id]
            auto_created = True
    else:
        obj = get_or_create_object(state, obj_id, obj_label or f"object-{obj_id}")
        auto_created = True
    if obj.completed:
        obj.completed = False
        state.pending_box_start = None
    click_x, click_y = evt.index
    x, y = display_to_original(state, float(click_x), float(click_y))
    frame_idx = state.current_frame_idx

    if mode == "正点":
        x, y = clamp_point(state, x, y)
        obj.points.append((frame_idx, x, y, 1))
        state.pending_box_start = None
        msg = objects_summary(state) + maybe_auto_save(state)
    elif mode == "负点":
        x, y = clamp_point(state, x, y)
        obj.points.append((frame_idx, x, y, 0))
        state.pending_box_start = None
        msg = objects_summary(state) + maybe_auto_save(state)
    else:
        if state.pending_box_start is None or state.pending_box_start[0] != frame_idx:
            state.pending_box_start = (frame_idx, x, y)
            msg = (
                f"已记录包围盒第一个角: ({x:.1f}, {y:.1f})。"
                "\n请在同一帧点击第二个角；可以点到图像外灰色边距，保存时会自动裁剪到图像边界。"
                f"\n{objects_summary(state)}"
            )
        else:
            _, x0, y0 = state.pending_box_start
            box = clamp_box(state, (
                min(x0, x),
                min(y0, y),
                max(x0, x),
                max(y0, y),
            ))
            width, height = box_size(box)
            if width < 2 or height < 2:
                msg = (
                    "包围盒太小或第二个点裁剪后与第一个点重合，没有创建框。"
                    "\n请重新点击第二个角，尽量拉开一点距离。"
                    f"\n{objects_summary(state)}"
                )
            else:
                obj.boxes[frame_idx] = box
                state.pending_box_start = None
                msg = (
                    f"已创建包围盒: ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})"
                    f"\n{objects_summary(state)}"
                    + maybe_auto_save(state)
                )
    if auto_created:
        msg = f"已自动新建对象 {obj_id}: {obj.name}\n" + msg
    return draw_prompts(state), msg, state, object_dropdown(state, int(obj_id)), all_object_dropdown(state, int(obj_id)), annotation_dropdown(state), annotated_frame_dropdown(state), current_object_color_picker(state)


def clear_current_object(
    state: AppState | None, obj_id: int | None
) -> tuple[Image.Image | None, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.ColorPicker]:
    state = state or AppState()
    if obj_id is None and state.active_obj_id is not None:
        obj_id = state.active_obj_id
    if obj_id is not None and int(obj_id) in state.objects:
        del state.objects[int(obj_id)]
    state.active_obj_id = min(state.objects) if state.objects else None
    sync_active_object_for_current_frame(state)
    state.next_obj_id = next_available_obj_id(state)
    state.pending_box_start = None
    return draw_prompts(state), objects_summary(state) + maybe_auto_save(state), state, object_dropdown(state, state.active_obj_id), all_object_dropdown(state, state.active_obj_id), annotation_dropdown(state), annotated_frame_dropdown(state), current_object_color_picker(state)


def clear_all(state: AppState | None) -> tuple[Image.Image | None, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.ColorPicker]:
    state = state or AppState()
    state.objects = {}
    state.next_obj_id = 1
    state.active_obj_id = None
    state.pending_box_start = None
    return draw_prompts(state), objects_summary(state) + maybe_auto_save(state), state, object_dropdown(state), all_object_dropdown(state), annotation_dropdown(state), annotated_frame_dropdown(state), current_object_color_picker(state)


def delete_annotation(annotation_key: str | None, state: AppState | None) -> tuple[Image.Image | None, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.ColorPicker]:
    state = state or AppState()
    if not annotation_key:
        raise gr.Error("当前帧没有可删除的标注。")
    parts = annotation_key.split(":")
    kind = parts[0]
    obj_id = int(parts[1])
    frame_idx = int(parts[2])
    if obj_id not in state.objects:
        raise gr.Error("对象不存在。")
    obj = state.objects[obj_id]
    if kind == "box":
        obj.boxes.pop(frame_idx, None)
    elif kind == "point":
        point_idx = int(parts[3])
        kept = []
        seen = 0
        for point in obj.points:
            if point[0] == frame_idx:
                if seen == point_idx:
                    seen += 1
                    continue
                seen += 1
            kept.append(point)
        obj.points = kept
    else:
        raise gr.Error("未知标注类型。")
    sync_active_object_for_current_frame(state)
    msg = "已删除当前帧标注。\n" + objects_summary(state) + maybe_auto_save(state)
    return draw_prompts(state), msg, state, object_dropdown(state, state.active_obj_id), all_object_dropdown(state, state.active_obj_id), annotation_dropdown(state), annotated_frame_dropdown(state), current_object_color_picker(state)


def draw_prompts(state: AppState) -> Image.Image | None:
    if state.first_frame is None:
        return None

    if state.preview_size is None:
        state.preview_size = get_preview_size(state.first_frame, state.annotation_zoom)
    frame_image = state.first_frame.resize(state.preview_size, Image.Resampling.BILINEAR)
    image = Image.new(
        "RGB",
        (state.preview_size[0] + 2 * ANNOTATION_MARGIN, state.preview_size[1] + 2 * ANNOTATION_MARGIN),
        (245, 245, 245),
    )
    image.paste(frame_image, (ANNOTATION_MARGIN, ANNOTATION_MARGIN))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle(
        (
            ANNOTATION_MARGIN,
            ANNOTATION_MARGIN,
            ANNOTATION_MARGIN + state.preview_size[0] - 1,
            ANNOTATION_MARGIN + state.preview_size[1] - 1,
        ),
        outline=(180, 180, 180, 255),
        width=1,
    )
    draw_saved_mask_overlay(state, image)
    for obj_id, obj in state.objects.items():
        if state.display_mode == "隐藏标注":
            continue
        if state.display_mode == "当前对象" and obj_id != state.active_obj_id:
            continue
        color = object_color(state, obj_id)
        rgba = (*color, 230)
        if state.current_frame_idx in obj.boxes:
            box = obj.boxes[state.current_frame_idx]
            x0, y0 = original_to_display(state, box[0], box[1])
            x1, y1 = original_to_display(state, box[2], box[3])
            draw.rectangle((x0, y0, x1, y1), outline=rgba, width=4)
        for frame_idx, x, y, label in obj.points:
            if frame_idx != state.current_frame_idx:
                continue
            x, y = original_to_display(state, x, y)
            r = 7
            fill = (52, 211, 153, 240) if label == 1 else (248, 113, 113, 240)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(255, 255, 255, 240), width=2)

    if state.pending_box_start is not None:
        frame_idx, x, y = state.pending_box_start
        if frame_idx == state.current_frame_idx:
            x, y = original_to_display(state, x, y)
            draw.line((x - 12, y, x + 12, y), fill=(20, 20, 20, 255), width=3)
            draw.line((x, y - 12, x, y + 12), fill=(20, 20, 20, 255), width=3)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 255, 255, 255), width=2)
    return image


def draw_saved_mask_overlay(state: AppState, image: Image.Image) -> None:
    if not state.show_saved_masks or not state.saved_mask_dir or state.preview_size is None:
        return
    if state.display_mode == "隐藏标注":
        return
    if (
        state.display_mode == "当前对象"
        and state.active_obj_id in state.objects
        and not object_has_prompt_on_frame(state.objects[state.active_obj_id], state.current_frame_idx)
    ):
        return
    mask_path = Path(state.saved_mask_dir) / f"{state.current_frame_idx:05d}.png"
    if not mask_path.exists():
        return
    try:
        mask = Image.open(mask_path).convert("L")
    except Exception:
        return
    mask = mask.resize(state.preview_size, Image.Resampling.NEAREST)
    mask_arr = np.array(mask)
    rgba = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8)
    for obj_id in np.unique(mask_arr):
        if obj_id == 0:
            continue
        if state.display_mode == "当前对象" and int(obj_id) != state.active_obj_id:
            continue
        color = object_color(state, int(obj_id))
        rgba[mask_arr == obj_id] = (*color, state.mask_alpha)
    overlay = Image.fromarray(rgba, mode="RGBA")
    image_rgba = image.convert("RGBA")
    image_rgba.alpha_composite(overlay, dest=(ANNOTATION_MARGIN, ANNOTATION_MARGIN))
    image.paste(image_rgba.convert("RGB"))


def objects_summary(state: AppState | None) -> str:
    if not state or not state.objects:
        return "还没有对象标注。先新建对象，再在第一帧上点击。"

    lines = []
    for obj_id, obj in state.objects.items():
        state_text = "已完成" if obj.completed else "编辑中"
        pos = sum(1 for *_, label in obj.points if label == 1)
        neg = sum(1 for *_, label in obj.points if label == 0)
        prompt_frames = sorted({frame_idx for frame_idx, *_ in obj.points} | set(obj.boxes))
        current_pos = sum(
            1
            for frame_idx, *_xy, label in obj.points
            if frame_idx == state.current_frame_idx and label == 1
        )
        current_neg = sum(
            1
            for frame_idx, *_xy, label in obj.points
            if frame_idx == state.current_frame_idx and label == 0
        )
        current_box = "有框" if state.current_frame_idx in obj.boxes else "无框"
        frames = ",".join(str(idx) for idx in prompt_frames) if prompt_frames else "-"
        lines.append(
            f"{obj_id}: {obj.name} [{state_text}] | 总正点 {pos} | 总负点 {neg} | 标注帧 {frames} | "
            f"当前帧 正点 {current_pos} 负点 {current_neg} {current_box}"
        )
    return "\n".join(lines)


def build_predictor(model_name: str, vos_optimized: bool):
    cache_key = f"{model_name}|{vos_optimized}|{DEVICE}"
    if cache_key in PREDICTOR_CACHE:
        return PREDICTOR_CACHE[cache_key]

    model_cfg, ckpt = MODEL_OPTIONS[model_name]
    ckpt_path = abs_path(ckpt)
    if not ckpt_path.exists():
        raise gr.Error(f"找不到 checkpoint: {ckpt_path}")
    predictor = build_sam2_video_predictor(
        model_cfg,
        str(ckpt_path),
        device=DEVICE,
        vos_optimized=vos_optimized,
        hydra_overrides_extra=[
            "++model.clear_non_cond_mem_around_input=true",
        ],
    )
    add_clear_obj_non_cond_mem_compat(predictor)
    PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def add_clear_obj_non_cond_mem_compat(predictor: Any) -> None:
    if hasattr(predictor, "_clear_obj_non_cond_mem_around_input"):
        return

    def _clear_obj_non_cond_mem_around_input(self, inference_state, frame_idx, obj_idx):
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)

    predictor._clear_obj_non_cond_mem_around_input = types.MethodType(
        _clear_obj_non_cond_mem_around_input, predictor
    )


def maybe_autocast():
    if DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.inference_mode()


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def release_inference_state(predictor: Any | None, inference_state: Any | None) -> None:
    if predictor is not None and inference_state is not None and hasattr(predictor, "reset_state"):
        try:
            predictor.reset_state(inference_state)
        except Exception:
            pass
    if isinstance(inference_state, dict):
        inference_state.clear()
    clear_cuda_memory()


def clear_gpu_cache() -> str:
    PREDICTOR_CACHE.clear()
    clear_cuda_memory()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"已卸载缓存模型并清理 CUDA cache。当前 PyTorch allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB。"
    return "当前未使用 CUDA。"


def yolo_device_arg() -> str:
    return "0" if DEVICE.type == "cuda" else "cpu"


def resolve_yolo_class_name(model_names: Any, class_id: int, labels: list[str]) -> str | None:
    model_name = None
    if isinstance(model_names, dict):
        model_name = model_names.get(class_id)
    elif isinstance(model_names, (list, tuple)) and 0 <= class_id < len(model_names):
        model_name = model_names[class_id]
    if isinstance(model_name, str) and model_name in labels:
        return model_name
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return None


def build_yolo_model(weights_path: str):
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise gr.Error(
            "当前环境缺少 ultralytics，请安装到 sam2 环境: "
            "conda run -n sam2 python -m pip install ultralytics"
        ) from exc

    weights = abs_path(weights_path)
    if not weights.exists():
        raise gr.Error(f"找不到 YOLO 权重: {weights}")
    return YOLO(str(weights))


def yolo_predictions_to_prompts(results: Any, model_names: Any, labels: list[str]) -> dict[int, dict[str, Any]]:
    prompts: dict[int, dict[str, Any]] = {}
    for result in results:
        frame_idx = int(Path(result.path).stem)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        scores = boxes.conf.detach().cpu().numpy()
        frame_best: dict[int, tuple[float, list[float], str]] = {}
        for box, class_id, score in zip(xyxy, cls, scores):
            label = resolve_yolo_class_name(model_names, int(class_id), labels)
            if label is None:
                continue
            obj_id = labels.index(label) + 1
            current = frame_best.get(obj_id)
            if current is None or float(score) > current[0]:
                frame_best[obj_id] = (float(score), [float(v) for v in box.tolist()], label)
        for obj_id, (score, box, label) in frame_best.items():
            current = prompts.get(obj_id)
            if current is None or score > float(current["confidence"]):
                prompts[obj_id] = {
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "name": label,
                    "box": box,
                    "confidence": score,
                }
    return prompts


def select_yolo_prompt_boxes(
    frame_dir: str,
    weights_path: str,
    labels: list[str],
    conf: float,
    imgsz: int = 640,
    target_frame_idx: int | None = None,
    progress: gr.Progress | None = None,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    if not labels:
        raise gr.Error("当前 task 没有 labels.txt，无法把 YOLO 类别映射成对象类别。")

    model = build_yolo_model(weights_path)
    prompts: dict[int, dict[str, Any]] = {}
    frame_paths = sorted(
        [p for p in Path(frame_dir).iterdir() if p.suffix in FRAME_EXTS],
        key=lambda p: int(p.stem),
    )
    if target_frame_idx is not None:
        frame_path = Path(frame_dir) / f"{int(target_frame_idx):05d}.jpg"
        if not frame_path.exists():
            raise gr.Error(f"当前帧图片不存在: {frame_path}")
        source: str | list[str] = str(frame_path)
        frame_count = 1
    else:
        source = [str(p) for p in frame_paths]
        frame_count = len(frame_paths)
    results = model.predict(
        source=source,
        stream=True,
        conf=float(conf),
        imgsz=int(imgsz),
        device=yolo_device_arg(),
        verbose=False,
    )
    for seen, result in enumerate(results, start=1):
        if progress and seen % 20 == 0:
            progress(min(0.35, 0.05 + 0.30 * seen / max(frame_count, 1)), desc=f"YOLO 检测第 {seen}/{frame_count} 帧")
        for obj_id, prompt in yolo_predictions_to_prompts([result], model.names, labels).items():
            if obj_id not in prompts:
                prompts[obj_id] = prompt
        if len(prompts) >= len(labels):
            break

    skipped = [label for idx, label in enumerate(labels, start=1) if idx not in prompts]
    return prompts, skipped


def apply_yolo_prompts_to_state(
    state: AppState,
    prompts: dict[int, dict[str, Any]],
    add_center_point: bool,
    replace_existing: bool,
) -> None:
    if replace_existing:
        state.objects = {}
        state.next_obj_id = 1
        state.active_obj_id = None
        state.pending_box_start = None
    for obj_id, prompt in sorted(prompts.items()):
        obj = get_or_create_object(state, obj_id, str(prompt["name"]))
        obj.completed = True
        box = tuple(float(v) for v in prompt["box"])
        frame_idx = int(prompt["frame_idx"])
        obj.boxes[frame_idx] = clamp_box(state, box)
        if add_center_point:
            x0, y0, x1, y1 = obj.boxes[frame_idx]
            obj.points.append((frame_idx, (x0 + x1) / 2, (y0 + y1) / 2, 1))
    ensure_object_colors(state)
    if prompts:
        first_prompt = min(prompts.values(), key=lambda item: int(item["frame_idx"]))
        state.current_frame_idx = int(first_prompt["frame_idx"])
        state.first_frame = load_frame_image(state, state.current_frame_idx)
        state.preview_size = get_preview_size(state.first_frame, state.annotation_zoom)
        state.active_obj_id = int(first_prompt["obj_id"])


def yolo_generate_prompts(
    weights_path: str,
    conf: float,
    frame_mode: str,
    add_center_point: bool,
    replace_existing: bool,
    state: AppState | None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Image.Image | None, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.ColorPicker, gr.Slider]:
    state = state or AppState()
    if state.frame_dir is None:
        raise gr.Error("请先加载视频。")
    progress(0.02, desc="加载 YOLO 模型")
    target_frame_idx = state.current_frame_idx if frame_mode == "当前帧" else None
    prompts, skipped = select_yolo_prompt_boxes(
        state.frame_dir,
        weights_path,
        state.task_labels,
        float(conf),
        target_frame_idx=target_frame_idx,
        progress=progress,
    )
    if not prompts:
        raise gr.Error(f"YOLO 没有在置信度 >= {conf:.2f} 时找到任何可用 prompt。")
    apply_yolo_prompts_to_state(state, prompts, bool(add_center_point), bool(replace_existing))
    save_msg = maybe_auto_save(state)
    lines = [
        f"YOLO 已生成 {len(prompts)} 个 box prompt。",
        *[
            f"{item['obj_id']}: {item['name']} | frame {item['frame_idx']} | conf {item['confidence']:.3f}"
            for item in sorted(prompts.values(), key=lambda p: int(p["obj_id"]))
        ],
    ]
    if skipped:
        lines.append("未检测到: " + ", ".join(skipped))
    lines.append(objects_summary(state))
    msg = "\n".join(lines) + save_msg
    return (
        draw_prompts(state),
        msg,
        state,
        object_dropdown(state, state.active_obj_id),
        all_object_dropdown(state, state.active_obj_id),
        annotation_dropdown(state),
        annotated_frame_dropdown(state),
        current_object_color_picker(state),
        gr.Slider(value=state.current_frame_idx),
    )


def yolo_generate_and_segment(
    weights_path: str,
    conf: float,
    frame_mode: str,
    add_center_point: bool,
    replace_existing: bool,
    model_name: str,
    vos_optimized: bool,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    state: AppState | None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Image.Image | None, str | None, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.ColorPicker, gr.Slider]:
    image, yolo_msg, state, obj_dd, all_obj_dd, anno_dd, frame_dd, color_picker, frame_slider = yolo_generate_prompts(
        weights_path,
        conf,
        frame_mode,
        add_center_point,
        replace_existing,
        state,
        progress,
    )
    image, video, seg_msg, state = run_segmentation(
        model_name,
        vos_optimized,
        offload_video_to_cpu,
        offload_state_to_cpu,
        state,
        progress,
    )
    return (
        image,
        video,
        yolo_msg + "\n\n" + seg_msg,
        state,
        object_dropdown(state, state.active_obj_id),
        all_object_dropdown(state, state.active_obj_id),
        annotation_dropdown(state),
        annotated_frame_dropdown(state),
        current_object_color_picker(state),
        gr.Slider(value=state.current_frame_idx),
    )


def video_candidates(root: str) -> list[str]:
    root_path = abs_path(root)
    if not root_path.exists() or not root_path.is_dir():
        raise gr.Error(f"视频目录不存在: {root_path}")
    candidates: list[str] = []
    for item in sorted(root_path.iterdir()):
        if item.is_file() and item.suffix.lower() in VIDEO_EXTS:
            candidates.append(str(item))
        elif item.is_dir() and contains_jpeg_frames(item):
            candidates.append(str(item))
    return candidates


def video_output_dir(task_path: Path, video_path: str | Path) -> Path:
    return task_path / f"{Path(video_path).stem}_sam2_masks"


def video_has_completed_output(task_path: Path, video_path: str | Path) -> bool:
    out_dir = video_output_dir(task_path, video_path)
    mask_dir = out_dir / "mask_frames"
    return (out_dir / "overlay.mp4").exists() and mask_dir.exists() and any(mask_dir.glob("*.png"))


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "未知"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


class ProgressSlice:
    def __init__(self, progress: gr.Progress, start: float, end: float, prefix: str):
        self.progress = progress
        self.start = start
        self.end = end
        self.prefix = prefix

    def __call__(self, value: float, desc: str | None = None):
        value = min(max(float(value), 0.0), 1.0)
        mapped = self.start + (self.end - self.start) * value
        full_desc = self.prefix if not desc else f"{self.prefix} | {desc}"
        return self.progress(mapped, desc=full_desc)


def init_state_for_batch_video(
    video_path: str,
    output_root: str,
    task_name: str,
    task_path: Path,
    task_labels: list[str],
) -> AppState:
    frame_dir, frame_names = prepare_frame_dir(video_path)
    first = Image.open(frame_dir / frame_names[0]).convert("RGB")
    state = AppState()
    state.output_root = str(abs_path(output_root or (REPO_ROOT / "outputs")))
    state.task_name = task_name.strip()
    state.task_dir = str(task_path)
    state.task_labels = task_labels
    state.video_path = str(abs_path(video_path))
    state.frame_dir = str(frame_dir)
    state.frame_names = frame_names
    state.first_frame = first
    state.current_frame_idx = 0
    state.annotation_zoom = 1.0
    state.preview_size = get_preview_size(first, state.annotation_zoom)
    state.objects = {}
    state.next_obj_id = 1
    state.active_obj_id = None
    state.pending_box_start = None
    state.label_dir = label_dir_from_root(task_path, video_path)
    state.saved_mask_dir = saved_mask_dir_from_root(task_path, video_path)
    return state


def add_yolo_prompts_at_batch_frames(
    state: AppState,
    yolo_model: Any,
    conf: float,
    add_center_point: bool,
) -> tuple[list[str], list[int]]:
    total = len(state.frame_names)
    target_frames = sorted({min(max(round((total - 1) * ratio), 0), total - 1) for ratio in (3 / 5, 4 / 5)})
    detected: set[str] = set()
    for frame_idx in target_frames:
        frame_path = Path(state.frame_dir) / f"{frame_idx:05d}.jpg"
        results = yolo_model.predict(
            source=str(frame_path),
            stream=False,
            conf=float(conf),
            imgsz=640,
            device=yolo_device_arg(),
            verbose=False,
        )
        prompts = yolo_predictions_to_prompts(results, yolo_model.names, state.task_labels)
        if not prompts:
            continue
        apply_yolo_prompts_to_state(state, prompts, bool(add_center_point), replace_existing=False)
        detected.update(str(prompt["name"]) for prompt in prompts.values())
    missed = [label for label in state.task_labels if label not in detected]
    return missed, target_frames


def batch_segment_remaining_videos(
    video_root: str,
    output_root: str,
    task_name: str,
    weights_path: str,
    conf: float,
    add_center_point: bool,
    model_name: str,
    vos_optimized: bool,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Image.Image | None, str | None, str, AppState, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.ColorPicker, gr.Slider]:
    if not task_name:
        raise gr.Error("请先选择或创建 task。")

    task_path = task_dir_from_root(output_root or (REPO_ROOT / "outputs"), task_name)
    task_path.mkdir(parents=True, exist_ok=True)
    task_labels = load_task_labels(task_path)
    if not task_labels:
        raise gr.Error(f"当前 task 没有标签，请先维护 {task_labels_file(task_path)}。")

    candidates = video_candidates(video_root)
    pending = [path for path in candidates if not video_has_completed_output(task_path, path)]
    skipped = len(candidates) - len(pending)
    if not pending:
        return None, None, f"没有待处理视频。共扫描 {len(candidates)} 个，已处理 {skipped} 个。", AppState(), gr.Dropdown(), gr.Dropdown(), gr.Dropdown(), gr.Dropdown(), gr.ColorPicker(value=generated_color(1)), gr.Slider(value=0)

    progress(0.01, desc="加载 YOLO 模型")
    yolo_model = build_yolo_model(weights_path)
    started_at = time.monotonic()
    processed: list[str] = []
    failed: list[str] = []
    last_state = AppState()
    last_video: str | None = None

    for index, video_path in enumerate(pending, start=1):
        elapsed = time.monotonic() - started_at
        eta = None if index == 1 else elapsed / (index - 1) * (len(pending) - index + 1)
        prefix = (
            f"批处理 {index}/{len(pending)}: {Path(video_path).name} | "
            f"已跳过 {skipped} | ETA {format_duration(eta)}"
        )
        start = (index - 1) / len(pending)
        end = index / len(pending)
        progress(start, desc=prefix)
        try:
            state = init_state_for_batch_video(video_path, output_root, task_name, task_path, task_labels)
            missed, target_frames = add_yolo_prompts_at_batch_frames(
                state,
                yolo_model,
                float(conf),
                bool(add_center_point),
            )
            if not any(obj.points or obj.boxes for obj in state.objects.values()):
                raise RuntimeError(f"YOLO 在帧 {target_frames} 没有生成任何 prompt")
            save_prompts_to_disk(state.label_dir, state)
            slice_progress = ProgressSlice(progress, start + 0.10 * (end - start), end, prefix)
            _, overlay_video, _, state = run_segmentation(
                model_name,
                vos_optimized,
                offload_video_to_cpu,
                offload_state_to_cpu,
                state,
                slice_progress,
            )
            processed.append(
                f"{Path(video_path).name}: prompts frames={target_frames}"
                + (f", missed={','.join(missed)}" if missed else "")
            )
            last_state = state
            last_video = overlay_video
        except Exception as exc:
            failed.append(f"{Path(video_path).name}: {exc}")
            clear_cuda_memory()
        progress(end, desc=f"{prefix} | 完成")

    elapsed = time.monotonic() - started_at
    msg = [
        f"批处理完成。扫描 {len(candidates)} 个，跳过已处理 {skipped} 个，成功 {len(processed)} 个，失败 {len(failed)} 个。",
        f"总耗时: {format_duration(elapsed)}",
    ]
    if processed:
        msg.append("成功:")
        msg.extend(processed[-10:])
    if failed:
        msg.append("失败:")
        msg.extend(failed)
    return (
        draw_prompts(last_state) if last_state.frame_dir else None,
        last_video,
        "\n".join(msg),
        last_state,
        object_dropdown(last_state, last_state.active_obj_id),
        all_object_dropdown(last_state, last_state.active_obj_id),
        annotation_dropdown(last_state),
        annotated_frame_dropdown(last_state),
        current_object_color_picker(last_state),
        gr.Slider(value=last_state.current_frame_idx if last_state.frame_dir else 0),
    )


def run_segmentation(
    model_name: str,
    vos_optimized: bool,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    state: AppState | None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[Image.Image | None, str | None, str, AppState]:
    state = state or AppState()
    if state.frame_dir is None:
        raise gr.Error("请先加载视频。")
    if not state.objects:
        raise gr.Error("请至少添加一个对象 prompt。")
    unfinished = [obj_id for obj_id, obj in state.objects.items() if not obj.completed]
    if unfinished:
        raise gr.Error(f"对象 {unfinished[0]} 仍在编辑中，请先点击“完成当前对象”。")

    prompted = [(obj_id, obj) for obj_id, obj in state.objects.items() if obj.points or obj.boxes]
    if not prompted:
        raise gr.Error("对象还没有点或框。")

    progress(0.05, desc="加载 SAM 2 模型")
    clear_cuda_memory()
    predictor = build_predictor(model_name, vos_optimized)
    inference_state = None
    video_segments: dict[int, dict[int, np.ndarray]] = {}
    try:
        with torch.inference_mode(), maybe_autocast():
            progress(0.12, desc="初始化视频状态")
            inference_state = predictor.init_state(
                state.frame_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )

            prompt_frame_indices: list[int] = []
            for obj_id, obj in prompted:
                frame_indices = sorted({frame_idx for frame_idx, *_ in obj.points} | set(obj.boxes))
                prompt_frame_indices.extend(frame_indices)
                for frame_idx in frame_indices:
                    frame_points = [
                        (*clamp_point(state, x, y), label)
                        for point_frame_idx, x, y, label in obj.points
                        if point_frame_idx == frame_idx
                    ]
                    points = np.array([(x, y) for x, y, _ in frame_points], dtype=np.float32)
                    labels = np.array([label for _, _, label in frame_points], dtype=np.int32)
                    if len(points) == 0:
                        points = None
                        labels = None

                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                        box=np.array(clamp_box(state, obj.boxes[frame_idx]), dtype=np.float32)
                        if frame_idx in obj.boxes
                        else None,
                    )
                    del out_mask_logits

            progress(0.25, desc="传播到全视频")
            total = len(state.frame_names)
            start_frame_idx = min(prompt_frame_indices)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=start_frame_idx, reverse=False
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                del out_mask_logits
                progress(0.25 + 0.55 * (out_frame_idx + 1) / max(total, 1), desc=f"传播第 {out_frame_idx + 1}/{total} 帧")

            reverse_start_idx = max(prompt_frame_indices)
            if reverse_start_idx > 0:
                progress(0.80, desc="向前反向传播")
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state, start_frame_idx=reverse_start_idx, reverse=True
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    del out_mask_logits
    except torch.cuda.OutOfMemoryError as exc:
        release_inference_state(predictor, inference_state)
        if not (offload_video_to_cpu and offload_state_to_cpu):
            progress(0.05, desc="显存不足，已清理并自动启用 CPU offload 重试")
            return run_segmentation(
                model_name,
                vos_optimized,
                True,
                True,
                state,
                progress,
            )
        raise gr.Error(
            "CUDA 显存不足，已启用 CPU offload 仍无法完成。建议改用 SAM 2.1 Base+/Small，"
            "减少同一轮 prompt 帧或对象数，或点击“卸载模型并释放显存”后重试。"
        ) from exc
    finally:
        release_inference_state(predictor, inference_state)
        inference_state = None

    progress(0.84, desc="导出结果")
    out_dir = export_results(state, video_segments)
    state.saved_mask_dir = str(out_dir / "mask_frames")
    state.show_saved_masks = True
    overlay_video = encode_overlay_video(out_dir)
    msg = f"完成。结果目录: {out_dir}"
    msg += "\nmask_frames 是类别 ID 图，像素值 0=背景、1/2/...=对象，直接查看会很暗。"
    msg += "\n可视化 mask 请看 mask_visual_frames；单对象 0/255 二值 mask 请看 object_mask_frames。"
    if overlay_video:
        msg += f"\nOverlay 视频: {overlay_video}"
    return draw_prompts(state), overlay_video, msg, state


def render_overlay(state: AppState, frame: Image.Image | None, masks: dict[int, np.ndarray]) -> Image.Image | None:
    if frame is None:
        return None
    base = frame.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for obj_id, mask in masks_in_export_priority(state, masks):
        mask_2d = np.squeeze(mask).astype(bool)
        color = object_color(state, int(obj_id))
        rgba = np.zeros((mask_2d.shape[0], mask_2d.shape[1], 4), dtype=np.uint8)
        rgba[mask_2d] = (*color, state.mask_alpha)
        mask_img = Image.fromarray(rgba, mode="RGBA").resize(base.size, Image.Resampling.NEAREST)
        overlay.alpha_composite(mask_img)
    return Image.alpha_composite(base, overlay).convert("RGB")


def export_results(state: AppState, video_segments: dict[int, dict[int, np.ndarray]]) -> Path:
    source_name = Path(state.video_path or state.frame_dir or "video").stem
    out_root = Path(state.task_dir) if state.task_dir else (REPO_ROOT / "outputs")
    out_dir = out_root / f"{source_name}_sam2_masks"
    overlay_dir = out_dir / "overlay_frames"
    mask_dir = out_dir / "mask_frames"
    mask_visual_dir = out_dir / "mask_visual_frames"
    object_mask_dir = out_dir / "object_mask_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for generated_path in (overlay_dir, mask_dir, mask_visual_dir, object_mask_dir):
        if generated_path.exists():
            shutil.rmtree(generated_path)
    old_overlay = out_dir / "overlay.mp4"
    if old_overlay.exists():
        old_overlay.unlink()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_visual_dir.mkdir(parents=True, exist_ok=True)
    object_mask_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = Path(state.frame_dir)
    for idx, frame_name in enumerate(state.frame_names):
        frame = Image.open(frame_dir / frame_name).convert("RGB")
        masks = video_segments.get(idx, {})
        render_overlay(state, frame, masks).save(overlay_dir / f"{idx:05d}.jpg", quality=92)

        class_mask = np.zeros((frame.height, frame.width), dtype=np.uint8)
        visual_mask = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)
        for obj_id, mask in masks_in_export_priority(state, masks):
            mask_img = Image.fromarray(np.squeeze(mask).astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize((frame.width, frame.height), Image.Resampling.NEAREST)
            binary_mask = np.array(mask_img) > 0
            class_mask[binary_mask] = int(obj_id)
            visual_mask[binary_mask] = object_color(state, int(obj_id))
            mask_img.save(object_mask_dir / f"{idx:05d}_obj{int(obj_id):03d}.png")
        Image.fromarray(class_mask, mode="L").save(mask_dir / f"{idx:05d}.png")
        Image.fromarray(visual_mask, mode="RGB").save(mask_visual_dir / f"{idx:05d}.png")

    return out_dir


def encode_overlay_video(out_dir: Path) -> str | None:
    overlay_dir = out_dir / "overlay_frames"
    output_mp4 = out_dir / "overlay.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "24",
        "-i",
        str(overlay_dir / "%05d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return str(output_mp4)


def build_ui() -> gr.Blocks:
    css = """
    #prompt_image .icon-buttons,
    #prompt_image button[title*="fullscreen" i],
    #prompt_image button[aria-label*="fullscreen" i],
    #prompt_image button[title*="download" i],
    #prompt_image button[aria-label*="download" i] {
        display: none !important;
    }
    """
    with gr.Blocks(title="SAM 2 视频分割工作台") as demo:
        demo.css = css
        state = gr.State(AppState())
        gr.Markdown("## SAM 2 视频分割工作台")
        default_video_root = "/home/romilab/Projects/IsaacLab/source/msr-surgical/saved_data/cube1/videos/observation.images.camera/chunk-000"
        default_output_root = str(REPO_ROOT / "outputs")
        default_tasks = list_tasks(default_output_root)
        default_task = default_tasks[0] if default_tasks else ""

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("视频与任务", open=True):
                    video_root = gr.Textbox(
                        label="当前视频目录",
                        value=default_video_root,
                        interactive=True,
                    )
                    dir_dropdown = gr.Dropdown(
                        label="子目录",
                        choices=list_subdirectories(default_video_root),
                        value=None,
                        interactive=True,
                    )
                    with gr.Row():
                        parent_btn = gr.Button("上一级")
                        enter_dir_btn = gr.Button("进入")
                        use_dir_btn = gr.Button("使用当前目录")
                    scan_btn = gr.Button("扫描视频")
                    video_dropdown = gr.Dropdown(label="视频 / JPEG 帧目录", choices=[], interactive=True)

                    output_root_box = gr.Textbox(
                        label="输出根目录",
                        value=default_output_root,
                        interactive=True,
                    )
                    with gr.Row():
                        scan_task_btn = gr.Button("扫描 task")
                        use_output_root_btn = gr.Button("当前目录作输出根目录")
                    task_dropdown_ui = gr.Dropdown(label="Task 类型", choices=default_tasks, value=default_task or None, interactive=True)
                    task_name_box = gr.Textbox(label="新建/选择 task", value=default_task, interactive=True)
                    use_task_btn = gr.Button("使用 task", variant="secondary")
                    load_btn = gr.Button("加载视频", variant="primary")

                with gr.Accordion("标注操作", open=True):
                    task_label_dropdown = gr.Dropdown(label="对象类别", choices=[], interactive=True)
                    with gr.Row():
                        add_obj_btn = gr.Button("新增对象", variant="primary")
                        rename_obj_btn = gr.Button("修改当前对象类别", variant="secondary")
                        finish_obj_btn = gr.Button("完成当前对象", variant="secondary")
                    obj_dropdown = gr.Dropdown(label="当前对象", choices=[], interactive=True)
                    with gr.Row():
                        object_color_picker = gr.ColorPicker(label="当前对象颜色", value=generated_color(1), interactive=True)
                        mask_alpha_slider = gr.Slider(
                            label="mask 透明度",
                            minimum=0,
                            maximum=255,
                            step=1,
                            value=80,
                            interactive=True,
                        )
                    apply_color_btn = gr.Button("应用对象颜色")
                    mode = gr.Radio(["正点", "负点", "框"], label="标注模式", value="正点")
                    all_obj_dropdown = gr.Dropdown(label="全部已有对象（跨帧继续标注）", choices=[], interactive=True)
                    display_mode = gr.Radio(
                        ["当前对象", "全部显示", "隐藏标注"],
                        label="标注显示",
                        value="当前对象",
                    )
                    show_saved_masks = gr.Checkbox(label="显示已有传播 mask", value=True)
                    with gr.Row():
                        clear_obj_btn = gr.Button("删除当前对象")
                        clear_all_btn = gr.Button("清空全部标注")

                with gr.Accordion("Task 标签维护", open=False):
                    new_task_label = gr.Textbox(label="新增标签", value="", interactive=True)
                    add_task_label_btn = gr.Button("写入 task 标签文件")
                    save_label_btn = gr.Button("保存当前视频标注")

                with gr.Accordion("模型与导出", open=False):
                    model_name = gr.Dropdown(
                        label="模型",
                        choices=list(MODEL_OPTIONS),
                        value="SAM 2.1 Large",
                        interactive=True,
                    )
                    vos_optimized = gr.Checkbox(label="启用 VOS optimized", value=False)
                    offload_video = gr.Checkbox(label="视频帧 offload 到 CPU", value=True)
                    offload_state = gr.Checkbox(label="状态 offload 到 CPU", value=True)
                    run_btn = gr.Button("开始分割并导出", variant="primary")
                    clear_gpu_btn = gr.Button("卸载模型并释放显存", variant="secondary")

                with gr.Accordion("YOLO 自动 Prompt", open=False):
                    yolo_weights = gr.Textbox(
                        label="YOLO 权重",
                        value=str(REPO_ROOT / "runs/yolo/task1_detect/weights/best.pt"),
                        interactive=True,
                    )
                    yolo_conf = gr.Slider(
                        label="YOLO 置信度阈值",
                        minimum=0.01,
                        maximum=0.95,
                        step=0.01,
                        value=0.25,
                        interactive=True,
                    )
                    yolo_frame_mode = gr.Radio(
                        ["当前帧", "全视频首次检出"],
                        label="YOLO 使用帧",
                        value="当前帧",
                    )
                    yolo_add_center = gr.Checkbox(label="同时添加 box 中心正点", value=False)
                    yolo_replace = gr.Checkbox(label="覆盖当前标注", value=True)
                    yolo_prompt_btn = gr.Button("YOLO 生成框 Prompt", variant="secondary")
                    yolo_run_btn = gr.Button("YOLO 生成 Prompt 并分割导出", variant="primary")

                with gr.Accordion("批量自动分割", open=False):
                    batch_run_btn = gr.Button("分割当前目录剩余视频", variant="primary")

            with gr.Column(scale=2):
                frame_slider = gr.Slider(
                    label="当前标注帧",
                    minimum=0,
                    maximum=1,
                    step=1,
                    value=0,
                    interactive=True,
                )
                zoom_slider = gr.Slider(
                    label="标注放大倍数",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    interactive=True,
                )
                with gr.Row():
                    annotated_frame_select = gr.Dropdown(label="已人工标注帧", choices=[], interactive=True)
                    annotation_select = gr.Dropdown(label="当前帧标注信息", choices=[], interactive=True)
                    delete_annotation_btn = gr.Button("删除选中标注")
                gr.Markdown("当前帧标注")
                prompt_image = gr.Image(
                    type="pil",
                    interactive=False,
                    show_label=False,
                    elem_id="prompt_image",
                )
                status = gr.Textbox(label="状态", lines=5, interactive=False)
                result_video = gr.Video(label="Overlay 视频")

        parent_btn.click(go_parent_dir, inputs=video_root, outputs=[video_root, dir_dropdown, status])
        enter_dir_btn.click(
            enter_selected_dir,
            inputs=[video_root, dir_dropdown],
            outputs=[video_root, dir_dropdown, status],
        )
        use_dir_btn.click(use_current_dir, inputs=video_root, outputs=[video_root, video_dropdown, status])
        scan_btn.click(list_videos, inputs=video_root, outputs=[video_dropdown, status])
        scan_task_btn.click(scan_tasks, inputs=output_root_box, outputs=[task_dropdown_ui, status])
        use_output_root_btn.click(
            lambda current_dir: (str(abs_path(current_dir)), task_dropdown(str(abs_path(current_dir))), f"输出根目录: {abs_path(current_dir)}"),
            inputs=video_root,
            outputs=[output_root_box, task_dropdown_ui, status],
        )
        task_dropdown_ui.change(lambda task: task or "", inputs=task_dropdown_ui, outputs=task_name_box)
        use_task_btn.click(
            use_task,
            inputs=[output_root_box, task_name_box, state],
            outputs=[task_dropdown_ui, task_label_dropdown, status, state],
        )
        add_task_label_btn.click(
            add_task_label,
            inputs=[output_root_box, task_name_box, new_task_label, state],
            outputs=[task_label_dropdown, status, state],
        )
        load_btn.click(
            load_video,
            inputs=[video_dropdown, output_root_box, task_name_box, state],
            outputs=[
                prompt_image,
                status,
                state,
                obj_dropdown,
                all_obj_dropdown,
                annotation_select,
                annotated_frame_select,
                frame_slider,
                zoom_slider,
                output_root_box,
                task_label_dropdown,
                object_color_picker,
                mask_alpha_slider,
                result_video,
            ],
        )
        save_label_btn.click(save_label_dir, inputs=[output_root_box, state], outputs=[status, state])
        frame_slider.change(
            change_frame,
            inputs=[frame_slider, state],
            outputs=[prompt_image, status, state, obj_dropdown, annotation_select, annotated_frame_select, object_color_picker],
        )
        annotated_frame_select.change(
            jump_to_annotated_frame,
            inputs=[annotated_frame_select, state],
            outputs=[prompt_image, status, state, obj_dropdown, annotation_select, annotated_frame_select, object_color_picker, frame_slider],
        )
        zoom_slider.change(change_zoom, inputs=[zoom_slider, state], outputs=[prompt_image, status, state])
        add_obj_btn.click(add_object, inputs=[task_label_dropdown, state], outputs=[obj_dropdown, all_obj_dropdown, object_color_picker, mode, status, state])
        all_obj_dropdown.change(
            select_existing_object,
            inputs=[all_obj_dropdown, state],
            outputs=[prompt_image, status, state, obj_dropdown, task_label_dropdown, object_color_picker, mode],
        )
        rename_obj_btn.click(
            rename_current_object,
            inputs=[obj_dropdown, task_label_dropdown, state],
            outputs=[prompt_image, status, state, obj_dropdown, all_obj_dropdown, annotation_select, annotated_frame_select, object_color_picker],
        )
        apply_color_btn.click(
            update_object_color,
            inputs=[obj_dropdown, object_color_picker, state],
            outputs=[prompt_image, status, state, obj_dropdown, all_obj_dropdown, annotation_select, annotated_frame_select],
        )
        mask_alpha_slider.change(change_mask_alpha, inputs=[mask_alpha_slider, state], outputs=[prompt_image, status, state])
        obj_dropdown.change(change_object, inputs=[obj_dropdown, state], outputs=[prompt_image, status, state, task_label_dropdown, object_color_picker])
        finish_obj_btn.click(
            finish_current_object,
            inputs=[state, obj_dropdown],
            outputs=[prompt_image, status, state, obj_dropdown, all_obj_dropdown],
        )
        display_mode.change(change_display_mode, inputs=[display_mode, state], outputs=[prompt_image, status, state])
        show_saved_masks.change(
            change_saved_mask_visibility,
            inputs=[show_saved_masks, state],
            outputs=[prompt_image, status, state],
        )
        prompt_image.select(
            image_click,
            inputs=[state, obj_dropdown, task_label_dropdown, mode],
            outputs=[prompt_image, status, state, obj_dropdown, all_obj_dropdown, annotation_select, annotated_frame_select, object_color_picker],
        )
        delete_annotation_btn.click(
            delete_annotation,
            inputs=[annotation_select, state],
            outputs=[prompt_image, status, state, obj_dropdown, all_obj_dropdown, annotation_select, annotated_frame_select, object_color_picker],
        )
        clear_obj_btn.click(
            clear_current_object,
            inputs=[state, obj_dropdown],
            outputs=[prompt_image, status, state, obj_dropdown, all_obj_dropdown, annotation_select, annotated_frame_select, object_color_picker],
        )
        clear_all_btn.click(
            clear_all,
            inputs=state,
            outputs=[prompt_image, status, state, obj_dropdown, all_obj_dropdown, annotation_select, annotated_frame_select, object_color_picker],
        )
        run_btn.click(
            run_segmentation,
            inputs=[model_name, vos_optimized, offload_video, offload_state, state],
            outputs=[prompt_image, result_video, status, state],
        )
        yolo_prompt_btn.click(
            yolo_generate_prompts,
            inputs=[yolo_weights, yolo_conf, yolo_frame_mode, yolo_add_center, yolo_replace, state],
            outputs=[
                prompt_image,
                status,
                state,
                obj_dropdown,
                all_obj_dropdown,
                annotation_select,
                annotated_frame_select,
                object_color_picker,
                frame_slider,
            ],
        )
        yolo_run_btn.click(
            yolo_generate_and_segment,
            inputs=[
                yolo_weights,
                yolo_conf,
                yolo_frame_mode,
                yolo_add_center,
                yolo_replace,
                model_name,
                vos_optimized,
                offload_video,
                offload_state,
                state,
            ],
            outputs=[
                prompt_image,
                result_video,
                status,
                state,
                obj_dropdown,
                all_obj_dropdown,
                annotation_select,
                annotated_frame_select,
                object_color_picker,
                frame_slider,
            ],
        )
        batch_run_btn.click(
            batch_segment_remaining_videos,
            inputs=[
                video_root,
                output_root_box,
                task_name_box,
                yolo_weights,
                yolo_conf,
                yolo_add_center,
                model_name,
                vos_optimized,
                offload_video,
                offload_state,
            ],
            outputs=[
                prompt_image,
                result_video,
                status,
                state,
                obj_dropdown,
                all_obj_dropdown,
                annotation_select,
                annotated_frame_select,
                object_color_picker,
                frame_slider,
            ],
        )
        clear_gpu_btn.click(clear_gpu_cache, outputs=status)
        demo.load(list_videos, inputs=video_root, outputs=[video_dropdown, status])

    return demo


if __name__ == "__main__":
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    server_name = os.environ.get("SAM2_GRADIO_HOST", "127.0.0.1")
    server_port = int(os.environ.get("SAM2_GRADIO_PORT", "7860"))
    build_ui().launch(server_name=server_name, server_port=server_port, inbrowser=False)
