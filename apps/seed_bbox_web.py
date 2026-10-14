#!/usr/bin/env python3
"""Local, two-view human seed-box editor for LeRobot or paired video datasets.

The source dataset is read-only. Drafts, per-view label files, and SAM2 prompt
JSON and propagation previews are written under --work-dir. SAM2 runs in a
separate process only when the user requests propagation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent / "tools"))
import sam2_direction_review as direction_review
PROPAGATE_SCRIPT = APP_DIR.parent / "tools" / "sam2_prompts_to_masks.py"
UNET_MODEL_DIR = APP_DIR.parent / 'segdata/temp/newur/unet_resnet18_front_seed4_r1'
ROUND_REVIEW_ROOT = APP_DIR.parent / 'segdata/temp/newur/unet_sam2_round1'
MODEL_VARIANTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
COLORS = ["#40a0ff", "#ff6961", "#77dd77", "#ffd166", "#c792ea", "#65d6ad", "#ff9f43", "#f78fb3"]
CANONICAL_COLORS = {
    "occluder": "#40a0ff", "tube": "#ff5c8a", "object": "#ff6961",
    "region": "#77dd77", "tool": "#ffd166", "lefttool": "#ffd166",
    "righttool": "#65d6ad", "leftarm": "#ea92c7", "rightarm": "#add665",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "item"


def parse_labels(text: str) -> tuple[list[dict], dict[str, str]]:
    labels: list[dict] = []
    guidance: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        legacy_view = re.fullmatch(r"([^:：\s]+?)视角\s*[-—–]\s*(.+)", line)
        if legacy_view:
            guidance[legacy_view.group(1)] = legacy_view.group(2).strip()
            continue
        parts = re.split(r"[:：]", line, maxsplit=1)
        name, description = parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        if not name or re.search(r"\s", name):
            raise ValueError(f"第 {number} 行类别名必须是单个名称: {line}")
        if name.endswith("视角"):
            if not description:
                raise ValueError(f"第 {number} 行视角说明不能为空")
            guidance[name[:-2]] = description
        else:
            labels.append({"id": len(labels) + 1, "name": name, "description": description,
                           "color": CANONICAL_COLORS.get(name, COLORS[len(labels) % len(COLORS)])})
    names = [entry["name"] for entry in labels]
    if len(names) != len(set(names)):
        raise ValueError("类别名不能重复")
    if len(labels) > 255:
        raise ValueError("最多支持 255 个类别")
    return labels, guidance


def split_global_labels(text: str, views: list[str]) -> dict[str, str]:
    """Accept both one shared list and legacy repeated per-view blocks."""
    blocks: dict[str, list[str]] = {"": []}
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        legacy = re.fullmatch(r"([^:：\s]+?)视角\s*[-—–]\s*(.+)", line)
        colon = re.fullmatch(r"([^:：\s]+?)视角\s*[:：]\s*(.+)", line)
        marker = legacy or colon
        if marker:
            current = marker.group(1)
            blocks.setdefault(current, [])
            blocks[current].append(f"{current}视角: {marker.group(2)}")
        else:
            blocks.setdefault(current, []).append(raw)
    common = "\n".join(blocks[""])
    candidates = ["\n".join(lines) for key, lines in blocks.items() if key and parse_labels("\n".join(lines))[0]]
    fallback = candidates[0] if candidates else common
    result = {}
    for view in views:
        section = "\n".join(blocks.get(view, []))
        view_labels, guidance = parse_labels(section) if section else ([], {})
        if view_labels:
            result[view] = section.rstrip() + "\n"
            continue
        shared_labels, _ = parse_labels(common) if common else ([], {})
        source = common if shared_labels else fallback
        source_labels, _ = parse_labels(source) if source else ([], {})
        lines = [f"{view}视角: {guidance[view]}"] if view in guidance else []
        lines += [f"{label['name']}: {label['description']}" if label["description"] else label["name"] for label in source_labels]
        result[view] = "\n".join(lines) + ("\n" if lines else "")
    return result


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(value, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {path}")
        result = {"frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                  "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  "fps": float(cap.get(cv2.CAP_PROP_FPS))}
        if min(result["frame_count"], result["width"], result["height"]) <= 0:
            raise ValueError(f"视频元数据无效: {path}")
        return result
    finally:
        cap.release()


def discover(dataset: Path, wanted_views: list[str] | None) -> tuple[list[str], list[dict]]:
    info_path = dataset / "meta" / "info.json"
    episode_dir = dataset / "meta" / "episodes"
    if info_path.is_file() and episode_dir.is_dir():
        import pyarrow.parquet as pq

        info = json.loads(info_path.read_text(encoding="utf-8"))
        available = [key for key, feature in info.get("features", {}).items()
                     if feature.get("dtype") == "video"]
        if wanted_views:
            selected = []
            for want in wanted_views:
                matches = [key for key in available if key == want or key.rsplit(".", 1)[-1] == want]
                if len(matches) != 1:
                    raise ValueError(f"视角 {want!r} 不唯一或不存在；可用视角: {available}")
                selected.append(matches[0])
        else:
            selected = available
        if len(selected) != 2 or len(set(selected)) != 2:
            raise ValueError(f"需要恰好两个视角；当前为 {selected}。可用 --views 指定。")
        template = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        items = []
        for parquet in sorted(episode_dir.rglob("*.parquet")):
            names = set(pq.read_schema(parquet).names)
            wanted = ["episode_index", "tasks", "length"]
            for key in selected:
                wanted.extend([f"videos/{key}/chunk_index", f"videos/{key}/file_index"])
            table = pq.read_table(parquet, columns=[name for name in wanted if name in names])
            for row in table.to_pylist():
                episode = int(row["episode_index"])
                tasks = row.get("tasks") or []
                task = tasks[0] if isinstance(tasks, list) and tasks else (tasks if isinstance(tasks, str) else f"episode-{episode:06d}")
                for key in selected:
                    view = key.rsplit(".", 1)[-1]
                    relative = template.format(video_key=key,
                                               chunk_index=int(row.get(f"videos/{key}/chunk_index", 0)),
                                               file_index=int(row.get(f"videos/{key}/file_index", episode)),
                                               episode_index=episode)
                    path = (dataset / relative).resolve()
                    if not path.is_file() or not path.is_relative_to(dataset):
                        raise FileNotFoundError(path)
                    shape = info["features"][key].get("shape", [])
                    meta = {"width": int(shape[1]), "height": int(shape[0]),
                            "frame_count": int(row.get("length") or 0),
                            "fps": float(info["features"][key].get("info", {}).get("video.fps") or info.get("fps") or 0)} if len(shape) >= 2 else video_meta(path)
                    if min(meta["frame_count"], meta["width"], meta["height"]) <= 0:
                        meta = video_meta(path)
                    items.append({"item_id": f"ep{episode:06d}__{slug(key)}", "task": str(task),
                                  "episode_index": episode, "video_key": key, "view": view,
                                  "video_path": str(path), **meta})
    else:
        root = dataset / "videos" if (dataset / "videos").is_dir() else dataset
        paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTS)
        available = sorted({path.relative_to(root).parts[0] for path in paths if len(path.relative_to(root).parts) > 1})
        selected = wanted_views or available
        if len(selected) != 2 or len(set(selected)) != 2:
            raise ValueError(f"普通视频数据集需有两个视角子目录；发现 {available}。可用 --views 指定。")
        if any(view not in available for view in selected):
            raise ValueError(f"指定视角不在视频目录中: {selected}")
        items = []
        for view in selected:
            for path in paths:
                parts = path.relative_to(root).parts
                if parts[0] != view:
                    continue
                token = slug(str(Path(*parts[1:]).with_suffix("")))
                items.append({"item_id": f"{slug(view)}__{token}", "task": path.stem,
                              "episode_index": -1, "video_key": view, "view": view,
                              "video_path": str(path.resolve()), **video_meta(path)})
    if not items:
        raise ValueError("未发现视频")
    views = [key.rsplit(".", 1)[-1] for key in selected]
    if len(set(views)) != 2:
        raise ValueError(f"两个视频键的短视角名相同: {selected}")
    return views, sorted(items, key=lambda x: (x["episode_index"], x["video_key"], x["item_id"]))


class Editor:
    def __init__(self, dataset: Path, work_dir: Path, views: list[str] | None):
        self.dataset = dataset.resolve()
        self.work_dir = work_dir.resolve()
        if not self.dataset.is_dir():
            raise ValueError(f"数据集目录不存在: {self.dataset}")
        if self.work_dir.is_relative_to(self.dataset) or self.dataset.is_relative_to(self.work_dir):
            raise ValueError("工作目录必须位于原始数据集之外，且不能是原始数据集的上级目录")
        self.views, items = discover(self.dataset, views)
        self.items = {item["item_id"]: item for item in items}
        self.lock = threading.RLock()
        manifest = self.work_dir / "dataset_manifest.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("dataset_root") != str(self.dataset) or existing.get("views") != self.views:
                raise ValueError(f"工作目录已属于其他数据集或视角组合: {manifest}")
        else:
            if self.work_dir.exists() and any(self.work_dir.iterdir()):
                raise ValueError(f"工作目录非空且没有本工具的 manifest，请选择空目录: {self.work_dir}")
            self.work_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(manifest, {"version": 1, "dataset": self.dataset.name,
                                   "dataset_root": str(self.dataset), "views": self.views,
                                   "items": items})
        self.sync_source_labels()

    def source_label_texts(self) -> dict[str, dict]:
        global_path = self.dataset / "labels.txt"
        base_text = global_path.read_text(encoding="utf-8") if global_path.is_file() else ""
        split_labels = split_global_labels(base_text, self.views)
        result = {}
        for view in self.views:
            source = next((path for path in (self.dataset / f"labels_{view}.txt", self.dataset / f"labels.{view}.txt") if path.is_file()), None)
            if source:
                text = source.read_text(encoding="utf-8")
            elif global_path.is_file():
                source, text = global_path, split_labels[view]
            else:
                text = ""
            if text:
                parse_labels(text)
            result[view] = {"path": str(source) if source else None, "text": text,
                            "hash": text_hash(text) if source else None,
                            "mtime_ns": source.stat().st_mtime_ns if source else None}
        return result

    def used_label_names(self, view: str) -> set[str]:
        used = set()
        for item in self.items.values():
            if item["view"] != view:
                continue
            draft = self.get_draft(item["item_id"])
            used.update(name for boxes in draft["frames"].values() for name in boxes)
            used.update(name for by_label in draft.get("points", {}).values() for name, points in by_label.items() if points)
        return used

    def sync_source_labels(self, force: bool = False) -> dict:
        """Refresh view copies when origin labels change; preserve divergent edits."""
        with self.lock:
            source_info = self.source_label_texts()
            sync_path = self.work_dir / "label_sync.json"
            state = json.loads(sync_path.read_text(encoding="utf-8")) if sync_path.is_file() else {"version": 1, "views": {}}
            records = state.setdefault("views", {})
            events, warnings = [], []
            for view in self.views:
                source = source_info[view]
                target = self.label_path(view)
                current = target.read_text(encoding="utf-8") if target.is_file() else None
                record = records.get(view)
                if not source["path"]:
                    if current is None:
                        atomic_text(target, "# 每行写一个类别；可用 类别: 说明\n")
                    continue
                desired = source["text"] or "# 每行写一个类别；可用 类别: 说明\n"
                desired_hash = text_hash(desired)
                if current is None:
                    atomic_text(target, desired)
                    records[view] = {"source_path": source["path"], "source_hash": source["hash"],
                                     "work_hash": desired_hash}
                    events.append(f"{view}: 已创建视角标签")
                    continue
                current_hash = text_hash(current)
                if current_hash == desired_hash:
                    records[view] = {"source_path": source["path"], "source_hash": source["hash"],
                                     "work_hash": current_hash}
                    continue
                source_changed = not record or record.get("source_hash") != source["hash"] or record.get("source_path") != source["path"]
                legacy_safe = (not record and source["mtime_ns"] > target.stat().st_mtime_ns
                               and not self.used_label_names(view))
                locally_unchanged = bool(record and current_hash == record.get("work_hash"))
                if not (force or (source_changed and (locally_unchanged or legacy_safe))):
                    if source_changed:
                        warnings.append(f"{view}: 原始标签已更新，但工作标签也有独立修改；点击“从原始标签同步”并检查后应用")
                    continue
                desired_names = {entry["name"] for entry in parse_labels(desired)[0]}
                removed_used = self.used_label_names(view) - desired_names
                if removed_used:
                    warnings.append(f"{view}: 原始标签删除了已有框使用的类别 {sorted(removed_used)}；请先处理这些框")
                    continue
                backup = self.work_dir / "label_backups" / f"labels_{view}_{time.time_ns()}.txt"
                atomic_text(backup, current)
                atomic_text(target, desired)
                records[view] = {"source_path": source["path"], "source_hash": source["hash"],
                                 "work_hash": desired_hash}
                events.append(f"{view}: 已同步原始标签；旧版备份于 {backup}")
                for item in self.items.values():
                    if item["view"] == view:
                        self.export_item(item["item_id"])
            atomic_json(sync_path, state)
            return {"events": events, "warnings": warnings,
                    "sources": {view: source_info[view]["path"] for view in self.views}}

    def label_path(self, view: str) -> Path:
        if view not in self.views:
            raise ValueError("未知视角")
        return self.work_dir / f"labels_{view}.txt"

    def labels(self, view: str) -> tuple[list[dict], dict[str, str]]:
        return parse_labels(self.label_path(view).read_text(encoding="utf-8"))

    def draft_path(self, item_id: str) -> Path:
        if item_id not in self.items:
            raise ValueError("未知视频")
        return self.work_dir / "drafts" / f"{item_id}.json"

    def get_draft(self, item_id: str) -> dict:
        path = self.draft_path(item_id)
        draft = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "revision": 0, "frames": {}, "notes": {}}
        draft.setdefault("points", {})
        return draft

    def prompt_path(self, item: dict) -> Path:
        episode = f"ep{item['episode_index']:06d}" if item["episode_index"] >= 0 else slug(item["item_id"])
        return self.work_dir / "prompts" / "json" / item["video_key"] / episode / "sam2_prompts.json"

    def preview_root(self, item_id: str) -> Path:
        if item_id not in self.items:
            raise ValueError("未知视频")
        return self.work_dir / "sam2_preview_integrated" / item_id

    def unet_root(self, item_id):
        if item_id not in self.items:
            raise ValueError('未知视频')
        return self.work_dir / 'unet_preview' / item_id

    def latest_unet(self, item_id):
        root = self.unet_root(item_id)
        if not (root / 'latest.json').is_file():
            return None
        record = json.loads((root / 'latest.json').read_text())
        run = record.get('run_id', '')
        if not re.fullmatch(r'[0-9a-f]{32}', run):
            return None
        output = root / run / 'output'
        if not all((output / name).is_file() for name in ('complete.json', 'mask_visual.mp4', 'overlay.mp4')):
            return None
        return {**json.loads((output / 'complete.json').read_text()), 'run_id': run, 'output_path': str(output)}

    def latest_preview(self, item_id: str) -> dict | None:
        latest = self.preview_root(item_id) / "latest.json"
        if not latest.is_file():
            return None
        record = json.loads(latest.read_text(encoding="utf-8"))
        run_id = record.get("run_id", "")
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            return None
        output = self.preview_root(item_id) / run_id / "output" / f"{Path(self.items[item_id]['video_path']).stem}_sam2_masks"
        if not (output / "complete.json").is_file() or not (output / "mask_visual.mp4").is_file():
            return None
        completion = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if not (completion.get("propagate_prompt_frames") is True
                and completion.get("conditioning_memory_preserved") is True
                and (completion.get("first_frame_reverse_refined") is True
                     or int(completion.get("frame_count", 0)) == 1)):
            return None
        prompt = self.prompt_path(self.items[item_id])
        current_hash = hashlib.sha256(prompt.read_bytes()).hexdigest() if prompt.is_file() else None
        return {"run_id": run_id, "output_path": str(output),
                "objects": [{k: obj[k] for k in ('id', 'name')} for obj in
                            json.loads((output / 'sam2_prompts.json').read_text())['objects']],
                "direction_review": direction_review.summary(output),
                "stale": current_hash != record.get("prompt_hash"),
                "settings": record.get("settings"),
                "timing_seconds": completion.get("timing_seconds"),
                "video_processing_fps": completion.get("video_processing_fps"),
                "frame_count": self.items[item_id]["frame_count"]}

    def export_item(self, item_id: str) -> None:
        item = self.items[item_id]
        draft = self.get_draft(item_id)
        labels, guidance = self.labels(item["view"])
        frames = draft["frames"]
        points_by_frame = draft["points"]
        path = self.prompt_path(item)
        objects = []
        for label in labels:
            name = label["name"]
            boxes = [{"frame": int(frame), "box": by_label[name]}
                     for frame, by_label in sorted(frames.items(), key=lambda entry: int(entry[0]))
                     if name in by_label]
            points = [{"frame": int(frame_key), "x": point["x"], "y": point["y"], "label": point["label"]}
                      for frame_key, by_label in sorted(points_by_frame.items(), key=lambda entry: int(entry[0]))
                      for point in by_label.get(name, [])]
            if boxes or any(point["label"] == 1 for point in points):
                objects.append({"id": label["id"], "name": name, "color": label["color"],
                                "completed": True, "points": points, "boxes": boxes})
        if not objects:
            path.unlink(missing_ok=True)
            return
        prompt_frames = sorted({entry["frame"] for obj in objects
                                for entry in [*obj["boxes"], *obj["points"]]})
        atomic_json(path, {"version": 1, "dataset": self.dataset.name, "task": item["task"],
                           "video_key": item["video_key"], "view_description": guidance.get(item["view"], ""),
                           "episode_index": item["episode_index"], "video_path": item["video_path"],
                           "frame_count": item["frame_count"], "prompt_frames": prompt_frames,
                           "labels_source": str(self.label_path(item["view"])), "objects": objects,
                           "review_notes": draft.get("notes", {})})

    def save_label_text(self, view: str, text: str) -> dict:
        with self.lock:
            labels, guidance = parse_labels(text)
            if not labels:
                raise ValueError("请至少写入一个类别")
            names = {entry["name"] for entry in labels}
            used = self.used_label_names(view)
            if used - names:
                raise ValueError(f"已有标注使用这些类别，不能删除: {sorted(used - names)}")
            atomic_text(self.label_path(view), text.rstrip() + "\n")
            for item in self.items.values():
                if item["view"] == view:
                    self.export_item(item["item_id"])
            return {"labels": labels, "guidance": guidance, "path": str(self.label_path(view))}

    def save_box(self, payload: dict) -> dict:
        item_id = str(payload.get("item_id", ""))
        item = self.items.get(item_id)
        if item is None:
            raise ValueError("未知视频")
        frame = int(payload.get("frame", -1))
        if not 0 <= frame < item["frame_count"]:
            raise ValueError("帧号超出范围")
        label = str(payload.get("label", ""))
        labels, _ = self.labels(item["view"])
        if label not in {entry["name"] for entry in labels}:
            raise ValueError("未知类别；请先保存视角标签")
        box = payload.get("box")
        if box is not None:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("包围盒应为 [x1,y1,x2,y2]")
            box = [int(value) for value in box]
            if not (0 <= box[0] < box[2] <= item["width"] and 0 <= box[1] < box[3] <= item["height"]):
                raise ValueError("包围盒超出画面或面积为零")
        with self.lock:
            draft = self.get_draft(item_id)
            if int(payload.get("revision", -1)) != draft["revision"]:
                raise RuntimeError("标注已被其他页面更新；请重新载入当前视频")
            boxes = draft["frames"].setdefault(str(frame), {})
            if box is None:
                boxes.pop(label, None)
            else:
                boxes[label] = box
            if not boxes:
                draft["frames"].pop(str(frame), None)
            draft["revision"] += 1
            atomic_json(self.draft_path(item_id), draft)
            self.export_item(item_id)
            return {"draft": draft, "prompt_path": str(self.prompt_path(item))}

    def save_point(self, payload: dict) -> dict:
        item_id = str(payload.get("item_id", ""))
        item = self.items.get(item_id)
        if item is None:
            raise ValueError("未知视频")
        frame = int(payload.get("frame", -1))
        if not 0 <= frame < item["frame_count"]:
            raise ValueError("帧号超出范围")
        name = str(payload.get("label", ""))
        labels, _ = self.labels(item["view"])
        if name not in {entry["name"] for entry in labels}:
            raise ValueError("未知类别；请先保存视角标签")
        action = str(payload.get("action", "add"))
        if action == "add":
            x, y = int(payload.get("x", -1)), int(payload.get("y", -1))
            point_label = int(payload.get("point_label", -1))
            if not (0 <= x < item["width"] and 0 <= y < item["height"]):
                raise ValueError("点坐标超出画面")
            if point_label not in (0, 1):
                raise ValueError("点类型必须是内部正点或排除负点")
        elif action == "delete":
            point_index = int(payload.get("point_index", -1))
        else:
            raise ValueError("未知点操作")
        with self.lock:
            draft = self.get_draft(item_id)
            if int(payload.get("revision", -1)) != draft["revision"]:
                raise RuntimeError("标注已被其他页面更新；请重新载入当前视频")
            frame_points = draft["points"].setdefault(str(frame), {})
            target = frame_points.setdefault(name, [])
            if action == "add":
                target.append({"x": x, "y": y, "label": point_label})
            else:
                if not 0 <= point_index < len(target):
                    raise ValueError("要删除的点不存在，请重新载入当前视频")
                target.pop(point_index)
            if not target:
                frame_points.pop(name, None)
            if not frame_points:
                draft["points"].pop(str(frame), None)
            draft["revision"] += 1
            atomic_json(self.draft_path(item_id), draft)
            self.export_item(item_id)
            return {"draft": draft, "prompt_path": str(self.prompt_path(item))}

    def save_note(self, payload: dict) -> dict:
        item_id = str(payload.get("item_id", ""))
        item = self.items.get(item_id)
        if item is None:
            raise ValueError("未知视频")
        frame = int(payload.get("frame", -1))
        if not 0 <= frame < item["frame_count"]:
            raise ValueError("帧号超出范围")
        note = str(payload.get("note", "")).strip()
        if len(note) > 2000:
            raise ValueError("备注最多 2000 字")
        with self.lock:
            draft = self.get_draft(item_id)
            if int(payload.get("revision", -1)) != draft["revision"]:
                raise RuntimeError("标注已被其他页面更新；请重新载入当前视频")
            if note:
                draft["notes"][str(frame)] = note
            else:
                draft["notes"].pop(str(frame), None)
            draft["revision"] += 1
            atomic_json(self.draft_path(item_id), draft)
            self.export_item(item_id)
            return {"draft": draft}

    def read_frame(self, item_id: str, frame: int) -> bytes:
        item = self.items.get(item_id)
        if item is None or not 0 <= frame < item["frame_count"]:
            raise ValueError("视频或帧号无效")
        cap = cv2.VideoCapture(item["video_path"])
        try:
            if not cap.isOpened():
                raise ValueError("无法读取视频")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"无法读取第 {frame} 帧；请检查视频帧数")
            success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                raise ValueError("帧编码失败")
            return encoded.tobytes()
        finally:
            cap.release()

    def bootstrap(self) -> dict:
        sync_result = self.sync_source_labels()
        items = []
        for item in self.items.values():
            draft = self.get_draft(item["item_id"])
            annotated_frames = set(draft["frames"]) | set(draft["points"])
            items.append({**item, "annotated_frames": len(annotated_frames),
                          "notes_count": len(draft.get("notes", {}))})
        return {"dataset": str(self.dataset), "work_dir": str(self.work_dir),
                "views": self.views, "items": items,
                "label_sources": sync_result["sources"],
                "label_sync_events": sync_result["events"],
                "label_sync_warnings": sync_result["warnings"],
                "labels": {view: {"text": self.label_path(view).read_text(encoding="utf-8"),
                                  "path": str(self.label_path(view)),
                                  "parsed": self.labels(view)[0]} for view in self.views}}


def default_work_dir(dataset: Path) -> Path:
    return APP_DIR.parent / "segdata" / "temp" / dataset.name / "manual_seed"


class EditorManager:
    def __init__(self, editor: Editor):
        self.editor = editor
        self.workspace_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.job: dict | None = None
        self.item_jobs: dict[str, dict] = {}

    def running_job(self, item_id=None):
        jobs = [self.job, *getattr(self, 'item_jobs', {}).values()]
        return next((j for j in jobs if j and j['state'] == 'running'
                     and (item_id is None or j['item_id'] == item_id)), None)

    def current(self) -> tuple[Editor, str]:
        with self.lock:
            return self.editor, self.workspace_id

    @staticmethod
    def propagation_options() -> dict:
        models = [{"name": name, "installed": (APP_DIR.parent / "checkpoints" / checkpoint).is_file()}
                  for name, (checkpoint, _) in MODEL_VARIANTS.items()]
        devices = [{"value": "auto", "label": "自动（默认 CUDA 设备）"},
                   {"value": "cpu", "label": "CPU（很慢）"}]
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            for row in result.stdout.splitlines():
                parts = [part.strip() for part in row.split(",", maxsplit=3)]
                if len(parts) == 4 and parts[0].isdigit():
                    devices.insert(-1, {"value": f"cuda:{parts[0]}",
                                        "label": f"GPU {parts[0]} · 负载 {parts[3]}% · 空闲 {parts[2]} MiB · {parts[1]}"})
        except (OSError, subprocess.SubprocessError):
            pass
        return {"models": models, "devices": devices}

    @staticmethod
    def require_workspace(provided: str, expected: str) -> None:
        if provided != expected:
            raise RuntimeError("标注数据集已切换；请刷新页面后继续")

    def bootstrap(self) -> dict:
        editor, workspace_id = self.current()
        return {**editor.bootstrap(), "workspace_id": workspace_id,
                "propagation_options": self.propagation_options()}

    def sync_labels(self, workspace_id: str) -> dict:
        editor, expected = self.current()
        self.require_workspace(workspace_id, expected)
        result = editor.sync_source_labels(force=True)
        return {**editor.bootstrap(), "workspace_id": expected, "sync_result": result,
                "propagation_options": self.propagation_options()}

    def open(self, payload: dict) -> dict:
        with self.lock:
            if self.running_job():
                raise RuntimeError("仍有传播、合并或修正任务正在运行；请等待完成后切换数据集")
        dataset_text = str(payload.get("dataset", "")).strip()
        if not dataset_text:
            raise ValueError("请输入服务器上的数据集目录")
        dataset = Path(dataset_text).expanduser().resolve()
        work_text = str(payload.get("work_dir", "")).strip()
        work_dir = Path(work_text).expanduser().resolve() if work_text else default_work_dir(dataset)
        views_text = str(payload.get("views", "")).strip()
        views = [part.strip() for part in views_text.split(",")] if views_text else None
        if views is not None and (len(views) != 2 or any(not part for part in views)):
            raise ValueError("视角需填写两个名称，用英文逗号分隔，例如 front,side")
        candidate = Editor(dataset, work_dir, views)
        with self.lock:
            if self.running_job():
                raise RuntimeError("仍有传播、合并或修正任务正在运行；请等待完成后切换数据集")
            self.editor = candidate
            self.workspace_id = uuid.uuid4().hex
            self.job = None
            self.item_jobs = {}
            workspace_id = self.workspace_id
        return {**candidate.bootstrap(), "workspace_id": workspace_id,
                "propagation_options": self.propagation_options()}

    def start_unet(self, payload):
        with self.lock:
            self.require_workspace(str(payload.get('workspace_id', '')), self.workspace_id)
            item_id = str(payload.get('item_id', ''))
            if item_id not in self.editor.items:
                raise ValueError('未知视频')
            editor = self.editor
            item = editor.items[item_id]
            if item['view'] != 'front':
                raise ValueError('当前 U-Net 仅支持 front 前视角')
            config = json.loads((UNET_MODEL_DIR / 'model_config.json').read_text())
            if [o['name'] for o in editor.labels(item['view'])[0]] != config['labels'][1:]:
                raise ValueError('当前数据集类别或顺序与 U-Net 模型不一致')
            if (self.job and self.job['state'] == 'running') or self.running_job(item_id):
                raise RuntimeError('已有分割任务或当前视频的修改任务运行中，请等待完成')
            filter_existing = payload.get('filter_existing', False)
            if type(filter_existing) is not bool:
                raise ValueError('过滤参数必须为布尔值')
            source = editor.latest_unet(item_id) if filter_existing else None
            if filter_existing and not source:
                raise ValueError('当前视频没有可过滤的U-Net预测')
            device = str(payload.get('device', 'cuda:1'))
            if filter_existing:
                device = 'cpu'
            if device not in {d['value'] for d in self.propagation_options()['devices']}:
                raise ValueError('计算设备不可用，请刷新页面')
            if device == 'auto':
                device = 'cuda:0'
            run_id = uuid.uuid4().hex
            root = editor.unet_root(item_id) / run_id
            root.mkdir(parents=True, exist_ok=False)
            job = {'kind': 'unet', 'item_id': item_id, 'workspace_id': self.workspace_id,
                   'run_id': run_id, 'state': 'running', 'started_at': time.time(),
                   'log_path': str(root / 'infer.log'), 'settings': {'model': 'ResNet18 + U-Net', 'device': device, 'filter_existing': filter_existing}}
            self.job = job
            command = [sys.executable, str(APP_DIR.parent / 'tools/unet_video_preview.py'),
                       '--model-dir', str(UNET_MODEL_DIR), '--input', item['video_path'],
                       '--output', str(root / 'output'), '--device', device]
            if source:
                command += ['--source-output', source['output_path']]
            def worker():
                try:
                    with (root / 'infer.log').open('w') as log:
                        result = subprocess.run(command, cwd=APP_DIR.parent, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'U-Net 分割失败（退出码 {result.returncode}），请查看日志')
                    completion = json.loads((root / 'output/complete.json').read_text())
                    if completion['frame_count'] != item['frame_count']:
                        raise RuntimeError('U-Net 输出帧数与原视频不一致')
                    atomic_json(editor.unet_root(item_id) / 'latest.json', {'run_id': run_id})
                    with self.lock:
                        job['state'] = 'complete'
                except Exception as exc:
                    with self.lock:
                        job.update(state='failed', error=str(exc))
            threading.Thread(target=worker, daemon=True).start()
            return {'state': 'running', 'run_id': run_id}

    def unet_video(self, workspace_id, item_id, run_id, mode):
        with self.lock:
            self.require_workspace(workspace_id, self.workspace_id)
            preview = self.editor.latest_unet(item_id)
            if not preview or preview['run_id'] != run_id:
                raise RuntimeError('U-Net 版本已变化，请刷新')
            if mode not in ('overlay', 'segmentation'):
                raise ValueError('无效显示模式')
            return Path(preview['output_path']) / ('overlay.mp4' if mode == 'overlay' else 'mask_visual.mp4')

    def start_propagation(self, workspace_id: str, item_id: str,
                          requested: dict | None = None) -> dict:
        requested = requested or {}
        model = str(requested.get("model", "base_plus"))
        if model not in MODEL_VARIANTS:
            raise ValueError("未知 SAM2 模型")
        checkpoint_name, config = MODEL_VARIANTS[model]
        checkpoint = APP_DIR.parent / "checkpoints" / checkpoint_name
        if not checkpoint.is_file():
            raise ValueError(f"模型权重未安装：{checkpoint}")
        options = self.propagation_options()
        device = str(requested.get("device", "auto"))
        if device not in {row["value"] for row in options["devices"]}:
            raise ValueError("GPU 设备不可用，请刷新页面")
        compile_mode = str(requested.get("compile_mode", "none"))
        if compile_mode not in ("none", "encoder", "full"):
            raise ValueError("未知编译模式")
        if device == "cpu" and compile_mode != "none":
            raise ValueError("CPU 模式不能启用模型编译")
        video_cpu = requested.get("offload_video_to_cpu", True)
        state_cpu = requested.get("offload_state_to_cpu", False)
        if not isinstance(video_cpu, bool) or not isinstance(state_cpu, bool):
            raise ValueError("CPU 卸载选项必须是布尔值")
        settings = {"model": model, "checkpoint": str(checkpoint), "config": config,
                    "propagation_method": "integrated_bidirectional_review_v1",
                    "save_bidirectional": True,
                    "propagate_prompt_frames": True,
                    "device": device, "compile_mode": compile_mode,
                    "offload_video_to_cpu": video_cpu,
                    "offload_state_to_cpu": state_cpu}
        with self.lock:
            self.require_workspace(workspace_id, self.workspace_id)
            editor = self.editor
            if item_id not in editor.items:
                raise ValueError("未知视频")
            if self.job and self.job["state"] == "running":
                raise RuntimeError("已有视频正在分割或传播，请等待它完成")
            if self.running_job(item_id):
                raise RuntimeError('当前视频正在合并或保存修正，请等待完成后传播')
            prompt = editor.prompt_path(editor.items[item_id])
            if not prompt.is_file():
                raise ValueError("当前视频没有可传播的提示；请先保存框或正点")
            prompt_bytes = prompt.read_bytes()
            run_id = uuid.uuid4().hex
            run_root = editor.preview_root(item_id) / run_id
            run_root.mkdir(parents=True, exist_ok=False)
            snapshot = run_root / "input" / "sam2_prompts.json"
            snapshot.parent.mkdir()
            snapshot.write_bytes(prompt_bytes)
            atomic_json(run_root / "settings.json", settings)
            job = {"item_id": item_id, "workspace_id": workspace_id, "run_id": run_id,
                   "state": "running", "started_at": time.time(), "kind": "propagation",
                   "prompt_hash": hashlib.sha256(prompt_bytes).hexdigest(),
                   "settings": settings,
                   "log_path": str(run_root / "propagate.log")}
            self.job = job
            threading.Thread(target=self._run_propagation,
                             args=(editor, job, snapshot, run_root), daemon=True).start()
            return self.propagation_status(workspace_id, item_id)

    def _run_propagation(self, editor: Editor, job: dict, snapshot: Path, run_root: Path) -> None:
        settings = job["settings"]
        command = [sys.executable, str(PROPAGATE_SCRIPT), "--prompt-json", str(snapshot),
                   "--propagate-prompt-frames",
                   "--save-bidirectional",
                   "--output-root", str(run_root / "output"),
                   "--sam2-checkpoint", settings["checkpoint"],
                   "--sam2-config", settings["config"],
                   "--device", settings["device"], "--compile-mode", settings["compile_mode"],
                   "--offload-video-to-cpu" if settings["offload_video_to_cpu"] else "--no-offload-video-to-cpu",
                   "--offload-state-to-cpu" if settings["offload_state_to_cpu"] else "--no-offload-state-to-cpu"]
        try:
            with Path(job["log_path"]).open("w", encoding="utf-8") as log:
                log.write("Running: " + " ".join(command) + "\n")
                log.flush()
                result = subprocess.run(command, cwd=APP_DIR.parent, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            output = run_root / "output" / f"{Path(editor.items[job['item_id']]['video_path']).stem}_sam2_masks"
            if result.returncode or not (output / "complete.json").is_file():
                raise RuntimeError(f"SAM2 传播失败（退出码 {result.returncode}）；查看日志")
            completion = json.loads((output / "complete.json").read_text(encoding="utf-8"))
            if not (completion.get("propagate_prompt_frames") is True
                    and completion.get("conditioning_memory_preserved") is True
                    and (completion.get("first_frame_reverse_refined") is True
                         or int(completion.get("frame_count", 0)) == 1)):
                raise RuntimeError("传播结果未完成标准整合流程；未更新预览")
            atomic_json(editor.preview_root(job["item_id"]) / "latest.json",
                        {"run_id": job["run_id"], "prompt_hash": job["prompt_hash"],
                         "completed_at": time.time(), "settings": settings})
            with self.lock:
                job["state"] = "complete"
        except Exception as exc:
            with self.lock:
                job["state"] = "failed"
                job["error"] = str(exc)

    def propagation_status(self, workspace_id: str, item_id: str) -> dict:
        with self.lock:
            self.require_workspace(workspace_id, self.workspace_id)
            if item_id not in self.editor.items:
                raise ValueError("未知视频")
            local_jobs = [j for j in [self.job, self.item_jobs.get(item_id)]
                          if j and j['workspace_id'] == workspace_id and j['item_id'] == item_id]
            job = max(local_jobs, key=lambda j: j['started_at'], default=None)
            summary = {key: job[key] for key in ("item_id", "run_id", "state", "started_at", "error", "settings", "kind")
                       if job and key in job} if job else None
            if summary and job.get("log_path") and Path(job["log_path"]).is_file():
                summary["log_tail"] = Path(job["log_path"]).read_text(encoding="utf-8", errors="replace")[-2500:]
            propagation = self.job if self.job and self.job['workspace_id'] == workspace_id else None
            propagation_summary = {key: propagation[key] for key in ('item_id', 'run_id', 'state', 'kind', 'started_at') if key in propagation} if propagation else None
            return {"job": summary, "propagation_job": propagation_summary, "supports_batch_fill_holes": True,
                    "supports_unet": (UNET_MODEL_DIR / 'model.ts').is_file(),
                    "unet_filter_version": direction_review.FILTER_VERSION,
                    "unet_preview": self.editor.latest_unet(item_id),
                    "preview": self.editor.latest_preview(item_id)}

    def review_action(self, payload: dict, finalize: bool = False, reanalyze: bool = False) -> dict:
        with self.lock:
            self.require_workspace(str(payload.get("workspace_id", "")), self.workspace_id)
            item_id = str(payload.get("item_id", ""))
            preview = self.editor.latest_preview(item_id)
            if not preview or preview['run_id'] != payload.get('run_id'):
                raise RuntimeError('传播版本已变化，请刷新后再选择')
            if preview['stale']:
                raise RuntimeError('提示已经修改，请重新传播后再确认最终结果')
            if self.running_job(item_id):
                raise RuntimeError('当前视频的任务正在运行，请等待完成')
            root = Path(preview['output_path'])
            current_review = direction_review.summary(root)
            if 'review_revision' in payload and (not current_review or payload['review_revision'] != current_review['revision']):
                raise RuntimeError('候选结果或选择已变化，请刷新后重试')
            if finalize and current_review.get('candidate_filter', {}).get('version') != direction_review.FILTER_VERSION:
                reanalyze = True  # First upgrade old candidates; require confirmation on the new report.
            if not finalize and not reanalyze:
                return direction_review.choose(root, str(payload.get('direction', '')), payload.get('interval_id'))
            review = direction_review.summary(root)
            if not review or (review['unresolved'] and not reanalyze):
                raise ValueError('请先为全部冲突片段选择方向')
            job = {'item_id': item_id, 'workspace_id': self.workspace_id, 'run_id': preview['run_id'],
                   'state': 'running', 'started_at': time.time(), 'settings': preview.get('settings'),
                   'kind': 'reanalysis' if reanalyze else 'finalize'}
            self.item_jobs[item_id] = job
            def worker():
                try:
                    if reanalyze:
                        direction_review.filter_candidates(root)
                    else:
                        direction_review.finalize(root)
                    with self.lock:
                        job['state'] = 'complete'
                except Exception as exc:
                    with self.lock:
                        job.update(state='failed', error=str(exc))
            threading.Thread(target=worker, daemon=True).start()
            return {'state': 'running'}

    def mark_revision_frame(self, payload):
        with self.lock:
            self.require_workspace(str(payload.get('workspace_id', '')), self.workspace_id)
            preview = self.editor.latest_preview(str(payload.get('item_id', '')))
            if not preview or preview['run_id'] != payload.get('run_id') or preview['stale']:
                raise RuntimeError('传播版本已变化，请刷新')
            root = Path(preview['output_path'])
            frame = payload.get('frame')
            review = direction_review.summary(root)
            if type(frame) is not int or not 0 <= frame < review['frame_count']:
                raise ValueError('无效帧号')
            path = root / 'manual_revision_frames.json'
            data = json.loads(path.read_text()) if path.is_file() else {'frames': []}
            marks = {m['frame']: m for m in data['frames']}
            action = payload.get('action', 'add')
            if action == 'remove':
                marks.pop(frame, None)
            elif action in ('add', 'done'):
                marks[frame] = {'frame': frame, 'done': action == 'done'}
            else:
                raise ValueError('无效标记操作')
            data['frames'] = [marks[k] for k in sorted(marks)]
            direction_review.write_json(path, data)
            return data

    def component_preview(self, payload):
        with self.lock:
            self.require_workspace(str(payload.get('workspace_id', '')), self.workspace_id)
            preview = self.editor.latest_preview(str(payload.get('item_id', '')))
            if not preview or preview['run_id'] != payload.get('run_id') or preview['stale']:
                raise RuntimeError('候选或提示已变化，请刷新后重试')
            root = Path(preview['output_path'])
            review = direction_review.summary(root)
            if not review or review['revision'] != payload.get('review_revision'):
                raise RuntimeError('候选版本已变化，请重新载入编辑帧')
            # Preview with no operations is valid; correction itself still requires a change request.
            options = direction_review.correction_options(root, {**payload, 'fill_area': 1})
            if options['start'] != options['end']:
                raise ValueError('编辑预览仅支持单帧')
            if options['direction'] != 'final':
                raise ValueError('请在合并结果上进行单帧修正')
            path = direction_review.final_root(root, review) / 'mask_frames' / f"{options['start']:05d}.png"
            mask = cv2.imread(str(path), 0)
            if mask is None:
                raise ValueError('无法读取掩码')
            if payload.get('frame_edits'):
                edits = direction_review.multiple_correction_options(root, {'edits': payload['frame_edits']})['edits']
                for entry in edits:
                    if entry['start'] != options['start']:
                        raise ValueError('预览草稿必须为同一帧')
                    mask, _, _ = direction_review.repair_mask(mask, entry['class_id'], entry['remove_area'], entry['fill_area'], entry['keep_largest'])
                    mask = direction_review.edit_components(mask, entry['class_id'], entry['operations'])
                result = mask
            else:
                result = direction_review.edit_components(mask, options['class_id'], options['operations'])
            return direction_review.component_preview(result == options['class_id'])

    def scan_final(self, payload):
        with self.lock:
            self.require_workspace(str(payload.get('workspace_id', '')), self.workspace_id)
            item_id = str(payload.get('item_id', ''))
            preview = self.editor.latest_preview(item_id)
            if not preview or preview['stale'] or preview['run_id'] != payload.get('run_id'):
                raise RuntimeError('结果版本已变化，请刷新')
            root = Path(preview['output_path'])
            review = direction_review.summary(root)
            if not review or review['revision'] != payload.get('review_revision'):
                raise RuntimeError('结果版本已变化，请刷新')
            source = direction_review.final_root(root, review)
            options = direction_review.correction_options(root, {**payload, 'direction': 'final', 'start': 0, 'end': 0, 'remove_area': 0, 'fill_area': 1})
            thresholds = {k: float(payload.get(k, default)) for k, default in
                          [('iou', .65), ('area_change', .5), ('centroid_shift', .1), ('min_pixels', 16)]}
            if any(not 0 <= thresholds[k] <= 1 for k in ('iou', 'area_change', 'centroid_shift')) or not 1 <= thresholds['min_pixels'] <= 100_000_000:
                raise ValueError('突变检测阈值无效')
            key = hashlib.sha256(json.dumps([options['class_id'], thresholds], sort_keys=True).encode()).hexdigest()[:16]
            cache = source / 'quality_scan' / f'{key}.json'
            if cache.is_file():
                return {'state': 'complete', 'report': json.loads(cache.read_text()), 'final_subdir': review['final_subdir']}
            prior = self.item_jobs.get(item_id)
            if prior and prior.get('scan_cache') == str(cache) and prior['state'] == 'failed' and not payload.get('retry'):
                return {'state': 'failed', 'error': prior.get('error')}
            if self.running_job(item_id):
                return {'state': 'running'}
            job = {'item_id': item_id, 'workspace_id': self.workspace_id, 'run_id': preview['run_id'],
                   'state': 'running', 'started_at': time.time(), 'kind': 'scan', 'scan_cache': str(cache)}
            self.item_jobs[item_id] = job
            def worker():
                try:
                    report = direction_review.scan_final(source, options['class_id'], review['frame_count'], thresholds)
                    cache.parent.mkdir(exist_ok=True)
                    direction_review.write_json(cache, report)
                    with self.lock:
                        job['state'] = 'complete'
                except Exception as exc:
                    with self.lock:
                        job.update(state='failed', error=str(exc))
            threading.Thread(target=worker, daemon=True).start()
            return {'state': 'running'}

    def correct_candidate(self, payload: dict, undo=False) -> dict:
        with self.lock:
            self.require_workspace(str(payload.get('workspace_id', '')), self.workspace_id)
            item_id = str(payload.get('item_id', ''))
            preview = self.editor.latest_preview(item_id)
            if not preview or preview['run_id'] != payload.get('run_id'):
                raise RuntimeError('传播版本已变化，请刷新后重试')
            if preview['stale']:
                raise RuntimeError('提示已修改，请重新传播后修正')
            if self.running_job(item_id):
                raise RuntimeError('当前视频的任务正在运行，请等待完成')
            root = Path(preview['output_path'])
            review = direction_review.summary(root)
            if not review or payload.get('review_revision') != review['revision']:
                raise RuntimeError('候选版本已变化，请刷新后重试')
            if undo:
                return direction_review.undo_final_correction(root)
            options = direction_review.multiple_correction_options(root, payload) if 'edits' in payload else direction_review.correction_options(root, payload)
            direction_review.final_root(root, review)
            if options['direction'] != 'final' or (options['start'] != options['end'] and not options.get('batch_keep_largest') and not options.get('batch_fill_holes') and not options.get('edits')):
                raise ValueError('仅允许修正合并后的单帧结果')
            job = {'item_id': item_id, 'workspace_id': self.workspace_id, 'run_id': preview['run_id'],
                   'state': 'running', 'started_at': time.time(), 'settings': preview.get('settings'),
                   'kind': 'correction'}
            self.item_jobs[item_id] = job
            def worker():
                try:
                    direction_review.correct_final(root, options)
                    with self.lock:
                        job['state'] = 'complete'
                except Exception as exc:
                    with self.lock:
                        job.update(state='failed', error=str(exc))
            threading.Thread(target=worker, daemon=True).start()
            return {'state': 'running'}

    def preview_file(self, workspace_id: str, item_id: str, kind: str, frame: int = -1,
                     source: str = 'final', run_id: str = '') -> Path:
        editor, expected = self.current()
        self.require_workspace(workspace_id, expected)
        preview = editor.latest_preview(item_id)
        if not preview:
            raise FileNotFoundError("当前视频尚无传播预览")
        if run_id and run_id != preview['run_id']:
            raise RuntimeError('预览版本已变化，请刷新')
        output = Path(preview["output_path"])
        if source not in ('forward', 'reverse', 'final'):
            raise ValueError('无效的预览方向')
        review = preview.get('direction_review')
        if review:
            if source == 'final':
                if review['status'] != 'finalized':
                    raise ValueError('请先人工选定并生成最终结果')
                output = output / review['final_subdir']
            else:
                output = direction_review.candidate_root(output, review)
                if kind == 'video':
                    return output / f'mask_visual_{source}.mp4'
                if kind == 'frame' and 0 <= frame < preview['frame_count']:
                    return output / 'directional_mask_frames' / source / f'{frame:05d}.png'
        elif source != 'final':
            raise ValueError('此版本没有双向候选，请重新传播')
        if kind == "video":
            return output / "mask_visual.mp4"
        if kind == "frame" and 0 <= frame < preview["frame_count"]:
            return output / "mask_frames" / f"{frame:05d}.png"
        raise ValueError("预览类型或帧号无效")

    def difference_frame(self, workspace_id, item_id, frame, run_id, candidate):
        with self.lock:
            self.require_workspace(workspace_id, self.workspace_id)
            preview = self.editor.latest_preview(item_id)
            if not preview or not preview.get('direction_review') or preview['run_id'] != run_id:
                raise ValueError('需要当前版本的双向候选结果')
            review = preview['direction_review']
            if candidate != review.get('candidate_revision', 'original'):
                raise RuntimeError('候选版本已变化，请刷新')
            if not 0 <= frame < preview['frame_count']:
                raise ValueError('帧号无效')
            root = direction_review.candidate_root(Path(preview['output_path']), review)
            masks = [cv2.imread(str(root / 'directional_mask_frames' / d / f'{frame:05d}.png'), 0) for d in ('forward', 'reverse')]
            if any(m is None for m in masks) or masks[0].shape != masks[1].shape:
                raise ValueError('正反向掩码缺失或尺寸不一致')
            a, b = masks
            forward = (a != 0) & (b == 0)
            reverse = (a == 0) & (b != 0)
            changed_class = (a != 0) & (b != 0) & (a != b)
            rgba = np.zeros((*a.shape, 4), np.uint8)
            rgba[forward] = (60, 100, 255, 255)
            rgba[reverse] = (255, 200, 30, 255)
            rgba[changed_class] = (255, 80, 210, 255)
            ok, encoded = cv2.imencode('.png', rgba)
            if not ok:
                raise RuntimeError('差异图编码失败')
            return {'frame': frame, 'candidate': candidate, 'different_pixels': int((a != b).sum()),
                    'forward_only': int(forward.sum()), 'reverse_only': int(reverse.sum()),
                    'class_mismatch': int(changed_class.sum()),
                    'image': 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')}

    def preview_frame(self, workspace_id: str, item_id: str, frame: int,
                      label_name: str = "", source: str = 'final', run_id: str = '') -> bytes:
        mask_path = self.preview_file(workspace_id, item_id, "frame", frame, source, run_id)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        prompt_path = Path(self.editor.latest_preview(item_id)['output_path']) / 'sam2_prompts.json'
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        for obj in prompt["objects"]:
            if label_name and obj["name"] != label_name:
                continue
            color = str(obj.get("color", "#ffffff"))
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                color = "#ffffff"
            rgb = bytes.fromhex(color[1:])
            selected = mask == int(obj["id"])
            rgba[selected, :3] = tuple(rgb[::-1])
            rgba[selected, 3] = 255
        ok, encoded = cv2.imencode(".png", rgba)
        if not ok:
            raise RuntimeError("预览帧编码失败")
        return encoded.tobytes()


def make_handler(manager: EditorManager):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def send_video(self, path: Path) -> None:
            size = path.stat().st_size
            requested = self.headers.get("Range", "")
            start, end = 0, size - 1
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if not match or (not match[1] and not match[2]):
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                if match[1]:
                    start = int(match[1])
                    if match[2]:
                        end = min(int(match[2]), size - 1)
                else:
                    start = max(0, size - int(match[2]))
                if start >= size or end < start:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
            self.send_response(HTTPStatus.PARTIAL_CONTENT if requested else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as source:
                source.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_HEAD(self) -> None:
            if urlsplit(self.path).path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            html_size = (APP_DIR / "seed_bbox_web.html").stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(html_size))
            self.end_headers()

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    self.send_bytes((APP_DIR / "seed_bbox_web.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == '/round-review':
                    latest = ROUND_REVIEW_ROOT / 'contact_filter_r1' / 'unet_global_r2'
                    if (latest / 'complete.json').is_file():
                        target = 'contact_filter_r1/unet_global_r2/index.html'
                    else:
                        prefix = 'contact_filter_r1/' if (latest.parent / 'unet_global_r1' / 'complete.json').is_file() else ''
                        target = prefix + 'unet_global_r1/index.html'
                    self.send_response(302)
                    self.send_header('Location', '/round-data/' + target)
                    self.end_headers()
                elif url.path.startswith('/round-data/'):
                    from urllib.parse import unquote
                    import mimetypes
                    relative = unquote(url.path[len('/round-data/'):])
                    data_root = ROUND_REVIEW_ROOT
                    round_roots = {'pseudo8_r2': ROUND_REVIEW_ROOT.parent / 'unet_pseudo8_r2',
                                   'pseudo24_r3': ROUND_REVIEW_ROOT.parent / 'unet_pseudo24_r3'}
                    alias = relative.split('/', 1)[0]
                    if alias in round_roots:
                        data_root = round_roots[alias]
                        relative = relative[len(alias)+1:]
                    path = (data_root / relative).resolve()
                    # The six reused U-Net folders are read-only links to the prior round.
                    allowed_roots = [data_root.resolve(), ROUND_REVIEW_ROOT.resolve(), round_roots['pseudo8_r2'].resolve()]
                    if not any(path.is_relative_to(r) for r in allowed_roots) or not path.is_file():
                        raise ValueError('无效的本轮文件路径')
                    if path.suffix == '.mp4':
                        self.send_video(path)
                    else:
                        self.send_bytes(path.read_bytes(), mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
                elif url.path == "/api/bootstrap":
                    self.send_json(manager.bootstrap())
                elif url.path == "/api/item":
                    editor, workspace_id = manager.current()
                    manager.require_workspace(query.get("workspace_id", [""])[0], workspace_id)
                    item_id = query.get("id", [""])[0]
                    self.send_json({"item": editor.items[item_id], "draft": editor.get_draft(item_id),
                                    "prompt_path": str(editor.prompt_path(editor.items[item_id]))})
                elif url.path == '/api/source-video':
                    editor, workspace_id = manager.current()
                    manager.require_workspace(query.get('workspace_id', [''])[0], workspace_id)
                    self.send_video(Path(editor.items[query.get('id', [''])[0]]['video_path']))
                elif url.path == "/api/frame":
                    editor, workspace_id = manager.current()
                    manager.require_workspace(query.get("workspace_id", [""])[0], workspace_id)
                    self.send_bytes(editor.read_frame(query.get("id", [""])[0], int(query.get("frame", ["-1"])[0])), "image/jpeg")
                elif url.path == "/api/propagation":
                    self.send_json(manager.propagation_status(query.get("workspace_id", [""])[0], query.get("id", [""])[0]))
                elif url.path == '/api/unet-video':
                    self.send_video(manager.unet_video(query.get('workspace_id', [''])[0],
                        query.get('id', [''])[0], query.get('run', [''])[0], query.get('mode', ['overlay'])[0]))
                elif url.path == '/api/difference-frame':
                    self.send_json(manager.difference_frame(query.get('workspace_id', [''])[0], query.get('id', [''])[0],
                                   int(query.get('frame', ['-1'])[0]), query.get('run', [''])[0], query.get('candidate', ['original'])[0]))
                elif url.path == "/api/preview-frame":
                    self.send_bytes(manager.preview_frame(query.get("workspace_id", [""])[0],
                                                          query.get("id", [""])[0],
                                                          int(query.get("frame", ["-1"])[0]),
                                                          query.get("label", [""])[0],
                                                          query.get("source", ["final"])[0],
                                                          query.get("run", [""])[0]), "image/png")
                elif url.path == "/api/preview-video":
                    self.send_video(manager.preview_file(query.get("workspace_id", [""])[0],
                                                         query.get("id", [""])[0], "video",
                                                         source=query.get("source", ["final"])[0],
                                                         run_id=query.get("run", [""])[0]))
                else:
                    self.send_json({"error": "路径不存在"}, HTTPStatus.NOT_FOUND)
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 1_000_000:
                    raise ValueError("请求大小无效")
                payload = json.loads(self.rfile.read(size))
                if self.path == "/api/open":
                    result = manager.open(payload)
                elif self.path == "/api/sync-labels":
                    result = manager.sync_labels(str(payload.get("workspace_id", "")))
                elif self.path == "/api/propagate":
                    result = manager.start_propagation(str(payload.get("workspace_id", "")),
                                                       str(payload.get("item_id", "")), payload)
                elif self.path == '/api/unet-segment':
                    result = manager.start_unet(payload)
                elif self.path in ('/api/review-choice', '/api/review-finalize', '/api/review-reanalyze'):
                    result = manager.review_action(payload, finalize=self.path.endswith('review-finalize'), reanalyze=self.path.endswith('review-reanalyze'))
                elif self.path in ('/api/correct-candidate', '/api/undo-correction'):
                    result = manager.correct_candidate(payload, undo=self.path.endswith('undo-correction'))
                elif self.path == '/api/component-preview':
                    result = manager.component_preview(payload)
                elif self.path == '/api/revision-frame':
                    result = manager.mark_revision_frame(payload)
                elif self.path == '/api/final-scan':
                    result = manager.scan_final(payload)
                elif self.path in ("/api/labels", "/api/box", "/api/point", "/api/note"):
                    editor, workspace_id = manager.current()
                    manager.require_workspace(str(payload.get("workspace_id", "")), workspace_id)
                    if self.path == "/api/labels":
                        result = editor.save_label_text(str(payload.get("view", "")), str(payload.get("text", "")))
                    elif self.path == "/api/box":
                        result = editor.save_box(payload)
                    elif self.path == "/api/point":
                        result = editor.save_point(payload)
                    else:
                        result = editor.save_note(payload)
                else:
                    self.send_json({"error": "路径不存在"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--views", help="两个视频键或短视角名，逗号分隔；默认用数据集全部视角")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        parser.error(f"数据集不存在: {dataset}")
    work_dir = args.work_dir or default_work_dir(dataset)
    views = [value.strip() for value in args.views.split(",")] if args.views else None
    editor = Editor(dataset, work_dir, views)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(EditorManager(editor)))
    print(f"初始框标注: http://{args.host}:{args.port}", flush=True)
    print(f"数据集: {dataset}; 视角: {', '.join(editor.views)}; 视频: {len(editor.items)}", flush=True)
    print(f"输出: {editor.work_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
