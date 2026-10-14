#!/usr/bin/env python3
"""Run one quality-filtered U-Net iteration on eight front-view videos.

This orchestrator is deliberately resumable and never overwrites its source
round.  It evaluates teacher masks, builds a cache from accepted refined masks,
trains a fresh ImageNet-ResNet18 U-Net, evaluates the fixed seed references,
re-segments the same videos, and produces a before/after research report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sam2_direction_review import write_json
from train_reviewed_resnet_unet import LABELS
from train_unet_pseudo_iteration import SEEDS, seed_test
from unet_video_preview import run as segment_video
from video_segmentation_quality import POLICY, run_round


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()


def _load(path: Path):
    return json.loads(path.read_text())


def build_cache(source_round: Path, quality_root: Path, model_dir: Path, split_seed: int) -> Path:
    cache = model_dir / "cache"
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest = _load(manifest_path)
        if manifest.get("quality_policy_version") != POLICY["version"]:
            raise ValueError("Existing cache was built with another quality policy")
        return cache
    source = _load(source_round / "manifest.json")
    accepted: list[tuple[int, int, float]] = []
    excluded: list[dict] = []
    for record in source["records"]:
        episode = int(record["item"]["episode_index"])
        selection = _load(quality_root / record["id"] / "selection.json")["frames"]
        if len(selection) != int(record["item"]["frame_count"]):
            raise ValueError(f"Selection length mismatch: {record['id']}")
        for row in selection:
            if row["accepted"]:
                accepted.append((episode, int(row["frame"]), float(row["quality_score"])))
            else:
                excluded.append({"episode": episode, **row})
    rng = random.Random(split_seed)
    shuffled = accepted.copy(); rng.shuffle(shuffled)
    train_count = int(0.8 * len(shuffled))
    assignment = {(ep, frame): ("train" if index < train_count else "val")
                  for index, (ep, frame, _) in enumerate(shuffled)}
    score_by_frame = {(ep, frame): score for ep, frame, score in accepted}
    if set(assignment) & {(ep, frame) for ep in SEEDS for frame in range(1)}:
        raise AssertionError("Seed episode leaked into quality cache")
    cache.mkdir(parents=True, exist_ok=False)
    counts = {split: np.zeros(len(LABELS), np.int64) for split in ("train", "val")}
    samples = []
    for record in source["records"]:
        item = record["item"]
        episode = int(item["episode_index"])
        if episode in SEEDS:
            raise AssertionError("Seed video cannot enter pseudo-label training")
        cap = cv2.VideoCapture(item["video_path"])
        if not cap.isOpened():
            raise ValueError(item["video_path"])
        try:
            for frame in range(int(item["frame_count"])):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError((episode, frame, "unreadable RGB"))
                pair = (episode, frame)
                if pair not in assignment:
                    continue
                mask_path = source_round / record["id"] / "unet" / "mask_frames" / f"{frame:05d}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None or mask.shape != bgr.shape[:2] or mask.dtype != np.uint8 or mask.max() >= len(LABELS):
                    raise ValueError((episode, frame, "invalid pseudo mask"))
                split = assignment[pair]
                rel = Path(split) / item["video_key"] / f"file-{episode:03d}" / f"{frame:05d}"
                image_out = cache / "images" / rel.with_suffix(".jpg")
                mask_out = cache / "masks" / rel.with_suffix(".png")
                image_out.parent.mkdir(parents=True, exist_ok=True)
                mask_out.parent.mkdir(parents=True, exist_ok=True)
                resized_image = cv2.resize(bgr, (600, 456), interpolation=cv2.INTER_AREA)
                resized_mask = cv2.resize(mask, (600, 456), interpolation=cv2.INTER_NEAREST)
                if not cv2.imwrite(str(image_out), resized_image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(image_out)
                if not cv2.imwrite(str(mask_out), resized_mask):
                    raise RuntimeError(mask_out)
                counts[split] += np.bincount(resized_mask.ravel(), minlength=len(LABELS))
                samples.append({
                    "episode": episode, "frame": frame, "split": split,
                    "quality_score": score_by_frame[pair],
                    "source_mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                })
        finally:
            cap.release()
        print("CACHE COMPLETE", record["id"], flush=True)
    if len(samples) != len(accepted):
        raise AssertionError((len(samples), len(accepted)))
    manifest = {
        "labels": LABELS, "seed": split_seed, "image_size": [600, 456],
        "split_method": "global_random_frame_80_20_after_reference_light_quality_exclusion",
        "warning": "Adjacent frames share train/validation; validation measures fit, not cross-video generalization.",
        "test_episodes": SEEDS, "samples": samples, "train": train_count,
        "val": len(samples) - train_count,
        "pixel_counts": {split: value.tolist() for split, value in counts.items()},
        "source_round": str(source_round), "quality_root": str(quality_root),
        "quality_policy_version": POLICY["version"], "accepted": len(accepted),
        "excluded": len(excluded), "excluded_frames": excluded,
    }
    write_json(manifest_path, manifest)
    return cache


def make_student_round(source_round: Path, student_root: Path, model_dir: Path, device: str) -> None:
    source = _load(source_round / "manifest.json")
    student_root.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = {
        "workflow": "quality_filtered_student_predictions", "model_dir": str(model_dir),
        "source_round": str(source_round), "records": records,
    }
    for source_record in source["records"]:
        record = {"id": source_record["id"], "item": source_record["item"], "status": "pending"}
        records.append(record)
    write_json(student_root / "manifest.json", manifest)
    for record in records:
        output = student_root / record["id"] / "unet"
        if not (output / "complete.json").exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                output.rename(output.parent / f"unet.incomplete.{time.time_ns()}")
            segment_video(model_dir, Path(record["item"]["video_path"]), output, device)
        record["status"] = "complete"
        record["media"] = {"unet": {
            "overlay": str((output / "overlay.mp4").relative_to(student_root)),
            "segmentation": str((output / "mask_visual.mp4").relative_to(student_root)),
        }}
        write_json(student_root / "manifest.json", manifest)
        print("STUDENT PREDICTION COMPLETE", record["id"], flush=True)


def _aggregate_classes(manifest: dict) -> dict:
    totals: dict[str, dict[str, float]] = {}
    for record in manifest["records"]:
        for name, branches in record["summary"]["per_class"].items():
            target = totals.setdefault(name, {})
            for branch in ("raw", "filtered"):
                key = branch + "_hard_anomaly_frames"
                target[key] = target.get(key, 0) + branches[branch]["hard_anomaly_frames"]
    return totals


def write_report(source_round: Path, source_quality: Path, student_round: Path,
                 student_quality: Path, model_dir: Path, output: Path) -> dict:
    before_manifest = _load(source_quality / "manifest.json")
    after_manifest = _load(student_quality / "manifest.json")
    before = _load(source_quality / "complete.json")
    after = _load(student_quality / "complete.json")
    test = _load(model_dir / "test_seed4.json")
    evaluation = _load(model_dir / "evaluation.json")
    comparisons = {}
    for name in ("unet_resnet18_front_pseudo8_r2", "unet_resnet18_front_pseudo24_r3"):
        path = source_round.parent / name / "test_seed4.json"
        if path.exists():
            comparisons[name] = _load(path)
    class_before = _aggregate_classes(before_manifest)
    class_after = _aggregate_classes(after_manifest)
    result = {
        "method": POLICY, "source_quality": before, "student_quality": after,
        "student_validation": evaluation, "student_seed4_test": test, "baselines": comparisons,
        "per_class_before": class_before, "per_class_after": class_after,
        "caveats": [
            "The eight quality videos are reused for student fitting, so their before/after temporal scores are diagnostic, not held-out generalization.",
            "The seed4 references were used to train the original teacher; they are held out from student fitting but are not independent of the label-generation ancestry.",
            "Temporal/structural consistency cannot certify a stable semantic error; random human audits remain necessary.",
        ],
    }
    write_json(output / "results.json", result)
    lines = [
        "# 类别感知时空质量评估与 8 视频迭代训练结果", "",
        "## 方法", "",
        "对每个类别同时分析原始预测和连通分量过滤后的掩码。描述符包括面积、质心、包围盒宽高、",
        "连通分量数（Betti-0）、孔洞数量/面积（Betti-1）以及经前后向光流一致性验证的对齐 IoU。",
        "整段鲁棒轨迹用于保留持续变化，只在面积、空间形态、拓扑、运动四组证据中至少两组同时异常时拒绝。", "",
        "## 数据筛选", "",
        "| 阶段 | 视频 | 总帧 | 接受帧 | 排除帧 | 接受率 | 原始异常帧 | 过滤后异常帧 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| 初始模型 | {before['videos']} | {before['frames']} | {before['accepted_frames']} | {before['excluded_frames']} | {before['acceptance_rate']:.2%} | {before['raw_any_anomaly_frames']} | {before['filtered_any_anomaly_frames']} |",
        f"| 迭代模型 | {after['videos']} | {after['frames']} | {after['accepted_frames']} | {after['excluded_frames']} | {after['acceptance_rate']:.2%} | {after['raw_any_anomaly_frames']} | {after['filtered_any_anomaly_frames']} |", "",
        "## 总体变化", "",
        f"- raw 异常帧：{before['raw_any_anomaly_frames']} → {after['raw_any_anomaly_frames']}（{(after['raw_any_anomaly_frames']/before['raw_any_anomaly_frames']-1):.2%}）。",
        f"- filtered 异常帧：{before['filtered_any_anomaly_frames']} → {after['filtered_any_anomaly_frames']}（{(after['filtered_any_anomaly_frames']/before['filtered_any_anomaly_frames']-1):.2%}）。",
        f"- 扩展排除帧：{before['excluded_frames']} → {after['excluded_frames']}（{(after['excluded_frames']/before['excluded_frames']-1):.2%}）。",
        f"- 过滤器修改帧：{before['filter_changed_frames']} → {after['filter_changed_frames']}（{(after['filter_changed_frames']/before['filter_changed_frames']-1):.2%}）。", "",
        "这些变化是在用于学生拟合的相同 8 个视频上计算，表示训练集内时序/结构改善，不是独立泛化结果。", "",
        "## 逐视频对比", "",
        "| 视频 | 初始 raw | 迭代 raw | 初始 filtered | 迭代 filtered | 初始排除 | 迭代排除 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    after_by_id = {record["id"]: record for record in after_manifest["records"]}
    for record in before_manifest["records"]:
        previous = record["summary"]
        current = after_by_id[record["id"]]["summary"]
        lines.append(f"| {record['id']} | {previous['raw_any_anomaly_frames']} | {current['raw_any_anomaly_frames']} | {previous['filtered_any_anomaly_frames']} | {current['filtered_any_anomaly_frames']} | {previous['excluded_frames']} | {current['excluded_frames']} |")
    validation_fg = float(np.mean([evaluation["per_class_iou"][name] for name in LABELS[1:]]))
    lines += ["", "## 训练收敛", "",
              f"最佳 epoch：{evaluation['best_epoch']}；随机帧验证前景 mIoU：{validation_fg:.6f}。", "",
        "## 每类别异常帧", "",
        "| 类别 | 初始 raw | 初始 filtered | 迭代 raw | 迭代 filtered |", "|---|---:|---:|---:|---:|",
    ]
    for name in LABELS[1:]:
        b, a = class_before.get(name, {}), class_after.get(name, {})
        lines.append(f"| {name} | {int(b.get('raw_hard_anomaly_frames', 0))} | {int(b.get('filtered_hard_anomaly_frames', 0))} | {int(a.get('raw_hard_anomaly_frames', 0))} | {int(a.get('filtered_hard_anomaly_frames', 0))} |")
    lines += ["", "## 固定种子参考集像素指标", "",
              "该参考集未参与学生模型训练或选择，但初始教师曾由这些视频训练，因此只用于迭代回归检查。", "",
              "| 模型 | 前景 mIoU | occluder | object | region | lefttool | righttool |", "|---|---:|---:|---:|---:|---:|---:|"]
    table = [(model_dir.name, test)] + [(name, value) for name, value in comparisons.items()]
    for name, value in table:
        scores = value["all"]
        pc = scores["per_class_iou"]
        lines.append(f"| {name} | {scores['foreground_miou']:.6f} | {pc['occluder']:.6f} | {pc['object']:.6f} | {pc['region']:.6f} | {pc['lefttool']:.6f} | {pc['righttool']:.6f} |")
    lines += ["", "## 解释限制", "",
              "- 时间一致性不是语义正确性的充分条件；持续稳定的错误仍需随机人工审计。",
              "- 同一视频内随机拆帧用于模型收敛，不应作为论文中的跨视频泛化验证。",
              "- 正式论文应另设按视频划分的密集人工真值校准集和独立测试集。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-round", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    source_round = args.source_round.resolve()
    work_root = args.work_root.resolve()
    model_dir = args.model_dir.resolve()
    source_quality = source_round / "video_quality_v2"
    student_root = work_root / "student_predictions"
    student_quality = student_root / "video_quality_v2"
    work_root.mkdir(parents=True, exist_ok=True)
    status = work_root / "status.json"

    if not (source_quality / "complete.json").exists():
        write_json(status, {"stage": "teacher_quality", "updated_at": time.time()})
        run_round(source_round, "video_quality_v2", "video_quality_v1")
    write_json(status, {"stage": "cache", "updated_at": time.time()})
    cache = build_cache(source_round, source_quality, model_dir, split_seed=42)
    shutil.copy2(source_round / "seed_test_snapshot.json", work_root / "seed_test_snapshot.json")

    if not (model_dir / "complete.json").exists():
        write_json(status, {"stage": "training", "updated_at": time.time()})
        command = [str(PYTHON), str(REPO / "tools" / "train_reviewed_resnet_unet.py"),
                   "--work-dir", str(model_dir), "--prepared-cache", str(cache),
                   "--episodes", *[str(r["item"]["episode_index"]) for r in _load(source_round / "manifest.json")["records"]],
                   "--device", args.device, "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                   "--epochs", str(args.epochs), "--min-epochs", str(args.min_epochs), "--patience", str(args.patience)]
        if (model_dir / "latest.pt").exists():
            command.append("--resume")
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "train.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    if not (model_dir / "test_seed4.json").exists():
        write_json(status, {"stage": "seed_test", "updated_at": time.time()})
        seed_test(work_root, model_dir, args.device)
    write_json(status, {"stage": "student_prediction", "updated_at": time.time()})
    make_student_round(source_round, student_root, model_dir, args.device)
    if not (student_quality / "complete.json").exists():
        write_json(status, {"stage": "student_quality", "updated_at": time.time()})
        run_round(student_root, "video_quality_v2")
    write_json(status, {"stage": "report", "updated_at": time.time()})
    write_report(source_round, source_quality, student_root, student_quality, model_dir, work_root)
    write_json(work_root / "complete.json", {
        "workflow": "class_motion_topology_quality_filtered_iteration",
        "videos": len(_load(source_round / "manifest.json")["records"]),
        "source_round": str(source_round), "model_dir": str(model_dir),
        "report": str(work_root / "report.md"), "finished_at": time.time(),
    })
    write_json(status, {"stage": "complete", "updated_at": time.time()})
    print("QUALITY ITERATION COMPLETE", work_root, flush=True)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    main()
