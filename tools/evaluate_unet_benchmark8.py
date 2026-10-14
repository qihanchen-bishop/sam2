#!/usr/bin/env python3
"""Freeze and evaluate the 4-anchor + 4-unseen front-view U-Net benchmark.

The benchmark intentionally has two panels:

* anchor4 (episodes 0/20/40/60): dense reviewed masks, but the initial model
  was trained on them.  These measure retention/regression, not generalization.
* unseen4 (episodes 15/21/35/83): selected before inference with a fixed,
  task-stratified random seed and excluded from every compared model's training
  set.  These are the generalization panel; pixel IoU remains pending until
  their masks are manually reviewed.

All eight videos receive identical raw inference, connected-component filtering,
preview rendering and reference-light temporal/structural assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from sam2_direction_review import write_json
from train_reviewed_resnet_unet import LABELS
from unet_video_preview import run as segment_video
from video_segmentation_quality import POLICY, run_round


REPO = Path(__file__).resolve().parents[1]
DATASET = Path("/home/qihan/data/sam2/segdata/origin/newur")
DEFAULT_OUTPUT = REPO / "segdata/temp/newur/unet_benchmark8_r1"
SELECTION_SEED = 20261002
ANCHOR_EPISODES = [0, 20, 40, 60]
UNSEEN_EPISODES = [15, 21, 35, 83]
TASK_RANGES = {"o_n": range(0, 30), "b_n": range(30, 60), "g_n": range(60, 90)}
TASK_NAMES = {"o_n": "orange new", "b_n": "black new", "g_n": "green new"}
MODEL_SPECS = {
    "initial_seed4": {
        "generation": 0,
        "role": "initial_model",
        "model_dir": REPO / "segdata/temp/newur/unet_resnet18_front_seed4_r1",
    },
    "generation1_pseudo8": {
        "generation": 1,
        "role": "initial_teacher_to_quality_filtered_pseudo8",
        "model_dir": REPO / "segdata/temp/newur/unet_resnet18_front_pseudo8_r2",
    },
    "generation2_pseudo24": {
        "generation": 2,
        "role": "generation1_teacher_to_prior8_plus_new16",
        "model_dir": REPO / "segdata/temp/newur/unet_resnet18_front_pseudo24_r3",
    },
    "generation3_pseudo82": {
        "generation": 3,
        "role": "generation2_teacher_to_frozen_prior24_plus_all_nonbenchmark58",
        "model_dir": REPO / "segdata/temp/newur/unet_resnet18_front_pseudo82_r4",
    },
}
SELECTION_LINEAGE_KEYS = ("initial_seed4", "generation1_pseudo8", "generation2_pseudo24")
COMPARISON_MODEL_KEYS = ("generation1_pseudo8", "generation2_pseudo24", "generation3_pseudo82")


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_task(episode: int) -> str:
    for name, episodes in TASK_RANGES.items():
        if episode in episodes:
            return name
    raise ValueError(f"Unknown task for episode {episode}")


def video_metadata(episode: int) -> dict:
    path = DATASET / "videos/observation.images.front/chunk-000" / f"file-{episode:03d}.mp4"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open benchmark video: {path}")
    metadata = {
        "episode_index": episode,
        "task": episode_task(episode),
        "task_description": TASK_NAMES[episode_task(episode)],
        "view": "front",
        "video_key": "observation.images.front",
        "video_path": str(path),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "video_size_bytes": path.stat().st_size,
    }
    capture.release()
    return metadata


def training_lineage() -> tuple[dict, set[int]]:
    lineage = {}
    used: set[int] = set()
    for key, spec in MODEL_SPECS.items():
        model_dir = Path(spec["model_dir"])
        if not (model_dir / "training_config.json").exists() or not (model_dir / "model.ts").exists():
            continue
        config = load_json(model_dir / "training_config.json")
        episodes = [int(value) for value in config["episodes"]]
        if key in SELECTION_LINEAGE_KEYS:
            used.update(episodes)
        lineage[key] = {
            **{name: value for name, value in spec.items() if name != "model_dir"},
            "model_dir": str(model_dir),
            "model_sha256": sha256(model_dir / "model.ts"),
            "training_episodes": episodes,
            "training_episode_count": len(episodes),
        }
    return lineage, used


def reproduce_unseen_selection(used: set[int]) -> tuple[list[int], dict]:
    target_counts = {task: sum(episode_task(ep) == task for ep in ANCHOR_EPISODES)
                     for task in TASK_RANGES}
    rng = random.Random(SELECTION_SEED)
    selected: list[int] = []
    pools = {}
    for task, task_range in TASK_RANGES.items():
        pool = [ep for ep in task_range if ep not in used]
        pools[task] = pool
        selected.extend(sorted(rng.sample(pool, target_counts[task])))
    return sorted(selected), {"target_counts": target_counts, "candidate_pools": pools}


def seed_ground_truth() -> dict[int, str]:
    snapshot = load_json(REPO / "segdata/temp/newur/unet_pseudo8_r2/seed_test_snapshot.json")
    roots = {int(row["episode"]): row["final_root"] for row in snapshot}
    if set(roots) != set(ANCHOR_EPISODES):
        raise AssertionError("Reviewed anchor snapshot is incomplete")
    return roots


def prepare_manifest(output: Path) -> dict:
    lineage, used = training_lineage()
    selected, selection = reproduce_unseen_selection(used)
    if selected != UNSEEN_EPISODES:
        raise AssertionError((selected, UNSEEN_EPISODES, "frozen unseen selection changed"))
    if set(UNSEEN_EPISODES) & used:
        raise AssertionError("Unseen benchmark leaked into model training")
    benchmark_episodes = set(ANCHOR_EPISODES + UNSEEN_EPISODES)
    for key, model in lineage.items():
        if key != "initial_seed4" and benchmark_episodes & set(model["training_episodes"]):
            raise AssertionError(f"Benchmark leaked into compared model {key}")
    gt_roots = seed_ground_truth()
    episodes = []
    for episode in ANCHOR_EPISODES + UNSEEN_EPISODES:
        item = video_metadata(episode)
        anchor = episode in ANCHOR_EPISODES
        item.update({
            "panel": "anchor4_seen" if anchor else "unseen4_heldout",
            "ground_truth_status": "dense_human_reviewed" if anchor else "pending_dense_human_review",
            "ground_truth_root": gt_roots.get(episode),
            "allowed_for_future_training": False,
        })
        episodes.append(item)
    manifest = {
        "benchmark": "front_unet_benchmark8_r1",
        "created_at": time.time(),
        "labels": LABELS,
        "selection": {
            "seed": SELECTION_SEED,
            "method": "task-stratified random sampling before benchmark inference; unseen task counts match anchor4",
            "anchor_episodes": ANCHOR_EPISODES,
            "unseen_episodes": UNSEEN_EPISODES,
            "excluded_training_episode_union": sorted(used),
            **selection,
        },
        "interpretation": {
            "anchor4_seen": "Regression/retention panel only; the initial model was trained on these videos.",
            "unseen4_heldout": "Primary generalization panel; permanently excluded from training, validation, filtering-threshold fitting and checkpoint selection.",
            "combined8": "May be reported descriptively, but must not replace the unseen4 primary result.",
        },
        "models": lineage,
        "episodes": episodes,
        "quality_policy": POLICY,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "benchmark_manifest.json"
    if path.exists():
        old = load_json(path)
        immutable = (old["selection"], old["episodes"])
        current = (manifest["selection"], manifest["episodes"])
        if immutable != current:
            raise ValueError("Existing benchmark manifest differs; use a new benchmark version")
        for key, value in lineage.items():
            if key in old["models"] and old["models"][key] != value:
                raise ValueError(f"Frozen benchmark model changed: {key}")
            old["models"][key] = value
        write_json(path, old)
        return old
    write_json(path, manifest)
    return manifest


def prediction_round_manifest(output: Path, benchmark: dict, model_key: str) -> dict:
    records = []
    for item in benchmark["episodes"]:
        records.append({
            "id": f"ep{item['episode_index']:06d}",
            "item": {key: item[key] for key in (
                "episode_index", "task", "view", "video_key", "video_path", "frame_count", "fps"
            )},
            "panel": item["panel"],
            "status": "pending",
        })
    return {
        "workflow": "fixed_benchmark_unet_predictions",
        "benchmark": benchmark["benchmark"],
        "benchmark_manifest": str(output / "benchmark_manifest.json"),
        "model_key": model_key,
        "model": benchmark["models"][model_key],
        "records": records,
    }


def run_predictions(output: Path, benchmark: dict, model_keys: list[str], device: str) -> None:
    for model_key in model_keys:
        round_root = output / "predictions" / model_key
        round_root.mkdir(parents=True, exist_ok=True)
        manifest_path = round_root / "manifest.json"
        manifest = load_json(manifest_path) if manifest_path.exists() else prediction_round_manifest(output, benchmark, model_key)
        expected_hash = benchmark["models"][model_key]["model_sha256"]
        model_dir = Path(benchmark["models"][model_key]["model_dir"])
        for record in manifest["records"]:
            target = round_root / record["id"] / "unet"
            complete_path = target / "complete.json"
            if complete_path.exists():
                complete = load_json(complete_path)
                if complete["model_sha256"] != expected_hash or complete["frame_count"] != record["item"]["frame_count"]:
                    raise ValueError(f"Stale benchmark prediction: {target}")
            else:
                if target.exists():
                    target.rename(target.with_name(f"unet.incomplete.{time.time_ns()}"))
                target.parent.mkdir(parents=True, exist_ok=True)
                record["status"] = "segmenting"
                write_json(manifest_path, manifest)
                segment_video(model_dir, Path(record["item"]["video_path"]), target, device)
            record["status"] = "complete"
            record["media"] = {
                "overlay": str((target / "overlay.mp4").relative_to(round_root)),
                "segmentation": str((target / "mask_visual.mp4").relative_to(round_root)),
            }
            write_json(manifest_path, manifest)
            print("BENCHMARK PREDICTION COMPLETE", model_key, record["id"], flush=True)


def confusion_metrics(confusion: np.ndarray) -> dict:
    diagonal = np.diag(confusion).astype(np.float64)
    truth = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = truth + predicted - diagonal
    iou = np.divide(diagonal, union, out=np.full_like(diagonal, np.nan), where=union > 0)
    dice_denominator = truth + predicted
    dice = np.divide(2 * diagonal, dice_denominator, out=np.full_like(diagonal, np.nan), where=dice_denominator > 0)
    return {
        "frames": None,
        "pixels": int(confusion.sum()),
        "foreground_miou": float(np.nanmean(iou[1:])),
        "foreground_mdice": float(np.nanmean(dice[1:])),
        "per_class_iou": {name: float(iou[index]) for index, name in enumerate(LABELS)},
        "per_class_dice": {name: float(dice[index]) for index, name in enumerate(LABELS)},
        "confusion_matrix": confusion.tolist(),
    }


def anchor_pixel_metrics(output: Path, benchmark: dict, model_key: str) -> dict:
    round_root = output / "predictions" / model_key
    aggregate = {branch: np.zeros((len(LABELS), len(LABELS)), np.int64) for branch in ("raw", "filtered")}
    per_episode = {}
    total_frames = 0
    for item in benchmark["episodes"]:
        if item["panel"] != "anchor4_seen":
            continue
        episode = item["episode_index"]
        root = round_root / f"ep{episode:06d}" / "unet"
        gt_root = Path(item["ground_truth_root"]) / "mask_frames"
        matrices = {branch: np.zeros((len(LABELS), len(LABELS)), np.int64) for branch in aggregate}
        for frame in range(item["frame_count"]):
            truth = cv2.imread(str(gt_root / f"{frame:05d}.png"), cv2.IMREAD_UNCHANGED)
            if truth is None:
                raise ValueError(f"Missing reviewed mask: episode {episode} frame {frame}")
            for branch, folder in (("raw", "raw_mask_frames"), ("filtered", "mask_frames")):
                prediction = cv2.imread(str(root / folder / f"{frame:05d}.png"), cv2.IMREAD_UNCHANGED)
                if prediction is None or prediction.shape != truth.shape:
                    raise ValueError((episode, frame, branch, "prediction/GT mismatch"))
                valid = (truth >= 0) & (truth < len(LABELS))
                encoded = truth[valid].astype(np.int64) * len(LABELS) + prediction[valid].astype(np.int64)
                matrices[branch] += np.bincount(encoded, minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))
        total_frames += item["frame_count"]
        for branch in aggregate:
            aggregate[branch] += matrices[branch]
        per_episode[str(episode)] = {branch: confusion_metrics(matrix) for branch, matrix in matrices.items()}
        for branch in per_episode[str(episode)].values():
            branch["frames"] = item["frame_count"]
    results = {
        "panel": "anchor4_seen",
        "interpretation": "Dense reviewed pixel metric, but not independent generalization because the initial model trained on these videos.",
        "model_key": model_key,
        "episodes": ANCHOR_EPISODES,
        "all": {branch: confusion_metrics(matrix) for branch, matrix in aggregate.items()},
        "per_episode": per_episode,
    }
    for branch in results["all"].values():
        branch["frames"] = total_frames
    write_json(round_root / "anchor4_pixel_metrics.json", results)
    return results


def grouped_quality(round_root: Path, benchmark: dict) -> dict:
    quality = load_json(round_root / "video_quality_v2/manifest.json")
    panel_by_id = {f"ep{item['episode_index']:06d}": item["panel"] for item in benchmark["episodes"]}
    groups = {"anchor4_seen": [], "unseen4_heldout": [], "combined8": []}
    for record in quality["records"]:
        row = {"id": record["id"], **record["summary"]}
        groups[panel_by_id[record["id"]]].append(row)
        groups["combined8"].append(row)
    result = {}
    for name, rows in groups.items():
        frames = sum(row["frame_count"] for row in rows)
        raw = sum(row["raw_any_anomaly_frames"] for row in rows)
        filtered = sum(row["filtered_any_anomaly_frames"] for row in rows)
        excluded = sum(row["excluded_frames"] for row in rows)
        changed = sum(row["filter_changed_frames"] for row in rows)
        result[name] = {
            "videos": len(rows), "frames": frames,
            "raw_anomaly_frames": raw,
            "raw_anomaly_rate": raw / frames,
            "filtered_anomaly_frames": filtered,
            "filtered_anomaly_rate": filtered / frames,
            "excluded_frames": excluded,
            "exclusion_rate": excluded / frames,
            "filter_changed_frames": changed,
            "filter_changed_rate": changed / frames,
            "per_class_hard_anomaly_frames": {
                label: {
                    branch: sum(row["per_class"][label][branch]["hard_anomaly_frames"] for row in rows)
                    for branch in ("raw", "filtered")
                }
                for label in LABELS[1:]
            },
            "per_video": {
                row["id"]: {
                    key: row[key] for key in (
                        "frame_count", "raw_any_anomaly_frames", "filtered_any_anomaly_frames",
                        "excluded_frames", "filter_changed_frames",
                    )
                }
                for row in rows
            },
        }
    return result


def write_report(output: Path, benchmark: dict, model_keys: list[str]) -> dict:
    models = {}
    for model_key in model_keys:
        round_root = output / "predictions" / model_key
        pixel = load_json(round_root / "anchor4_pixel_metrics.json")
        quality = grouped_quality(round_root, benchmark)
        models[model_key] = {"anchor4_pixel": pixel, "quality": quality}
    result = {
        "benchmark": benchmark["benchmark"],
        "models": models,
        "primary_generalization_metric_status": "pending unseen4 dense human review",
        "valid_now": {
            "anchor4_pixel": "retention/regression only",
            "combined8_temporal_structural": "reference-light quality comparison, not semantic accuracy",
            "unseen4_temporal_structural": "held-out reference-light generalization proxy",
        },
        "invalid_claims": [
            "Do not call anchor4 or combined8 IoU an independent generalization result.",
            "Do not call temporal consistency semantic accuracy.",
            "Do not tune thresholds or select checkpoints on unseen4.",
        ],
    }
    write_json(output / "results.json", result)
    lines = [
        "# 固定 4+4 前视角 U-Net benchmark", "",
        "## 评估口径", "",
        "- anchor4：000、020、040、060，有逐像素人工复核标签；仅作保持性/退化检查。",
        "- unseen4：015、021、035、083，未进入任何被比较模型训练；是主要泛化面板。",
        "- 初始 seed4 模型只负责产生第一批伪标签，不进入模型性能比较。",
        "- unseen4 完成人工逐像素复核前，只报告参考较少的连续性/结构质量，不能报告真实 IoU。",
        "- combined8 可以描述总体运行表现，但不能替代 unseen4 的主要泛化结果。", "",
        "## 模型谱系", "",
        "| 模型 | 代次 | 训练视频 | 说明 |", "|---|---:|---:|---|",
    ]
    for key in model_keys:
        spec = benchmark["models"][key]
        lines.append(f"| {key} | {spec['generation']} | {spec['training_episode_count']} | {spec['role']} |")
    lines += ["", "## anchor4 raw 人工标签像素指标（主要保持性指标）", "",
              "| 模型 | raw 前景 mIoU | raw mDice | occluder | object | region | lefttool | righttool |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in model_keys:
        raw = models[key]["anchor4_pixel"]["all"]["raw"]
        pc = raw["per_class_iou"]
        lines.append(
            f"| {key} | {raw['foreground_miou']:.6f} | {raw['foreground_mdice']:.6f} | "
            f"{pc['occluder']:.6f} | {pc['object']:.6f} | "
            f"{pc['region']:.6f} | {pc['lefttool']:.6f} | {pc['righttool']:.6f} |"
        )
    lines += ["", "## anchor4 filtered 人工标签像素指标（部署后处理结果）", "",
              "| 模型 | filtered 前景 mIoU | filtered mDice | occluder | object | region | lefttool | righttool |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in model_keys:
        filtered = models[key]["anchor4_pixel"]["all"]["filtered"]
        pc = filtered["per_class_iou"]
        lines.append(
            f"| {key} | {filtered['foreground_miou']:.6f} | {filtered['foreground_mdice']:.6f} | "
            f"{pc['occluder']:.6f} | {pc['object']:.6f} | {pc['region']:.6f} | "
            f"{pc['lefttool']:.6f} | {pc['righttool']:.6f} |"
        )
    lines += ["", "## 全 8 视频连续性与结构质量", "",
              "| 模型 | raw 异常率 | filtered 异常率 | 排除率 | 过滤修改率 |",
              "|---|---:|---:|---:|---:|"]
    for key in model_keys:
        row = models[key]["quality"]["combined8"]
        lines.append(f"| {key} | {row['raw_anomaly_rate']:.2%} | {row['filtered_anomaly_rate']:.2%} | {row['exclusion_rate']:.2%} | {row['filter_changed_rate']:.2%} |")
    lines += ["", "## unseen4 连续性与结构质量（独立但仅为代理指标）", "",
              "| 模型 | raw 异常率 | filtered 异常率 | 排除率 | 过滤修改率 |",
              "|---|---:|---:|---:|---:|"]
    for key in model_keys:
        row = models[key]["quality"]["unseen4_heldout"]
        lines.append(f"| {key} | {row['raw_anomaly_rate']:.2%} | {row['filtered_anomaly_rate']:.2%} | {row['exclusion_rate']:.2%} | {row['filter_changed_rate']:.2%} |")
    lines += ["", "### unseen4 逐视频异常帧（raw / filtered / excluded）", "",
              "| 视频 | " + " | ".join(model_keys) + " |",
              "|---|" + "---:|" * len(model_keys)]
    for episode in UNSEEN_EPISODES:
        video_id = f"ep{episode:06d}"
        values = []
        for key in model_keys:
            row = models[key]["quality"]["unseen4_heldout"]["per_video"][video_id]
            values.append(
                f"{row['raw_any_anomaly_frames']} / {row['filtered_any_anomaly_frames']} / {row['excluded_frames']}"
            )
        lines.append(f"| {video_id} | " + " | ".join(values) + " |")
    lines += ["", "### unseen4 分类别 filtered 硬异常帧", "",
              "| 类别 | " + " | ".join(model_keys) + " |",
              "|---|" + "---:|" * len(model_keys)]
    for label in LABELS[1:]:
        values = [
            str(models[key]["quality"]["unseen4_heldout"]["per_class_hard_anomaly_frames"][label]["filtered"])
            for key in model_keys
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "## 解释限制", "",
              "- anchor4 被伪标签祖先模型见过，因此后续模型在此的 IoU 只用于保持性/退化比较。",
              "- 连续性好不代表语义一定正确；稳定的错误分割可能不触发异常检测。",
              "- unseen4 的人工真值完成后，应以其分类别 IoU/Dice 为主结论，并冻结本 benchmark。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    cv2.setNumThreads(1)
    benchmark = prepare_manifest(output)
    model_keys = args.models or [key for key in COMPARISON_MODEL_KEYS if key in benchmark["models"]]
    missing = [key for key in model_keys if key not in benchmark["models"]]
    if missing:
        raise ValueError(f"Models are not trained/exported yet: {missing}")
    write_json(output / "status.json", {"stage": "manifest_ready", "updated_at": time.time()})
    print("BENCHMARK", ANCHOR_EPISODES, "+", UNSEEN_EPISODES, flush=True)
    if args.prepare_only:
        return
    run_predictions(output, benchmark, model_keys, args.device)
    for model_key in model_keys:
        write_json(output / "status.json", {"stage": "anchor_pixel_metrics", "model": model_key, "updated_at": time.time()})
        anchor_pixel_metrics(output, benchmark, model_key)
        if not args.skip_quality:
            write_json(output / "status.json", {"stage": "video_quality", "model": model_key, "updated_at": time.time()})
            round_root = output / "predictions" / model_key
            if not (round_root / "video_quality_v2/complete.json").exists():
                run_round(round_root, "video_quality_v2")
    if args.skip_quality:
        write_json(output / "status.json", {"stage": "predictions_and_anchor_metrics_complete", "updated_at": time.time()})
        return
    write_report(output, benchmark, model_keys)
    write_json(output / "complete.json", {
        "workflow": "fixed_anchor4_plus_unseen4_model_iteration_benchmark",
        "models": model_keys,
        "episodes": ANCHOR_EPISODES + UNSEEN_EPISODES,
        "report": str(output / "report.md"),
        "finished_at": time.time(),
    })
    write_json(output / "status.json", {"stage": "complete", "updated_at": time.time()})
    print("BENCHMARK COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
