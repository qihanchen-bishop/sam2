#!/usr/bin/env python3
"""Compare generations 1--3 on the same 82 non-benchmark videos.

This is an all-data temporal/structural diagnostic.  Generation 3 trains on
these videos, so this output must never be described as held-out semantic
accuracy.  The fixed benchmark8 evaluator remains the generalization test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2

from sam2_direction_review import write_json
from unet_video_preview import run as segment_video
from video_segmentation_quality import POLICY, assess_video


REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO / "segdata/temp/newur/unet_pseudo82_r4/manifest.json"
DEFAULT_OUTPUT = REPO / "segdata/temp/newur/unet_full82_quality_r1"
BENCHMARK_EPISODES = {0, 15, 20, 21, 35, 40, 60, 83}
MODEL_SPECS = {
    "generation1_pseudo8": REPO / "segdata/temp/newur/unet_resnet18_front_pseudo8_r2",
    "generation2_pseudo24": REPO / "segdata/temp/newur/unet_resnet18_front_pseudo24_r3",
    "generation3_pseudo82": REPO / "segdata/temp/newur/unet_resnet18_front_pseudo82_r4",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_metadata(model_key: str) -> dict:
    model_dir = MODEL_SPECS[model_key]
    config = load_json(model_dir / "training_config.json")
    return {
        "model_key": model_key, "model_dir": str(model_dir),
        "model_sha256": sha256(model_dir / "model.ts"),
        "training_episodes": [int(value) for value in config["episodes"]],
    }


def prepare(output: Path, model_keys: list[str]) -> dict:
    source = load_json(SOURCE_MANIFEST)
    records = [{"id": row["id"], "item": row["item"]} for row in source["records"]]
    episodes = {int(row["item"]["episode_index"]) for row in records}
    if len(records) != 82 or episodes & BENCHMARK_EPISODES:
        raise AssertionError("Full82 source set is not the benchmark-excluded population")
    models = {key: model_metadata(key) for key in model_keys}
    benchmark = {
        "workflow": "same_population_full82_temporal_structural_comparison",
        "created_at": time.time(), "records": records,
        "benchmark_episodes_excluded": sorted(BENCHMARK_EPISODES),
        "quality_policy": POLICY, "models": models,
        "interpretation": (
            "Same 82-video operational population for all models. Generation 3 trained on this population, "
            "so its result is an in-domain diagnostic, not held-out accuracy."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        old = load_json(path)
        if old["records"] != records or old["benchmark_episodes_excluded"] != sorted(BENCHMARK_EPISODES):
            raise ValueError("Existing full82 population differs")
        for key, metadata in models.items():
            if key in old["models"] and old["models"][key] != metadata:
                raise ValueError(f"Frozen model changed: {key}")
            old["models"][key] = metadata
        write_json(path, old)
        return old
    write_json(path, benchmark)
    return benchmark


def reusable_prediction(model_key: str, record: dict, expected_hash: str) -> Path | None:
    episode = int(record["item"]["episode_index"])
    candidates = []
    if model_key == "generation1_pseudo8":
        candidates.append(REPO / f"segdata/temp/newur/unet_pseudo24_r3/ep{episode:06d}/unet")
    elif model_key == "generation2_pseudo24":
        candidates.append(REPO / f"segdata/temp/newur/unet_pseudo82_r4/teacher_new58/ep{episode:06d}/unet")
    for candidate in candidates:
        complete_path = candidate / "complete.json"
        if not complete_path.exists():
            continue
        complete = load_json(complete_path)
        if (complete.get("model_sha256") == expected_hash and
                int(complete.get("frame_count", -1)) == int(record["item"]["frame_count"])):
            return candidate.resolve()
    return None


def predict(output: Path, benchmark: dict, model_key: str, device: str) -> Path:
    model = benchmark["models"][model_key]
    root = output / "predictions" / model_key
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {
            "workflow": "full82_model_predictions", "model": model,
            "records": [{**row, "status": "pending"} for row in benchmark["records"]],
        }
        write_json(manifest_path, manifest)
    for record in manifest["records"]:
        target = root / record["id"] / "unet"
        complete_path = target / "complete.json"
        if complete_path.exists():
            complete = load_json(complete_path)
            if complete["model_sha256"] != model["model_sha256"]:
                raise ValueError(f"Stale full82 prediction: {target}")
        else:
            reuse = reusable_prediction(model_key, record, model["model_sha256"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if reuse is not None and not target.exists():
                target.symlink_to(reuse, target_is_directory=True)
            elif not target.exists():
                record["status"] = "segmenting"
                write_json(manifest_path, manifest)
                segment_video(Path(model["model_dir"]), Path(record["item"]["video_path"]), target, device)
        complete = load_json(target / "complete.json")
        if int(complete["frame_count"]) != int(record["item"]["frame_count"]):
            raise ValueError(f"Frame mismatch: {model_key} {record['id']}")
        record["status"] = "complete"
        write_json(manifest_path, manifest)
        print("FULL82 PREDICTION COMPLETE", model_key, record["id"], flush=True)
    return root


def score(root: Path) -> dict:
    source = load_json(root / "manifest.json")
    quality_root = root / "video_quality_v2"
    quality_root.mkdir(exist_ok=True)
    records = []
    manifest = {
        "workflow": "class_motion_topology_quality_assessment",
        "policy": POLICY, "source_manifest": str((root / "manifest.json").resolve()),
        "records": records,
    }
    for source_record in source["records"]:
        target = quality_root / source_record["id"]
        summary_path = target / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else assess_video(
            Path(source_record["item"]["video_path"]),
            root / source_record["id"] / "unet", target, float(source_record["item"]["fps"]),
        )
        records.append({
            "id": source_record["id"], "item": source_record["item"],
            "status": "complete", "summary": summary,
        })
        write_json(quality_root / "manifest.json", manifest)
        print("FULL82 QUALITY COMPLETE", root.name, source_record["id"], flush=True)
    frames = sum(row["summary"]["frame_count"] for row in records)
    accepted = sum(row["summary"]["accepted_frames"] for row in records)
    result = {
        "videos": len(records), "frames": frames, "accepted_frames": accepted,
        "excluded_frames": frames - accepted, "acceptance_rate": accepted / frames,
        "raw_any_anomaly_frames": sum(row["summary"]["raw_any_anomaly_frames"] for row in records),
        "filtered_any_anomaly_frames": sum(row["summary"]["filtered_any_anomaly_frames"] for row in records),
        "filter_changed_frames": sum(row["summary"]["filter_changed_frames"] for row in records),
        "per_class": {
            label: {
                branch: sum(
                    row["summary"]["per_class"][label][branch]["hard_anomaly_frames"] for row in records
                ) for branch in ("raw", "filtered")
            } for label in ("occluder", "object", "region", "lefttool", "righttool")
        },
        "policy_version": POLICY["version"], "finished_at": time.time(),
    }
    write_json(quality_root / "complete.json", result)
    return result


def report(output: Path, benchmark: dict, results: dict[str, dict]) -> None:
    write_json(output / "results.json", {
        "interpretation": benchmark["interpretation"], "models": results,
    })
    lines = [
        "# 同一 full82 视频集合的连续性/结构质量比较", "",
        "固定排除 benchmark 8 个视频，三代模型均在相同 82 个视频上计算。第三代使用这 82 个视频训练，",
        "因此本表只表示训练域质量诊断；跨模型泛化结论必须使用 benchmark8。", "",
        "| 模型 | 训练视频 | raw 异常率 | filtered 异常率 | 排除率 | 过滤修改率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, values in results.items():
        model = benchmark["models"][key]
        frames = values["frames"]
        lines.append(
            f"| {key} | {len(model['training_episodes'])} | "
            f"{values['raw_any_anomaly_frames']/frames:.2%} | "
            f"{values['filtered_any_anomaly_frames']/frames:.2%} | "
            f"{values['excluded_frames']/frames:.2%} | "
            f"{values['filter_changed_frames']/frames:.2%} |"
        )
    lines += ["", "## 分类别 filtered 硬异常帧", "",
              "| 类别 | " + " | ".join(results) + " |",
              "|---|" + "---:|" * len(results)]
    for label in ("occluder", "object", "region", "lefttool", "righttool"):
        lines.append("| " + label + " | " + " | ".join(
            str(results[key]["per_class"][label]["filtered"]) for key in results
        ) + " |")
    lines += ["", "## 限制", "",
              "- 这些指标衡量时间连续性和结构异常，不等同于语义正确率。",
              "- 第三代在 full82 上属于训练域评估，不能据此宣称泛化提升。",
              "- 模型选择和最终结论以固定 benchmark8 为准。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS),
                        default=list(MODEL_SPECS))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    output = args.output.resolve()
    benchmark = prepare(output, args.models)
    if args.prepare_only:
        print("FULL82 MANIFEST READY", output, flush=True)
        return
    results = {}
    for model_key in args.models:
        root = predict(output, benchmark, model_key, args.device)
        results[model_key] = score(root)
    report(output, benchmark, results)
    write_json(output / "complete.json", {
        "workflow": benchmark["workflow"], "models": args.models,
        "videos": 82, "report": str(output / "report.md"), "finished_at": time.time(),
    })
    print("FULL82 QUALITY COMPARISON COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
