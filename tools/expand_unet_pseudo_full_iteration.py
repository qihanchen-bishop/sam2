#!/usr/bin/env python3
"""Train generation 3 on every non-benchmark front-view video.

The fixed benchmark episodes are never used for pseudo-label generation,
quality-threshold fitting, train/validation splitting or checkpoint selection.
The prior 24-video cache is frozen byte-for-byte.  Generation 2 labels the
remaining 58 eligible videos; class-wise temporal/structural quality gates
select their usable frames before a fresh ResNet18-U-Net is trained on 82
videos in total.
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
import torch

from sam2_direction_review import component_filter_policy, write_json
from train_reviewed_resnet_unet import LABELS
from train_unet_pseudo_iteration import REPO, get
from unet_video_preview import run as segment_video
from video_segmentation_quality import POLICY, assess_video


PYTHON = Path(sys.executable).resolve()
BENCHMARK_EPISODES = {0, 15, 20, 21, 35, 40, 60, 83}
PRIOR_ROUND = REPO / "segdata/temp/newur/unet_pseudo24_r3"
PRIOR_MODEL = REPO / "segdata/temp/newur/unet_resnet18_front_pseudo24_r3"


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize(work_root: Path, model_dir: Path) -> dict:
    manifest_path = work_root / "manifest.json"
    teacher_hash = sha256(PRIOR_MODEL / "model.ts")
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest["teacher_sha256"] != teacher_hash:
            raise ValueError("Teacher checkpoint changed after round initialization")
        if set(manifest["benchmark_episodes"]) != BENCHMARK_EPISODES:
            raise ValueError("Frozen benchmark set changed")
        return manifest

    bootstrap = get("/api/bootstrap")
    items = sorted(
        (item for item in bootstrap["items"] if item["view"] == "front"),
        key=lambda item: int(item["episode_index"]),
    )
    if len(items) != 90 or {int(item["episode_index"]) for item in items} != set(range(90)):
        raise ValueError("Expected exactly front-view episodes 0..89")
    prior = load_json(PRIOR_ROUND / "manifest.json")
    prior_episodes = {int(record["item"]["episode_index"]) for record in prior["records"]}
    if len(prior_episodes) != 24 or prior_episodes & BENCHMARK_EPISODES:
        raise AssertionError("Prior24 is invalid or overlaps the benchmark")
    eligible = [item for item in items if int(item["episode_index"]) not in BENCHMARK_EPISODES]
    new_items = [item for item in eligible if int(item["episode_index"]) not in prior_episodes]
    if len(eligible) != 82 or len(new_items) != 58:
        raise AssertionError((len(eligible), len(new_items)))

    records = []
    for item in eligible:
        episode = int(item["episode_index"])
        records.append({
            "id": f"ep{episode:06d}", "item": item,
            "cohort": "prior24_frozen" if episode in prior_episodes else "new58_generation2",
            "status": "frozen" if episode in prior_episodes else "pending",
        })
    manifest = {
        "workflow": "generation2_teacher_prior24_plus_all_nonbenchmark58",
        "created_at": time.time(),
        "teacher_model": str(PRIOR_MODEL),
        "teacher_sha256": teacher_hash,
        "prior_round": str(PRIOR_ROUND),
        "prior_cache": str(PRIOR_MODEL / "cache"),
        "student_model": str(model_dir),
        "benchmark_episodes": sorted(BENCHMARK_EPISODES),
        "training_episode_count": 82,
        "prior_episode_count": 24,
        "new_episode_count": 58,
        "split_seed": 42,
        "filter": component_filter_policy(),
        "new_frame_quality_policy": POLICY,
        "training_initialization": "fresh ImageNet ResNet18 encoder and new decoder",
        "split_method": "prior24 assignments frozen; accepted new58 frames random 80/20",
        "records": records,
    }
    work_root.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    return manifest


def new_round_manifest(work_root: Path, manifest: dict) -> tuple[Path, dict]:
    root = work_root / "teacher_new58"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    expected = [{
        "id": record["id"], "item": record["item"], "status": record["status"]
    } for record in manifest["records"] if record["cohort"] == "new58_generation2"]
    if path.exists():
        current = load_json(path)
        if [(row["id"], row["item"]) for row in current["records"]] != [
            (row["id"], row["item"]) for row in expected
        ]:
            raise ValueError("Existing new58 manifest differs")
        return root, current
    result = {
        "workflow": "generation2_predictions_for_nonbenchmark_new58",
        "model_dir": manifest["teacher_model"],
        "model_sha256": manifest["teacher_sha256"],
        "benchmark_episodes": manifest["benchmark_episodes"],
        "records": expected,
    }
    write_json(path, result)
    return root, result


def generate_new_predictions(root: Path, round_manifest: dict, device: str) -> None:
    manifest_path = root / "manifest.json"
    model_dir = Path(round_manifest["model_dir"])
    for record in round_manifest["records"]:
        output = root / record["id"] / "unet"
        complete_path = output / "complete.json"
        if complete_path.exists():
            complete = load_json(complete_path)
            if complete["model_sha256"] != round_manifest["model_sha256"]:
                raise ValueError(f"Stale prediction model: {record['id']}")
            if int(complete["frame_count"]) != int(record["item"]["frame_count"]):
                raise ValueError(f"Prediction frame mismatch: {record['id']}")
        else:
            if output.exists():
                output.rename(output.parent / f"unet.incomplete.{time.time_ns()}")
            output.parent.mkdir(parents=True, exist_ok=True)
            record["status"] = "segmenting"
            write_json(manifest_path, round_manifest)
            segment_video(model_dir, Path(record["item"]["video_path"]), output, device)
        record["status"] = "predicted"
        record["media"] = {"unet": {
            "overlay": str((output / "overlay.mp4").relative_to(root)),
            "segmentation": str((output / "mask_visual.mp4").relative_to(root)),
        }}
        write_json(manifest_path, round_manifest)
        print("PREDICTION COMPLETE", record["id"], flush=True)


def score_new_predictions(root: Path, round_manifest: dict) -> Path:
    quality_root = root / "video_quality_v2"
    quality_root.mkdir(exist_ok=True)
    quality_manifest_path = quality_root / "manifest.json"
    quality_records = []
    quality_manifest = {
        "workflow": "class_motion_topology_quality_assessment",
        "policy": POLICY,
        "source_manifest": str((root / "manifest.json").resolve()),
        "records": quality_records,
    }
    for source in round_manifest["records"]:
        target = quality_root / source["id"]
        summary_path = target / "summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
        else:
            summary = assess_video(
                Path(source["item"]["video_path"]),
                root / source["id"] / "unet", target, float(source["item"]["fps"]),
            )
        source["status"] = "quality_complete"
        quality_records.append({
            "id": source["id"], "item": source["item"],
            "status": "complete", "summary": summary,
        })
        write_json(quality_manifest_path, quality_manifest)
        write_json(root / "manifest.json", round_manifest)
        print("QUALITY COMPLETE", source["id"], summary["accepted_frames"], flush=True)
    total_frames = sum(row["summary"]["frame_count"] for row in quality_records)
    accepted = sum(row["summary"]["accepted_frames"] for row in quality_records)
    aggregate = {
        "videos": len(quality_records), "frames": total_frames,
        "accepted_frames": accepted, "excluded_frames": total_frames - accepted,
        "acceptance_rate": accepted / total_frames,
        "raw_any_anomaly_frames": sum(row["summary"]["raw_any_anomaly_frames"] for row in quality_records),
        "filtered_any_anomaly_frames": sum(row["summary"]["filtered_any_anomaly_frames"] for row in quality_records),
        "filter_changed_frames": sum(row["summary"]["filter_changed_frames"] for row in quality_records),
        "policy_version": POLICY["version"], "finished_at": time.time(),
    }
    write_json(quality_root / "complete.json", aggregate)
    return quality_root


def _hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    target.hardlink_to(source)


def build_cache(work_root: Path, manifest: dict, new_root: Path, quality_root: Path,
                model_dir: Path) -> Path:
    cache = model_dir / "cache"
    cache_manifest_path = cache / "manifest.json"
    if cache_manifest_path.exists():
        existing = load_json(cache_manifest_path)
        if existing.get("benchmark_episodes") != sorted(BENCHMARK_EPISODES):
            raise ValueError("Existing cache has another benchmark exclusion")
        if existing.get("quality_policy_version") != POLICY["version"]:
            raise ValueError("Existing cache has another quality policy")
        return cache

    prior_cache = Path(manifest["prior_cache"])
    prior = load_json(prior_cache / "manifest.json")
    prior_samples = prior["samples"]
    prior_episodes = {int(sample["episode"]) for sample in prior_samples}
    expected_prior = {
        int(record["item"]["episode_index"])
        for record in manifest["records"] if record["cohort"] == "prior24_frozen"
    }
    if prior_episodes != expected_prior or prior_episodes & BENCHMARK_EPISODES:
        raise AssertionError("Frozen prior cache episode set is invalid")

    accepted_new: list[tuple[int, int, float]] = []
    excluded_new: list[dict] = []
    new_records = [record for record in manifest["records"] if record["cohort"] == "new58_generation2"]
    for record in new_records:
        episode = int(record["item"]["episode_index"])
        frames = load_json(quality_root / record["id"] / "selection.json")["frames"]
        if len(frames) != int(record["item"]["frame_count"]):
            raise ValueError(f"Selection length mismatch: {record['id']}")
        for row in frames:
            if row["accepted"]:
                accepted_new.append((episode, int(row["frame"]), float(row["quality_score"])))
            else:
                excluded_new.append({"episode": episode, **row})

    shuffled = accepted_new.copy()
    random.Random(int(manifest["split_seed"])).shuffle(shuffled)
    new_train_count = int(0.8 * len(shuffled))
    new_split = {
        (episode, frame): "train" if index < new_train_count else "val"
        for index, (episode, frame, _) in enumerate(shuffled)
    }
    new_scores = {(episode, frame): score for episode, frame, score in accepted_new}
    cache.mkdir(parents=True, exist_ok=False)
    counts = {split: np.zeros(len(LABELS), np.int64) for split in ("train", "val")}
    samples = []

    for sample in prior_samples:
        episode, frame, split = int(sample["episode"]), int(sample["frame"]), sample["split"]
        rel = Path(split) / "observation.images.front" / f"file-{episode:03d}" / f"{frame:05d}"
        source_image = prior_cache / "images" / rel.with_suffix(".jpg")
        source_mask = prior_cache / "masks" / rel.with_suffix(".png")
        _hardlink(source_image, cache / "images" / rel.with_suffix(".jpg"))
        _hardlink(source_mask, cache / "masks" / rel.with_suffix(".png"))
        mask = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (456, 600):
            raise ValueError(source_mask)
        counts[split] += np.bincount(mask.ravel(), minlength=len(LABELS))
        samples.append({**sample, "cohort": "prior24_frozen"})

    for record in new_records:
        item = record["item"]
        episode = int(item["episode_index"])
        capture = cv2.VideoCapture(item["video_path"])
        if not capture.isOpened():
            raise ValueError(item["video_path"])
        try:
            for frame in range(int(item["frame_count"])):
                ok, image = capture.read()
                if not ok:
                    raise ValueError((episode, frame, "unreadable RGB"))
                pair = (episode, frame)
                if pair not in new_split:
                    continue
                mask_path = new_root / record["id"] / "unet/mask_frames" / f"{frame:05d}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None or mask.shape != image.shape[:2] or mask.dtype != np.uint8:
                    raise ValueError((episode, frame, "invalid pseudo mask"))
                split = new_split[pair]
                rel = Path(split) / item["video_key"] / f"file-{episode:03d}" / f"{frame:05d}"
                image_out = cache / "images" / rel.with_suffix(".jpg")
                mask_out = cache / "masks" / rel.with_suffix(".png")
                image_out.parent.mkdir(parents=True, exist_ok=True)
                mask_out.parent.mkdir(parents=True, exist_ok=True)
                resized_image = cv2.resize(image, (600, 456), interpolation=cv2.INTER_AREA)
                resized_mask = cv2.resize(mask, (600, 456), interpolation=cv2.INTER_NEAREST)
                if not cv2.imwrite(str(image_out), resized_image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(image_out)
                if not cv2.imwrite(str(mask_out), resized_mask):
                    raise RuntimeError(mask_out)
                counts[split] += np.bincount(resized_mask.ravel(), minlength=len(LABELS))
                samples.append({
                    "episode": episode, "frame": frame, "split": split,
                    "cohort": "new58_generation2", "quality_score": new_scores[pair],
                    "source_mask_sha256": sha256(mask_path),
                })
        finally:
            capture.release()
        print("CACHE COMPLETE", record["id"], flush=True)

    train_count = sum(sample["split"] == "train" for sample in samples)
    cache_manifest = {
        "labels": LABELS, "seed": int(manifest["split_seed"]), "image_size": [600, 456],
        "split_method": manifest["split_method"],
        "warning": "Frame-random validation measures fit, not cross-video generalization.",
        "benchmark_episodes": sorted(BENCHMARK_EPISODES),
        "test_episodes": sorted(BENCHMARK_EPISODES),
        "prior_cache": str(prior_cache), "prior_samples": len(prior_samples),
        "new_accepted": len(accepted_new), "new_excluded": len(excluded_new),
        "quality_policy_version": POLICY["version"],
        "samples": samples, "train": train_count, "val": len(samples) - train_count,
        "pixel_counts": {split: values.tolist() for split, values in counts.items()},
        "source_round": str(work_root), "new_teacher_round": str(new_root),
    }
    if {int(sample["episode"]) for sample in samples} & BENCHMARK_EPISODES:
        raise AssertionError("Benchmark leaked into the training cache")
    write_json(cache_manifest_path, cache_manifest)
    write_json(work_root / "frame_selection.json", {
        "prior_accepted": len(prior_samples), "new_accepted": len(accepted_new),
        "new_excluded": len(excluded_new), "total_accepted": len(samples),
        "train": train_count, "val": len(samples) - train_count,
        "new_excluded_frames": excluded_new,
    })
    return cache


def train(work_root: Path, manifest: dict, model_dir: Path, cache: Path, args) -> None:
    episodes = [int(record["item"]["episode_index"]) for record in manifest["records"]]
    if set(episodes) & BENCHMARK_EPISODES or len(episodes) != 82:
        raise AssertionError("Invalid student training episode set")
    protocol = {
        "workflow": manifest["workflow"],
        "teacher_model": manifest["teacher_model"],
        "teacher_sha256": manifest["teacher_sha256"],
        "benchmark_episodes": sorted(BENCHMARK_EPISODES),
        "benchmark_used_for_training_validation_or_selection": False,
        "prior_training_episodes": [
            int(record["item"]["episode_index"])
            for record in manifest["records"] if record["cohort"] == "prior24_frozen"
        ],
        "new_training_episodes": [
            int(record["item"]["episode_index"])
            for record in manifest["records"] if record["cohort"] == "new58_generation2"
        ],
        "training_episodes": episodes,
        "quality_policy": POLICY,
        "component_filter": manifest["filter"],
        "training_initialization": manifest["training_initialization"],
        "split_method": manifest["split_method"],
        "cache": str(cache),
        "cache_counts": {key: load_json(cache / "manifest.json")[key] for key in (
            "prior_samples", "new_accepted", "new_excluded", "train", "val"
        )},
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    write_json(model_dir / "data_protocol.json", protocol)
    if (model_dir / "complete.json").exists():
        return
    command = [
        str(PYTHON), str(REPO / "tools/train_reviewed_resnet_unet.py"),
        "--work-dir", str(model_dir), "--prepared-cache", str(cache),
        "--episodes", *[str(episode) for episode in episodes],
        "--device", args.device, "--batch-size", str(args.batch_size),
        "--workers", str(args.workers), "--epochs", str(args.epochs),
        "--min-epochs", str(args.min_epochs), "--patience", str(args.patience),
    ]
    if (model_dir / "latest.pt").exists():
        command.append("--resume")
    with (model_dir / "train.log").open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    work_root = args.work_dir.resolve()
    model_dir = args.model_dir.resolve()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)

    try:
        manifest = initialize(work_root, model_dir)
        write_json(work_root / "status.json", {"stage": "teacher_predictions", "updated_at": time.time()})
        new_root, round_manifest = new_round_manifest(work_root, manifest)
        generate_new_predictions(new_root, round_manifest, args.device)
        write_json(work_root / "status.json", {"stage": "quality_scoring", "updated_at": time.time()})
        quality_root = score_new_predictions(new_root, round_manifest)
        write_json(work_root / "status.json", {"stage": "cache_building", "updated_at": time.time()})
        cache = build_cache(work_root, manifest, new_root, quality_root, model_dir)
        write_json(work_root / "status.json", {"stage": "training", "updated_at": time.time()})
        train(work_root, manifest, model_dir, cache, args)
        write_json(work_root / "complete.json", {
            "workflow": manifest["workflow"], "videos": 82,
            "prior_videos": 24, "new_videos": 58,
            "benchmark_episodes": sorted(BENCHMARK_EPISODES),
            "model_dir": str(model_dir), "finished_at": time.time(),
        })
        write_json(work_root / "status.json", {
            "stage": "complete", "model_dir": str(model_dir), "updated_at": time.time(),
        })
        print("FULL ITERATION COMPLETE", work_root, flush=True)
    except Exception as error:
        work_root.mkdir(parents=True, exist_ok=True)
        write_json(work_root / "status.json", {
            "stage": "failed", "error": repr(error), "updated_at": time.time(),
        })
        raise


if __name__ == "__main__":
    main()
