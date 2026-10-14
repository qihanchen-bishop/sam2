#!/usr/bin/env python3
"""Wait for generation 3, then run benchmark8 and same-population full82 evaluation."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
ROUND = REPO / "segdata/temp/newur/unet_pseudo82_r4"
MODEL = REPO / "segdata/temp/newur/unet_resnet18_front_pseudo82_r4"
STATUS = REPO / "segdata/temp/newur/unet_generation3_finalize_status.json"


def write_status(stage: str, **extra) -> None:
    STATUS.write_text(json.dumps({"stage": stage, "updated_at": time.time(), **extra}, indent=2))


def run(command: list[str], log_path: Path) -> None:
    with log_path.open("a") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    write_status("waiting_for_generation3")
    while not (ROUND / "complete.json").exists():
        status_path = ROUND / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("stage") == "failed":
                raise RuntimeError(f"Generation 3 failed: {status.get('error')}")
        time.sleep(30)
    if not (MODEL / "model.ts").exists():
        raise RuntimeError("Generation 3 round completed without exported model.ts")

    write_status("benchmark8")
    run([
        str(PYTHON), str(REPO / "tools/evaluate_unet_benchmark8.py"),
        "--device", "cuda:0", "--models",
        "generation1_pseudo8", "generation2_pseudo24", "generation3_pseudo82",
    ], REPO / "segdata/temp/newur/unet_benchmark8_r1/generation3_evaluation.log")

    write_status("full82_quality")
    full_root = REPO / "segdata/temp/newur/unet_full82_quality_r1"
    full_root.mkdir(parents=True, exist_ok=True)
    run([
        str(PYTHON), str(REPO / "tools/evaluate_unet_full82_quality.py"),
        "--device", "cuda:0", "--models",
        "generation1_pseudo8", "generation2_pseudo24", "generation3_pseudo82",
    ], full_root / "evaluation.log")

    write_status(
        "complete",
        benchmark_report=str(REPO / "segdata/temp/newur/unet_benchmark8_r1/report.md"),
        full82_report=str(full_root / "report.md"),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("failed", error=repr(error))
        raise
