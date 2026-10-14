"""Shared labels.txt parser for segmentation classes and view guidance."""

from __future__ import annotations

import re
from pathlib import Path


def parse_label_metadata(path: Path) -> dict:
    labels = []
    view_descriptions = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[:：]", line, maxsplit=1)
        if len(parts) == 1:
            if len(line.split()) > 1:
                raise RuntimeError(
                    "A label without ':' or '：' must be a single token; "
                    f"found unexpected line {line_number} in {path}: {line!r}"
                )
            parts = line.split(maxsplit=1)
        name = parts[0].strip()
        description = parts[1].strip() if len(parts) == 2 else ""
        compact_name = re.sub(r"\s+", "", name)
        if compact_name.endswith("视角"):
            view = compact_name[: -len("视角")]
            if not view or not description:
                raise RuntimeError(
                    f"Invalid view description on line {line_number} of {path}"
                )
            if view in view_descriptions:
                raise RuntimeError(f"Duplicate view description {view!r} in {path}")
            view_descriptions[view] = description
            continue
        if name:
            labels.append(name)
    if not labels:
        raise RuntimeError(f"No segmentation labels found in {path}")
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"Duplicate segmentation labels in {path}: {labels}")
    return {"labels": labels, "view_descriptions": view_descriptions}


def read_label_names(path: Path) -> list[str]:
    return parse_label_metadata(path)["labels"]
