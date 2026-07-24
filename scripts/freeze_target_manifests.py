#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.frozen_split import freeze_one_shot_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ids(path: Path, key) -> list[str]:
    return [str(key(row)) for row in json.loads(path.read_text(encoding="utf-8"))]


def _alfworld_ids(root: Path, split: str) -> list[str]:
    base = root / split
    # AlfredTWEnv registers executable TextWorld games, not every raw
    # trajectory annotation. Selecting traj_data.json would admit rows that
    # can never be reset by the official text environment.
    return [str(path.relative_to(base)) for path in base.rglob("game.tw-pddl")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=Path("/fs/gamma-projects/vlm-robot"))
    parser.add_argument("--output", type=Path, default=REPO / "configs/target_manifests_v1.json")
    args = parser.parse_args()
    root = args.workspace_root
    datasets = root / "datasets"
    browser = root / "emnlp2026_download/workspace/main_project/Cold-start-out-browsergym"
    alf = root / "Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data/json_2.1.1"

    vtb_path = datasets / "VisualToolBench/test.parquet"
    # The public VTB artifact has 1204 rows but its nested columns trigger a
    # PyArrow repetition-histogram bug. Row indices are stable under the
    # pinned file hash and avoid content-dependent selection.
    vtb_ids = [f"row:{index}" for index in range(1204)]
    tir_path = datasets / "TIR-Bench/TIR-Bench.json"
    tir_ids = _json_ids(tir_path, lambda row: row["id"])
    vh_root = datasets / "Video-Holmes/Benchmark"
    vh_train = _json_ids(vh_root / "train_Video-Holmes.json", lambda row: f"{row['video ID']}.Q{row['Question ID']}")
    vh_test = _json_ids(vh_root / "test_Video-Holmes.json", lambda row: f"{row['video ID']}.Q{row['Question ID']}")
    miniwob_ids = [path.name for path in browser.glob("miniwob.*") if path.is_dir()]
    webshop_ids = sorted({path.name for run in browser.glob("webshop_50task_*") for path in run.glob("webshop.*") if path.is_dir()})
    alf_train = _alfworld_ids(alf, "train")
    alf_seen = _alfworld_ids(alf, "valid_seen")
    alf_unseen = _alfworld_ids(alf, "valid_unseen")

    specs = {
        "visual_toolbench": (vtb_ids, vtb_ids, "public_test_internal_holdout", {"dataset_sha256": _sha256(vtb_path)}),
        "tir_bench": (tir_ids, tir_ids, "public_test_internal_holdout", {"dataset_sha256": _sha256(tir_path)}),
        "video_holmes": (vh_train, vh_test, "official_train_to_test", {
            "train_sha256": _sha256(vh_root / "train_Video-Holmes.json"),
            "test_sha256": _sha256(vh_root / "test_Video-Holmes.json"),
        }),
        "miniwob": (miniwob_ids, miniwob_ids, "task_id_internal_holdout", {}),
        "webshop": (webshop_ids, webshop_ids, "task_id_internal_holdout", {}),
        "alfworld_valid_seen": (alf_train, alf_seen, "official_train_to_valid_seen", {}),
        "alfworld_valid_unseen": (alf_train, alf_unseen, "official_train_to_valid_unseen", {}),
    }
    cells = {}
    for cell, (adaptation_ids, test_ids, split_kind, provenance) in specs.items():
        row = freeze_one_shot_split(adaptation_ids, test_ids, namespace=f"motif-transfer:{cell}:v1")
        row["split_kind"] = split_kind
        row["provenance"] = provenance
        row["smoke_test_ids"] = row["test_ids"][:2]
        cells[cell] = row
    payload = {
        "schema_version": 1,
        "matrix": "4-domain/7-cell",
        "excluded": ["siv_bench"],
        "frozen_before_target_runs": True,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({cell: {k: row[k] for k in ("adaptation_id", "test_pool_size", "smoke_test_ids", "split_kind")}
                      for cell, row in cells.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
