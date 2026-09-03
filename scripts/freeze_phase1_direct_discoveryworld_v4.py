#!/usr/bin/env python3
"""Freeze uniform V4 confirmation after the V3 applicability failure."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    file_sha256,
    read_object,
    validate_manifest,
)


ROOT = REPO / "configs/phase1_direct_prospective_v4"
OUTPUT = ROOT / "discoveryworld_manifest.json"
V1_PATH = REPO / "configs/phase1_direct_prospective_v1/manifest.json"
V3_PATH = REPO / "configs/phase1_direct_prospective_v3/discoveryworld_manifest.json"
V3_ROOT = REPO / "configs/phase1_direct_prospective_v3"
SEEDS = tuple(range(39, 45))
SCHEMA = "phase1-direct-discoveryworld-confirmation-v4"
STATUS = "FROZEN_BEFORE_ANY_V4_TARGET_RESET_PROVIDER_CALL_OR_OUTCOME"


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def _assert_fresh(task_ids: tuple[str, ...]) -> None:
    for task_id in task_ids:
        result = subprocess.run(
            ["rg", "-l", "-F", task_id, "runs", "docs/results"],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        if result.stdout.strip():
            raise SystemExit(f"V4 target identity already appears: {task_id}")


def main() -> None:
    frozen = [
        OUTPUT,
        ROOT / "discoveryworld_target_manifest.json",
        ROOT / "discoveryworld_target_only.json",
        ROOT / "discoveryworld_protocol.json",
    ]
    if any(path.exists() for path in frozen):
        raise SystemExit("DiscoveryWorld V4 frozen files exist; refusing overwrite")
    v1 = read_object(V1_PATH)
    validate_manifest(v1, repo=REPO)
    v3 = read_object(V3_PATH)
    task_ids = tuple(f"proteomics.easy.seed{seed}" for seed in SEEDS)
    _assert_fresh(task_ids)
    tasks = [
        {
            "task_id": task_id, "scenario": "Proteomics",
            "difficulty": "Easy", "seed": seed,
        }
        for task_id, seed in zip(task_ids, SEEDS)
    ]

    target_manifest_body = {
        "schema_version": "discoveryworld-phase1-direct-reserve-v4",
        "status": "FROZEN_BEFORE_TARGET_RESET",
        "official_environment_commit": "fd591323920be0d3786ef350955de1945aa571e5",
        "roles": {"formal_reserve": tasks},
    }
    target_manifest = target_manifest_body | {
        "manifest_sha256": stable_hash(target_manifest_body),
    }
    target_manifest_path = ROOT / "discoveryworld_target_manifest.json"
    _write_once(target_manifest_path, target_manifest)

    target_config = deepcopy(read_object(V3_ROOT / "discoveryworld_target_only.json"))
    target_config.update({
        "schema_version": "discoveryworld-target-only-phase1-direct-v4",
        "claim_boundary": (
            "Six fresh Proteomics Easy seeds39-44, with the same independently "
            "qualified Qwen target acquisition frozen in V3."
        ),
        "manifest": str(target_manifest_path.relative_to(REPO)),
    })
    target_config_path = ROOT / "discoveryworld_target_only.json"
    _write_once(target_config_path, target_config)

    protocol = deepcopy(read_object(V3_ROOT / "discoveryworld_protocol.json"))
    protocol.update({
        "schema_version": "discoveryworld-search-automaton-direct-v4",
        "claim_boundary": (
            "Uniform six-source confirmation with neural candidate sets "
            "rejected unless the symbolic source policy has a safe branch."
        ),
        "target_baseline_config": str(target_config_path.relative_to(REPO)),
        "task_ids": list(task_ids),
        "applicability_completeness_contract": {
            "accept_if": (
                "EXISTS_NEURAL_TYPED_PARSER_VALIDATED_REVERSIBLE_POSITION OR "
                "EXISTS_SYMBOLICALLY_WITNESSED_COMMIT"
            ),
            "on_failure": "REJECT_NEURAL_SET_AND_RESAMPLE_WITHIN_FROZEN_ATTEMPT_BUDGET",
            "unsafe_legacy_fallback": (
                "IF_UNWITNESSED_COMMIT_WOULD_BE_SELECTED_CHOOSE_MAX_INFORMATION_GAIN_"
                "POSITION_FROM_THE_SAME_NEURAL_SET"
            ),
            "creates_native_action": False,
            "weakens_source_gate": False,
            "uses_outcome_or_scorecard": False,
        },
    })
    protocol_path = ROOT / "discoveryworld_protocol.json"
    _write_once(protocol_path, protocol)

    cells = []
    for game, task in zip(SOURCE_GAMES, tasks):
        source = v1["sources"][game]
        cells.append({
            "cell_id": f"{game}__to__discoveryworld",
            "source_game": game,
            "target_domain": "discoveryworld",
            "target_task_id": task["task_id"],
            "target_task": task,
            "target_task_multiplicity": 1,
            "selected_target_previously_executed": False,
            "source_artifact": source["artifact"],
            "source_artifact_sha256": source["artifact_sha256"],
            "source_artifact_file_sha256": source["artifact_file_sha256"],
        })

    runtime_files = list(v3["runtime_file_sha256"])
    runtime_files.extend([
        "src/motif_transfer/discoveryworld_applicability_grounder_v4.py",
        "scripts/freeze_phase1_direct_discoveryworld_v4.py",
        "scripts/run_phase1_direct_discoveryworld_v4.py",
    ])
    runtime_files = list(dict.fromkeys(runtime_files))
    body = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "claim_boundary": (
            "Six direct prospective source-lineage-to-DiscoveryWorld executions; "
            "all V1-V3 DiscoveryWorld outcomes are excluded."
        ),
        "parent_v1_manifest": str(V1_PATH.relative_to(REPO)),
        "parent_v1_manifest_sha256": v1["manifest_sha256"],
        "parent_v1_manifest_file_sha256": file_sha256(V1_PATH),
        "consumed_v3_manifest": str(V3_PATH.relative_to(REPO)),
        "consumed_v3_manifest_sha256": v3["manifest_sha256"],
        "selection_read_target_outcome": False,
        "historical_target_outcome_reuse_allowed": False,
        "uniform_confirmation_for_all_six_sources": True,
        "conditions": deepcopy(v3["conditions"]),
        "target_manifest": str(target_manifest_path.relative_to(REPO)),
        "target_config": str(target_config_path.relative_to(REPO)),
        "protocol": str(protocol_path.relative_to(REPO)),
        "cells": cells,
        "runtime_file_sha256": {
            path: file_sha256(REPO / path) for path in runtime_files
        },
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write_once(OUTPUT, manifest)
    print(json.dumps({
        "status": manifest["status"], "cells": len(cells),
        "tasks": list(task_ids), "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
