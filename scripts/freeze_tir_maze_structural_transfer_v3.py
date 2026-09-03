#!/usr/bin/env python3
"""Freeze all remaining never-assigned TIR maze IDs before model calls."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _allocated_maze_ids(maze_ids: set[str]) -> tuple[set[str], str, int]:
    """Audit prior TIR configs/receipts without opening prompts or outcomes."""

    allocated: set[str] = set()
    audited = []
    patterns = (
        "configs/*tir*.json",
        "runs/*tir*/**/*.json",
        "runs/tir*/**/*.json",
    )
    paths = sorted({
        Path(value)
        for pattern in patterns
        for value in glob.glob(str(REPO / pattern), recursive=True)
    })

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                if child_key in {"sample_id", "task_id"} and str(child) in maze_ids:
                    allocated.add(str(child))
                walk(child, child_key)
        elif isinstance(value, list):
            if key in {
                "qualification", "heldout", "formal", "train", "validation",
                "development", "reserve", "task_ids", "sample_ids",
            }:
                allocated.update(str(child) for child in value if str(child) in maze_ids)
            for child in value:
                walk(child, key)

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        walk(payload)
        audited.append({
            "path": str(path.resolve().relative_to(REPO)),
            "file_sha256": _sha256(path),
        })
    return allocated, stable_hash(audited), len(audited)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/TIR-Bench"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/tir_maze_structural_transfer_v3_frozen.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen config: {args.output}")
    dataset_path = args.dataset_root / "TIR-Bench.json"
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    # Selection authority intentionally reads schema only, never prompt, image,
    # answer, or any model outcome.
    maze_ids = {
        str(row["id"])
        for row in rows
        if row.get("task") == "maze" and not row.get("image_2")
    }
    allocated, audit_sha256, audited_files = _allocated_maze_ids(maze_ids)
    unassigned = sorted(
        maze_ids - allocated,
        key=lambda sample_id: stable_hash({
            "allocation": "tir-maze-structural-v3", "sample_id": sample_id,
        }),
    )
    if len(unassigned) != 27:
        raise SystemExit(
            f"expected exactly 27 never-assigned maze IDs, found {len(unassigned)}"
        )
    qualification = unassigned[:9]
    heldout = unassigned[9:]
    source_artifact = REPO / "runs/sokoban_relational_structural_v2/artifact.json"
    source_confirmation = (
        REPO / "runs/sokoban_relational_structural_v2/fresh_confirmation_report.json"
    )
    target_authority = (
        REPO / "runs/tir_maze_topology_v2_consumed_development/consumed_development_report.json"
    )
    integrity_paths = (
        REPO / "src/motif_transfer/relational_structural_induction.py",
        REPO / "src/motif_transfer/tir_maze_topology.py",
        REPO / "src/motif_transfer/sokoban_topology_skill.py",
        REPO / "scripts/run_tir_maze_structural_transfer_v2.py",
        REPO / "scripts/run_tir_maze_topology_v2.py",
        source_artifact,
        source_confirmation,
        target_authority,
    )
    body = {
        "schema_version": "tir-maze-source-induced-structural-freeze-v3",
        "status": "FROZEN_BEFORE_FRESH_QUALIFICATION",
        "claim_boundary": {
            "qualification": (
                "ONE_SHOT_FRESH_QUALIFICATION_OF_SOURCE_INDUCED_SHARED_IR_WITH_"
                "TARGET_NATIVE_NEURAL_MAZE_GROUNDING"
            ),
            "heldout": (
                "ONE_SHOT_FRESH_FORMAL_GAME_TO_TIR_MAZE_STRUCTURAL_TRANSFER;"
                "CLAIM_LIMITED_TO_RELATIONAL_PATH_EXECUTION_AND_NOT_SOURCE_"
                "PROVENANCE_NECESSITY_BECAUSE_AN_ISOMORPHIC_TARGET_CEILING_EXISTS"
            ),
        },
        "source": {
            "artifact_path": str(source_artifact.relative_to(REPO)),
            "artifact_file_sha256": _sha256(source_artifact),
            "confirmation_path": str(source_confirmation.relative_to(REPO)),
            "confirmation_file_sha256": _sha256(source_confirmation),
            "required_confirmation_status": (
                "SOURCE_RELATIONAL_STRUCTURAL_FRESH_VALIDATED"
            ),
        },
        "target_interface_authority": {
            "report_path": str(target_authority.relative_to(REPO)),
            "report_file_sha256": _sha256(target_authority),
            "required_status": "CONSUMED_DEVELOPMENT_GATE_PASSED",
            "reuse_boundary": (
                "TARGET_NATIVE_MOVE_COLOR_BINDER_AND_PIXEL_GRAPH_INTERFACE_ONLY;"
                "NO_PRIOR_TARGET_TRANSFER_OUTCOME_USED_TO_SELECT_FRESH_IDS"
            ),
        },
        "qualification_authority": {
            "report_path": "runs/tir_maze_structural_transfer_v3/qualification_report.json",
            "required_status": "FRESH_QUALIFICATION_GATE_PASSED",
        },
        "dataset": {
            "file_sha256": _sha256(dataset_path),
            "single_image_maze_count": len(maze_ids),
            "prior_allocated_count": len(allocated),
            "never_assigned_count": len(unassigned),
            "prior_allocated_ids": sorted(allocated, key=int),
            "prior_allocation_audit_sha256": audit_sha256,
            "prior_allocation_audited_json_files": audited_files,
            "selection_contract": (
                "READ_ID_TASK_AND_IMAGE2_SCHEMA_ONLY;EXCLUDE_EVERY_ID_PRESENT_IN_"
                "ANY_PRIOR_TIR_CONFIG_OR_RECEIPT;RANK_BY_CONTENT_HASH;NEVER_READ_"
                "PROMPT_IMAGE_ANSWER_OR_MODEL_OUTCOME_BEFORE_FREEZE"
            ),
        },
        "model": {
            "provider": "openrouter",
            "id": "openai/gpt-4.1-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "timeout_seconds": 180,
            "max_retries": 2,
            "maximum_output_tokens": 1200,
            "temperature": 0,
            "authority": (
                "BIND_NATIVE_MOVE_RELATIONS_AND_START_GOAL_VISUAL_ENTITIES;"
                "BASELINE_SOLVES_IN_SEPARATE_CALL;NO_GOLD"
            ),
        },
        "media": {"max_side": 768, "jpeg_quality": 90},
        "splits": {"qualification": qualification, "heldout": heldout},
        "controls": [
            "neural_only", "source_induced", "alpha_renamed_source",
            "source_relation_permuted", "generic_scaffold",
            "target_native_ceiling",
        ],
        "preregistered_gates": {
            "qualification": (
                "ALL_GATES_EXCEPT_FORMAL_PAIRED_SIGNIFICANCE_MUST_PASS"
            ),
            "formal": (
                "ZERO_NEGATIVE_TRANSFER;STRICTLY_BEAT_NEURAL_PERMUTED_GENERIC;"
                "MATCH_TARGET_CEILING;ALPHA_INVARIANT;EXACT_P_LE_0.05_VS_"
                "NEURAL_AND_PERMUTED"
            ),
        },
        "integrity": {
            "file_sha256": {
                str(path.relative_to(REPO)): _sha256(path) for path in integrity_paths
            },
        },
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "qualification": qualification,
        "heldout": heldout,
        "prior_allocated_count": len(allocated),
        "manifest_sha256": manifest["manifest_sha256"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
