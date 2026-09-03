#!/usr/bin/env python3
"""Freeze fresh ALFWorld multiplicity tasks and all structural-transfer dependencies."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_structural_induction import (  # noqa: E402
    validate_grounder,
    validate_target_sequence_program,
)
from motif_transfer.alfworld_structural_runtime_v1 import CONDITIONS  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_delta_induction import validate_structural_program  # noqa: E402


GROUNDING = REPO / "artifacts/alfworld_structural_grounder_v1/artifact.json.gz"
SOURCE = REPO / "configs/source_structural_v5c_frozen/programs/put_near.json"
PERMUTED = REPO / "configs/source_structural_v5c_frozen/programs/unlock_pickup.json"
GROUNDING_REPORT = REPO / "runs/alfworld_structural_grounder_v1_development/report.json"
CLOSED_LOOP_REPORT = REPO / "runs/alfworld_structural_transfer_v2b_development/report.json"
OUTPUT = REPO / "configs/alfworld_structural_transfer_v2/manifest.json"
DATA = Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data")
ALFWORLD_CONFIG = Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1/configs/alfworld_base_config.yaml")
TASK_PATTERN = re.compile(r"pick_two_obj_and_place-[^\"\s]+/game\.tw-pddl")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _executed_task_ids() -> tuple[set[str], list[str]]:
    executed: set[str] = set()
    scanned = []
    for path in sorted((REPO / "runs").rglob("*.json")):
        lower = path.name.lower()
        if any(token in lower for token in ("enumeration", "manifest", "plan")):
            continue
        scanned.append(str(path.relative_to(REPO)))
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                executed.update(TASK_PATTERN.findall(line))
    return executed, scanned


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {OUTPUT}")
    grounder = _read(GROUNDING)
    validate_grounder(grounder)
    if grounder.get("status") != "QUALIFIED":
        raise SystemExit("target-native structural grounder is not qualified")
    validate_target_sequence_program(grounder["target_program"])
    source = _read(SOURCE)
    permuted = _read(PERMUTED)
    grounding_report = _read(GROUNDING_REPORT)
    closed_loop_report = _read(CLOSED_LOOP_REPORT)
    validate_structural_program(source)
    validate_structural_program(permuted)
    if source["program_sha256"] != grounder["selected_source_program_sha256"]:
        raise SystemExit("development applicability did not select frozen authentic source")
    if grounding_report.get("status") != "ALFWORLD_STRUCTURAL_GROUNDER_QUALIFIED":
        raise SystemExit("target structural grounder development did not qualify")
    if closed_loop_report.get("status") != "ALFWORLD_STRUCTURAL_DEVELOPMENT_PASSED":
        raise SystemExit("closed-loop target development did not qualify")
    if closed_loop_report.get("grounder_sha256") != grounder["grounder_sha256"]:
        raise SystemExit("closed-loop qualification used a different target grounder")
    executed, scanned = _executed_task_ids()
    root = DATA / "json_2.1.1" / "train"
    candidates = sorted(
        path.relative_to(root).as_posix()
        for path in root.glob("pick_two_obj_and_place-*/trial_*/game.tw-pddl")
        if path.relative_to(root).as_posix() not in executed
    )
    ranked = sorted(candidates, key=lambda task_id: stable_hash({
        "salt": "ALFWORLD_STRUCTURAL_TRANSFER_V2_FRESH_RESERVE_20260816",
        "task_id": task_id,
    }))
    task_ids = ranked[:12]
    if len(task_ids) < 12:
        raise SystemExit("fewer than twelve execution-untouched ALFWorld tasks remain")
    integrity_paths = (
        "src/motif_transfer/alfworld_env.py",
        "src/motif_transfer/alfworld_structural_induction.py",
        "src/motif_transfer/alfworld_structural_runtime_v1.py",
        "src/motif_transfer/structural_delta_induction.py",
        "scripts/run_alfworld_structural_transfer_v1.py",
        str(GROUNDING.relative_to(REPO)),
        str(SOURCE.relative_to(REPO)),
        str(PERMUTED.relative_to(REPO)),
        str(GROUNDING_REPORT.relative_to(REPO)),
        str(CLOSED_LOOP_REPORT.relative_to(REPO)),
    )
    body = {
        "schema_version": "alfworld-structural-transfer-frozen-manifest-v2",
        "status": "FROZEN_BEFORE_FRESH_EXECUTION",
        "role": "SECOND_TARGET_STRUCTURAL_REPLICATION",
        "conditions": list(CONDITIONS),
        "grounder": {
            "path": str(GROUNDING.relative_to(REPO)),
            "file_sha256": _sha256(GROUNDING),
            "grounder_sha256": grounder["grounder_sha256"],
            "threshold": grounder["threshold"],
            "qualification_thresholds_frozen": True,
            "formal_target_outcome_read": False,
        },
        "development_qualification": {
            "grounding_report_path": str(GROUNDING_REPORT.relative_to(REPO)),
            "grounding_report_file_sha256": _sha256(GROUNDING_REPORT),
            "closed_loop_report_path": str(CLOSED_LOOP_REPORT.relative_to(REPO)),
            "closed_loop_report_file_sha256": _sha256(CLOSED_LOOP_REPORT),
            "closed_loop_status": closed_loop_report["status"],
            "formal_task_ids_read": [],
        },
        "target_program": {
            "program_sha256": grounder["target_program"]["program_sha256"],
            "induced_sequence": grounder["target_program"]["induced_sequence"],
            "source_program_copied_as_target_body": False,
        },
        "source_induced": {
            "path": str(SOURCE.relative_to(REPO)),
            "file_sha256": _sha256(SOURCE),
            "program_sha256": source["program_sha256"],
            "source_name_evaluator_label_only": "put_near",
            "selection_rule": "UNIQUE_EXACT_REPEATED_STRUCTURAL_SUBPROGRAM_MATCH",
            "source_identity_used_as_runtime_feature": False,
        },
        "source_permuted": {
            "path": str(PERMUTED.relative_to(REPO)),
            "file_sha256": _sha256(PERMUTED),
            "program_sha256": permuted["program_sha256"],
            "source_name_evaluator_label_only": "unlock_pickup",
            "control": "DETERMINISTIC_SOURCE_PROGRAM_IDENTITY_PERMUTATION",
        },
        "target": {
            "alfworld_config": str(ALFWORLD_CONFIG),
            "alfworld_data": str(DATA),
            "split": "train",
            "seed": 2026081617,
            "max_steps": 180,
            "task_ids": task_ids,
            "task_selection": "STABLE_HASH_RANK_FROM_EXECUTION_UNTOUCHED_MULTIPLICITY_POOL",
            "execution_untouched_candidate_pool_size": len(candidates),
            "prior_execution_evidence_file_count_scanned": len(scanned),
            "selected_task_prior_execution_occurrences": {
                task_id: int(task_id in executed) for task_id in task_ids
            },
        },
        "preregistered_gates": {
            "minimum_tasks": 12,
            "strictly_beats": ["neural_only", "source_permuted", "generic_scaffold"],
            "two_sided_exact_sign_p_vs_neural_max": 0.05,
            "negative_transfer_rate_vs_neural_max": 0.10,
            "target_native_ceiling_success_rate_min": 0.60,
            "changed_from_neural_action_count_min": 1,
            "source_operator_admissions_min": 24,
        },
        "integrity": {
            "file_sha256": {
                relative: _sha256((REPO / relative).resolve())
                for relative in integrity_paths
            },
        },
        "formal_results_may_change_protocol": False,
        "claim_boundary": (
            "Fresh tasks are absent from prior non-enumeration execution JSON. "
            "The target domain itself is not pristine: its previous failed formal "
            "run motivated this structural redesign."
        ),
    }
    manifest = body | {"config_sha256": stable_hash(body)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT), "config_sha256": manifest["config_sha256"],
        "candidate_pool_size": len(candidates), "task_ids": task_ids,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
