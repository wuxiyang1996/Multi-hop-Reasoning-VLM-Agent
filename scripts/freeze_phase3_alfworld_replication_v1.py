#!/usr/bin/env python3
"""Freeze consumed qualification and untouched valid-seen multiplicity reserve."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_alfworld_transfer import CONDITIONS  # noqa: E402
from motif_transfer.phase3_alfworld_typed_grounder import (  # noqa: E402
    validate_artifact,
)


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, body: dict[str, Any]) -> None:
    payload = body | {"config_sha256": stable_hash(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounder", type=Path, required=True)
    parser.add_argument("--qualification-output", type=Path, required=True)
    parser.add_argument("--formal-output", type=Path, required=True)
    args = parser.parse_args()
    if args.qualification_output.exists() or args.formal_output.exists():
        raise SystemExit("refusing to overwrite a frozen ALFWorld config")
    grounder_path = args.grounder.resolve()
    grounder = _read(grounder_path)
    validate_artifact(grounder)
    if grounder.get("status") not in {
        "ALFWORLD_TYPED_GROUNDING_QUALIFIED",
        "ALFWORLD_TYPED_GROUNDING_AND_ABSTENTION_QUALIFIED",
        "ALFWORLD_TYPED_OPTION_BINDING_QUALIFIED",
        "ALFWORLD_INTERVENTION_GROUNDER_QUALIFIED",
        "ALFWORLD_INTERVENTION_GROUNDER_AND_ABSTENTION_QUALIFIED",
    }:
        raise SystemExit("ALFWorld typed grounder did not pass qualification")
    consumed_path = REPO / "configs/alfworld_multiplicity_v1_consumed_development_manifest.json"
    consumed = _read(consumed_path)
    qualification_ids = tuple(map(str,
        consumed["cells"]["alfworld_valid_unseen"]["splits"]["qualification"]
    ))
    data = Path(
        "/fs/gamma-projects/vlm-robot/"
        "Multi-hop-Reasoning-VLM-Agent-github-main/.cache/alfworld_data"
    ).resolve()
    valid_seen = (data / "json_2.1.1/valid_seen").resolve()
    formal_reserve = tuple(sorted(
        path.relative_to(valid_seen).as_posix()
        for path in valid_seen.glob("pick_two_obj_and_place-*/*/game.tw-pddl")
    ))
    if len(formal_reserve) != 24:
        raise SystemExit(
            "installed executable valid_seen multiplicity population changed: "
            f"expected 24, observed {len(formal_reserve)}"
        )
    source_rows = []
    for row in grounder["source_programs"]:
        path = Path(str(row["path"])).resolve()
        source_rows.append({
            "path": str(path.relative_to(REPO)),
            "file_sha256": _sha256(path),
            "artifact_sha256": row["artifact_sha256"],
        })
    dependencies = (
        "scripts/run_phase3_alfworld_replication.py",
        "src/motif_transfer/phase3_alfworld_transfer.py",
        "src/motif_transfer/phase3_alfworld_typed_grounder.py",
        "src/motif_transfer/alfworld_env.py",
        "src/motif_transfer/phase3_attempt_runtime.py",
        "src/motif_transfer/phase3_source_portfolio.py",
        "src/motif_transfer/phase3_typed_effect_induction.py",
    )
    integrity = {
        relative: _sha256(REPO / relative) for relative in dependencies
    }
    common = {
        "schema_version": "phase3-alfworld-replication-config-v1",
        "conditions": list(CONDITIONS),
        "grounder": {
            "path": str(grounder_path.relative_to(REPO)),
            "file_sha256": _sha256(grounder_path),
            "artifact_sha256": grounder["artifact_sha256"],
            "formal_success_read_for_training_or_qualification": False,
            "qualification_thresholds_frozen": True,
        },
        "source_programs": source_rows,
        "target": {
            "alfworld_config": (
                "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-"
                "source-fresh-v1/configs/alfworld_base_config.yaml"
            ),
            "alfworld_data": str(data),
            "seed": 2026081602,
            "max_steps": 180,
        },
        "integrity": {"file_sha256": integrity},
        "same_symbolic_ir_as_discoveryworld": True,
        "only_target_native_grounder_replaced": True,
        "source_identity_used_as_runtime_feature": False,
        "adaptive_history": grounder.get("abstention_calibration"),
        "target_capability_definition": (
            "PER_TASK_UNION_OF_TARGET_NATIVE_OFFICIAL_EXPERT_"
            "NEURAL_ONLY_AND_SOURCE_FREE_GENERIC;SOURCE_INDUCED_EXCLUDED"
        ),
    }
    qualification = common | {
        "status": "FROZEN_BEFORE_CONSUMED_QUALIFICATION_RESET",
        "role": "qualification",
        "claim_boundary": (
            "Already-consumed valid_unseen multiplicity tasks; may qualify the "
            "same-IR adapter but cannot support a fresh replication claim."
        ),
        "target": common["target"] | {
            "split": "eval_out_of_distribution",
            "task_ids": list(qualification_ids),
        },
        "gates": {
            "expected_tasks": len(qualification_ids),
            "minimum_ceiling_successes": 8,
            "maximum_negative_transfer_rate": 0.10,
            "minimum_changed_actions": 4,
            "minimum_permuted_first_action_contrasts": 4,
            "minimum_selected_effect_types": 2,
        },
        "formal_reserve_read": False,
    }
    formal = common | {
        "status": "FROZEN_BEFORE_QUALIFICATION_AND_ANY_FORMAL_RESET",
        "role": "formal",
        "claim_boundary": (
            "Prospective same-IR evaluation over the complete executable "
            "valid_seen multiplicity population. Historical target exposure is "
            "possible, so this is an in-distribution mechanism replication, not "
            "an untouched or new valid_unseen claim."
        ),
        "target": common["target"] | {
            "split": "eval_in_distribution",
            "task_ids": list(formal_reserve),
        },
        "reserve_audit": {
            "selection_rule": (
                "All executable pick_two_obj_and_place tasks in the pinned "
                "valid_seen installation; no outcome-based task selection."
            ),
            "historical_target_exposure_possible": True,
            "outcomes_used_for_selection": False,
            "tasks": len(formal_reserve),
        },
        "gates": {
            "expected_tasks": len(formal_reserve),
            "minimum_ceiling_successes": 20,
            "maximum_negative_transfer_rate": 0.0,
            "minimum_changed_actions": 8,
            "minimum_permuted_first_action_contrasts": 8,
            "minimum_selected_effect_types": 2,
        },
        "formal_results_may_change_protocol": False,
    }
    _write(args.qualification_output.resolve(), qualification)
    _write(args.formal_output.resolve(), formal)
    print(json.dumps({
        "qualification": str(args.qualification_output.resolve()),
        "qualification_tasks": len(qualification_ids),
        "formal": str(args.formal_output.resolve()),
        "formal_tasks": len(formal_reserve),
        "grounder_artifact_sha256": grounder["artifact_sha256"],
        "formal_reserve_outcomes_used_for_selection": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
