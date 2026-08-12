#!/usr/bin/env python3
"""Freeze a V8 candidate before any selected adaptation reset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from motif_transfer.alfworld_masked_effect_grounder import (
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.parameterized_alfworld_harness import (
    validate_property_router,
)
from motif_transfer.slot_aware_alfworld_harness import (
    parameterize_slot_source_ir,
    validate_slot_source_ir,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise SystemExit(f"invalid frozen artifact hash: {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-harness", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--slot-harness-code", type=Path, required=True)
    parser.add_argument("--runner-code", type=Path, required=True)
    parser.add_argument(
        "--transfer-scope",
        choices=("full", "relational"),
        default="full",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V8 candidate: {args.output}")
    v7 = _read(args.v7_harness)
    _validate_hash(v7, "harness_sha256")
    if v7.get("status") != "FRESH_CONFIRMATION_AUTHORIZED":
        raise SystemExit("V7 dependency was not a frozen authorized Harness")
    manifest = _read(args.manifest)
    _validate_hash(manifest, "manifest_sha256")
    if not str(manifest.get("schema_version", "")).startswith(
        "slot-aware-alfworld-manifest-v8"
    ):
        raise SystemExit("wrong V8 manifest schema")
    if manifest.get("status") not in {
        "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET",
        "FROZEN_BEFORE_ANY_REVISED_ADAPTATION_RESET",
    }:
        raise SystemExit("V8 manifest was not frozen before reset")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("V8 manifest selection used target outcomes")
    if set(manifest.get("splits", {})) != {
        "adaptation_gate", "fresh_confirmation"
    }:
        raise SystemExit("V8 manifest lacks its two frozen splits")
    source_path = Path(str(v7["source_report"]["path"]))
    target_path = Path(str(v7["target_grounder"]["path"]))
    if _sha256(source_path) != v7["source_report"]["file_sha256"]:
        raise SystemExit("V7 source report changed")
    if _sha256(target_path) != v7["target_grounder"]["file_sha256"]:
        raise SystemExit("V7 target grounder changed")
    source = _read(source_path)
    target = _read(target_path)
    validate_target_artifact(target)
    router = dict(v7["property_router"])
    validate_property_router(router)
    source_ir = parameterize_slot_source_ir(source["effect_ir"])
    validate_slot_source_ir(source_ir)
    development_report = None
    if args.development_report:
        development = _read(args.development_report)
        _validate_hash(development, "report_sha256")
        if development.get("status") != "ADAPTATION_GATE_FAILED_STOP":
            raise SystemExit("V8 development report is not a stopped gate")
        development_report = {
            "path": str(args.development_report.resolve()),
            "file_sha256": _sha256(args.development_report),
            "report_sha256": development["report_sha256"],
            "use_authority": "CONSUMED_DEVELOPMENT_DIAGNOSIS_ONLY",
        }
    inherited = v7["thresholds"]
    thresholds = {
        "selection_authority": (
            "INHERITED_FROM_CONSUMED_V7_BEFORE_V8_ADAPTATION_RESET"
        ),
        "minimum_property_confidence": float(
            inherited["minimum_property_confidence"]
        ),
        "minimum_role_binding": float(inherited["minimum_role_binding"]),
        "minimum_realization_score": float(
            inherited["selected_minimum_realization_score"]
        ),
        "minimum_target_policy_ratio": float(
            inherited["selected_minimum_target_policy_ratio"]
        ),
    }
    if args.transfer_scope == "relational":
        transfer_scope = {
            "name": "CROSS_ENGINE_BIND_RELATE_SLOT_TRANSFER",
            "allowed_source_effects": ["BIND", "RELATE"],
            "active_required_properties": ["NONE"],
            "claimed_changed_effects": ["BIND", "RELATE"],
            "primary_target_families": [
                "pick_and_place_simple",
                "pick_two_obj_and_place",
            ],
            "excluded_claim": (
                "NO_MUTATE_OR_PROPERTY_TRANSFER_CLAIM_AFTER_V8_FULL_GRAPH_GATE"
            ),
        }
    else:
        transfer_scope = {
            "name": "FULL_BIND_MUTATE_RELATE_TRANSFER",
            "allowed_source_effects": ["BIND", "MUTATE", "RELATE"],
            "active_required_properties": list(
                ("NONE", "CLEAN", "HEAT", "COOL", "LIGHT")
            ),
            "claimed_changed_effects": ["BIND", "MUTATE", "RELATE"],
            "primary_target_families": list(
                ("look_at_obj_in_light", "pick_and_place_simple",
                 "pick_clean_then_place_in_recep",
                 "pick_cool_then_place_in_recep",
                 "pick_heat_then_place_in_recep",
                 "pick_two_obj_and_place")
            ),
            "excluded_claim": None,
        }
    body = {
        "schema_version": "slot-aware-alfworld-candidate-v8",
        "status": "ADAPTATION_GATE_ONLY",
        "claim_boundary": (
            "FROZEN_BEFORE_V8_ADAPTATION_RESET; CONFIRMATION_FORBIDDEN_UNTIL_"
            "CLOSED_LOOP_GATE_PASSES; EXISTING_VALID_UNSEEN_HELDOUT_FORBIDDEN"
        ),
        "v7_harness": {
            "path": str(args.v7_harness.resolve()),
            "file_sha256": _sha256(args.v7_harness),
            "harness_sha256": v7["harness_sha256"],
        },
        "source_report": {
            "path": str(source_path.resolve()),
            "file_sha256": _sha256(source_path),
            "parent_ir_sha256": source["effect_ir"]["ir_sha256"],
        },
        "slot_source_ir": source_ir,
        "target_grounder": {
            "path": str(target_path.resolve()),
            "file_sha256": _sha256(target_path),
            "artifact_sha256": target["artifact_sha256"],
        },
        "property_router": router,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "file_sha256": _sha256(args.manifest),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "development_report": development_report,
        "implementation": {
            "slot_harness": {
                "path": str(args.slot_harness_code.resolve()),
                "file_sha256": _sha256(args.slot_harness_code),
            },
            "runner": {
                "path": str(args.runner_code.resolve()),
                "file_sha256": _sha256(args.runner_code),
            },
        },
        "thresholds": thresholds,
        "transfer_scope": transfer_scope,
        "adaptation_gates": {
            "authentic_success_noninferior_to_target_only": True,
            "paired_net_win_nonnegative": True,
            "changed_effects_each_claimed_effect": 1,
            "changed_tasks": 3,
            "source_admission_rate_range": [0.005, 0.30],
            "reopened_completed_slots": 0,
            "failed_selected_postconditions": 0,
        },
        "permissions": {
            "source_ir": [
                "SELECT_TYPED_EFFECT",
                "ORDER_SLOT_CONDITIONED_EFFECTS",
            ],
            "target_native": [
                "PREDICT_REQUIRED_PROPERTY",
                "GROUND_TARGET_ACTIONS",
                "PARSE_TARGET_GOAL_ROLES",
                "UPDATE_LEDGER_FROM_OBSERVED_POSTCONDITIONS",
                "ABSTAIN_TO_EXACT_TARGET_POLICY",
            ],
            "forbidden": [
                "SOURCE_ACTION_OR_COORDINATE_AT_RUNTIME",
                "SOURCE_TASK_ID_AT_RUNTIME",
                "OFFICIAL_SUCCESS_FOR_ACTION_SELECTION",
                "FRESH_CONFIRMATION_BEFORE_GATE",
                "EXISTING_VALID_UNSEEN_RESET",
            ],
        },
    }
    result = body | {"candidate_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": result["status"],
        "candidate_sha256": result["candidate_sha256"],
        "slot_source_ir_sha256": source_ir["ir_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "thresholds": thresholds,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
