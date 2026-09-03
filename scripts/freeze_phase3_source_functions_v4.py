#!/usr/bin/env python3
"""Freeze source-induced domain functions and a fourth untouched reserve."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_source_function_induction import (  # noqa: E402
    ABSTAINING,
    QUALIFIED,
    evaluate_source_function,
    function_weights,
    induce_source_function_program,
    recalibrate_source_function_program,
    validate_source_function_program,
)
from motif_transfer.phase3_source_induction import read_jsonl  # noqa: E402
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    typed_intervention_sets_from_rows,
)


SOURCE_SPECS = (
    ("tetris", 16, 401001),
    ("candy_crush", 16, 402001),
    ("gymv_columns", 50, 403001),
    ("gymv_streets_of_rage_2", 50, 404001),
    ("gymv_thunder_force_iii", 23, 405001),
    ("gymv_strider", 22, 406001),
)
DEVELOPMENT_ROOT = REPO / "runs/phase3_source_confirmation_v1"
CALIBRATION_ROOT = REPO / "runs/phase3_typed_effect_source_reserve_v3"
BASE_CONFIG_ROOT = REPO / "configs/phase3_source_induction_v3/frozen_reserve"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _configured_seeds() -> set[int]:
    values = set()
    for path in (REPO / "configs").rglob("*.json"):
        try:
            root = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        stack = [root]
        while stack:
            item = stack.pop()
            if isinstance(item, Mapping):
                seeds = item.get("seeds")
                if isinstance(seeds, list):
                    values.update(
                        int(seed) for seed in seeds
                        if isinstance(seed, int) and not isinstance(seed, bool)
                    )
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "configs/phase3_source_function_v4/frozen_reserve",
    )
    parser.add_argument("--seeds-per-game", type=int, default=24)
    args = parser.parse_args()
    if args.seeds_per_game < 24 or args.seeds_per_game % 3:
        raise SystemExit("seeds-per-game must be a multiple of three and at least 24")
    if (args.output_dir / "manifest.json").exists():
        raise SystemExit("refusing to overwrite V4 source-function manifest")

    historical = _configured_seeds()
    allocated = set()
    receipts = []
    for game, primary_horizon, seed_start in SOURCE_SPECS:
        development_path = DEVELOPMENT_ROOT / game / "rows.jsonl"
        calibration_path = CALIBRATION_ROOT / game / "rows.jsonl"
        development, development_audit = typed_intervention_sets_from_rows(
            read_jsonl(development_path), primary_horizon=primary_horizon,
        )
        induced = induce_source_function_program(
            development,
            source_receipts_sha256=stable_hash({
                "development_rows_file_sha256": _sha(development_path),
                "eligible_snapshot_sha256s": [
                    row.snapshot_sha256 for row in development
                    if row.source_split in {"discovery", "qualification"}
                ],
                "audit": development_audit,
            }),
        )
        calibration, calibration_audit = typed_intervention_sets_from_rows(
            read_jsonl(calibration_path), primary_horizon=primary_horizon,
        )
        weights = function_weights(induced)
        authentic = evaluate_source_function(
            calibration, weights=weights, source_split="heldout",
        )
        shuffled = evaluate_source_function(
            calibration, weights=weights, source_split="heldout",
            shuffled_effects=True,
        )
        calibrated = recalibrate_source_function_program(
            induced,
            calibration_metrics={
                "authentic": authentic,
                "shuffled_effect_binding": shuffled,
            },
            calibration_receipt_sha256=stable_hash({
                "calibration_rows_file_sha256": _sha(calibration_path),
                "eligible_snapshot_sha256s": [
                    row.snapshot_sha256 for row in calibration
                    if row.source_split == "heldout"
                ],
                "audit": calibration_audit,
                "authentic": authentic,
                "shuffled": shuffled,
            }),
        )
        validate_source_function_program(calibrated)
        artifact_body = {
            "schema_version": "phase3-frozen-source-domain-function-v4",
            "status": "FROZEN_BEFORE_FOURTH_UNTOUCHED_SOURCE_RESERVE",
            "source_function_program": calibrated,
            "development_rows_file_sha256": _sha(development_path),
            "cross_batch_calibration_rows_file_sha256": _sha(calibration_path),
            "program_body_fields_frozen": [
                "source_function.terms",
                "source_function.required_observation_horizon",
                "source_function.retry_after_low",
                "source_function.maximum_trials",
                "transition_graph",
                "abstention_rule",
            ],
            "target_data_read_for_program_induction_or_calibration": False,
            "source_identity_used_as_runtime_feature": False,
            "legacy_canonical_attempt_program_used": False,
        }
        artifact = artifact_body | {"artifact_sha256": stable_hash(artifact_body)}
        artifact_path = args.output_dir / "programs" / f"{game}.json"
        _write(artifact_path, artifact)

        seeds = list(range(seed_start, seed_start + args.seeds_per_game))
        overlap = set(seeds) & (historical | allocated)
        if overlap:
            raise SystemExit(f"V4 source reserve seed overlap for {game}: {sorted(overlap)}")
        allocated.update(seeds)
        config = deepcopy(_read(BASE_CONFIG_ROOT / "source_configs" / f"{game}.json"))
        config.update({
            "schema_version": "phase3-source-domain-function-reserve-config-v4",
            "namespace": f"phase3-source-domain-function-v4-reserve:{game}",
            "seeds": seeds,
            "frozen_program_path": str(artifact_path.relative_to(REPO)),
            "frozen_program_file_sha256": _sha(artifact_path),
            "source_function_reserve_gates": {
                "minimum_planned_heldout_seeds": args.seeds_per_game // 3,
                "minimum_fresh_eligible_ledgers": 6,
                "minimum_fresh_eligible_fraction": 0.50,
                "minimum_qualified_accuracy": 0.50,
                "minimum_qualified_authentic_minus_shuffled": 0.25,
                "minimum_qualified_varying_effect_fraction": 0.50,
                "maximum_intervention_failed_rows": 0,
            },
            "claim_boundary": (
                "FOURTH_UNTOUCHED_SOURCE_RESERVE_FOR_SOURCE_INDUCED_DOMAIN_"
                "FUNCTIONS;NO_REINDUCTION;NO_TARGET_DATA"
            ),
        })
        config_path = args.output_dir / "source_configs" / f"{game}.json"
        _write(config_path, config)
        receipts.append({
            "source_game": game,
            "primary_horizon": primary_horizon,
            "program_path": str(artifact_path.relative_to(REPO)),
            "program_file_sha256": _sha(artifact_path),
            "program_artifact_sha256": artifact["artifact_sha256"],
            "source_function_program_sha256": calibrated["program_sha256"],
            "qualification_status": calibrated["status"],
            "function_terms": calibrated["source_function"]["terms"],
            "required_observation_horizon": calibrated["source_function"][
                "required_observation_horizon"
            ],
            "retry_after_low": calibrated["source_function"]["retry_after_low"],
            "config_path": str(config_path.relative_to(REPO)),
            "config_file_sha256": _sha(config_path),
            "seed_min": min(seeds), "seed_max": max(seeds),
            "seed_count": len(seeds),
        })

    qualified = [row for row in receipts if row["qualification_status"] == QUALIFIED]
    abstaining = [row for row in receipts if row["qualification_status"] == ABSTAINING]
    prefreeze_gates = {
        "exact_six_source_lineages": len(receipts) == 6,
        "at_least_three_qualified_source_functions": len(qualified) >= 3,
        "qualified_and_abstaining_functions_present": bool(qualified and abstaining),
        "qualified_program_bodies_are_content_distinct": len({
            stable_hash({
                "terms": row["function_terms"],
                "horizon": row["required_observation_horizon"],
                "retry": row["retry_after_low"],
            }) for row in qualified
        }) == len(qualified),
        "qualified_retry_and_no_retry_graphs_present": {
            bool(row["retry_after_low"]) for row in qualified
        } == {False, True},
        "all_fourth_reserve_seeds_new": not (allocated & historical),
        "legacy_canonical_attempt_program_disabled": True,
        "no_target_data_read_for_freeze": True,
    }
    if not all(prefreeze_gates.values()):
        raise SystemExit(f"V4 source-function prefreeze gates failed: {prefreeze_gates}")
    code_paths = (
        "src/motif_transfer/phase3_source_function_induction.py",
        "src/motif_transfer/phase3_typed_effect_induction.py",
        "src/motif_transfer/phase1_common_search_ir.py",
        "scripts/run_phase1_common_search_ir.py",
        "scripts/freeze_phase3_source_functions_v4.py",
    )
    body = {
        "schema_version": "phase3-source-domain-function-reserve-manifest-v4",
        "status": "FROZEN_BEFORE_ANY_RESERVE_PLAN_OR_INTERVENTION_OUTCOME",
        "source_receipts": receipts,
        "fresh_seed_count": len(allocated),
        "prefreeze_gates": prefreeze_gates,
        "runtime_file_sha256": {path: _sha(REPO / path) for path in code_paths},
        "reserve_protocol": {
            "reward_blind_plan": True,
            "matched_repeats": 2,
            "heldout_split_only_for_confirmation": True,
            "controls": [
                "FULL_TYPED_EFFECT_VECTOR_BINDING_SHUFFLE_V1",
                "QUALIFICATION_STATUS_MATCHED_SOURCE_FUNCTION_DERANGEMENT_V1",
            ],
            "failure_policy": "FAIL_CLOSED_NO_REPLACEMENT_SEED",
        },
        "claim_boundary": (
            "PROSPECTIVE_SOURCE_ONLY_VALIDATION_OF_DOMAIN_SPECIFIC_FUNCTION_"
            "BODY_HORIZON_RETRY_GRAPH_AND_ABSTENTION;NO_TARGET_CLAIM"
        ),
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "qualified": [row["source_game"] for row in qualified],
        "abstaining": [row["source_game"] for row in abstaining],
        "program_bodies": [{
            key: row[key] for key in (
                "source_game", "function_terms", "required_observation_horizon",
                "retry_after_low",
            )
        } for row in receipts],
        "fresh_seed_count": len(allocated),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
