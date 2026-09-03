#!/usr/bin/env python3
"""Freeze cross-batch-calibrated typed programs and a third source reserve."""

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
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    recalibrate_typed_effect_program,
    validate_typed_effect_program,
)


SOURCE_SPECS = (
    ("tetris", 16, 301001),
    ("candy_crush", 16, 302001),
    ("gymv_columns", 50, 303001),
    ("gymv_streets_of_rage_2", 50, 304001),
    ("gymv_thunder_force_iii", 23, 305001),
    ("gymv_strider", 22, 306001),
)
V2_ROOT = REPO / "configs/phase3_source_induction_v2/frozen_reserve"
V2_REPORT = REPO / "runs/phase3_typed_effect_source_reserve_v2/report.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _configured_seeds() -> set[int]:
    values: set[int] = set()
    for path in (REPO / "configs").rglob("*.json"):
        try:
            root = json.loads(path.read_text(encoding="utf-8"))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "configs/phase3_source_induction_v3/frozen_reserve",
    )
    parser.add_argument("--seeds-per-game", type=int, default=24)
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {manifest_path}")
    if args.seeds_per_game < 24 or args.seeds_per_game % 3:
        raise SystemExit("seeds-per-game must be a multiple of three and at least 24")

    v2_report = _read(V2_REPORT)
    _self_hash(v2_report, "report_sha256")
    if v2_report.get("status") != "SOURCE_SPECIFIC_TYPED_EFFECT_TRANSFER_GATE_FAILED":
        raise SystemExit("V2 calibration authority must be the frozen failed report")
    report_lineages = {
        row["source_game"]: row for row in v2_report["lineages"]
    }
    historical = _configured_seeds()
    allocated: set[int] = set()
    receipts = []
    for game, horizon, seed_start in SOURCE_SPECS:
        v2_artifact_path = V2_ROOT / "programs" / f"{game}.json"
        v2_artifact = _read(v2_artifact_path)
        _self_hash(v2_artifact, "artifact_sha256")
        v2_program = v2_artifact["typed_effect_program"]
        calibration = report_lineages[game]["evaluations"]
        calibrated = recalibrate_typed_effect_program(
            v2_program,
            calibration_metrics=calibration,
            calibration_receipt_sha256=stable_hash({
                "v2_report_sha256": v2_report["report_sha256"],
                "source_game": game,
                "rows_file_sha256": report_lineages[game]["rows_file_sha256"],
                "heldout_metrics": calibration,
            }),
        )
        validate_typed_effect_program(calibrated)
        artifact_body = {
            "schema_version": "phase3-frozen-typed-effect-lineage-v3",
            "status": "FROZEN_BEFORE_THIRD_UNTOUCHED_SOURCE_RESERVE",
            "source_game": game,
            "typed_effect_program": calibrated,
            "anonymous_transition_program": v2_artifact[
                "anonymous_transition_program"
            ],
            "v2_program_artifact_sha256": v2_artifact["artifact_sha256"],
            "v2_failed_report_sha256": v2_report["report_sha256"],
            "calibration_policy": (
                "KEEP_V2_EFFECT_TYPE_FIXED;REMOVE_OPERATOR_IF_INDEPENDENT_"
                "SOURCE_RESERVE_FAILS;NEVER_RESCUE_PRIOR_ABSTENTION"
            ),
            "target_data_read_for_program_induction_or_calibration": False,
            "source_identity_used_as_runtime_feature": False,
            "v1_rank_prior_disabled": True,
        }
        artifact = artifact_body | {"artifact_sha256": stable_hash(artifact_body)}
        artifact_path = args.output_dir / "programs" / f"{game}.json"
        _write(artifact_path, artifact)

        seeds = list(range(seed_start, seed_start + args.seeds_per_game))
        overlap = set(seeds) & (historical | allocated)
        if overlap:
            raise SystemExit(f"third reserve seed overlap for {game}: {sorted(overlap)}")
        allocated.update(seeds)
        config = deepcopy(_read(V2_ROOT / "source_configs" / f"{game}.json"))
        config.update({
            "schema_version": "phase3-typed-effect-source-reserve-config-v3",
            "namespace": f"phase3-source-induced-typed-effect-v3-reserve:{game}",
            "seeds": seeds,
            "frozen_program_path": str(artifact_path.relative_to(REPO)),
            "frozen_program_file_sha256": _sha(artifact_path),
            "typed_effect_reserve_gates": {
                "minimum_planned_heldout_seeds": args.seeds_per_game // 3,
                "minimum_fresh_eligible_ledgers": 6,
                "minimum_fresh_eligible_fraction": 0.50,
                "minimum_qualified_accuracy": 0.50,
                "minimum_qualified_authentic_minus_shuffled": 0.25,
                "minimum_qualified_varying_effect_fraction": 0.50,
                "maximum_intervention_failed_rows": 0,
            },
            "claim_boundary": (
                "THIRD_UNTOUCHED_SOURCE_RESERVE_FOR_CROSS_BATCH_CALIBRATED_"
                "TYPED_EFFECT_PROGRAM;NO_REINDUCTION;NO_TARGET_DATA"
            ),
        })
        config_path = args.output_dir / "source_configs" / f"{game}.json"
        _write(config_path, config)
        receipts.append({
            "source_game": game,
            "primary_horizon": horizon,
            "program_path": str(artifact_path.relative_to(REPO)),
            "program_file_sha256": _sha(artifact_path),
            "program_artifact_sha256": artifact["artifact_sha256"],
            "typed_effect_program_sha256": calibrated["program_sha256"],
            "qualification_status": calibrated["status"],
            "selected_effect_type": calibrated["selected_effect_type"],
            "config_path": str(config_path.relative_to(REPO)),
            "config_file_sha256": _sha(config_path),
            "seed_min": min(seeds), "seed_max": max(seeds),
            "seed_count": len(seeds),
        })

    qualified = [
        row for row in receipts
        if row["qualification_status"] == "SOURCE_TYPED_EFFECT_PROGRAM_QUALIFIED"
    ]
    abstaining = [row for row in receipts if row not in qualified]
    prefreeze_gates = {
        "exact_six_source_lineages": len(receipts) == 6,
        "exact_three_cross_batch_qualified_sources": len(qualified) == 3,
        "exact_three_cross_batch_abstaining_sources": len(abstaining) == 3,
        "qualified_sources_have_three_distinct_effect_types": len({
            row["selected_effect_type"] for row in qualified
        }) == 3,
        "candy_false_positive_now_abstains": next(
            row for row in receipts if row["source_game"] == "candy_crush"
        )["qualification_status"] == "SOURCE_TYPED_EFFECT_ABSTENTION_INDUCED",
        "all_third_reserve_seeds_new": not (allocated & historical),
        "no_target_data_read_for_freeze": True,
    }
    if not all(prefreeze_gates.values()):
        raise SystemExit(f"V3 prefreeze gates failed: {prefreeze_gates}")
    code_paths = (
        "src/motif_transfer/phase3_typed_effect_induction.py",
        "src/motif_transfer/phase1_common_search_ir.py",
        "scripts/run_phase1_common_search_ir.py",
        "scripts/freeze_phase3_typed_source_reserve_v3.py",
    )
    body = {
        "schema_version": "phase3-typed-effect-source-reserve-manifest-v3",
        "status": "FROZEN_BEFORE_ANY_RESERVE_PLAN_OR_INTERVENTION_OUTCOME",
        "source_receipts": receipts,
        "fresh_seed_count": len(allocated),
        "v2_calibration_report_sha256": v2_report["report_sha256"],
        "prefreeze_gates": prefreeze_gates,
        "runtime_file_sha256": {
            path: _sha(REPO / path) for path in code_paths
        },
        "reserve_protocol": {
            "reward_blind_plan": True,
            "matched_repeats": 2,
            "heldout_split_only_for_confirmation": True,
            "controls": [
                "DETERMINISTIC_EFFECT_BINDING_SHUFFLE_V1",
                "QUALIFICATION_STATUS_MATCHED_SOURCE_PROGRAM_DERANGEMENT_V1",
            ],
            "failure_policy": "FAIL_CLOSED_NO_REPLACEMENT_SEED",
        },
        "claim_boundary": (
            "PROSPECTIVE_CONFIRMATION_OF_CROSS_BATCH_CALIBRATED_SOURCE_"
            "SPECIFIC_TYPED_EFFECT_APPLICABILITY_AND_ABSTENTION;NO_TARGET_CLAIM"
        ),
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "qualified": [row["source_game"] for row in qualified],
        "abstaining": [row["source_game"] for row in abstaining],
        "fresh_seed_count": len(allocated),
    }, indent=2))


if __name__ == "__main__":
    main()
