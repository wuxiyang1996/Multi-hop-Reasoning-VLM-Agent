#!/usr/bin/env python3
"""Freeze source-induced typed-effect programs and untouched reserve plans."""

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
from motif_transfer.phase3_source_induction import read_jsonl  # noqa: E402
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    induce_typed_effect_program,
    typed_intervention_sets_from_rows,
)


SOURCE_SPECS = (
    ("tetris", 16, 201001),
    ("candy_crush", 16, 202001),
    ("gymv_columns", 50, 203001),
    ("gymv_streets_of_rage_2", 50, 204001),
    ("gymv_thunder_force_iii", 23, 205001),
    ("gymv_strider", 22, 206001),
)
SOURCE_RUN = REPO / "runs/phase3_source_confirmation_v1"
V1_CONFIGS = (
    REPO / "configs/phase3_source_induction_v1/frozen_confirmation/source_configs"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        default=REPO / "configs/phase3_source_induction_v2/frozen_reserve",
    )
    parser.add_argument("--seeds-per-game", type=int, default=24)
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {manifest_path}")
    if args.seeds_per_game < 24 or args.seeds_per_game % 3:
        raise SystemExit("seeds-per-game must be a multiple of three and at least 24")

    historical = _configured_seeds()
    allocated: set[int] = set()
    receipts = []
    for game, horizon, seed_start in SOURCE_SPECS:
        rows_path = SOURCE_RUN / game / "rows.jsonl"
        examples, audit = typed_intervention_sets_from_rows(
            read_jsonl(rows_path), primary_horizon=horizon,
        )
        source_receipts_sha256 = stable_hash({
            "rows_file_sha256": _sha(rows_path),
            "eligible_snapshot_sha256s": [
                row.snapshot_sha256 for row in examples
            ],
            "audit": audit,
        })
        typed_program = induce_typed_effect_program(
            examples, source_receipts_sha256=source_receipts_sha256,
        )
        v1_artifact_path = (
            REPO / "configs/phase3_source_induction_v1/frozen_confirmation/"
            f"programs/{game}.json"
        )
        v1_artifact = _read(v1_artifact_path)
        artifact_body = {
            "schema_version": "phase3-frozen-typed-effect-lineage-v2",
            "status": "FROZEN_BEFORE_UNTOUCHED_TYPED_EFFECT_SOURCE_RESERVE",
            "source_game": game,
            "typed_effect_program": typed_program,
            # The state-machine structure remains the previously source-induced
            # anonymous program. Candidate order is now exclusively V2 typed
            # effect content, never the V1 rank profile.
            "anonymous_transition_program": v1_artifact["authentic_program"],
            "development_rows_file_sha256": _sha(rows_path),
            "development_rows_role": (
                "DISCOVERY_FOR_EFFECT_TYPE_INDUCTION;QUALIFICATION_FOR_"
                "APPLICABILITY_AND_ABSTENTION;HELDOUT_NOT_READ_BY_PROGRAM"
            ),
            "target_data_read_for_program_induction": False,
            "source_identity_used_as_runtime_feature": False,
            "v1_rank_prior_disabled": True,
        }
        artifact = artifact_body | {"artifact_sha256": stable_hash(artifact_body)}
        artifact_path = args.output_dir / "programs" / f"{game}.json"
        _write(artifact_path, artifact)

        seeds = list(range(seed_start, seed_start + args.seeds_per_game))
        overlap = set(seeds) & (historical | allocated)
        if overlap:
            raise SystemExit(f"reserve seed overlap for {game}: {sorted(overlap)}")
        allocated.update(seeds)
        base = deepcopy(_read(V1_CONFIGS / f"{game}.json"))
        base.update({
            "schema_version": "phase3-typed-effect-source-reserve-config-v2",
            "namespace": f"phase3-source-induced-typed-effect-v2-reserve:{game}",
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
                "maximum_unqualified_accuracy_for_calibration": 0.49,
                "maximum_intervention_failed_rows": 0,
            },
            "claim_boundary": (
                "UNTOUCHED_SOURCE_RESERVE_FOR_FROZEN_TYPED_EFFECT_PROGRAM;"
                "NO_REINDUCTION;NO_TARGET_DATA"
            ),
        })
        config_path = args.output_dir / "source_configs" / f"{game}.json"
        _write(config_path, base)
        receipts.append({
            "source_game": game,
            "primary_horizon": horizon,
            "program_path": str(artifact_path.relative_to(REPO)),
            "program_file_sha256": _sha(artifact_path),
            "program_artifact_sha256": artifact["artifact_sha256"],
            "typed_effect_program_sha256": typed_program["program_sha256"],
            "qualification_status": typed_program["status"],
            "selected_effect_type": typed_program["selected_effect_type"],
            "config_path": str(config_path.relative_to(REPO)),
            "config_file_sha256": _sha(config_path),
            "seed_min": min(seeds), "seed_max": max(seeds),
            "seed_count": len(seeds),
        })

    statuses = {row["qualification_status"] for row in receipts}
    effect_types = {row["selected_effect_type"] for row in receipts}
    prefreeze_gates = {
        "exact_six_source_lineages": len(receipts) == 6,
        "both_qualified_and_abstaining_sources_present": len(statuses) == 2,
        "at_least_three_source_induced_effect_types": len(effect_types) >= 3,
        "all_reserve_seeds_new": not (allocated & historical),
        "no_target_data_read_for_freeze": True,
    }
    if not all(prefreeze_gates.values()):
        raise SystemExit(f"typed-effect prefreeze gates failed: {prefreeze_gates}")
    code_paths = (
        "src/motif_transfer/phase3_typed_effect_induction.py",
        "src/motif_transfer/phase1_common_search_ir.py",
        "scripts/run_phase1_common_search_ir.py",
        "scripts/freeze_phase3_typed_source_reserve_v2.py",
    )
    body = {
        "schema_version": "phase3-typed-effect-source-reserve-manifest-v2",
        "status": "FROZEN_BEFORE_ANY_RESERVE_PLAN_OR_INTERVENTION_OUTCOME",
        "source_receipts": receipts,
        "fresh_seed_count": len(allocated),
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
                "SOURCE_PROGRAM_DERANGEMENT_V1",
            ],
            "failure_policy": "FAIL_CLOSED_NO_REPLACEMENT_SEED",
        },
        "claim_boundary": (
            "PROSPECTIVE_CONFIRMATION_OF_SOURCE_SPECIFIC_TYPED_EFFECT_"
            "APPLICABILITY_AND_ABSTENTION;NO_TARGET_TRANSFER_CLAIM"
        ),
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "fresh_seed_count": len(allocated),
        "source_profiles": [{
            key: row[key] for key in (
                "source_game", "qualification_status", "selected_effect_type"
            )
        } for row in receipts],
    }, indent=2))


if __name__ == "__main__":
    main()
