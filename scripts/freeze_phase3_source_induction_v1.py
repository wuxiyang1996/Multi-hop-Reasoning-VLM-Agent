#!/usr/bin/env python3
"""Freeze anonymous source programs and fresh confirmation configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_source_induction import (  # noqa: E402
    build_lineage_report,
    file_sha256,
)


SOURCE_SPECS = (
    (
        "tetris",
        "configs/phase1_common_search_ir_formal_v1/tetris.json",
        "runs/phase1_common_search_ir_formal_v1/tetris/rows.jsonl",
        16,
        101001,
    ),
    (
        "candy_crush",
        "configs/phase1_common_search_ir_formal_v1/candy_crush.json",
        "runs/phase1_common_search_ir_formal_v1/candy_crush/rows.jsonl",
        16,
        102001,
    ),
    (
        "gymv_columns",
        "configs/phase1_common_search_ir_formal_v1/gymv_columns.json",
        "runs/phase1_common_search_ir_formal_v1/gymv_columns/rows.jsonl",
        50,
        103001,
    ),
    (
        "gymv_streets_of_rage_2",
        "configs/phase1_common_search_ir_streets_formal_v2/gymv_streets_of_rage_2.json",
        "runs/phase1_common_search_ir_streets_formal_v2/gymv_streets_of_rage_2/rows.jsonl",
        50,
        104001,
    ),
    (
        "gymv_thunder_force_iii",
        "configs/phase1_common_search_ir_formal_v1/gymv_thunder_force_iii.json",
        "runs/phase1_common_search_ir_formal_v1/gymv_thunder_force_iii/rows.jsonl",
        23,
        105001,
    ),
    (
        "gymv_strider",
        "configs/phase1_common_search_ir_formal_v1/gymv_strider.json",
        "runs/phase1_common_search_ir_formal_v1/gymv_strider/rows.jsonl",
        22,
        106001,
    ),
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _historical_seed_inventory() -> set[int]:
    values: set[int] = set()
    for path in sorted((REPO / "configs").rglob("*.json")):
        try:
            root = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        stack = [root]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                raw = value.get("seeds")
                if isinstance(raw, list):
                    values.update(
                        int(seed) for seed in raw
                        if isinstance(seed, int) and not isinstance(seed, bool)
                    )
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-config",
        type=Path,
        default=REPO / "configs/phase3_source_induction_v1/development.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "configs/phase3_source_induction_v1/frozen_confirmation",
    )
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {manifest_path}")
    development = _read(args.development_config)
    thresholds = dict(development["thresholds"])
    historical_seeds = _historical_seed_inventory()
    all_new_seeds: set[int] = set()
    receipts = []

    for game, base_relative, rows_relative, horizon, seed_start in SOURCE_SPECS:
        base_path = REPO / base_relative
        rows_path = REPO / rows_relative
        report = build_lineage_report(
            source_game=game,
            rows_path=rows_relative,
            primary_horizon=horizon,
            thresholds=thresholds,
        )
        if report["authentic_program"]["status"] != "SOURCE_INDUCED_PROGRAM_QUALIFIED":
            raise SystemExit(f"historical source induction did not qualify: {game}")
        program_body = {
            "schema_version": "phase3-frozen-lineage-program-v1",
            "status": "FROZEN_BEFORE_FRESH_SOURCE_CONFIRMATION",
            "source_game": game,
            "authentic_program": report["authentic_program"],
            "shuffled_effect_program": report["shuffled_effect_program"],
            "source_only_profile": report["source_profile"],
            "historical_rows_file_sha256": file_sha256(rows_path),
            "historical_data_role": (
                "DEVELOPMENT_AND_QUALIFICATION_ONLY_FOR_PROGRAM_CONTENT;"
                "OLD_HELDOUT_REPLAY_NOT_A_NEW_CONFIRMATORY_CLAIM"
            ),
            "target_data_read_for_program_induction": False,
            "forbidden_named_policy_tokens": [
                "EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY",
            ],
        }
        program = program_body | {"artifact_sha256": stable_hash(program_body)}
        program_path = args.output_dir / "programs" / f"{game}.json"
        _write(program_path, program)

        seeds = list(range(seed_start, seed_start + 48))
        overlap = (set(seeds) & historical_seeds) | (set(seeds) & all_new_seeds)
        if overlap:
            raise SystemExit(f"fresh source seed overlap for {game}: {sorted(overlap)}")
        all_new_seeds.update(seeds)
        config = deepcopy(_read(base_path))
        config.update({
            "schema_version": "phase3-source-confirmation-config-v1",
            "namespace": f"phase3-source-induction-v1-confirmation:{game}",
            "seeds": seeds,
            "expected_policy_sha256": None,
            "frozen_program_path": str(program_path.relative_to(REPO)),
            "frozen_program_file_sha256": file_sha256(program_path),
            "phase3_confirmation_gates": {
                "minimum_planned_heldout_seeds": 16,
                "minimum_fresh_eligible_ledgers": 8,
                "minimum_fresh_eligible_fraction": 0.40,
                "minimum_authentic_closed_loop_success_rate": 1.0,
                "maximum_each_destructive_control_success_rate": 0.25,
                "maximum_intervention_failed_rows": 0
            },
            "claim_boundary": (
                "FRESH_SOURCE_CONFIRMATION_ONLY;FROZEN_ANONYMOUS_PROGRAM;"
                "NO_REINDUCTION_FROM_CONFIRMATION_ROWS;NO_TARGET_DATA"
            ),
        })
        config_path = args.output_dir / "source_configs" / f"{game}.json"
        _write(config_path, config)
        extra_receipts = {}
        for key in ("option_template_artifact", "source_runtime_script"):
            raw = config.get(key)
            if raw:
                path = Path(str(raw))
                extra_receipts[key] = {
                    "path": str(path), "file_sha256": file_sha256(path),
                }
        receipts.append({
            "source_game": game,
            "base_config_path": base_relative,
            "base_config_file_sha256": file_sha256(base_path),
            "historical_rows_path": rows_relative,
            "historical_rows_file_sha256": file_sha256(rows_path),
            "program_path": str(program_path.relative_to(REPO)),
            "program_file_sha256": file_sha256(program_path),
            "program_artifact_sha256": program["artifact_sha256"],
            "config_path": str(config_path.relative_to(REPO)),
            "config_file_sha256": file_sha256(config_path),
            "seed_min": min(seeds),
            "seed_max": max(seeds),
            "seed_count": len(seeds),
            "extra_runtime_receipts": extra_receipts,
        })

    code_paths = (
        REPO / "src/motif_transfer/phase3_source_induction.py",
        REPO / "scripts/run_phase3_source_induction_v1.py",
        REPO / "scripts/freeze_phase3_source_induction_v1.py",
        REPO / "src/motif_transfer/phase1_common_search_ir.py",
        REPO / "scripts/run_phase1_common_search_ir.py",
    )
    body = {
        "schema_version": "phase3-source-confirmation-manifest-v1",
        "status": "FROZEN_BEFORE_FRESH_SOURCE_PLAN_OR_OUTCOME",
        "source_games": [row[0] for row in SOURCE_SPECS],
        "source_confirmation_outcomes_visible_at_freeze": False,
        "target_data_read_for_freeze": False,
        "program_induction_uses_historical_heldout": False,
        "fresh_seed_count": len(all_new_seeds),
        "source_receipts": receipts,
        "code_receipts": [
            {
                "path": str(path.relative_to(REPO)),
                "file_sha256": file_sha256(path),
            }
            for path in code_paths
        ],
        "confirmation_protocol": {
            "plan_selection": "REWARD_BLIND_HASH_RANKED_OBSERVED_PREFIX",
            "matched_duplicate_forks": 2,
            "program_update_from_confirmation_rows": False,
            "primary_split": "HELDOUT_SEEDS_ONLY",
            "conditions": [
                "frozen_source_induced_authentic",
                "frozen_shuffled_effect_program",
                "authentic_program_with_runtime_effect_permutation"
            ],
            "failure_policy": "FAIL_CLOSED_NO_REPLACEMENT_SEED",
        },
        "claim_boundary": (
            "PROSPECTIVE_SOURCE_CONFIRMATION_OF_INDUCED_ANONYMOUS_PROGRAM;"
            "NO_TARGET_TRANSFER_UTILITY_CLAIM"
        ),
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    _write(manifest_path, manifest)
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "programs": len(receipts),
        "fresh_source_seeds": len(all_new_seeds),
    }, indent=2))


if __name__ == "__main__":
    main()
