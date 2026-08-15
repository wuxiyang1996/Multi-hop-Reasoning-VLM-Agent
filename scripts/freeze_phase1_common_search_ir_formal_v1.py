#!/usr/bin/env python3
"""Freeze the six-game common-search-IR formal protocol before collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FORMAL_DIR = REPO / "configs" / "phase1_common_search_ir_formal_v1"
GAMES = (
    "tetris",
    "candy_crush",
    "gymv_columns",
    "gymv_streets_of_rage_2",
    "gymv_thunder_force_iii",
    "gymv_strider",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=FORMAL_DIR / "manifest.json"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {args.output}")

    pilot_seeds: set[int] = set()
    for path in sorted((REPO / "configs").glob("phase1_common_search_ir_v*_pilot/*.json")):
        pilot_seeds.update(map(int, json.loads(path.read_text())["seeds"]))

    config_receipts = []
    formal_seeds: set[int] = set()
    for game in GAMES:
        path = FORMAL_DIR / f"{game}.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        seeds = set(map(int, config["seeds"]))
        if seeds & pilot_seeds:
            raise SystemExit(f"{game}: formal seeds overlap pilot seeds")
        if seeds & formal_seeds:
            raise SystemExit(f"{game}: formal seeds overlap another formal game")
        formal_seeds.update(seeds)
        expected = {
            "maximum_intervention_failed_rows": 0,
            "minimum_eligible_fraction_each_split": 0.40,
            "maximum_infrastructure_attempts_per_snapshot": 2,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise SystemExit(f"{game}: formal top-level gate drift")
        gate = config["source_gate"]
        if gate["minimum_fresh_eligible_states"] != 8:
            raise SystemExit(f"{game}: fresh-state gate drift")
        if gate["minimum_fresh_examples_per_selected_action"] != 8:
            raise SystemExit(f"{game}: per-action support gate drift")
        if gate["minimum_authentic_minus_each_destructive_control"] != 0.40:
            raise SystemExit(f"{game}: destructive-control gate drift")
        config_receipts.append({
            "game": game,
            "path": str(path.resolve()),
            "file_sha256": _sha256(path),
            "seed_count": len(seeds),
            "option_grounding": (
                config.get("template_strategy")
                or f"native_candidates_{config['continuation_mode']}"
            ),
        })

    code_paths = (
        REPO / "src" / "motif_transfer" / "phase1_common_search_ir.py",
        REPO / "scripts" / "run_phase1_common_search_ir.py",
        REPO / "src" / "motif_transfer" / "sokoban_search_automaton_v16.py",
    )
    pilot_reports = sorted(
        path for path in (REPO / "runs").glob(
            "phase1_common_search_ir_v*_pilot*/**/report*.json"
        ) if path.is_file()
    )
    body = {
        "schema_version": "phase1-common-search-ir-formal-manifest-v1",
        "status": "FROZEN_BEFORE_FORMAL_SOURCE_COLLECTION",
        "games": list(GAMES),
        "target_data_read_for_freeze": False,
        "protocol": {
            "reward_blind_prefix_and_snapshot_selection": True,
            "matched_duplicate_option_forks": 2,
            "whole_snapshot_retry_only_on_infrastructure_failure": True,
            "outcome_used_for_retry": False,
            "native_action_tokens_exported_to_common_ir": False,
            "required_canonical_policy_sha256": (
                "02d3adc83616688ae0b51b152bae3ce8ab468f0fc3a99ac0f27b7d399e696fdf"
            ),
        },
        "config_receipts": config_receipts,
        "code_receipts": [
            {"path": str(path.resolve()), "file_sha256": _sha256(path)}
            for path in code_paths
        ],
        "pilot_report_receipts": [
            {"path": str(path.resolve()), "file_sha256": _sha256(path)}
            for path in pilot_reports
        ],
        "claim_boundary": (
            "SOURCE_QUALIFICATION_ONLY;TARGET_UNREAD_DURING_SOURCE_FREEZE;"
            "FORMAL_OUTCOMES_NOT_USED_TO_CHANGE_THIS_MANIFEST"
        ),
    }
    artifact = body | {"manifest_sha256": _stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(args.output.resolve()),
        "manifest_sha256": artifact["manifest_sha256"],
        "games": len(GAMES),
        "formal_seeds": len(formal_seeds),
        "pilot_seeds_excluded": len(pilot_seeds),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
