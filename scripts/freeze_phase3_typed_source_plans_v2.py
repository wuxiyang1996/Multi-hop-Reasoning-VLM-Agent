#!/usr/bin/env python3
"""Freeze reward-blind V2 source plans before reserve fork outcomes exist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase3_source_induction_v2/frozen_reserve/manifest.json",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=REPO / "runs/phase3_typed_effect_source_reserve_v2",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            REPO / "configs/phase3_source_induction_v2/frozen_reserve/"
            "plan_receipt.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen plan receipt: {args.output}")
    manifest = _read(args.manifest)
    _self_hash(manifest, "manifest_sha256")
    if manifest.get("status") != (
        "FROZEN_BEFORE_ANY_RESERVE_PLAN_OR_INTERVENTION_OUTCOME"
    ):
        raise SystemExit("typed source reserve manifest is not frozen")
    for relative, expected in manifest["runtime_file_sha256"].items():
        if _sha(REPO / relative) != expected:
            raise SystemExit(f"frozen runtime changed before plan freeze: {relative}")

    plans = []
    for receipt in manifest["source_receipts"]:
        game = str(receipt["source_game"])
        config_path = REPO / str(receipt["config_path"])
        program_path = REPO / str(receipt["program_path"])
        if _sha(config_path) != receipt["config_file_sha256"]:
            raise SystemExit(f"frozen source config changed: {game}")
        if _sha(program_path) != receipt["program_file_sha256"]:
            raise SystemExit(f"frozen typed program changed: {game}")
        config = _read(config_path)
        plan_path = args.run_dir / game / "plan.json"
        rows_path = args.run_dir / game / "rows.jsonl"
        if rows_path.exists():
            raise SystemExit(f"reserve outcomes exist before plan freeze: {game}")
        plan = _read(plan_path)
        _self_hash(plan, "plan_sha256")
        snapshots = list(plan.get("snapshots") or ())
        seeds = set(map(int, config["seeds"]))
        if {int(row["seed"]) for row in snapshots} != seeds:
            raise SystemExit(f"plan seeds differ from frozen config: {game}")
        if len(snapshots) != len(seeds):
            raise SystemExit(f"plan must contain one snapshot per seed: {game}")
        heldout = sum(str(row["split"]) == "heldout" for row in snapshots)
        expected_heldout = len(seeds) // 3
        if heldout != expected_heldout:
            raise SystemExit(f"unexpected heldout plan count: {game}")
        selection = plan.get("selection") or {}
        if selection.get("reward_read_during_plan_collection") is not False:
            raise SystemExit(f"plan read source rewards: {game}")
        if selection.get("content_or_outcome_used_for_selection") is not False:
            raise SystemExit(f"plan used source content/outcome: {game}")
        plans.append({
            "source_game": game,
            "plan_path": str(plan_path.relative_to(REPO)),
            "plan_file_sha256": _sha(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "snapshots": len(snapshots),
            "heldout_snapshots": heldout,
            "source_outcome_visible_at_freeze": False,
        })

    evaluator_path = REPO / "scripts/evaluate_phase3_typed_source_reserve_v2.py"
    body = {
        "schema_version": "phase3-typed-effect-source-plan-receipt-v2",
        "status": "FROZEN_AFTER_REWARD_BLIND_PLANS_BEFORE_RESERVE_OUTCOMES",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": _sha(args.manifest),
        "plans": plans,
        "source_fork_outcomes_visible_at_freeze": False,
        "program_update_after_plan": False,
        "evaluator_file_sha256": _sha(evaluator_path),
        "freeze_script_file_sha256": _sha(Path(__file__)),
        "claim_boundary": manifest["claim_boundary"],
    }
    receipt = body | {"plan_receipt_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": receipt["status"],
        "plan_receipt_sha256": receipt["plan_receipt_sha256"],
        "plans": len(plans),
        "heldout_snapshots": sum(row["heldout_snapshots"] for row in plans),
    }, indent=2))


if __name__ == "__main__":
    main()
