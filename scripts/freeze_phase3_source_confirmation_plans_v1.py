#!/usr/bin/env python3
"""Freeze reward-blind fresh source plans before any intervention fork."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_source_induction import file_sha256  # noqa: E402


def _read(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "configs/phase3_source_induction_v1/frozen_confirmation/manifest.json",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO / "runs/phase3_source_confirmation_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "configs/phase3_source_induction_v1/frozen_confirmation/plan_receipt.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen plan receipt: {args.output}")
    manifest = _read(args.manifest)
    manifest_body = dict(manifest)
    claimed = manifest_body.pop("manifest_sha256", None)
    if claimed != stable_hash(manifest_body):
        raise SystemExit("source confirmation manifest hash mismatch")
    receipts = []
    for source in manifest["source_receipts"]:
        game = str(source["source_game"])
        config = _read(REPO / str(source["config_path"]))
        plan_path = args.run_dir / game / "plan.json"
        rows_path = args.run_dir / game / "rows.jsonl"
        if rows_path.exists():
            raise SystemExit(f"source outcomes already exist before plan freeze: {game}")
        plan = _read(plan_path)
        plan_body = dict(plan)
        plan_claimed = plan_body.pop("plan_sha256", None)
        if plan_claimed != stable_hash(plan_body):
            raise SystemExit(f"plan hash mismatch: {game}")
        snapshots = list(plan.get("snapshots") or ())
        seeds = {int(row["seed"]) for row in snapshots}
        if seeds != set(map(int, config["seeds"])):
            raise SystemExit(f"plan seeds differ from frozen config: {game}")
        if not (len(snapshots) == len(seeds) == 48):
            raise SystemExit(f"plan must contain exactly one snapshot per seed: {game}")
        heldout = sum(str(row["split"]) == "heldout" for row in snapshots)
        if heldout != 16:
            raise SystemExit(f"plan must have exactly 16 heldout seeds: {game}")
        selection = plan.get("selection") or {}
        if selection.get("reward_read_during_plan_collection") is not False:
            raise SystemExit(f"plan selection read source rewards: {game}")
        if selection.get("content_or_outcome_used_for_selection") is not False:
            raise SystemExit(f"plan selection used source outcome: {game}")
        receipts.append({
            "source_game": game,
            "plan_path": str(plan_path.relative_to(REPO)),
            "plan_file_sha256": file_sha256(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "snapshots": len(snapshots),
            "heldout_snapshots": heldout,
            "reward_read_during_plan_collection": False,
            "source_outcome_visible_at_plan_freeze": False,
        })
    body = {
        "schema_version": "phase3-source-confirmation-plan-receipt-v1",
        "status": "FROZEN_AFTER_REWARD_BLIND_PLANS_BEFORE_SOURCE_FORK_OUTCOMES",
        "manifest_sha256": manifest["manifest_sha256"],
        "plans": receipts,
        "source_fork_outcomes_visible_at_freeze": False,
        "program_update_after_plan": False,
        "claim_boundary": manifest["claim_boundary"],
    }
    receipt = body | {"plan_receipt_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": receipt["status"],
        "plan_receipt_sha256": receipt["plan_receipt_sha256"],
        "plans": len(receipts),
        "heldout_snapshots": sum(row["heldout_snapshots"] for row in receipts),
    }, indent=2))


if __name__ == "__main__":
    main()
