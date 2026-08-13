#!/usr/bin/env python3
"""Aggregate and integrity-check a completed WebShop V11 receipt directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402


def _verified_receipts(receipt_dir: Path, config: dict) -> list[dict]:
    expected_tasks = set(config["task_ids"])
    expected_conditions = set(config["conditions"])
    expected_pairs = {
        (task_id, condition)
        for task_id in expected_tasks
        for condition in expected_conditions
    }
    rows = []
    observed_pairs = set()
    for path in sorted(receipt_dir.glob("webshop.*.*.json")):
        row = json.loads(path.read_text())
        pair = (row.get("task_id"), row.get("condition"))
        if pair in observed_pairs:
            raise ValueError(f"duplicate receipt pair: {pair}")
        observed_pairs.add(pair)
        claimed_hash = row.get("receipt_sha256")
        unhashed = dict(row)
        unhashed.pop("receipt_sha256", None)
        if claimed_hash != stable_hash(unhashed):
            raise ValueError(f"receipt hash mismatch: {path}")
        rows.append(row)
    missing = sorted(expected_pairs - observed_pairs)
    unexpected = sorted(observed_pairs - expected_pairs)
    if missing or unexpected:
        raise ValueError(
            f"receipt matrix mismatch: missing={missing}, unexpected={unexpected}"
        )
    return rows


def aggregate(
    receipt_dir: Path,
    frozen_config: Path,
    config: dict,
    retry_manifest: Path | None,
) -> dict:
    rows = _verified_receipts(receipt_dir, config)
    expected_tasks = set(config["task_ids"])
    conditions = list(config["conditions"])
    expected_hashes = config["runtime_hashes"]
    runner_hashes = {row["runtime_hashes"]["runner"] for row in rows}
    grounder_hashes = {row["runtime_hashes"]["grounder"] for row in rows}
    if runner_hashes != {expected_hashes["runner"]}:
        raise ValueError(f"runner hash mismatch: {sorted(runner_hashes)}")
    if grounder_hashes != {expected_hashes["grounder"]}:
        raise ValueError(f"grounder hash mismatch: {sorted(grounder_hashes)}")
    matched_initial_states = all(
        len({
            row["initial_state_hash"]
            for row in rows
            if row["task_id"] == task_id
        })
        == 1
        for task_id in expected_tasks
    )
    if not matched_initial_states:
        raise ValueError("condition initial states are not matched")
    retry = None
    if retry_manifest is not None:
        retry = {
            "path": str(retry_manifest),
            "sha256": file_sha256(retry_manifest),
            "record": json.loads(retry_manifest.read_text()),
        }
    report = {
        "schema_version": 1,
        "experiment": "webshop_neural_symbolic_transfer_v11_receipt_aggregate",
        "tasks": sorted(expected_tasks, key=lambda item: int(item.split(".")[-1])),
        "receipt_count": len(rows),
        "expected_receipt_count": len(expected_tasks) * len(conditions),
        "receipt_hashes_verified": True,
        "matched_initial_state_hashes": matched_initial_states,
        "zero_failures": all(row["failure"] is None for row in rows),
        "conditions": {
            condition: {
                "strict_successes": sum(
                    row["strict_success"]
                    for row in rows
                    if row["condition"] == condition
                ),
                "mean_reward": float(np.mean([
                    row["official_reward"]
                    for row in rows
                    if row["condition"] == condition
                ])),
                "mean_steps": float(np.mean([
                    row["step_count"]
                    for row in rows
                    if row["condition"] == condition
                ])),
                "changed_from_target_rank_zero": sum(
                    row["changed_from_target_rank_zero_count"]
                    for row in rows
                    if row["condition"] == condition
                ),
                "source_decisions": sum(
                    row["source_decision_count"]
                    for row in rows
                    if row["condition"] == condition
                ),
                "failures": sum(
                    row["failure"] is not None
                    for row in rows
                    if row["condition"] == condition
                ),
            }
            for condition in conditions
        },
        "operational_retry": retry,
        "runtime_hashes": {
            "frozen_config": file_sha256(frozen_config),
            "runner": next(iter(runner_hashes)),
            "grounder": next(iter(grounder_hashes)),
        },
    }
    report["summary_sha256"] = stable_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operational-retry-manifest", type=Path)
    args = parser.parse_args()
    config = json.loads(args.frozen_config.read_text())
    report = aggregate(
        args.receipt_dir,
        args.frozen_config,
        config,
        args.operational_retry_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
