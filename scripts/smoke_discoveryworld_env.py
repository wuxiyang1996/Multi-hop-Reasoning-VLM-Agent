#!/usr/bin/env python3
"""Run a no-model deterministic smoke test against official DiscoveryWorld."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.discoveryworld_env import DiscoveryWorldEnvironment, stable_hash  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", default="Proteomics")
    parser.add_argument("--difficulty", default="Easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thread-id", type=int, default=93000)
    args = parser.parse_args()
    action = {"action": "TELEPORT_TO_LOCATION", "arg1": "Animal Area 1"}
    rows = []
    for repetition in range(2):
        frame_dir = args.output.parent / "frames" / f"repeat_{repetition}"
        env = DiscoveryWorldEnvironment(
            scenario=args.scenario,
            difficulty=args.difficulty,
            seed=args.seed,
            max_steps=4,
            thread_id=args.thread_id + repetition,
            include_vision=False,
            frame_dir=frame_dir,
        )
        initial = env.reset()
        after, receipt = env.step(action)
        rows.append({
            "initial_policy_state_sha256": initial.policy_state_sha256,
            "initial_audit_world_sha256": receipt.before_audit_world_sha256,
            "after_policy_state_sha256": after.policy_state_sha256,
            "after_audit_world_sha256": receipt.after_audit_world_sha256,
            "transition_receipt": asdict(receipt),
        })
    gates = {
        "matched_initial_policy_state": rows[0]["initial_policy_state_sha256"] == rows[1]["initial_policy_state_sha256"],
        "matched_initial_hidden_state": rows[0]["initial_audit_world_sha256"] == rows[1]["initial_audit_world_sha256"],
        "matched_after_policy_state": rows[0]["after_policy_state_sha256"] == rows[1]["after_policy_state_sha256"],
        "matched_after_hidden_state": rows[0]["after_audit_world_sha256"] == rows[1]["after_audit_world_sha256"],
        "action_succeeded": all(row["transition_receipt"]["action_succeeded"] for row in rows),
        "receipts_valid": all(
            row["transition_receipt"]["receipt_sha256"]
            == stable_hash({
                key: value for key, value in row["transition_receipt"].items()
                if key != "receipt_sha256"
            })
            for row in rows
        ),
        "zero_oracle_scorecard_use": all(
            not row["transition_receipt"]["runtime_saw_oracle_scorecard"] for row in rows
        ),
    }
    payload = {
        "schema_version": "discoveryworld-environment-smoke-v1",
        "official_environment_commit": "fd591323920be0d3786ef350955de1945aa571e5",
        "scenario": args.scenario,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "action": action,
        "gates": gates,
        "passed": all(gates.values()),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "gates": gates}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
