#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

from motif_transfer.real_source_interventions import content_hash, summarize_source_gate, validate_plan


def _load_adapter(source_script: Path):
    spec = importlib.util.spec_from_file_location("real_source_runtime", source_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source runtime: {source_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._SourceReplayAdapter


def _sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _run_fork(adapter_class, game: str, snapshot: dict[str, Any], action: str) -> dict[str, Any]:
    status = "VALID"
    diagnostic = None
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            adapter = adapter_class(game, int(snapshot["max_steps"]))
            try:
                adapter.reset(seed=int(snapshot["seed"]))
                for prefix_action in snapshot["prefix_actions"]:
                    adapter.step(str(prefix_action))
                    if adapter.last_terminated or adapter.last_truncated:
                        status = "PREFIX_TERMINATED_EARLY"
                        diagnostic = f"terminated before fork at prefix length {len(snapshot['prefix_actions'])}"
                        break
                replayed_fork_hash = _sha256_text(adapter.state_receipt())
                replayed_actions = tuple(str(item) for item in adapter.admissible_actions())
                replayed_actions_hash = content_hash(list(replayed_actions))
                if status == "VALID" and replayed_fork_hash != snapshot["expected_fork_state_sha256"]:
                    status = "FORK_STATE_MISMATCH"
                    diagnostic = "observable state hash differs from frozen evidence"
                if status == "VALID" and action not in replayed_actions:
                    status = "ACTION_NOT_ADMISSIBLE"
                    diagnostic = "frozen candidate is absent from replayed native actions"
                after_hash = None
                reward = None
                terminated = None
                truncated = None
                if status == "VALID":
                    adapter.step(action)
                    after_hash = _sha256_text(adapter.state_receipt())
                    reward = float(adapter.last_reward)
                    terminated = bool(adapter.last_terminated)
                    truncated = bool(adapter.last_truncated)
            finally:
                adapter.close()
    row = {
        "schema_version": "real-source-intervention-receipt-v1",
        "snapshot_id": snapshot["snapshot_id"],
        "split": snapshot["split"],
        "condition": snapshot["condition"],
        "episode_id": snapshot["episode_id"],
        "seed": snapshot["seed"],
        "step": snapshot["step"],
        "action": action,
        "grounding_state": snapshot.get("grounding_state", ""),
        "is_logged_action": action == snapshot["logged_action"],
        "expected_fork_state_sha256": snapshot["expected_fork_state_sha256"],
        "replayed_fork_state_sha256": replayed_fork_hash,
        "replayed_native_actions_sha256": replayed_actions_hash,
        "after_observable_state_sha256": after_hash,
        "immediate_reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "status": status,
        "diagnostic": diagnostic,
    }
    row["receipt_sha256"] = content_hash(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan_path = Path(config["plan_path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    adapter_class = _load_adapter(Path(config["source_runtime_script"]))
    output_path = Path(config["receipts_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output_path.open("w", encoding="utf-8") as handle:
        for snapshot in plan["snapshots"]:
            for action in snapshot["selected_actions"]:
                row = _run_fork(adapter_class, str(plan["source"]["game"]), snapshot, str(action))
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    report = summarize_source_gate(rows)
    report["plan_sha256"] = plan["plan_sha256"]
    report["receipts_path"] = str(output_path.resolve())
    report["receipts_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    report_path = Path(config["gate_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "SOURCE_GATE_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
