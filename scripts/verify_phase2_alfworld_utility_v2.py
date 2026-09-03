#!/usr/bin/env python3
"""One-task-environment reset-only preflight for ALFWorld Phase-2 V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase2_alfworld_utility_v2 import validate_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite preflight: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest, repo=REPO)
    split_root = Path(manifest["alfworld_data_root"]) / "json_2.1.1" / "valid_seen"
    rows = []
    for task in manifest["tasks"]:
        expected = str(task["target_identity"])
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(manifest["alfworld_config"]), data_path=str(manifest["alfworld_data_root"]),
            split=str(manifest["target_split"]), seed=int(manifest["seed"]),
            game_ids=[expected], max_steps=int(manifest["max_steps"]),
        )
        try:
            observation = environment.reset()
            actual = str(Path(environment.resolved_game_file).relative_to(split_root))
            rows.append({
                "expected_target_identity": expected, "actual_target_identity": actual,
                "initial_state_sha256": stable_hash({
                    "observation": observation.state.get("observation", ""),
                    "goal": observation.state.get("task_goal", ""),
                    "native_actions": observation.native_actions,
                }),
            })
        finally:
            environment.close()
    gates = {
        "manifest_valid_before_preflight": True,
        "exactly_32_reset_only_checks": len(rows) == 32,
        "one_task_environment_identity_exact": all(row["expected_target_identity"] == row["actual_target_identity"] for row in rows),
        "unique_initial_state_hashes": len({row["initial_state_sha256"] for row in rows}) == 32,
        "zero_actions": True, "zero_outcomes_read": True, "zero_provider_calls": True,
    }
    body = {
        "schema_version": "phase2-alfworld-single-task-preflight-v2",
        "status": "PHASE2_ALFWORLD_V2_PREFLIGHT_PASSED" if all(gates.values()) else "PHASE2_ALFWORLD_V2_PREFLIGHT_FAILED",
        "manifest_sha256": manifest["manifest_sha256"], "rows": rows, "gates": gates,
    }
    result = body | {"preflight_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "tasks": len(rows), "gates": gates, "preflight_sha256": result["preflight_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
