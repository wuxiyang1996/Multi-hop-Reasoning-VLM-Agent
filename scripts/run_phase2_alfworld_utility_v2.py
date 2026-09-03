#!/usr/bin/env python3
"""Run the frozen ALFWorld V2 matrix with one task per environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase2_alfworld_utility_v2 import build_report, validate_manifest  # noqa: E402
from motif_transfer.search_automaton_transfer_v16 import SourceSearchAutomaton  # noqa: E402
from motif_transfer.webshop_search_automaton_v16 import CONDITIONS  # noqa: E402
from scripts.run_alfworld_search_automaton_v16 import _run_episode  # noqa: E402


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite formal evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest, repo=REPO)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    preflight_body = dict(preflight)
    claimed = preflight_body.pop("preflight_sha256", None)
    if stable_hash(preflight_body) != claimed:
        raise RuntimeError("preflight hash mismatch")
    if preflight.get("status") != "PHASE2_ALFWORLD_V2_PREFLIGHT_PASSED" or preflight.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError("matching V2 preflight did not pass")
    if args.run_dir.exists():
        raise RuntimeError(f"formal run directory already exists: {args.run_dir}")
    receipts_dir, started_dir = args.run_dir / "receipts", args.run_dir / "started"
    receipts_dir.mkdir(parents=True)
    started_dir.mkdir(parents=True)
    target_grounder = json.loads((REPO / manifest["target_grounder"]).read_text(encoding="utf-8"))["target_grounder"]
    split_root = Path(manifest["alfworld_data_root"]) / "json_2.1.1" / "valid_seen"
    receipts = []
    for condition in CONDITIONS:
        for index, task in enumerate(manifest["tasks"]):
            identity = str(task["target_identity"])
            marker_body = {
                "manifest_sha256": manifest["manifest_sha256"], "target_identity": identity,
                "condition": condition, "reset_count_before_marker": 0,
                "action_count_before_marker": 0, "outcome_count_before_marker": 0,
            }
            _write_once(started_dir / f"cell_{index:02d}.{condition}.json", marker_body | {"marker_sha256": stable_hash(marker_body)})
            environment = ALFWorldTextBatchEnvironment(
                config_path=str(manifest["alfworld_config"]), data_path=str(manifest["alfworld_data_root"]),
                split=str(manifest["target_split"]), seed=int(manifest["seed"]),
                game_ids=[identity], max_steps=int(manifest["max_steps"]),
            )
            try:
                source_artifact = json.loads((REPO / task["source_artifact"]).read_text(encoding="utf-8"))
                source = SourceSearchAutomaton(source_artifact, expected_sha256=str(task["source_artifact_sha256"]))
                episode = _run_episode(
                    environment=environment, condition=condition, source=source,
                    target_grounder=target_grounder, max_steps=int(manifest["max_steps"]),
                )
                actual = str(Path(environment.resolved_game_file).relative_to(split_root))
                if actual != identity:
                    raise RuntimeError(f"single-task environment mismatch: {actual} != {identity}")
            finally:
                environment.close()
            body = {
                "schema_version": "phase2-alfworld-cell-receipt-v2",
                "manifest_sha256": manifest["manifest_sha256"], "target_identity": identity,
                "task_family": task["task_family"], "condition": condition,
                "source_game": task["source_game"], "source_artifact_sha256": task["source_artifact_sha256"],
                "initial_state_hash": str(episode["records"][0]["before_state_sha256"]) if episode["records"] else "",
                "strict_success": bool(episode["official_success"]),
                "pass_success": bool(episode["official_success"]),
                "official_reward": float(bool(episode["official_success"])),
                "step_count": int(episode["steps"]), "steps": episode["records"],
                "v16_controller": {
                    "source_decisions": int(episode["source_decisions"]),
                    "source_action_counts": dict(episode["source_action_counts"]),
                },
                "target_outcomes": dict(episode["target_outcomes"]),
                "failure": None, "unsafe_commits": [],
                "target_reset_or_sample_open_count": 1, "historical_target_outcome_reused": False,
            }
            receipt = body | {"receipt_sha256": stable_hash(body)}
            _write_once(receipts_dir / f"cell_{index:02d}.{condition}.json", receipt)
            receipts.append(receipt)
            print(json.dumps({
                "condition": condition, "task_index": index, "source_game": task["source_game"],
                "success": episode["official_success"], "steps": episode["steps"],
            }), flush=True)
    report = build_report(manifest, receipts)
    _write_once(args.run_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"], "summaries": report["summaries"],
        "paired": report["paired"], "gates": report["gates"], "report_sha256": report["report_sha256"],
    }, indent=2), flush=True)
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
