#!/usr/bin/env python3
"""Execute the frozen selective ALFWorld V3 matrix once."""

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
from motif_transfer.phase2_alfworld_utility_v3 import build_report, validate_manifest  # noqa: E402
from motif_transfer.search_automaton_transfer_v16 import SourceSearchAutomaton  # noqa: E402
from motif_transfer.webshop_search_automaton_v16 import CONDITIONS, RAW  # noqa: E402
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
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, repo=REPO)
    preflight = json.loads(args.preflight.read_text())
    preflight_body = dict(preflight)
    claimed = preflight_body.pop("preflight_sha256", None)
    if stable_hash(preflight_body) != claimed:
        raise RuntimeError("preflight self-hash mismatch")
    if preflight.get("status") != "PHASE2_ALFWORLD_V3_PREFLIGHT_PASSED" or preflight.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError("matching V3 preflight did not pass")
    if args.run_dir.exists():
        raise RuntimeError(f"formal run already exists: {args.run_dir}")
    receipts_dir, started_dir = args.run_dir / "receipts", args.run_dir / "started"
    receipts_dir.mkdir(parents=True)
    started_dir.mkdir(parents=True)
    target_grounder = json.loads((REPO / manifest["target_grounder"]).read_text())["target_grounder"]
    split_root = Path(manifest["alfworld_data_root"]) / "json_2.1.1" / "valid_seen"
    receipts = []
    for condition in CONDITIONS:
        for index, task in enumerate(manifest["tasks"]):
            identity = task["target_identity"]
            effective_condition = condition if task["transfer_applicable"] or condition == RAW else RAW
            marker_body = {
                "manifest_sha256": manifest["manifest_sha256"], "target_identity": identity,
                "condition": condition, "effective_condition": effective_condition,
                "reset_count_before_marker": 0, "action_count_before_marker": 0,
                "outcome_count_before_marker": 0,
            }
            _write_once(started_dir / f"cell_{index:02d}.{condition}.json", marker_body | {"marker_sha256": stable_hash(marker_body)})
            env = ALFWorldTextBatchEnvironment(
                config_path=manifest["alfworld_config"], data_path=manifest["alfworld_data_root"],
                split=manifest["target_split"], seed=manifest["seed"],
                game_ids=[identity], max_steps=manifest["max_steps"],
            )
            try:
                artifact = json.loads((REPO / task["source_artifact"]).read_text())
                source = SourceSearchAutomaton(artifact, expected_sha256=task["source_artifact_sha256"])
                episode = _run_episode(
                    environment=env, condition=effective_condition, source=source,
                    target_grounder=target_grounder, max_steps=manifest["max_steps"],
                )
                actual = str(Path(env.resolved_game_file).relative_to(split_root))
                if actual != identity:
                    raise RuntimeError(f"target identity mismatch: {actual} != {identity}")
            finally:
                env.close()
            body = {
                "schema_version": "phase2-alfworld-selective-cell-v3",
                "manifest_sha256": manifest["manifest_sha256"], "target_identity": identity,
                "task_family": task["task_family"], "condition": condition,
                "effective_condition": effective_condition,
                "transfer_applicable": task["transfer_applicable"],
                "source_game": task["source_game"], "source_artifact_sha256": task["source_artifact_sha256"],
                "initial_state_hash": episode["records"][0]["before_state_sha256"] if episode["records"] else "",
                "strict_success": bool(episode["official_success"]),
                "pass_success": bool(episode["official_success"]),
                "official_reward": float(bool(episode["official_success"])),
                "step_count": episode["steps"], "steps": episode["records"],
                "v16_controller": {
                    "source_decisions": episode["source_decisions"],
                    "source_action_counts": episode["source_action_counts"],
                },
                "target_outcomes": episode["target_outcomes"], "failure": None,
                "unsafe_commits": [], "target_reset_or_sample_open_count": 1,
                "historical_target_outcome_reused": False,
            }
            receipt = body | {"receipt_sha256": stable_hash(body)}
            _write_once(receipts_dir / f"cell_{index:02d}.{condition}.json", receipt)
            receipts.append(receipt)
            print(json.dumps({
                "condition": condition, "task_index": index,
                "applicable": task["transfer_applicable"],
                "success": episode["official_success"], "steps": episode["steps"],
            }), flush=True)
    report = build_report(manifest, receipts)
    _write_once(args.run_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"], "summaries": report["summaries"],
        "paired": report["paired"], "negative_transfer_rate": report["discordant_negative_transfer_rate"],
        "gates": report["gates"], "report_sha256": report["report_sha256"],
    }, indent=2), flush=True)
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
