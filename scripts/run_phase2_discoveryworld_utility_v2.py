#!/usr/bin/env python3
"""Run every eligible V2 matched fork and one deterministic abstention."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.discoveryworld_applicability_grounder_v4 import (  # noqa: E402
    call_applicability_complete_grounder, select_source_safe_candidate,
)
from motif_transfer.phase2_discoveryworld_utility_v2 import (  # noqa: E402
    CONDITIONS, file_sha256, make_cell, read_object, validate_manifest,
    validate_self_hash,
)
from motif_transfer.search_automaton_transfer_v16 import SourceSearchAutomaton  # noqa: E402
import scripts.run_discoveryworld_commit_recovery_v1 as matched_runner  # noqa: E402
import scripts.run_phase1_direct_discoveryworld_v1 as direct_runner  # noqa: E402


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_one(manifest: dict, task_id: str, keys: Path, output_root: Path) -> dict:
    task = next(row for row in manifest["tasks"] if row["task_id"] == task_id)
    cell_dir = output_root / "cells" / task_id
    cell_path = cell_dir / "cell.json"
    if cell_path.is_file():
        cell = read_object(cell_path)
        validate_self_hash(cell, "cell_sha256")
        return cell
    start_body = {
        "schema_version": "phase2-discoveryworld-selective-start-v2",
        "manifest_sha256": manifest["manifest_sha256"], "task_id": task_id,
        "applicable": task["applicable"],
    }
    write(cell_dir / "started.json", start_body | {"start_sha256": stable_hash(start_body)})
    if not task["applicable"]:
        episode = read_object(REPO / task["target_episode"])
        outcome = bool(episode["evaluation"]["official_success"])
        cell = make_cell(
            manifest_sha256=manifest["manifest_sha256"], task=task,
            outcomes={condition: outcome for condition in CONDITIONS},
            recovery_steps={condition: 0 for condition in CONDITIONS}, routes=[],
            matched_result_file_sha256=None, all_matched_forks=True,
            all_selection_receipts_valid=True, runtime_error=None,
        )
        write(cell_path, cell)
        return cell
    source = SourceSearchAutomaton(
        read_object(REPO / task["source_artifact"]),
        expected_sha256=task["source_artifact_sha256"],
    )
    matched_runner.call_grounder = call_applicability_complete_grounder
    matched_runner.select_candidate = select_source_safe_candidate
    suffix = "\nReturn one valid json object."
    if not matched_runner.TARGET_BINDER_SYSTEM_PROMPT.endswith(suffix):
        matched_runner.TARGET_BINDER_SYSTEM_PROMPT += suffix
    result_path = cell_dir / "matched_result.json"
    result, routes, runtime_error = direct_runner._run_matched_with_online_source(
        config_path=REPO / task["fork_config"], keys=keys, output_path=result_path,
        source=source, task_id=task_id,
    )
    conditions = result.get("conditions", {})
    outcomes = {
        condition: bool(conditions.get(condition, {}).get("official_success"))
        for condition in CONDITIONS
    }
    recovery = {
        condition: len(conditions.get(condition, {}).get("recovery") or ())
        for condition in CONDITIONS
    }
    if tuple(name for name in CONDITIONS if name in conditions) != CONDITIONS:
        runtime_error = runtime_error or "matched conditions incomplete"
    cell = make_cell(
        manifest_sha256=manifest["manifest_sha256"], task=task,
        outcomes=outcomes, recovery_steps=recovery, routes=routes,
        matched_result_file_sha256=file_sha256(result_path) if result_path.is_file() else "",
        all_matched_forks=bool(result.get("all_matched_forks")),
        all_selection_receipts_valid=bool(result.get("all_selection_receipts_valid")),
        runtime_error=runtime_error,
    )
    write(cell_path, cell)
    return cell


def run_child(command: list[str], log: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    return {"task_id": command[command.index("--task-id") + 1], "exit_code": completed.returncode}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "configs/phase2_discoveryworld_utility_v2/manifest.json")
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--output-root", type=Path, default=REPO / "runs/phase2_discoveryworld_utility_v2")
    parser.add_argument("--task-id")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    validate_manifest(manifest, repo=REPO)
    if args.task_id:
        cell = run_one(manifest, args.task_id, args.keys, args.output_root)
        print(json.dumps({"task_id": args.task_id, "outcomes": cell["outcomes"], "error": cell["runtime_error"]}))
        return 0 if cell["runtime_error"] is None else 2
    tasks = [row["task_id"] for row in manifest["tasks"]]
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        for task_id in tasks:
            command = [sys.executable, str(Path(__file__).resolve()), "--manifest", str(args.manifest), "--keys", str(args.keys), "--output-root", str(args.output_root), "--task-id", task_id]
            futures.append(pool.submit(run_child, command, args.output_root / "logs" / f"{task_id}.log"))
        for future in as_completed(futures):
            row = future.result(); results.append(row); print(json.dumps(row), flush=True)
    failed = [row for row in results if row["exit_code"]]
    print(json.dumps({"complete": len(results)-len(failed), "failed": failed}, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
