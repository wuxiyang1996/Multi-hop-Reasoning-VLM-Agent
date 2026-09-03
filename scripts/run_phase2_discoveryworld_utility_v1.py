#!/usr/bin/env python3
"""Execute the frozen 36-task DiscoveryWorld matched utility matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.discoveryworld_applicability_grounder_v4 import (  # noqa: E402
    call_applicability_complete_grounder,
    select_source_safe_candidate,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase2_discoveryworld_utility_v1 import (  # noqa: E402
    CONDITIONS, file_sha256, make_cell_receipt, read_object,
    validate_manifest, validate_self_hash,
)
from motif_transfer.search_automaton_transfer_v16 import SourceSearchAutomaton  # noqa: E402
import scripts.run_discoveryworld_commit_recovery_v1 as matched_runner  # noqa: E402
import scripts.run_phase1_direct_discoveryworld_v1 as direct_runner  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_by_id(manifest: dict[str, Any], task_id: str) -> dict[str, Any]:
    rows = [row for row in manifest["tasks"] if row["task_id"] == task_id]
    if len(rows) != 1:
        raise ValueError(f"expected one frozen task: {task_id}")
    return dict(rows[0])


def run_one(
    *, manifest: dict[str, Any], task_id: str, keys: Path, output_root: Path,
) -> dict[str, Any]:
    task = task_by_id(manifest, task_id)
    cell_dir = output_root / "cells" / task_id
    cell_path = cell_dir / "cell.json"
    if cell_path.is_file():
        existing = read_object(cell_path)
        validate_self_hash(existing, "cell_sha256")
        if existing.get("source_artifact_sha256") != task["source_artifact_sha256"]:
            raise RuntimeError(f"incompatible resume cell: {task_id}")
        return existing
    fork_config = output_root / "frozen_forks" / f"{task_id}.json"
    if not fork_config.is_file():
        raise RuntimeError(f"frozen eligible fork missing: {task_id}")
    source_path = REPO / str(task["source_artifact"])
    source = SourceSearchAutomaton(
        read_object(source_path), expected_sha256=str(task["source_artifact_sha256"]),
    )

    # V4's applicability-complete neural grounder is a frozen component of this
    # manifest.  The suffix is transport-only and does not alter candidate or
    # symbolic semantics.
    matched_runner.call_grounder = call_applicability_complete_grounder
    matched_runner.select_candidate = select_source_safe_candidate
    suffix = "\nReturn one valid json object."
    if not matched_runner.TARGET_BINDER_SYSTEM_PROMPT.endswith(suffix):
        matched_runner.TARGET_BINDER_SYSTEM_PROMPT += suffix

    matched_path = cell_dir / "matched_result.json"
    start_body = {
        "schema_version": "phase2-discoveryworld-cell-start-v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "task_id": task_id,
        "source_game": task["source_game"],
        "source_artifact_sha256": task["source_artifact_sha256"],
    }
    write_json(
        cell_dir / "started.json",
        start_body | {"start_sha256": stable_hash(start_body)},
    )
    result, routes, runtime_error = direct_runner._run_matched_with_online_source(
        config_path=fork_config,
        keys=keys,
        output_path=matched_path,
        source=source,
        task_id=task_id,
    )
    if result and tuple(
        name for name in CONDITIONS if name in result.get("conditions", {})
    ) != CONDITIONS:
        runtime_error = runtime_error or "matched conditions incomplete or reordered"
    receipt = make_cell_receipt(
        task=task,
        result=result,
        routes=routes,
        matched_result_file_sha256=(file_sha256(matched_path) if matched_path.is_file() else ""),
        runtime_error=runtime_error,
    )
    write_json(cell_path, receipt)
    return receipt


def run_child(command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    task_id = command[command.index("--task-id") + 1]
    return {"task_id": task_id, "exit_code": int(completed.returncode)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase2_discoveryworld_utility_v1/manifest.json",
    )
    parser.add_argument(
        "--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "runs/phase2_discoveryworld_utility_v1",
    )
    parser.add_argument("--task-id")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    validate_manifest(manifest, repo=REPO)
    if args.task_id:
        receipt = run_one(
            manifest=manifest, task_id=args.task_id,
            keys=args.keys, output_root=args.output_root,
        )
        print(json.dumps({
            "task_id": args.task_id,
            "complete": receipt["runtime_error"] is None,
            "outcomes": receipt["outcomes"],
        }, indent=2))
        return 0 if receipt["runtime_error"] is None else 2

    preparation = args.output_root / "preparation_receipt.json"
    if not preparation.is_file():
        raise SystemExit("run preparation before matched execution")
    prep = read_object(preparation)
    validate_self_hash(prep, "preparation_receipt_sha256")
    if prep.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("preparation receipt belongs to another manifest")
    tasks = [str(row["task_id"]) for row in manifest["tasks"]]
    workers = max(1, min(args.workers, len(tasks)))
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task_id in tasks:
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--manifest", str(args.manifest), "--keys", str(args.keys),
                "--output-root", str(args.output_root), "--task-id", task_id,
            ]
            futures.append(pool.submit(
                run_child, command, args.output_root / "logs" / f"{task_id}.matched.log",
            ))
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    failed = [row for row in results if row["exit_code"] != 0]
    print(json.dumps({
        "status": "MATCHED_MATRIX_COMPLETE" if not failed else "MATCHED_MATRIX_INCOMPLETE",
        "completed": len(results) - len(failed), "failed": failed,
    }, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
