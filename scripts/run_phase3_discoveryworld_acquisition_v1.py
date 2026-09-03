#!/usr/bin/env python3
"""Collect frozen Phase-3 DiscoveryWorld acquisition episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
DISCOVERYWORLD_CHECKOUT = REPO.parent / "discoveryworld-official"
if DISCOVERYWORLD_CHECKOUT.is_dir():
    sys.path.insert(0, str(DISCOVERYWORLD_CHECKOUT))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.phase3_discoveryworld_grounding import (  # noqa: E402
    call_qualified_decision,
)
import scripts.run_discoveryworld_target_only_v1 as base  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _backend(config, key: str, cache_path: Path):
    os.environ["PHASE3_DISCOVERYWORLD_OPENROUTER_KEY"] = key
    return MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            str(config["model"]["base_url"]),
            {
                "decision": str(config["model"]["model"]),
                "affordance": str(config["model"]["affordance_model"]),
            },
            api_key_env="PHASE3_DISCOVERYWORLD_OPENROUTER_KEY",
            json_mode=True, temperature=float(config["model"]["temperature"]),
            timeout_seconds=180,
            request_overrides={
                "max_tokens": int(config["model"]["maximum_output_tokens"]),
            },
            transport_attempts=3,
        ),
        cache_path=cache_path,
    )


def _qualified_call(*, backend, observation, memory, hypotheses, recent, attempts):
    decision, action, raw, audit, fallback = call_qualified_decision(
        backend=backend, observation=observation, memory=memory,
        hypotheses=hypotheses, recent=recent, attempts=attempts,
    )
    if fallback:
        decision = {**decision, "reason": "SCHEMA_FAILURE_FALLBACK"}
    return decision, action, raw, audit


def run_one(
    *, manifest_path: Path, task: Mapping[str, Any], key: str,
    output_dir: Path, task_index: int,
) -> dict[str, Any]:
    manifest = _read(manifest_path)
    task_id = str(task["task_id"])
    runtime_hashes = {
        "manifest": _file_sha256(manifest_path),
        "runner": _file_sha256(Path(__file__)),
        "target_episode_runner": _file_sha256(
            REPO / "scripts/run_discoveryworld_target_only_v1.py"
        ),
        "grounder": _file_sha256(
            REPO / "src/motif_transfer/phase3_discoveryworld_grounding.py"
        ),
        "environment_wrapper": _file_sha256(
            REPO / "src/motif_transfer/discoveryworld_env.py"
        ),
        "official_environment_commit": manifest["official_environment_commit"],
    }
    config = {
        "claim_boundary": manifest["claim_boundary"],
        "model": dict(manifest["acquisition_model"]),
        "runtime": dict(manifest["acquisition_runtime"]),
    }
    backend = _backend(
        config, key,
        output_dir / "decision_caches" / f"{task_id}.json",
    )
    old_call = base._call_decision
    base._call_decision = _qualified_call
    try:
        return base.run_episode(
            task={
                "scenario": task["scenario"],
                "difficulty": task["difficulty"],
                "seed": task["seed"],
            },
            config=config, backend=backend, output_dir=output_dir,
            runtime_hashes=runtime_hashes,
            thread_id=int(manifest["acquisition_runtime"]["thread_id_base"]) + task_index,
        )
    finally:
        base._call_decision = old_call


def _worker(manifest_path: str, task: dict, key: str, output_dir: str, index: int):
    try:
        receipt = run_one(
            manifest_path=Path(manifest_path), task=task, key=key,
            output_dir=Path(output_dir), task_index=index,
        )
        return {
            "task_id": task["task_id"], "error": None,
            "official_success": bool(receipt["evaluation"]["official_success"]),
            "steps": len(receipt["steps"]),
            "schema_fallback_steps": receipt["schema_fallback_steps"],
            "invalid_native_actions": receipt["invalid_native_actions"],
            "episode_sha256": receipt["episode_sha256"],
        }
    except BaseException as exc:
        return {"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    manifest = _read(args.manifest)
    _self_hash(manifest, "manifest_sha256")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_PHASE3_TARGET_RESET_OR_OUTCOME":
        raise SystemExit("Phase-3 manifest is not frozen")
    for path, expected in manifest["runtime_file_sha256"].items():
        if _file_sha256(REPO / path) != expected:
            raise SystemExit(f"frozen runtime changed: {path}")
    tasks = list(manifest["tasks"])
    if args.task_id:
        selected = set(args.task_id)
        tasks = [row for row in tasks if row["task_id"] in selected]
        if len(tasks) != len(selected):
            raise SystemExit("unknown task ID")
    values = runpy.run_path(str(args.keys))
    key = values.get(manifest["acquisition_model"]["api_key_name"])
    if not key:
        raise SystemExit("configured OpenRouter key is missing")
    workers = min(
        len(tasks), args.workers or int(manifest["acquisition_runtime"]["task_workers"]),
    )
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _worker, str(args.manifest), dict(task), str(key),
                str(args.output_dir), manifest["tasks"].index(task),
            ): task["task_id"]
            for task in tasks
        }
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: row["task_id"])
    complete = [row for row in rows if row["error"] is None]
    body = {
        "schema_version": "phase3-discoveryworld-acquisition-summary-v1",
        "status": (
            "PHASE3_DISCOVERYWORLD_ACQUISITION_COMPLETE"
            if len(complete) == len(rows) else
            "PHASE3_DISCOVERYWORLD_ACQUISITION_INCOMPLETE"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": len(rows), "complete_tasks": len(complete),
        "official_successes": sum(row["official_success"] for row in complete),
        "steps": sum(row["steps"] for row in complete),
        "schema_fallback_steps": sum(
            row["schema_fallback_steps"] for row in complete
        ),
        "invalid_native_actions": sum(
            row["invalid_native_actions"] for row in complete
        ),
        "rows": rows,
        "claim_boundary": manifest["claim_boundary"],
    }
    summary = body | {"summary_sha256": stable_hash(body)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if len(complete) != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
