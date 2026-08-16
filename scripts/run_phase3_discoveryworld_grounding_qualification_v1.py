#!/usr/bin/env python3
"""Run fixed-horizon, evaluator-free DiscoveryWorld grounding qualification."""

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
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
# DiscoveryWorld is kept as an adjacent, pinned source checkout rather than an
# installed wheel in the benchmark environment.  Resolve that dependency here
# so a qualification receipt does not depend on an undocumented caller-side
# PYTHONPATH.  An installed package remains valid when the checkout is absent.
DISCOVERYWORLD_CHECKOUT = REPO.parent / "discoveryworld-official"
if DISCOVERYWORLD_CHECKOUT.is_dir() and str(DISCOVERYWORLD_CHECKOUT) not in sys.path:
    sys.path.insert(0, str(DISCOVERYWORLD_CHECKOUT))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.discoveryworld_env import DiscoveryWorldEnvironment  # noqa: E402
from motif_transfer.discoveryworld_policy import updated_memory  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.phase3_discoveryworld_grounding import (  # noqa: E402
    call_qualified_decision,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _task_id(task: Mapping[str, Any]) -> str:
    return f"proteomics.easy.seed{int(task['seed'])}"


def _make_backend(
    *, config: Mapping[str, Any], key: str, cache_path: Path,
) -> MemoizedCompletionBackend:
    os.environ["PHASE3_DISCOVERYWORLD_OPENROUTER_KEY"] = key
    return MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            str(config["model"]["base_url"]),
            {"decision": str(config["model"]["model"])},
            api_key_env="PHASE3_DISCOVERYWORLD_OPENROUTER_KEY",
            json_mode=True,
            temperature=float(config["model"]["temperature"]),
            timeout_seconds=180,
            request_overrides={
                "max_tokens": int(config["model"]["maximum_output_tokens"]),
            },
            transport_attempts=3,
        ),
        cache_path=cache_path,
    )


def run_task(
    *,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    backend,
    output_dir: Path,
    thread_id: int,
) -> dict[str, Any]:
    task_id = _task_id(task)
    output = output_dir / f"{task_id}.json"
    if output.exists():
        existing = _read(output)
        if existing.get("status") == "GROUNDING_QUALIFICATION_TASK_COMPLETE":
            return existing
        raise RuntimeError(f"refusing to overwrite incomplete qualification: {output}")
    runtime = config["runtime"]
    env = DiscoveryWorldEnvironment(
        scenario="Proteomics",
        difficulty="Easy",
        seed=int(task["seed"]),
        max_steps=int(runtime["environment_maximum_steps"]),
        thread_id=thread_id,
        include_vision=False,
        frame_dir=output_dir / "frames" / task_id,
    )
    observation = env.reset()
    memory = ""
    hypotheses: tuple[str, ...] = ()
    recent: list[dict[str, Any]] = []
    steps = []
    for index in range(int(runtime["qualification_steps"])):
        if observation.terminal:
            break
        decision, action, raw, attempts, fallback = call_qualified_decision(
            backend=backend,
            observation=observation,
            memory=memory,
            hypotheses=hypotheses,
            recent=recent,
            attempts=int(config["model"]["schema_attempts"]),
        )
        after, transition = env.step(action)
        memory, hypotheses = updated_memory(decision, memory, hypotheses)
        row = {
            "step": index + 1,
            "action": dict(action),
            "action_succeeded": bool(transition.action_succeeded),
            "schema_fallback": bool(fallback),
            "schema_attempts": attempts,
            "model_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "transition_receipt_sha256": transition.receipt_sha256,
            "terminal": bool(after.terminal),
            "policy_runtime_saw_oracle_scorecard": False,
        }
        steps.append(row)
        recent.append({
            "step": index + 1,
            "action": dict(action),
            "action_succeeded": bool(transition.action_succeeded),
            "last_action_message": str(after.ui.get("lastActionMessage") or "")[:1000],
        })
        observation = after
    body = {
        "schema_version": "phase3-discoveryworld-grounding-task-v1",
        "status": "GROUNDING_QUALIFICATION_TASK_COMPLETE",
        "role": config["role"],
        "task_id": task_id,
        "task": dict(task),
        "steps": steps,
        "schema_fallback_steps": sum(row["schema_fallback"] for row in steps),
        "invalid_native_actions": sum(not row["action_succeeded"] for row in steps),
        "policy_runtime_saw_oracle_scorecard": False,
        "finalize_evaluation_called": False,
        "official_success_persisted": False,
        "claim_boundary": config["claim_boundary"],
    }
    receipt = body | {"receipt_sha256": stable_hash(body)}
    _write(output, receipt)
    return receipt


def _run_task_process(
    *, task: Mapping[str, Any], config: Mapping[str, Any], key: str,
    output_dir: Path, thread_id: int,
) -> dict[str, Any]:
    """Process-isolated task execution with a task-local completion cache."""

    task_id = _task_id(task)
    backend = _make_backend(
        config=config,
        key=key,
        cache_path=output_dir / "decision_caches" / f"{task_id}.json",
    )
    return run_task(
        task=task,
        config=config,
        backend=backend,
        output_dir=output_dir,
        thread_id=thread_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = _read(args.config)
    if config.get("schema_version") != "phase3-discoveryworld-grounding-config-v1":
        raise SystemExit("unsupported grounding qualification config")
    if config.get("reads_target_success") is not False:
        raise SystemExit("grounding qualification must not read target success")
    values = runpy.run_path(str(args.keys))
    key = values.get(config["model"]["api_key_name"])
    if not key:
        raise SystemExit("configured OpenRouter key is missing")
    tasks = list(config["tasks"])
    maximum_workers = min(
        len(tasks), max(1, int(config["runtime"].get("task_workers", 1))),
    )
    receipts = []
    with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
        futures = {
            executor.submit(
                _run_task_process,
                task=task,
                config=config,
                key=str(key),
                output_dir=args.output_dir,
                thread_id=int(config["runtime"]["thread_id_base"]) + index,
            ): _task_id(task)
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            print(json.dumps({
                "task_id": receipt["task_id"],
                "steps": len(receipt["steps"]),
                "schema_fallback_steps": receipt["schema_fallback_steps"],
                "invalid_native_actions": receipt["invalid_native_actions"],
            }), flush=True)
    receipts.sort(key=lambda row: row["task_id"])
    steps = sum(len(row["steps"]) for row in receipts)
    fallbacks = sum(row["schema_fallback_steps"] for row in receipts)
    invalid = sum(row["invalid_native_actions"] for row in receipts)
    gates_config = config["qualification_gates"]
    gates = {
        "minimum_steps": steps >= int(gates_config["minimum_steps"]),
        "schema_fallback_rate": (
            fallbacks / steps if steps else 1.0
        ) <= float(gates_config["maximum_schema_fallback_rate"]),
        "zero_invalid_native_actions": invalid <= int(
            gates_config["maximum_invalid_native_actions"]
        ),
        "zero_evaluator_calls": all(
            not row["finalize_evaluation_called"] for row in receipts
        ),
        "zero_oracle_scorecard_use": all(
            not row["policy_runtime_saw_oracle_scorecard"] for row in receipts
        ),
    }
    required = list(gates)
    body = {
        "schema_version": "phase3-discoveryworld-grounding-summary-v1",
        "status": (
            "DISCOVERYWORLD_GROUNDING_QUALIFICATION_PASSED"
            if all(gates.values()) else "DISCOVERYWORLD_GROUNDING_QUALIFICATION_FAILED"
        ),
        "role": config["role"],
        "tasks": len(receipts),
        "steps": steps,
        "schema_fallback_steps": fallbacks,
        "schema_fallback_rate": fallbacks / steps if steps else 1.0,
        "invalid_native_actions": invalid,
        "gates": gates,
        "required_gates": required,
        "task_receipt_sha256s": [row["receipt_sha256"] for row in receipts],
        "claim_boundary": config["claim_boundary"],
    }
    summary = body | {"summary_sha256": stable_hash(body)}
    _write(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
