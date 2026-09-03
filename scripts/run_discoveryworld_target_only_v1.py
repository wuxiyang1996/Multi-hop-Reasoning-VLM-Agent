#!/usr/bin/env python3
"""Run an oracle-free target-only DiscoveryWorld development baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from motif_transfer.discoveryworld_env import (  # noqa: E402
    DETERMINISM_PROTOCOL,
    DiscoveryWorldEnvironment,
    stable_hash,
)
from motif_transfer.discoveryworld_policy import (  # noqa: E402
    TARGET_ONLY_SYSTEM_PROMPT,
    native_action_from_decision,
    parse_json_object,
    prompt_payload,
    target_native_facts,
    updated_memory,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.cross_domain_fairness import (  # noqa: E402
    require_formal_suite_audit,
    require_nonpilot_embedding,
)
from motif_transfer.cross_domain_memory_baselines import (  # noqa: E402
    LocalHashingEmbeddingBackend,
    LocalSentenceTransformerEmbeddingBackend,
    MemoryBaseline,
    validate_memory_artifact,
)
from motif_transfer.cross_domain_memory_runtime import MemoryAugmentedDecisionBackend  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_id(row: Mapping[str, Any]) -> str:
    name = str(row["scenario"]).lower().replace(" ", "_").replace("'", "")
    name = "".join(value for value in name if value.isalnum() or value == "_")
    return f"{name}.{str(row['difficulty']).lower()}.seed{int(row['seed'])}"


def _usage(backend: MemoizedCompletionBackend) -> dict[str, Any]:
    return json.loads(json.dumps(dict(backend.last_usage or {}), default=str))


def _call_decision(
    *, backend: MemoizedCompletionBackend, observation, memory: str,
    hypotheses: tuple[str, ...], recent: list[dict[str, Any]], attempts: int,
) -> tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]:
    schema_error = None
    failures = []
    for attempt in range(attempts):
        payload = prompt_payload(
            observation,
            memory=memory,
            hypotheses=hypotheses,
            recent_decisions=recent,
            schema_error=schema_error,
        )
        raw = backend.complete("decision", TARGET_ONLY_SYSTEM_PROMPT, payload)
        usage = _usage(backend)
        try:
            decision = parse_json_object(raw)
            action = native_action_from_decision(decision, observation)
            return decision, action, raw, failures + [{
                "attempt": attempt + 1, "accepted": True, "usage": usage,
            }]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            failures.append({
                "attempt": attempt + 1,
                "accepted": False,
                "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "usage": usage,
            })
    # A valid native no-argument action keeps the environment auditable while
    # making schema failure visible. It is not credited as a model decision.
    fallback = {"action": "DISCOVERY_FEED_GET_UPDATES"}
    return ({
        **fallback,
        "memory": memory,
        "running_hypotheses": list(hypotheses),
        "expected_effect": "Read public feed updates after repeated schema failure.",
        "reason": "SCHEMA_FAILURE_FALLBACK",
    }, fallback, "", failures)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_episode(
    *, task: Mapping[str, Any], config: Mapping[str, Any], backend,
    output_dir: Path, runtime_hashes: Mapping[str, str], thread_id: int,
    arm: str = "target_only",
) -> dict[str, Any]:
    task_id = _task_id(task)
    output_path = output_dir / f"{task_id}.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if existing.get("status") in {
            "TARGET_ONLY_EPISODE_COMPLETE", "CROSS_DOMAIN_MEMORY_EPISODE_COMPLETE",
        }:
            if existing.get("runtime_hashes") != dict(runtime_hashes):
                raise RuntimeError(
                    f"refusing to reuse {output_path}: frozen runtime hashes differ"
                )
            if existing.get("task") != dict(task):
                raise RuntimeError(f"refusing to reuse {output_path}: task differs")
            return existing
    runtime = config["runtime"]
    model = config["model"]
    env = DiscoveryWorldEnvironment(
        scenario=str(task["scenario"]), difficulty=str(task["difficulty"]),
        seed=int(task["seed"]), max_steps=int(runtime["maximum_steps"]),
        thread_id=thread_id, include_vision=bool(runtime["include_vision"]),
        frame_dir=output_dir / "frames" / task_id,
    )
    observation = env.reset()
    initial_policy_hash = observation.policy_state_sha256
    initial_audit_hash = env.current_audit_hash
    memory = ""
    hypotheses: tuple[str, ...] = ()
    recent: list[dict[str, Any]] = []
    steps = []
    while not observation.terminal:
        before_facts = target_native_facts(observation)
        decision, action, raw, attempts = _call_decision(
            backend=backend,
            observation=observation,
            memory=memory,
            hypotheses=hypotheses,
            recent=recent,
            attempts=int(model["schema_attempts"]),
        )
        try:
            after, transition = env.step(action)
        except (TypeError, ValueError) as exc:
            # Local action validation occurs before the official environment is
            # touched, so one repair call is safe and does not alter the fork.
            repair_payload = prompt_payload(
                observation,
                memory=memory,
                hypotheses=hypotheses,
                recent_decisions=recent,
                schema_error=f"environment action validation: {exc}",
            )
            raw = backend.complete("decision", TARGET_ONLY_SYSTEM_PROMPT, repair_payload)
            repaired = parse_json_object(raw)
            action = native_action_from_decision(repaired, observation)
            decision = repaired
            attempts.append({"environment_repair": True, "accepted": True, "usage": _usage(backend)})
            after, transition = env.step(action)
        memory, hypotheses = updated_memory(decision, memory, hypotheses)
        summary = {
            "episode_step": transition.episode_step,
            "action": dict(action),
            "action_succeeded": transition.action_succeeded,
            "reason": str(decision.get("reason") or "")[:2000],
            "expected_effect": str(decision.get("expected_effect") or "")[:2000],
            "memory": memory,
            "running_hypotheses": list(hypotheses),
            "model_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "schema_attempts": attempts,
            "before_target_native_facts": before_facts,
            "after_target_native_facts": target_native_facts(after),
            "transition": asdict(transition),
        }
        steps.append(summary)
        recent.append({
            "episode_step": transition.episode_step,
            "action": dict(action),
            "action_succeeded": transition.action_succeeded,
            "last_action_message": str(after.ui.get("lastActionMessage") or "")[:1000],
            "expected_effect": summary["expected_effect"],
        })
        observation = after
        partial = {
            "schema_version": "discoveryworld-target-only-episode-v1",
            "status": (
                "TARGET_ONLY_EPISODE_RUNNING" if arm == "target_only"
                else "CROSS_DOMAIN_MEMORY_EPISODE_RUNNING"
            ),
            "arm": arm,
            "claim_boundary": config["claim_boundary"],
            "task_id": task_id,
            "task": dict(task),
            "initial_policy_state_sha256": initial_policy_hash,
            "initial_audit_world_sha256": initial_audit_hash,
            "runtime_hashes": dict(runtime_hashes),
            "determinism_protocol": DETERMINISM_PROTOCOL,
            "model": dict(model),
            "steps": steps,
            "policy_runtime_saw_oracle_scorecard": False,
        }
        _write(output_path, partial)
        print(json.dumps({
            "task_id": task_id,
            "step": transition.episode_step,
            "action": action,
            "action_succeeded": transition.action_succeeded,
            "terminal": observation.terminal,
        }), flush=True)
    evaluation = env.finalize_evaluation()
    payload = {
        "schema_version": "discoveryworld-target-only-episode-v1",
        "status": (
            "TARGET_ONLY_EPISODE_COMPLETE" if arm == "target_only"
            else "CROSS_DOMAIN_MEMORY_EPISODE_COMPLETE"
        ),
        "arm": arm,
        "claim_boundary": config["claim_boundary"],
        "task_id": task_id,
        "task": dict(task),
        "initial_policy_state_sha256": initial_policy_hash,
        "initial_audit_world_sha256": initial_audit_hash,
        "runtime_hashes": dict(runtime_hashes),
        "determinism_protocol": DETERMINISM_PROTOCOL,
        "model": dict(model),
        "steps": steps,
        "evaluation": asdict(evaluation),
        "policy_runtime_saw_oracle_scorecard": False,
        "schema_fallback_steps": sum(
            row["reason"] == "SCHEMA_FAILURE_FALLBACK" for row in steps
        ),
        "invalid_native_actions": sum(not row["action_succeeded"] for row in steps),
    }
    payload["episode_sha256"] = stable_hash(payload)
    _write(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--role", choices=(
            "development", "qualification", "formal_reserve", "consumed_adaptation",
        ),
        default="development",
    )
    parser.add_argument("--task-index", type=int, action="append")
    parser.add_argument("--maximum-tasks", type=int)
    parser.add_argument(
        "--arm", default="target_only",
        choices=["target_only", *[row.value for row in MemoryBaseline]],
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--run-mode", choices=["pilot", "formal"], default="pilot")
    parser.add_argument("--fairness-audit", type=Path)
    args = parser.parse_args()
    if args.arm != "target_only" and args.artifact is None:
        raise SystemExit("--artifact is required for a memory arm")
    config = json.loads(args.config.read_text())
    manifest_path = REPO / str(config["manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest_role = str(config.get("manifest_role") or args.role)
    if manifest_role not in manifest["roles"]:
        raise SystemExit(f"unknown manifest role: {manifest_role}")
    tasks = list(manifest["roles"][manifest_role])
    configured_indices = config.get("task_indices")
    if configured_indices is not None:
        if args.task_index:
            raise SystemExit("task indices are frozen in config; do not pass --task-index")
        if not isinstance(configured_indices, list) or not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in configured_indices
        ):
            raise SystemExit("config task_indices must be a list of integers")
        tasks = [tasks[index] for index in configured_indices]
    if args.task_index:
        tasks = [tasks[index] for index in args.task_index]
    if args.maximum_tasks is not None:
        tasks = tasks[:args.maximum_tasks]
    values = runpy.run_path(str(args.keys))
    key = values.get(config["model"]["api_key_name"])
    if not key:
        raise SystemExit("configured OpenRouter key is missing")
    os.environ["DISCOVERYWORLD_OPENROUTER_KEY"] = str(key)
    base_backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            str(config["model"]["base_url"]),
            {"decision": str(config["model"]["model"])},
            api_key_env="DISCOVERYWORLD_OPENROUTER_KEY",
            json_mode=True,
            temperature=float(config["model"]["temperature"]),
            timeout_seconds=180,
            request_overrides={"max_tokens": int(config["model"]["maximum_output_tokens"])},
            transport_attempts=3,
        ),
        cache_path=args.output_dir / "decision_cache.json",
    )
    memory_backend = None
    artifact = None
    if args.arm != "target_only":
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        validate_memory_artifact(artifact)
        if artifact["method"] != args.arm:
            raise SystemExit("memory artifact method does not match --arm")
        embedding_backend = (
            LocalHashingEmbeddingBackend()
            if args.embedding_model == "hashing-pilot"
            else LocalSentenceTransformerEmbeddingBackend(args.embedding_model)
        )
        require_nonpilot_embedding(embedding_backend.identity, run_mode=args.run_mode)
        memory_backend = MemoryAugmentedDecisionBackend(
            base_backend, artifact=artifact, domain="discoveryworld",
            embedding_backend=embedding_backend, top_k=3,
        )
    require_formal_suite_audit(
        args.fairness_audit,
        run_mode=args.run_mode,
        target_domain="discoveryworld",
        method=None if args.arm == "target_only" else args.arm,
        artifact_sha256=artifact["artifact_sha256"] if artifact else None,
    )
    backend = memory_backend or base_backend
    runtime_hashes = {
        "config": file_sha256(args.config),
        "manifest": file_sha256(manifest_path),
        "runner": file_sha256(Path(__file__)),
        "environment_wrapper": file_sha256(REPO / "src/motif_transfer/discoveryworld_env.py"),
        "policy": file_sha256(REPO / "src/motif_transfer/discoveryworld_policy.py"),
        "official_environment_commit": str(manifest["official_environment_commit"]),
    }
    if args.artifact is not None:
        runtime_hashes["memory_artifact"] = file_sha256(args.artifact)
    receipts = []
    for index, task in enumerate(tasks):
        receipt = run_episode(
            task=task, config=config, backend=backend,
            output_dir=args.output_dir, runtime_hashes=runtime_hashes,
            thread_id=96000 + index, arm=args.arm,
        )
        receipts.append(receipt)
        scorecards = receipt["evaluation"]["scorecard"] or []
        normalized = [float(row.get("scoreNormalized", 0.0)) for row in scorecards]
        print(json.dumps({
            "task_id": receipt["task_id"],
            "official_success": receipt["evaluation"]["official_success"],
            "score_normalized": normalized,
            "steps": len(receipt["steps"]),
            "invalid_native_actions": receipt["invalid_native_actions"],
        }), flush=True)
    scores = [
        max([float(row.get("scoreNormalized", 0.0)) for row in receipt["evaluation"]["scorecard"]] or [0.0])
        for receipt in receipts
    ]
    summary = {
        "schema_version": "discoveryworld-target-only-summary-v1",
        "status": {
            "development": "TARGET_ONLY_DEVELOPMENT_COMPLETE",
            "qualification": "TARGET_ONLY_QUALIFICATION_COMPLETE",
            "formal_reserve": "TARGET_ONLY_FORMAL_RESERVE_COMPLETE",
            "consumed_adaptation": "TARGET_ONLY_CONSUMED_ADAPTATION_COMPLETE",
        }[args.role],
        "role": args.role,
        "arm": args.arm,
        "run_mode": args.run_mode,
        "implementation_fidelity": "clean_room_style",
        "result_label": "target-only" if args.arm == "target_only" else f"{args.arm}-style",
        "claim_boundary": config["claim_boundary"],
        "tasks": len(receipts),
        "successes": sum(receipt["evaluation"]["official_success"] for receipt in receipts),
        "nonzero_progress_tasks": sum(score > 0 for score in scores),
        "mean_score_normalized": sum(scores) / len(scores) if scores else 0.0,
        "invalid_native_actions": sum(receipt["invalid_native_actions"] for receipt in receipts),
        "schema_fallback_steps": sum(receipt["schema_fallback_steps"] for receipt in receipts),
        "zero_policy_oracle_scorecard_use": all(
            not receipt["policy_runtime_saw_oracle_scorecard"] for receipt in receipts
        ),
        "runtime_hashes": runtime_hashes,
        "determinism_protocol": DETERMINISM_PROTOCOL,
        "episode_sha256": [receipt["episode_sha256"] for receipt in receipts],
        "memory_receipt": memory_backend.receipt() if memory_backend else None,
    }
    summary["summary_sha256"] = stable_hash(summary)
    _write(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
