#!/usr/bin/env python3
"""Run target-native structured acquisition and an optional neural continuation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
DISCOVERYWORLD = REPO.parent / "discoveryworld-official"
if DISCOVERYWORLD.is_dir():
    sys.path.insert(0, str(DISCOVERYWORLD))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.discoveryworld_env import (  # noqa: E402
    DETERMINISM_PROTOCOL, DiscoveryWorldEnvironment,
)
from motif_transfer.discoveryworld_policy import (  # noqa: E402
    target_native_facts, updated_memory,
)
from motif_transfer.discoveryworld_structured_acquisition_v2 import (  # noqa: E402
    call_structured_acquisition_grounder,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend, OpenAICompatibleBackend,
)
from motif_transfer.phase3_discoveryworld_formal import (  # noqa: E402
    select_outcome_blind_formal_fork,
)
from motif_transfer.phase3_discoveryworld_grounding import (  # noqa: E402
    call_qualified_decision,
)
from motif_transfer.phase3_discoveryworld_transfer import (  # noqa: E402
    extract_phase3_acquisition_evidence,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value); claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _backend(config, key: str, cache_path: Path):
    model = config["model"]
    os.environ["PHASE3_STRUCTURED_ACQUISITION_KEY"] = key
    raw = OpenAICompatibleBackend(
        str(model["base_url"]), {
            "acquisition": str(model["model"]),
            "decision": str(model["model"]),
            "affordance": str(model.get("affordance_model") or model["model"]),
        },
        api_key_env="PHASE3_STRUCTURED_ACQUISITION_KEY", json_mode=True,
        temperature=float(model["temperature"]), timeout_seconds=180,
        request_overrides={"max_tokens": int(model["maximum_output_tokens"])},
        transport_attempts=3,
    )
    return MemoizedCompletionBackend(raw, cache_path=cache_path)


def _evidence_memory(steps) -> str:
    evidence = extract_phase3_acquisition_evidence({"steps": steps}, len(steps))
    values = "; ".join(
        f"{row['subject']}: " + ", ".join(
            f"{key}={value}" for key, value in row["measurement_vector"].items()
        ) for row in evidence
    )
    return f"Raw target-native instrument measurements: {values}"[-3000:]


def _step_row(
    *, transition, action, before_facts, after, reason, expected_effect,
    memory, hypotheses, raw, attempts, phase,
):
    return {
        "episode_step": transition.episode_step,
        "phase": phase,
        "action": dict(action),
        "action_succeeded": transition.action_succeeded,
        "reason": str(reason)[:2000],
        "expected_effect": str(expected_effect)[:2000],
        "memory": memory,
        "running_hypotheses": list(hypotheses),
        "model_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "schema_attempts": list(attempts),
        "before_target_native_facts": before_facts,
        "after_target_native_facts": target_native_facts(after),
        "transition": asdict(transition),
    }


def run_one(*, config_path: Path, task, key: str, output_dir: Path, index: int):
    config = _read(config_path)
    runtime = config["runtime"]
    task_id = str(task["task_id"])
    output_path = output_dir / f"{task_id}.json"
    if output_path.is_file():
        existing = _read(output_path)
        if existing.get("status") == "TARGET_ONLY_EPISODE_COMPLETE":
            _self_hash(existing, "episode_sha256")
            return existing
    backend = _backend(
        config, key, output_dir / "caches" / f"{task_id}.json",
    )
    env = DiscoveryWorldEnvironment(
        scenario=str(task["scenario"]), difficulty=str(task["difficulty"]),
        seed=int(task["seed"]),
        max_steps=(
            int(runtime["maximum_acquisition_steps"])
            + int(runtime["continuation_horizon"])
        ),
        thread_id=int(runtime["thread_id_base"]) + index,
        include_vision=False, frame_dir=output_dir / "frames" / task_id,
    )
    observation = env.reset()
    steps = []
    acquisition_fallbacks = 0
    acquisition_repairs = 0
    fork_receipt = None
    while len(steps) < int(runtime["maximum_acquisition_steps"]):
        evidence = extract_phase3_acquisition_evidence({"steps": steps}, len(steps))
        measured = tuple(str(row["subject"]) for row in evidence)
        try:
            fork_receipt = select_outcome_blind_formal_fork({"steps": steps})
            break
        except ValueError:
            pass
        before = target_native_facts(observation)
        action, raw, attempts, fallback = call_structured_acquisition_grounder(
            backend=backend, observation=observation,
            measured_subjects=measured,
            attempts=int(config["model"]["schema_attempts"]),
        )
        acquisition_fallbacks += int(fallback)
        acquisition_repairs += int(any(not row["accepted"] for row in attempts))
        after, transition = env.step(action)
        memory = _evidence_memory(steps)
        hypotheses = (
            "Exactly one measured species is a multivariate outlier; compare full vectors.",
        )
        steps.append(_step_row(
            transition=transition, action=action, before_facts=before, after=after,
            reason="TARGET_NATIVE_STRUCTURED_ACQUISITION",
            expected_effect="Advance one missing measurement or held-object prerequisite.",
            memory=memory, hypotheses=hypotheses, raw=raw, attempts=attempts,
            phase="STRUCTURED_ACQUISITION",
        ))
        observation = after
        if observation.terminal:
            break
    if fork_receipt is None:
        try:
            fork_receipt = select_outcome_blind_formal_fork({"steps": steps})
        except ValueError:
            fork_receipt = None
    fork_step = int(fork_receipt["fork_after_episode_step"]) if fork_receipt else None

    memory = _evidence_memory(steps)
    hypotheses = (
        "Exactly one measured species is a multivariate outlier; compare full vectors.",
    )
    recent = []
    continuation_fallbacks = 0
    while (
        fork_receipt is not None and not observation.terminal
        and len(steps) < fork_step + int(runtime["continuation_horizon"])
    ):
        before = target_native_facts(observation)
        decision, action, raw, attempts, fallback = call_qualified_decision(
            backend=backend, observation=observation, memory=memory,
            hypotheses=hypotheses, recent=recent,
            attempts=int(config["model"]["schema_attempts"]),
        )
        continuation_fallbacks += int(fallback)
        after, transition = env.step(action)
        memory, hypotheses = updated_memory(decision, memory, hypotheses)
        steps.append(_step_row(
            transition=transition, action=action, before_facts=before, after=after,
            reason=decision.get("reason") or "",
            expected_effect=decision.get("expected_effect") or "",
            memory=memory, hypotheses=hypotheses, raw=raw, attempts=attempts,
            phase="NEURAL_ONLY_CONTINUATION",
        ))
        recent.append({
            "action": dict(action),
            "action_succeeded": transition.action_succeeded,
            "last_action_message": str(after.ui.get("lastActionMessage") or "")[:1000],
            "expected_effect": str(decision.get("expected_effect") or "")[:1000],
        })
        observation = after

    evaluator_finalized = bool(observation.terminal)
    evaluation = (
        asdict(env.finalize_evaluation()) if evaluator_finalized else {
            "schema_version": "discoveryworld-evaluation-not-finalized-v1",
            "terminal": False,
            "official_success": False,
            "scorecard": None,
        }
    )
    body = {
        "schema_version": "discoveryworld-structured-acquisition-episode-v2",
        "status": "TARGET_ONLY_EPISODE_COMPLETE",
        "claim_boundary": config["claim_boundary"],
        "task_id": task_id,
        "task": {key: task[key] for key in ("scenario", "difficulty", "seed")},
        "formal_manifest_sha256": config.get("manifest_sha256"),
        "determinism_protocol": DETERMINISM_PROTOCOL,
        "model": dict(config["model"]),
        "steps": steps,
        "fork_after_episode_step": fork_step,
        "fork_receipt": fork_receipt,
        "acquisition_ready": fork_receipt is not None,
        "acquisition_actions": fork_step or len(steps),
        "acquisition_schema_fallback_steps": acquisition_fallbacks,
        "acquisition_calls_requiring_repair": acquisition_repairs,
        "continuation_schema_fallback_steps": continuation_fallbacks,
        "schema_fallback_steps": acquisition_fallbacks + continuation_fallbacks,
        "invalid_native_actions": sum(not row["action_succeeded"] for row in steps),
        "evaluation": evaluation,
        "evaluator_finalized": evaluator_finalized,
        "policy_runtime_saw_oracle_scorecard": False,
        "formal_outcome_read_by_acquisition_or_fork_selection": False,
        "runtime_hashes": {
            "config": _file_sha256(config_path), "runner": _file_sha256(Path(__file__)),
            "grounder": _file_sha256(
                REPO / "src/motif_transfer/discoveryworld_structured_acquisition_v2.py"
            ),
        },
    }
    payload = body | {"episode_sha256": stable_hash(body)}
    _write(output_path, payload)
    return payload


def _worker(config, task, key, output, index):
    try:
        value = run_one(
            config_path=Path(config), task=task, key=key,
            output_dir=Path(output), index=index,
        )
        return {
            "task_id": task["task_id"], "error": None,
            "acquisition_ready": value["acquisition_ready"],
            "acquisition_actions": value["acquisition_actions"],
            "acquisition_schema_fallback_steps": value[
                "acquisition_schema_fallback_steps"
            ],
            "acquisition_calls_requiring_repair": value[
                "acquisition_calls_requiring_repair"
            ],
            "official_success": bool(value["evaluation"]["official_success"]),
            "evaluator_finalized": bool(value["evaluator_finalized"]),
            "formal_outcome_read_by_acquisition_or_fork_selection": bool(
                value["formal_outcome_read_by_acquisition_or_fork_selection"]
            ),
            "steps": len(value["steps"]),
            "schema_fallback_steps": value["schema_fallback_steps"],
            "invalid_native_actions": value["invalid_native_actions"],
            "episode_sha256": value["episode_sha256"],
        }
    except BaseException as exc:
        return {"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    config = _read(args.config)
    if config.get("status") == "FROZEN_BEFORE_STRUCTURED_ACQUISITION_QUALIFICATION":
        _self_hash(config, "manifest_sha256")
    tasks = list(config["tasks"])
    selected = set(args.task_id or ())
    if selected:
        tasks = [row for row in tasks if row["task_id"] in selected]
        if len(tasks) != len(selected):
            raise SystemExit("unknown task ID")
    if config.get("runtime_file_sha256"):
        for path, expected in config["runtime_file_sha256"].items():
            if _file_sha256(REPO / path) != expected:
                raise SystemExit(f"frozen runtime changed: {path}")
    values = runpy.run_path(str(args.keys))
    key = values.get(config["model"]["api_key_name"])
    if not key:
        raise SystemExit("configured OpenRouter key is missing")
    workers = min(len(tasks), args.workers or int(config["runtime"]["task_workers"]))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _worker, str(args.config), dict(task), str(key),
                str(args.output_dir), config["tasks"].index(task),
            ): task["task_id"] for task in tasks
        }
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: row["task_id"])
    complete = [row for row in rows if row["error"] is None]
    actions = sum(row["acquisition_actions"] for row in complete)
    fallback = sum(row["acquisition_schema_fallback_steps"] for row in complete)
    repairs = sum(row["acquisition_calls_requiring_repair"] for row in complete)
    gates = None
    status = (
        "PHASE3_DISCOVERYWORLD_ACQUISITION_COMPLETE"
        if len(complete) == len(rows) else
        "PHASE3_DISCOVERYWORLD_ACQUISITION_INCOMPLETE"
    )
    if config.get("frozen_qualification_gates"):
        gate_config = config["frozen_qualification_gates"]
        gates = {
            "all_states_ready": (
                len(complete) == int(gate_config["required_ready_states"])
                and sum(row["acquisition_ready"] for row in complete)
                == int(gate_config["required_ready_states"])
            ),
            "fallback_rate": (fallback / actions if actions else 1.0) <= float(
                gate_config["maximum_acquisition_schema_fallback_rate"]
            ),
            "repair_rate": (repairs / actions if actions else 1.0) <= float(
                gate_config["maximum_acquisition_repair_rate"]
            ),
            "zero_invalid_native_actions": sum(
                row["invalid_native_actions"] for row in complete
            ) == 0,
            "no_evaluator_or_formal_outcome": all(
                row["evaluator_finalized"] is False
                and row["formal_outcome_read_by_acquisition_or_fork_selection"]
                is False for row in complete
            ),
            "no_runtime_errors": len(complete) == len(rows),
        }
        status = (
            "DISCOVERYWORLD_STRUCTURED_ACQUISITION_QUALIFICATION_PASSED"
            if all(gates.values()) else
            "DISCOVERYWORLD_STRUCTURED_ACQUISITION_QUALIFICATION_FAILED"
        )
    body = {
        "schema_version": "phase3-discoveryworld-structured-acquisition-summary-v2",
        "status": status,
        "manifest_sha256": config.get("manifest_sha256"),
        "role": config["role"],
        "tasks": len(rows), "complete_tasks": len(complete),
        "acquisition_ready_tasks": sum(row["acquisition_ready"] for row in complete),
        "acquisition_actions": actions,
        "acquisition_schema_fallback_steps": fallback,
        "acquisition_schema_fallback_rate": fallback / actions if actions else 1.0,
        "acquisition_calls_requiring_repair": repairs,
        "acquisition_repair_rate": repairs / actions if actions else 1.0,
        "official_successes": sum(row["official_success"] for row in complete),
        "steps": sum(row["steps"] for row in complete),
        "schema_fallback_steps": sum(row["schema_fallback_steps"] for row in complete),
        "invalid_native_actions": sum(row["invalid_native_actions"] for row in complete),
        "rows": rows,
        "gates": gates,
        "formal_outcome_read_by_acquisition_or_fork_selection": False,
        "claim_boundary": config["claim_boundary"],
    }
    summary = body | {"summary_sha256": stable_hash(body)}
    _write(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "ready": summary["acquisition_ready_tasks"],
        "fallback_rate": summary["acquisition_schema_fallback_rate"],
        "repair_rate": summary["acquisition_repair_rate"],
        "summary_sha256": summary["summary_sha256"],
    }, indent=2))
    if len(complete) != len(rows) or (gates is not None and not all(gates.values())):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
