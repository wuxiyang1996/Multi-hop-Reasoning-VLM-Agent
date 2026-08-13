#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
from typing import Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.webshop_applicability_v8 import (  # noqa: E402
    candidate_semantics,
    exact_stall,
    safe_recovery_indices,
)
from motif_transfer.webshop_neural_grounder_v5 import (  # noqa: E402
    bids_from_axtree,
    mlp_predict,
    nearest_source_options,
    source_option_values,
    target_action_features,
    validate_browsergym_action,
)


CONDITIONS = (
    "target_only",
    "selective_minimum_repeat",
    "selective_safe_minimum_repeat",
    "authentic_game_source",
    "phase_permuted_source",
    "within_state_value_shuffle",
    "other_game_source",
    "selective_authentic_source",
    "selective_safe_authentic_source",
    "selective_phase_permuted_source",
    "selective_other_game_source",
)


class _OfflineCacheMissBackend:
    def __init__(self, identity: dict) -> None:
        self._identity = dict(identity)
        self.last_usage: dict = {}

    @property
    def identity(self) -> dict:
        return self._identity

    def complete(self, role: str, system: str, payload: dict) -> str:
        del role, system, payload
        raise RuntimeError("offline Decision cache miss")


def _source_for_condition(condition: str, authentic_source: dict, other_source: dict) -> dict:
    if condition in {"other_game_source", "selective_other_game_source"}:
        return other_source
    return authentic_source


def _canonicalize_session_text(text: object, actual_session: str, canonical_session: str) -> str:
    return str(text or "").replace(actual_session, canonical_session)


def _candidate_actions(raw: str, *, axtree: str, maximum: int) -> tuple[str, ...]:
    if raw.strip() in {"", "None", "null"}:
        raise ValueError("Decision model exhausted its output budget before emitting candidates")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Decision response must be a JSON object")
    rows = parsed.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("Decision response must contain a candidates list")
    valid_bids = bids_from_axtree(axtree)
    output = []
    for row in rows:
        action = row.get("action") if isinstance(row, dict) else row
        if not isinstance(action, str):
            continue
        action = action.strip()
        if action not in output and validate_browsergym_action(action, valid_bids):
            output.append(action)
        if len(output) >= maximum:
            break
    if not output:
        raise ValueError("Decision response contains no valid target-native action")
    return tuple(output)


def _completion_diagnostic(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
        json_type = type(parsed).__name__
    except json.JSONDecodeError:
        json_type = "invalid_json"
    return {
        "character_count": len(raw),
        "json_type": json_type,
        "empty_or_none": raw.strip() in {"", "None", "null"},
    }


def _decision_candidates(
    *,
    backend: MemoizedCompletionBackend,
    system: str,
    payload: dict,
    axtree: str,
    maximum: int,
    schema_retries: int,
    attempts_out: list[dict] | None = None,
) -> tuple[tuple[str, ...], str, list[dict]]:
    attempts: list[dict] = attempts_out if attempts_out is not None else []
    prior_response_sha256 = None
    last_error: Exception | None = None
    raw = ""
    for attempt in range(schema_retries + 1):
        request_payload = dict(payload)
        if attempt:
            request_payload["schema_retry"] = {
                "attempt": attempt,
                "prior_response_sha256": prior_response_sha256,
                "require_json_object_with_candidates": True,
            }
        raw = backend.complete("decision", system, request_payload)
        prior_response_sha256 = stable_hash(raw)
        attempt_receipt = {
            "attempt": attempt,
            "response_sha256": prior_response_sha256,
            "completion_diagnostic": _completion_diagnostic(raw),
            "cache_usage": dict(backend.last_usage),
        }
        attempts.append(attempt_receipt)
        try:
            candidates = _candidate_actions(raw, axtree=axtree, maximum=maximum)
            attempt_receipt["valid_candidates"] = len(candidates)
            return candidates, raw, attempts
        except (json.JSONDecodeError, ValueError) as exc:
            attempt_receipt["validation_error"] = f"{type(exc).__name__}:{exc}"
            last_error = exc
    assert last_error is not None
    raise last_error


def _select_action(
    *,
    condition: str,
    candidates: Sequence[str],
    effects: np.ndarray,
    authentic_source: dict,
    other_source: dict,
    context: Sequence[float],
    previous_option: int | None,
    candidate_semantics_rows: Sequence[dict] | None = None,
    observed_stall: bool = False,
) -> tuple[int, dict]:
    if condition == "target_only":
        return 0, {"source_admitted": False, "reason": "target_rank_zero"}
    if condition in {"selective_safe_minimum_repeat", "selective_safe_authentic_source"}:
        if candidate_semantics_rows is None:
            raise ValueError("safe transfer requires target-native candidate semantics")
        repeat_index = list(authentic_source["effect_feature_names"]).index("action_repeated")
        rank_zero_repeat = float(effects[0, repeat_index])
        safe_indices, guard_reason = safe_recovery_indices(candidate_semantics_rows)
        eligible = tuple(
            index for index in safe_indices
            if float(effects[index, repeat_index]) <= rank_zero_repeat - 0.2
        )
        option_ids = nearest_source_options(effects, authentic_source)
        values = source_option_values(
            option_ids,
            source_candidate=authentic_source,
            context_features=context,
            previous_option=previous_option,
        )[:, -1]
        neural_only = condition == "selective_safe_minimum_repeat"
        if neural_only:
            counterfactual_selected = min(
                eligible,
                key=lambda index: (float(effects[index, repeat_index]), index),
                default=0,
            )
            source_advantage = None
        else:
            counterfactual_selected = max(
                eligible,
                key=lambda index: (float(values[index]), -index),
                default=0,
            )
            source_advantage = (
                float(values[counterfactual_selected] - values[0])
                if counterfactual_selected else None
            )
        if not observed_stall:
            guard_reason = "not_exact_observed_stall"
        elif not eligible:
            guard_reason = guard_reason or "no_repeat_improving_safe_candidate"
        elif not neural_only and (source_advantage is None or source_advantage <= 0):
            guard_reason = "no_positive_source_value_advantage"
        gate_open = guard_reason is None and counterfactual_selected != 0
        selected = counterfactual_selected if gate_open else 0
        return selected, {
            "source_admitted": not neural_only and gate_open,
            "neural_only": neural_only,
            "candidate_option_ids": option_ids.tolist(),
            "candidate_h8_values": values.tolist(),
            "selected_option_id": int(option_ids[selected]),
            "applicability_gate": {
                "rule": (
                    "exact_stall_and_target_native_safety_and_repeat_improvement"
                    + ("_and_positive_source_advantage" if not neural_only else "")
                ),
                "observed_exact_stall": observed_stall,
                "safe_candidate_indices": list(safe_indices),
                "eligible_candidate_indices": list(eligible),
                "rank_zero_repeat_probability": rank_zero_repeat,
                "counterfactual_selected_index": counterfactual_selected,
                "source_value_advantage": source_advantage,
                "gate_open": gate_open,
                "closed_reason": guard_reason,
            },
        }
    if condition == "selective_minimum_repeat":
        repeat_index = list(authentic_source["effect_feature_names"]).index("action_repeated")
        rank_zero_repeat = float(effects[0, repeat_index])
        counterfactual_selected = int(np.argmin(effects[:, repeat_index]))
        alternative_repeat = float(effects[counterfactual_selected, repeat_index])
        gate_open = (
            counterfactual_selected != 0
            and rank_zero_repeat >= 0.5
            and alternative_repeat <= rank_zero_repeat - 0.2
        )
        selected = counterfactual_selected if gate_open else 0
        return selected, {
            "source_admitted": False,
            "neural_only": True,
            "selected_option_id": None,
            "selective_gate": {
                "rule": "rank0_repeat>=0.5_and_minimum_repeat_improves_by>=0.2",
                "rank_zero_repeat_probability": rank_zero_repeat,
                "counterfactual_alternative_repeat_probability": alternative_repeat,
                "counterfactual_selected_index": counterfactual_selected,
                "gate_open": gate_open,
            },
        }
    source = _source_for_condition(condition, authentic_source, other_source)
    option_ids = nearest_source_options(effects, source)
    corruption = "phase_permuted" if condition in {
        "phase_permuted_source", "selective_phase_permuted_source"
    } else None
    values = source_option_values(
        option_ids,
        source_candidate=source,
        context_features=context,
        previous_option=previous_option,
        corruption=corruption,
    )[:, -1]
    scores = values.copy()
    if condition == "within_state_value_shuffle" and len(scores) > 1:
        scores = np.roll(scores, 1)
    counterfactual_selected = int(np.argmax(scores))
    selected = counterfactual_selected
    selective_gate = None
    if condition.startswith("selective_"):
        repeat_index = list(source["effect_feature_names"]).index("action_repeated")
        rank_zero_repeat = float(effects[0, repeat_index])
        alternative_repeat = float(effects[counterfactual_selected, repeat_index])
        gate_open = (
            counterfactual_selected != 0
            and rank_zero_repeat >= 0.5
            and alternative_repeat <= rank_zero_repeat - 0.2
        )
        if not gate_open:
            selected = 0
        selective_gate = {
            "rule": "rank0_repeat>=0.5_and_alternative_repeat_improves_by>=0.2",
            "rank_zero_repeat_probability": rank_zero_repeat,
            "counterfactual_alternative_repeat_probability": alternative_repeat,
            "counterfactual_selected_index": counterfactual_selected,
            "gate_open": gate_open,
        }
    return selected, {
        "source_admitted": len(candidates) > 1,
        "candidate_option_ids": option_ids.tolist(),
        "candidate_h8_values": values.tolist(),
        "selection_scores": scores.tolist(),
        "selected_option_id": int(option_ids[selected]),
        "selective_gate": selective_gate,
        "source_artifact_sha256": source["artifact_sha256"],
        "corruption": corruption,
    }


def _run_condition(
    *,
    task_id: str,
    condition: str,
    seed: int,
    maximum_steps: int,
    candidate_count: int,
    backend: MemoizedCompletionBackend,
    grounder: dict,
    authentic_source: dict,
    other_source: dict,
    wrapper_root: Path,
    session_namespace: str,
    schema_retries: int,
    expected_goal: str | None,
) -> dict:
    os.environ["WEBSHOP_BASE_URL"] = "http://127.0.0.1:3000"
    os.environ["WEBSHOP_NUM_GOALS"] = "50"
    os.environ["WEBSHOP_SESSION_NAMESPACE"] = session_namespace
    if str(wrapper_root) not in sys.path:
        sys.path.insert(0, str(wrapper_root))
    from webshop_wrapper import register_webshop_tasks

    register_webshop_tasks(50)
    import gymnasium as gym
    from browsergym.utils.obs import flatten_axtree_to_str

    gym_id = task_id if task_id.startswith("browsergym/") else f"browsergym/{task_id}"
    task_index = int(task_id.split(".")[-1])
    actual_session = f"{session_namespace}_fixed_{task_index}"
    canonical_session = f"fixed_{task_index}"
    env = gym.make(gym_id, headless=True)
    steps = []
    total_reward = 0.0
    terminated = truncated = False
    failure = None
    previous_action = None
    previous_before_hash = None
    previous_after_hash = None
    previous_option = None
    recent_rewards: list[float] = []
    active_step_index = None
    active_payload = None
    active_decision_attempts: list[dict] = []
    try:
        observation, info = env.reset(seed=seed)
        goal = str(observation.get("goal") or observation.get("goal_object") or "")
        if expected_goal is not None and goal != expected_goal:
            raise RuntimeError("live WebShop goal does not match frozen goal manifest")
        initial_axtree = flatten_axtree_to_str(
            observation["axtree_object"],
            extra_properties=observation.get("extra_element_properties", {}),
        )
        initial_axtree = _canonicalize_session_text(
            initial_axtree, actual_session, canonical_session
        )
        initial_url = _canonicalize_session_text(
            observation.get("url"), actual_session, canonical_session
        )
        initial_hash = stable_hash({
            "goal": goal,
            "axtree": initial_axtree,
            "url": initial_url,
        })
        for step_index in range(maximum_steps):
            active_step_index = step_index
            axtree = flatten_axtree_to_str(
                observation["axtree_object"],
                extra_properties=observation.get("extra_element_properties", {}),
            )
            axtree = _canonicalize_session_text(axtree, actual_session, canonical_session)
            canonical_url = _canonicalize_session_text(
                observation.get("url"), actual_session, canonical_session
            )
            payload = {
                "goal": goal,
                "accessibility_tree": axtree[:24000],
                "url": canonical_url,
                "last_action": observation.get("last_action"),
                "last_action_error": str(observation.get("last_action_error") or ""),
                "history": [
                    {"action": row["selected_action"], "reward": row["reward"]}
                    for row in steps[-6:]
                ],
                "candidate_count": candidate_count,
            }
            active_payload = payload
            system = (
                "You are a target-native BrowserGym WebShop Decision Agent. Return exactly one JSON "
                "object with key candidates, containing up to five distinct objects with string key "
                "action. Order them from best to worst. Every action must be immediately executable "
                "from the supplied accessibility tree and use only visible BIDs. Valid forms are "
                "click('bid'), fill('bid','text'), press('bid','KEY'), scroll(x,y), go_back(), "
                "go_forward(), and noop(). Propose useful alternatives when more than one action is "
                "plausible. Do not mention games, latent options, source skills, or explanations."
            )
            decision_attempts = []
            active_decision_attempts = decision_attempts
            candidates, raw, decision_attempts = _decision_candidates(
                backend=backend,
                system=system,
                payload=payload,
                axtree=axtree,
                maximum=candidate_count,
                schema_retries=schema_retries,
                attempts_out=decision_attempts,
            )
            before_hash = stable_hash({
                "axtree": axtree,
                "url": canonical_url,
            })
            semantics = [
                candidate_semantics(
                    observation_text=axtree,
                    url=canonical_url,
                    goal=goal,
                    action=action,
                )
                for action in candidates
            ]
            observed_exact_stall = exact_stall(
                previous_before_hash=previous_before_hash,
                previous_after_hash=previous_after_hash,
                rank_zero_action=candidates[0],
                previous_action=previous_action,
            )
            feature_rows = [
                target_action_features(
                    observation_text=axtree,
                    url=canonical_url,
                    goal=goal,
                    action=action,
                    step_index=step_index,
                    maximum_steps=maximum_steps,
                    previous_action=previous_action,
                )
                for action in candidates
            ]
            effects = mlp_predict(grounder, feature_rows)
            context = (
                step_index / max(1, maximum_steps - 1),
                (maximum_steps - step_index) / maximum_steps,
                float(np.tanh(sum(recent_rewards[-4:]) / 3.0)),
            )
            selected_index, source_receipt = _select_action(
                condition=condition,
                candidates=candidates,
                effects=effects,
                authentic_source=authentic_source,
                other_source=other_source,
                context=context,
                previous_option=previous_option,
                candidate_semantics_rows=semantics,
                observed_stall=observed_exact_stall,
            )
            selected_action = candidates[selected_index]
            observation, reward, terminated, truncated, info = env.step(selected_action)
            after_axtree = flatten_axtree_to_str(
                observation["axtree_object"],
                extra_properties=observation.get("extra_element_properties", {}),
            )
            after_axtree = _canonicalize_session_text(
                after_axtree, actual_session, canonical_session
            )
            after_url = _canonicalize_session_text(
                observation.get("url"), actual_session, canonical_session
            )
            reward = float(reward)
            total_reward += reward
            recent_rewards.append(reward)
            selected_options = nearest_source_options(
                effects[selected_index : selected_index + 1],
                _source_for_condition(condition, authentic_source, other_source),
            )
            previous_option = int(selected_options[0])
            after_hash = stable_hash({
                "axtree": after_axtree,
                "url": after_url,
            })
            steps.append({
                "step": step_index,
                "prompt_sha256": stable_hash(payload),
                "response_sha256": stable_hash(raw),
                "decision_attempts": decision_attempts,
                "before_hash": before_hash,
                "candidates": list(candidates),
                "candidate_semantics": semantics,
                "predicted_effects": effects.tolist(),
                "selected_index": selected_index,
                "selected_action": selected_action,
                "changed_from_target_rank_zero": selected_index != 0,
                "observed_exact_stall": observed_exact_stall,
                "source_receipt": source_receipt,
                "reward": reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "last_action_error": str(observation.get("last_action_error") or ""),
                "after_hash": after_hash,
            })
            previous_action = selected_action
            previous_before_hash = before_hash
            previous_after_hash = after_hash
            if terminated or truncated:
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}:{exc}"
        failure_traceback = traceback.format_exc(limit=8)
        if "initial_hash" not in locals():
            initial_hash = None
        if "goal" not in locals():
            goal = ""
    finally:
        env.close()
    changed = sum(row["changed_from_target_rank_zero"] for row in steps)
    admitted = sum(row["source_receipt"].get("source_admitted", False) for row in steps)
    return {
        "schema_version": 1,
        "task_id": task_id,
        "condition": condition,
        "seed": seed,
        "goal": goal,
        "initial_state_hash": initial_hash,
        "maximum_steps": maximum_steps,
        "steps": steps,
        "step_count": len(steps),
        "official_reward": total_reward,
        "strict_success": bool(total_reward >= 1.0 - 1e-9),
        "pass_success": bool(total_reward >= 0.5),
        "any_reward": bool(total_reward > 0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "failure": failure,
        "failure_traceback": failure_traceback if failure is not None else None,
        "failure_context": {
            "step": active_step_index,
            "prompt_sha256": stable_hash(active_payload) if active_payload else None,
            "decision_attempts": active_decision_attempts,
        } if failure is not None else None,
        "session_namespace": session_namespace,
        "canonical_session": canonical_session,
        "changed_from_target_rank_zero_count": changed,
        "option_change_rate": changed / max(1, len(steps)),
        "source_admission_rate": admitted / max(1, len(steps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/real_game_multitarget_v5_manifest.json",
    )
    parser.add_argument(
        "--experiment-config", type=Path,
        default=REPO / "configs/webshop_selective_neurosymbolic_v6.json",
    )
    parser.add_argument("--goal-manifest", type=Path)
    parser.add_argument(
        "--grounder", type=Path,
        default=REPO / "runs/real_game_multitarget_neurosymbolic_v5/webshop_grounder/frozen_grounder.json",
    )
    parser.add_argument(
        "--source-candidate", type=Path,
        default=REPO / "runs/real_game_multitarget_neurosymbolic_v5/source_development/frozen_candidate.json",
    )
    parser.add_argument(
        "--other-source-candidate", type=Path,
        default=REPO / "runs/real_game_multitarget_neurosymbolic_v5/source_other_game_control/frozen_candidate.json",
    )
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/real_game_multitarget_neurosymbolic_v5/webshop_qualification_smoke",
    )
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-name", default="openrouter_api_key")
    parser.add_argument("--decision-cache", type=Path)
    parser.add_argument("--offline-cache-only", action="store_true")
    parser.add_argument("--maximum-steps", type=int, default=12)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=3200)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--task-limit", type=int, default=2)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument(
        "--task-role", choices=("qualification", "reserve"), default="qualification"
    )
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--session-run-id")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()
    if args.schema_retries < 0:
        raise SystemExit("--schema-retries must be non-negative")

    values = runpy.run_path(str(args.keys))
    api_key = values.get(args.api_key_name)
    if api_key is None and args.api_key_name == "openrouter_api_key":
        api_key = values.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(f"{args.api_key_name} is missing")
    os.environ["V5_DECISION_API_KEY"] = str(api_key)
    manifest = json.loads(args.manifest.read_text())
    experiment_config = json.loads(args.experiment_config.read_text())
    goal_manifest = json.loads(args.goal_manifest.read_text()) if args.goal_manifest else None
    grounder = json.loads(args.grounder.read_text())
    authentic_source = json.loads(args.source_candidate.read_text())
    other_source = json.loads(args.other_source_candidate.read_text())
    if grounder["manifest_sha256"] != manifest["manifest_sha256"]:
        raise SystemExit("grounder/manifest hash mismatch")
    if grounder["source_candidate_artifact_sha256"] != authentic_source["artifact_sha256"]:
        raise SystemExit("grounder/source candidate hash mismatch")
    if args.task_ids:
        tasks = list(args.task_ids)
    else:
        tasks = manifest["targets"]["webshop"]["partition"]["roles"][args.task_role]
        tasks = tasks[args.task_offset : args.task_offset + args.task_limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    live_backend = OpenAICompatibleBackend(
        args.base_url,
        {"decision": args.model},
        api_key_env="V5_DECISION_API_KEY",
        json_mode=True,
        temperature=0,
        timeout_seconds=180,
        request_overrides={"max_tokens": args.max_output_tokens},
    )
    inner_backend = (
        _OfflineCacheMissBackend(dict(live_backend.identity))
        if args.offline_cache_only else live_backend
    )
    decision_cache = args.decision_cache or args.output_dir / "decision_cache.json"
    backend = MemoizedCompletionBackend(inner_backend, cache_path=decision_cache)
    receipts = []
    session_run_id = args.session_run_id or (
        "v7-" + stable_hash({"output_dir": str(args.output_dir.resolve()), "pid": os.getpid()})[:12]
    )
    for task_id in tasks:
        for condition in args.conditions:
            receipt_path = args.output_dir / f"{task_id}.{condition}.json"
            receipt = _run_condition(
                task_id=task_id,
                condition=condition,
                seed=args.seed,
                maximum_steps=args.maximum_steps,
                candidate_count=args.candidate_count,
                backend=backend,
                grounder=grounder,
                authentic_source=authentic_source,
                other_source=other_source,
                wrapper_root=args.wrapper_root,
                session_namespace=f"{session_run_id}.{condition.replace('_', '-')}",
                schema_retries=args.schema_retries,
                expected_goal=(
                    goal_manifest["goals"][task_id]["instruction_text"]
                    if goal_manifest is not None else None
                ),
            )
            receipt["decision_runtime"] = {
                "model": args.model,
                "maximum_output_tokens": args.max_output_tokens,
                "schema_retries": args.schema_retries,
                "candidate_count": args.candidate_count,
            }
            receipt["runtime_hashes"] = {
                "runner": file_sha256(Path(__file__)),
                "experiment_config": file_sha256(args.experiment_config),
                "manifest": file_sha256(args.manifest),
                "grounder": file_sha256(args.grounder),
                "source_candidate": file_sha256(args.source_candidate),
                "other_source_candidate": file_sha256(args.other_source_candidate),
            }
            if args.goal_manifest is not None:
                receipt["runtime_hashes"]["goal_manifest"] = file_sha256(args.goal_manifest)
            receipt["receipt_sha256"] = stable_hash(receipt)
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            receipts.append(receipt)
            print(json.dumps({
                "task_id": task_id,
                "condition": condition,
                "reward": receipt["official_reward"],
                "strict_success": receipt["strict_success"],
                "steps": receipt["step_count"],
                "option_change_rate": receipt["option_change_rate"],
                "failure": receipt["failure"],
            }, ensure_ascii=False))
    conditions = {}
    for condition in args.conditions:
        rows = [row for row in receipts if row["condition"] == condition]
        conditions[condition] = {
            "tasks": len(rows),
            "strict_successes": sum(row["strict_success"] for row in rows),
            "pass_successes": sum(row["pass_success"] for row in rows),
            "any_reward_tasks": sum(row["any_reward"] for row in rows),
            "mean_official_reward": float(np.mean([row["official_reward"] for row in rows])),
            "mean_steps": float(np.mean([row["step_count"] for row in rows])),
            "mean_option_change_rate": float(np.mean([row["option_change_rate"] for row in rows])),
            "failures": sum(row["failure"] is not None for row in rows),
        }
    matched_initial_hashes = all(
        len({
            row["initial_state_hash"] for row in receipts if row["task_id"] == task_id
        }) == 1
        for task_id in tasks
    )
    summary = {
        "schema_version": 1,
        "experiment": experiment_config["experiment"],
        "claim_limit": "Qualification smoke only; WebShop held-out remains unread",
        "tasks": tasks,
        "task_role": args.task_role,
        "conditions": conditions,
        "matched_initial_state_hashes": matched_initial_hashes,
        "model": args.model,
        "maximum_steps": args.maximum_steps,
        "candidate_count": args.candidate_count,
        "maximum_output_tokens": args.max_output_tokens,
        "schema_retries": args.schema_retries,
        "session_run_id": session_run_id,
        "experiment_config_sha256": file_sha256(args.experiment_config),
        "goal_manifest_sha256": (
            file_sha256(args.goal_manifest) if args.goal_manifest is not None else None
        ),
        "decision_cache_sha256": (
            file_sha256(decision_cache)
            if decision_cache.is_file() else None
        ),
    }
    summary["summary_sha256"] = stable_hash(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
