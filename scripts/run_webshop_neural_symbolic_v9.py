#!/usr/bin/env python3
"""Run matched WebShop episodes with a frozen V9 neural-symbolic selector."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.webshop_applicability_v8 import candidate_semantics  # noqa: E402
from motif_transfer.webshop_neural_symbolic_v9 import (  # noqa: E402
    TargetOutcomeMLP,
    build_source_value_models,
    choose_transfer_action,
    target_features,
    visible_goal_constraint_status,
)
from scripts.run_webshop_transfer_qualification_v5 import (  # noqa: E402
    _canonicalize_session_text,
    _decision_candidates,
)


def _run_condition(
    *,
    task_id: str,
    condition: str,
    backend: MemoizedCompletionBackend,
    grounder: TargetOutcomeMLP,
    source_models: dict,
    source_policy: dict,
    expected_goal: str | None,
    wrapper_root: Path,
    session_namespace: str,
    number_of_goals: int,
    maximum_steps: int,
    candidate_count: int,
    schema_retries: int,
) -> dict:
    os.environ["WEBSHOP_BASE_URL"] = "http://127.0.0.1:3000"
    os.environ["WEBSHOP_NUM_GOALS"] = str(number_of_goals)
    os.environ["WEBSHOP_SESSION_NAMESPACE"] = session_namespace
    if str(wrapper_root) not in sys.path:
        sys.path.insert(0, str(wrapper_root))
    from webshop_wrapper import register_webshop_tasks

    register_webshop_tasks(number_of_goals)
    import gymnasium as gym
    from browsergym.utils.obs import flatten_axtree_to_str

    task_index = int(task_id.split(".")[-1])
    actual_session = f"{session_namespace}_fixed_{task_index}"
    canonical_session = f"fixed_{task_index}"
    env = gym.make(f"browsergym/{task_id}", headless=True)
    steps = []
    total_reward = 0.0
    terminated = truncated = False
    failure = None
    failure_traceback = None
    previous_action = None
    previous_before_hash = None
    previous_after_hash = None
    try:
        observation, _ = env.reset(seed=0)
        goal = str(observation.get("goal") or observation.get("goal_object") or "")
        if expected_goal is not None and goal != expected_goal:
            raise RuntimeError("live goal does not match frozen manifest")
        initial_tree = flatten_axtree_to_str(
            observation["axtree_object"],
            extra_properties=observation.get("extra_element_properties", {}),
        )
        initial_tree = _canonicalize_session_text(
            initial_tree, actual_session, canonical_session,
        )
        initial_url = _canonicalize_session_text(
            observation.get("url"), actual_session, canonical_session,
        )
        initial_hash = stable_hash({
            "goal": goal, "axtree": initial_tree, "url": initial_url,
        })
        for step_index in range(maximum_steps):
            axtree = flatten_axtree_to_str(
                observation["axtree_object"],
                extra_properties=observation.get("extra_element_properties", {}),
            )
            axtree = _canonicalize_session_text(
                axtree, actual_session, canonical_session,
            )
            url = _canonicalize_session_text(
                observation.get("url"), actual_session, canonical_session,
            )
            payload = {
                "goal": goal,
                "accessibility_tree": axtree[:24000],
                "url": url,
                "last_action": observation.get("last_action"),
                "last_action_error": str(observation.get("last_action_error") or ""),
                "history": [
                    {"action": row["selected_action"], "reward": row["reward"]}
                    for row in steps[-6:]
                ],
                "candidate_count": candidate_count,
            }
            system = (
                "You are a target-native BrowserGym WebShop Decision Agent. Return exactly one "
                "JSON object with key candidates, containing up to five distinct objects with "
                "string key action. Order them from best to worst. Every action must be immediately "
                "executable from the supplied accessibility tree and use only visible BIDs. Valid "
                "forms are click('bid'), fill('bid','text'), press('bid','KEY'), scroll(x,y), "
                "go_back(), go_forward(), and noop(). Propose useful alternatives when more than "
                "one action is plausible. Do not mention games, latent options, source skills, or "
                "explanations."
            )
            candidates, raw, attempts = _decision_candidates(
                backend=backend,
                system=system,
                payload=payload,
                axtree=axtree,
                maximum=candidate_count,
                schema_retries=schema_retries,
            )
            before_hash = stable_hash({"axtree": axtree, "url": url})
            prior_no_effect = bool(
                previous_before_hash
                and previous_after_hash
                and previous_before_hash == previous_after_hash
            )
            satisfied, unsatisfied = visible_goal_constraint_status(axtree, goal)
            semantics = [
                candidate_semantics(
                    observation_text=axtree, url=url, goal=goal, action=action,
                )
                for action in candidates
            ]
            features = [
                target_features(
                    row,
                    visible_satisfied=satisfied,
                    visible_unsatisfied=unsatisfied,
                    prior_no_effect=prior_no_effect,
                    step_index=step_index,
                    maximum_steps=maximum_steps,
                )
                for row in semantics
            ]
            predictions = grounder.predict(features)
            decision = choose_transfer_action(
                condition=condition,
                predictions=predictions,
                semantics=semantics,
                source_models=source_models,
                visible_satisfied=satisfied,
                visible_unsatisfied=unsatisfied,
                prior_no_effect=prior_no_effect,
                remaining_fraction=(maximum_steps - step_index) / maximum_steps,
                previous_action=previous_action,
                candidates=candidates,
                uncertainty_scale=float(source_policy["uncertainty_scale"]),
                decision_margin=float(source_policy["decision_margin"]),
            )
            action = candidates[decision.selected_index]
            observation, reward, terminated, truncated, _ = env.step(action)
            reward = float(reward)
            total_reward += reward
            after_tree = flatten_axtree_to_str(
                observation["axtree_object"],
                extra_properties=observation.get("extra_element_properties", {}),
            )
            after_tree = _canonicalize_session_text(
                after_tree, actual_session, canonical_session,
            )
            after_url = _canonicalize_session_text(
                observation.get("url"), actual_session, canonical_session,
            )
            after_hash = stable_hash({"axtree": after_tree, "url": after_url})
            steps.append({
                "step": step_index,
                "prompt_sha256": stable_hash(payload),
                "response_sha256": stable_hash(raw),
                "decision_attempts": attempts,
                "before_hash": before_hash,
                "candidates": list(candidates),
                "candidate_semantics": semantics,
                "target_features": [list(row) for row in features],
                "target_predictions": predictions.tolist(),
                "state_context": {
                    "visible_goal_constraint_satisfied": satisfied,
                    "visible_goal_constraint_unsatisfied": unsatisfied,
                    "prior_action_had_no_effect": prior_no_effect,
                },
                "selected_index": decision.selected_index,
                "selected_action": action,
                "changed_from_target_rank_zero": decision.selected_index != 0,
                "abstract_kind": decision.abstract_kind,
                "source_abstained": decision.source_abstained,
                "source_test_value": decision.source_test_value,
                "source_commit_value": decision.source_commit_value,
                "selection_reason": decision.reason,
                "reward": reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "after_hash": after_hash,
            })
            previous_action = action
            previous_before_hash = before_hash
            previous_after_hash = after_hash
            if terminated or truncated:
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}:{exc}"
        failure_traceback = traceback.format_exc(limit=8)
        goal = locals().get("goal", "")
        initial_hash = locals().get("initial_hash")
    finally:
        env.close()
    return {
        "schema_version": 1,
        "task_id": task_id,
        "condition": condition,
        "goal": goal,
        "initial_state_hash": initial_hash,
        "maximum_steps": maximum_steps,
        "steps": steps,
        "step_count": len(steps),
        "official_reward": total_reward,
        "strict_success": bool(total_reward >= 1.0 - 1e-9),
        "pass_success": bool(total_reward >= 0.5),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "failure": failure,
        "failure_traceback": failure_traceback,
        "changed_from_target_rank_zero_count": sum(
            row["changed_from_target_rank_zero"] for row in steps
        ),
        "source_decision_count": sum(
            not row["source_abstained"] for row in steps
        ),
        "session_namespace": session_namespace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--role", choices=("calibration", "confirmation"), required=True)
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument(
        "--goal-manifest", type=Path,
        default=REPO / "configs/webshop_consumed_goals_v7.json",
    )
    parser.add_argument(
        "--grounder", type=Path,
        default=REPO / "runs/webshop_neurosymbolic_applicability_v9/frozen_grounder.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--maximum-output-tokens", type=int, default=3200)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--maximum-steps", type=int, default=12)
    parser.add_argument("--number-of-goals", type=int, default=50)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    artifact = json.loads(args.grounder.read_text())
    if not artifact["preflight_passed"]:
        raise SystemExit("frozen V9 grounder did not pass preflight")
    grounder = TargetOutcomeMLP.from_dict(artifact["grounder"])
    source_path = Path(artifact["source_contract"]["config"])
    if file_sha256(source_path) != artifact["source_contract"]["config_sha256"]:
        raise SystemExit("frozen source config hash mismatch")
    source_config = json.loads(source_path.read_text())
    source_models = build_source_value_models(
        source_config, seed=int(source_config["model"]["seed"]),
    )
    values = runpy.run_path(str(args.keys))
    api_key = values.get("OPENROUTER_API_KEY") or values.get("openrouter_api_key")
    if not api_key:
        raise SystemExit("OpenRouter API key is missing")
    os.environ["V9_OPENROUTER_API_KEY"] = str(api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url,
            {"decision": args.model},
            api_key_env="V9_OPENROUTER_API_KEY",
            json_mode=True,
            temperature=0,
            timeout_seconds=180,
            request_overrides={"max_tokens": args.maximum_output_tokens},
        ),
        cache_path=args.output_dir / "decision_cache.json",
    )
    goals = json.loads(args.goal_manifest.read_text())
    conditions = list(artifact["conditions"])
    receipts = []
    for task_id in args.task_ids:
        task_expected_goal = (
            goals.get("goals", {}).get(task_id, {}).get("instruction_text")
        )
        for condition in conditions:
            receipt = _run_condition(
                task_id=task_id,
                condition=condition,
                backend=backend,
                grounder=grounder,
                source_models=source_models,
                source_policy=artifact["policy"],
                expected_goal=task_expected_goal,
                wrapper_root=args.wrapper_root,
                session_namespace=f"{args.run_id}.{condition.replace('_', '-')}",
                number_of_goals=args.number_of_goals,
                maximum_steps=args.maximum_steps,
                candidate_count=args.candidate_count,
                schema_retries=args.schema_retries,
            )
            receipt["role"] = args.role
            receipt["runtime_hashes"] = {
                "runner": file_sha256(Path(__file__)),
                "grounder": file_sha256(args.grounder),
                "goal_manifest": file_sha256(args.goal_manifest),
            }
            receipt["receipt_sha256"] = stable_hash(receipt)
            path = args.output_dir / f"{task_id}.{condition}.json"
            path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            receipts.append(receipt)
            if task_expected_goal is None and receipt["failure"] is None:
                task_expected_goal = receipt["goal"]
            print(json.dumps({
                "task_id": task_id,
                "condition": condition,
                "reward": receipt["official_reward"],
                "strict": receipt["strict_success"],
                "steps": receipt["step_count"],
                "changes": receipt["changed_from_target_rank_zero_count"],
                "source_decisions": receipt["source_decision_count"],
                "failure": receipt["failure"],
            }), flush=True)
    summary = {
        "schema_version": 1,
        "experiment": "webshop_neural_symbolic_transfer_v9",
        "role": args.role,
        "claim_limit": f"Consumed WebShop {args.role} groups only; held-out unread.",
        "tasks": args.task_ids,
        "conditions": {
            condition: {
                "strict_successes": sum(
                    row["strict_success"] for row in receipts if row["condition"] == condition
                ),
                "mean_reward": float(np.mean([
                    row["official_reward"]
                    for row in receipts if row["condition"] == condition
                ])),
                "mean_steps": float(np.mean([
                    row["step_count"] for row in receipts if row["condition"] == condition
                ])),
                "changed_from_target_rank_zero": sum(
                    row["changed_from_target_rank_zero_count"]
                    for row in receipts if row["condition"] == condition
                ),
                "source_decisions": sum(
                    row["source_decision_count"]
                    for row in receipts if row["condition"] == condition
                ),
                "failures": sum(
                    row["failure"] is not None
                    for row in receipts if row["condition"] == condition
                ),
            }
            for condition in conditions
        },
        "matched_initial_state_hashes": all(
            len({
                row["initial_state_hash"] for row in receipts if row["task_id"] == task_id
            }) == 1
            for task_id in args.task_ids
        ),
        "model": args.model,
        "number_of_goals": args.number_of_goals,
        "run_id": args.run_id,
        "runtime_hashes": {
            "runner": file_sha256(Path(__file__)),
            "grounder": file_sha256(args.grounder),
            "decision_cache": file_sha256(args.output_dir / "decision_cache.json"),
        },
        "held_out_read_or_run": False,
    }
    summary["summary_sha256"] = stable_hash(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
