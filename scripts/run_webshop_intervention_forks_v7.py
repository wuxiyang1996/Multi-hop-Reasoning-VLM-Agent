#!/usr/bin/env python3
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
from motif_transfer.webshop_neural_grounder_v5 import (  # noqa: E402
    nearest_source_options,
)
from scripts.run_webshop_transfer_qualification_v5 import (  # noqa: E402
    _canonicalize_session_text,
    _decision_candidates,
    _select_action,
)


BRANCH_CONDITIONS = (
    "target_only",
    "selective_minimum_repeat",
    "selective_authentic_source",
    "selective_phase_permuted_source",
    "selective_other_game_source",
)
DECISION_SYSTEM = (
    "You are a target-native BrowserGym WebShop Decision Agent. Return exactly one JSON "
    "object with key candidates, containing up to five distinct objects with string key "
    "action. Order them from best to worst. Every action must be immediately executable "
    "from the supplied accessibility tree and use only visible BIDs. Valid forms are "
    "click('bid'), fill('bid','text'), press('bid','KEY'), scroll(x,y), go_back(), "
    "go_forward(), and noop(). Propose useful alternatives when more than one action is "
    "plausible. Do not mention games, latent options, source skills, or explanations."
)


def _load_target_receipts(directories: list[Path]) -> list[dict]:
    rows = []
    for directory in directories:
        rows.extend(
            json.loads(path.read_text())
            for path in sorted(directory.glob("webshop.*.target_only.json"))
        )
    return sorted(rows, key=lambda row: int(row["task_id"].split(".")[1]))


def _discover_opportunities(
    receipt: dict,
    authentic_source: dict,
    other_source: dict,
) -> list[dict]:
    previous_authentic = None
    previous_other = None
    recent_rewards: list[float] = []
    opportunities = []
    for row in receipt["steps"]:
        effects = np.asarray(row["predicted_effects"], dtype=np.float64)
        candidates = tuple(row["candidates"])
        step_index = int(row["step"])
        context = (
            step_index / max(1, receipt["maximum_steps"] - 1),
            (receipt["maximum_steps"] - step_index) / receipt["maximum_steps"],
            float(np.tanh(sum(recent_rewards[-4:]) / 3.0)),
        )
        selections = {"target_only": 0}
        selection_receipts = {}
        for condition in BRANCH_CONDITIONS[1:]:
            previous = previous_other if condition == "selective_other_game_source" else previous_authentic
            selected, selection_receipt = _select_action(
                condition=condition,
                candidates=candidates,
                effects=effects,
                authentic_source=authentic_source,
                other_source=other_source,
                context=context,
                previous_option=previous,
            )
            selections[condition] = selected
            selection_receipts[condition] = selection_receipt
        if len(set(selections.values())) > 1:
            opportunities.append({
                "task_id": receipt["task_id"],
                "step": step_index,
                "before_hash": row["before_hash"],
                "prefix_actions": [prior["selected_action"] for prior in receipt["steps"][:step_index]],
                "prefix_rewards": [prior["reward"] for prior in receipt["steps"][:step_index]],
                "candidates": list(candidates),
                "predicted_effects": effects.tolist(),
                "selected_indices": selections,
                "selection_receipts": selection_receipts,
            })
        baseline_index = int(row["selected_index"])
        previous_authentic = int(nearest_source_options(
            effects[baseline_index : baseline_index + 1], authentic_source
        )[0])
        previous_other = int(nearest_source_options(
            effects[baseline_index : baseline_index + 1], other_source
        )[0])
        recent_rewards.append(float(row["reward"]))
    return opportunities


def _flatten(observation: dict, actual_session: str, canonical_session: str) -> tuple[str, str]:
    from browsergym.utils.obs import flatten_axtree_to_str

    axtree = flatten_axtree_to_str(
        observation["axtree_object"],
        extra_properties=observation.get("extra_element_properties", {}),
    )
    return (
        _canonicalize_session_text(axtree, actual_session, canonical_session),
        _canonicalize_session_text(observation.get("url"), actual_session, canonical_session),
    )


def _run_branch(
    *,
    opportunity: dict,
    branch_action_index: int,
    branch_conditions: list[str],
    backend: MemoizedCompletionBackend,
    wrapper_root: Path,
    expected_goal: str,
    run_id: str,
    fork_horizon: int,
    candidate_count: int,
    schema_retries: int,
) -> dict:
    os.environ["WEBSHOP_BASE_URL"] = "http://127.0.0.1:3000"
    os.environ["WEBSHOP_NUM_GOALS"] = "50"
    task_id = opportunity["task_id"]
    task_index = int(task_id.split(".")[1])
    namespace = f"{run_id}.t{task_index}.s{opportunity['step']}.a{branch_action_index}"
    os.environ["WEBSHOP_SESSION_NAMESPACE"] = namespace
    if str(wrapper_root) not in sys.path:
        sys.path.insert(0, str(wrapper_root))
    from webshop_wrapper import register_webshop_tasks

    register_webshop_tasks(50)
    import gymnasium as gym

    actual_session = f"{namespace}_fixed_{task_index}"
    canonical_session = f"fixed_{task_index}"
    env = gym.make(f"browsergym/{task_id}", headless=True)
    actions = []
    failure = None
    failure_traceback = None
    horizon_reward = 0.0
    terminated = truncated = False
    state_match = False
    first_state_changed = None
    try:
        observation, info = env.reset(seed=0)
        goal = str(observation.get("goal") or observation.get("goal_object") or "")
        if goal != expected_goal:
            raise RuntimeError("fork goal does not match frozen goal manifest")
        for action, expected_reward in zip(
            opportunity["prefix_actions"], opportunity["prefix_rewards"], strict=True
        ):
            observation, reward, terminated, truncated, info = env.step(action)
            if abs(float(reward) - float(expected_reward)) > 1e-9:
                raise RuntimeError("fork prefix reward mismatch")
            if terminated or truncated:
                raise RuntimeError("fork prefix terminated before intervention state")
        axtree, canonical_url = _flatten(observation, actual_session, canonical_session)
        before_hash = stable_hash({"axtree": axtree, "url": canonical_url})
        state_match = before_hash == opportunity["before_hash"]
        if not state_match:
            raise RuntimeError("fork replay did not reconstruct intervention state")

        branch_action = opportunity["candidates"][branch_action_index]
        observation, reward, terminated, truncated, info = env.step(branch_action)
        reward = float(reward)
        horizon_reward += reward
        after_axtree, after_url = _flatten(observation, actual_session, canonical_session)
        after_hash = stable_hash({"axtree": after_axtree, "url": after_url})
        first_state_changed = after_hash != before_hash
        actions.append({
            "relative_step": 0,
            "action": branch_action,
            "reward": reward,
            "target_rank_zero": branch_action_index == 0,
            "after_hash": after_hash,
        })
        previous_action = branch_action
        history = [
            {"action": action, "reward": reward}
            for action, reward in zip(
                opportunity["prefix_actions"], opportunity["prefix_rewards"], strict=True
            )
        ] + [{"action": branch_action, "reward": reward}]
        for relative_step in range(1, fork_horizon):
            if terminated or truncated:
                break
            axtree, canonical_url = _flatten(observation, actual_session, canonical_session)
            payload = {
                "goal": goal,
                "accessibility_tree": axtree[:24000],
                "url": canonical_url,
                "last_action": previous_action,
                "last_action_error": str(observation.get("last_action_error") or ""),
                "history": history[-6:],
                "candidate_count": candidate_count,
            }
            decision_attempts: list[dict] = []
            candidates, raw, decision_attempts = _decision_candidates(
                backend=backend,
                system=DECISION_SYSTEM,
                payload=payload,
                axtree=axtree,
                maximum=candidate_count,
                schema_retries=schema_retries,
                attempts_out=decision_attempts,
            )
            selected_action = candidates[0]
            observation, reward, terminated, truncated, info = env.step(selected_action)
            reward = float(reward)
            horizon_reward += reward
            after_axtree, after_url = _flatten(observation, actual_session, canonical_session)
            actions.append({
                "relative_step": relative_step,
                "action": selected_action,
                "reward": reward,
                "response_sha256": stable_hash(raw),
                "decision_attempts": decision_attempts,
                "after_hash": stable_hash({"axtree": after_axtree, "url": after_url}),
            })
            previous_action = selected_action
            history.append({"action": selected_action, "reward": reward})
    except Exception as exc:
        failure = f"{type(exc).__name__}:{exc}"
        failure_traceback = traceback.format_exc(limit=8)
    finally:
        env.close()
    return {
        "schema_version": 1,
        "task_id": task_id,
        "fork_step": opportunity["step"],
        "branch_action_index": branch_action_index,
        "branch_conditions": branch_conditions,
        "branch_action": opportunity["candidates"][branch_action_index],
        "state_match": state_match,
        "first_state_changed": first_state_changed,
        "horizon_reward": horizon_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "actions": actions,
        "failure": failure,
        "failure_traceback": failure_traceback,
        "session_namespace": namespace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-receipt-dirs", type=Path, nargs="+", required=True)
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
    parser.add_argument(
        "--goal-manifest", type=Path,
        default=REPO / "configs/webshop_consumed_goals_v7.json",
    )
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--maximum-output-tokens", type=int, default=3200)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--fork-horizon", type=int, default=5)
    parser.add_argument("--maximum-opportunities", type=int, default=12)
    parser.add_argument("--run-id", default="causal-v7")
    args = parser.parse_args()

    values = runpy.run_path(str(args.keys))
    api_key = values.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    os.environ["V7_OPENROUTER_API_KEY"] = str(api_key)
    authentic_source = json.loads(args.source_candidate.read_text())
    other_source = json.loads(args.other_source_candidate.read_text())
    goal_manifest = json.loads(args.goal_manifest.read_text())
    target_receipts = _load_target_receipts(args.target_receipt_dirs)
    opportunities = []
    for receipt in target_receipts:
        if receipt["failure"] is None:
            opportunities.extend(
                _discover_opportunities(receipt, authentic_source, other_source)
            )
    opportunities = opportunities[: args.maximum_opportunities]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url,
            {"decision": args.model},
            api_key_env="V7_OPENROUTER_API_KEY",
            json_mode=True,
            temperature=0,
            timeout_seconds=180,
            request_overrides={"max_tokens": args.maximum_output_tokens},
        ),
        cache_path=args.output_dir / "decision_cache.json",
    )
    forks = []
    for opportunity_index, opportunity in enumerate(opportunities):
        by_action: dict[int, list[str]] = {}
        for condition, action_index in opportunity["selected_indices"].items():
            by_action.setdefault(int(action_index), []).append(condition)
        opportunity_receipts = []
        for action_index, conditions in sorted(by_action.items()):
            row = _run_branch(
                opportunity=opportunity,
                branch_action_index=action_index,
                branch_conditions=conditions,
                backend=backend,
                wrapper_root=args.wrapper_root,
                expected_goal=goal_manifest["goals"][opportunity["task_id"]]["instruction_text"],
                run_id=f"{args.run_id}.o{opportunity_index}",
                fork_horizon=args.fork_horizon,
                candidate_count=args.candidate_count,
                schema_retries=args.schema_retries,
            )
            row["receipt_sha256"] = stable_hash(row)
            opportunity_receipts.append(row)
            print(json.dumps({
                "task_id": row["task_id"],
                "fork_step": row["fork_step"],
                "conditions": conditions,
                "action": row["branch_action"],
                "horizon_reward": row["horizon_reward"],
                "steps": len(row["actions"]),
                "failure": row["failure"],
            }))
        forks.append({
            "opportunity": opportunity,
            "branches": opportunity_receipts,
        })
    report = {
        "schema_version": 1,
        "experiment": "webshop_intervention_forks_v7_development",
        "claim_limit": "Consumed tasks only; held-out remains unread.",
        "fork_horizon": args.fork_horizon,
        "opportunities": forks,
        "opportunity_count": len(forks),
        "branch_count": sum(len(row["branches"]) for row in forks),
        "runtime_hashes": {
            "runner": file_sha256(Path(__file__)),
            "episode_runner": file_sha256(REPO / "scripts/run_webshop_transfer_qualification_v5.py"),
            "grounder": file_sha256(args.grounder),
            "source_candidate": file_sha256(args.source_candidate),
            "other_source_candidate": file_sha256(args.other_source_candidate),
            "goal_manifest": file_sha256(args.goal_manifest),
            "decision_cache": file_sha256(args.output_dir / "decision_cache.json")
            if (args.output_dir / "decision_cache.json").exists() else None,
        },
        "held_out_read_or_run": False,
    }
    report["report_sha256"] = stable_hash(report)
    (args.output_dir / "fork_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "opportunities": report["opportunity_count"],
        "branches": report["branch_count"],
        "report_sha256": report["report_sha256"],
    }))


if __name__ == "__main__":
    main()
