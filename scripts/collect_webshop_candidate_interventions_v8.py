#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.webshop_applicability_v8 import candidate_semantics  # noqa: E402


def _load_target_receipts(directories: list[Path]) -> list[dict]:
    rows = []
    for directory in directories:
        rows.extend(
            json.loads(path.read_text())
            for path in sorted(directory.glob("webshop.*.target_only.json"))
        )
    return sorted(rows, key=lambda row: int(row["task_id"].split(".")[1]))


def discover_exact_stalls(receipts: list[dict]) -> list[tuple[dict, dict]]:
    return [
        (receipt, step)
        for receipt in receipts
        for step in receipt["steps"]
        if step.get("observed_exact_stall") is True
    ]


def discover_no_progress_states(
    receipts: list[dict], *, minimum_no_effect_steps: int = 2
) -> list[tuple[dict, dict]]:
    """Return decision states preceded by an exact, observed no-effect streak.

    This is computed only from environment hashes, not an LLM judgment or a
    reward proxy.  It catches short action cycles where rank zero alternates
    between two actions while neither changes the canonical state.
    """
    if minimum_no_effect_steps < 1:
        raise ValueError("minimum_no_effect_steps must be positive")
    opportunities = []
    for receipt in receipts:
        streak = 0
        for step in receipt["steps"]:
            if streak >= minimum_no_effect_steps:
                opportunities.append((receipt, step))
            streak = streak + 1 if step["before_hash"] == step["after_hash"] else 0
    return opportunities


def _flatten(observation: dict, actual_session: str, canonical_session: str) -> tuple[str, str]:
    from browsergym.utils.obs import flatten_axtree_to_str

    axtree = flatten_axtree_to_str(
        observation["axtree_object"],
        extra_properties=observation.get("extra_element_properties", {}),
    )
    return (
        str(axtree).replace(actual_session, canonical_session),
        str(observation.get("url") or "").replace(actual_session, canonical_session),
    )


def _run_branch(
    *,
    receipt: dict,
    step: dict,
    candidate_index: int,
    wrapper_root: Path,
    expected_goal: str,
    run_id: str,
) -> dict:
    task_id = receipt["task_id"]
    task_index = int(task_id.split(".")[1])
    namespace = f"{run_id}.t{task_index}.s{step['step']}.a{candidate_index}"
    os.environ["WEBSHOP_BASE_URL"] = "http://127.0.0.1:3000"
    os.environ["WEBSHOP_NUM_GOALS"] = "50"
    os.environ["WEBSHOP_SESSION_NAMESPACE"] = namespace
    if str(wrapper_root) not in sys.path:
        sys.path.insert(0, str(wrapper_root))
    from webshop_wrapper import register_webshop_tasks

    register_webshop_tasks(50)
    import gymnasium as gym

    actual_session = f"{namespace}_fixed_{task_index}"
    canonical_session = f"fixed_{task_index}"
    action = step["candidates"][candidate_index]
    env = gym.make(f"browsergym/{task_id}", headless=True)
    failure = None
    failure_traceback = None
    state_match = False
    actual_state_changed = None
    reward = 0.0
    terminated = truncated = False
    try:
        observation, info = env.reset(seed=0)
        goal = str(observation.get("goal") or observation.get("goal_object") or "")
        if goal != expected_goal:
            raise RuntimeError("candidate fork goal does not match frozen manifest")
        for prefix in receipt["steps"][: step["step"]]:
            observation, prefix_reward, terminated, truncated, info = env.step(
                prefix["selected_action"]
            )
            if abs(float(prefix_reward) - float(prefix["reward"])) > 1e-9:
                raise RuntimeError("candidate fork prefix reward mismatch")
            if terminated or truncated:
                raise RuntimeError("candidate fork prefix terminated early")
        axtree, url = _flatten(observation, actual_session, canonical_session)
        before_hash = stable_hash({"axtree": axtree, "url": url})
        state_match = before_hash == step["before_hash"]
        if not state_match:
            raise RuntimeError("candidate fork state hash mismatch")
        live_semantics = candidate_semantics(
            observation_text=axtree,
            url=url,
            goal=goal,
            action=action,
        )
        observation, reward, terminated, truncated, info = env.step(action)
        after_tree, after_url = _flatten(observation, actual_session, canonical_session)
        after_hash = stable_hash({"axtree": after_tree, "url": after_url})
        actual_state_changed = after_hash != before_hash
    except Exception as exc:
        failure = f"{type(exc).__name__}:{exc}"
        failure_traceback = traceback.format_exc(limit=8)
    finally:
        env.close()
    predicted = step["predicted_effects"][candidate_index]
    return {
        "task_id": task_id,
        "step": step["step"],
        "candidate_index": candidate_index,
        "action": action,
        "semantics": live_semantics if "live_semantics" in locals() else None,
        "predicted_effect": predicted,
        "predicted_state_changed": predicted[0],
        "predicted_terminated": predicted[6],
        "state_match": state_match,
        "actual_state_changed": actual_state_changed,
        "actual_reward": float(reward),
        "actual_terminated": bool(terminated),
        "actual_truncated": bool(truncated),
        "failure": failure,
        "failure_traceback": failure_traceback,
        "session_namespace": namespace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument(
        "--goal-manifest", type=Path,
        default=REPO / "configs/webshop_consumed_goals_v7.json",
    )
    parser.add_argument(
        "--grouped-split", type=Path,
        default=REPO / "configs/webshop_grouped_development_v8.json",
    )
    parser.add_argument("--run-id", default="candidate-v8")
    parser.add_argument(
        "--opportunity-kind",
        choices=("exact_stall", "no_progress"),
        default="exact_stall",
    )
    parser.add_argument("--minimum-no-effect-steps", type=int, default=2)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--candidate-indices", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipts = _load_target_receipts(args.receipt_dirs)
    stalls = (
        discover_exact_stalls(receipts)
        if args.opportunity_kind == "exact_stall"
        else discover_no_progress_states(
            receipts,
            minimum_no_effect_steps=args.minimum_no_effect_steps,
        )
    )
    if args.task_ids is not None:
        stalls = [row for row in stalls if row[0]["task_id"] in args.task_ids]
    if args.steps is not None:
        stalls = [row for row in stalls if int(row[1]["step"]) in args.steps]
    goals = json.loads(args.goal_manifest.read_text())
    split = json.loads(args.grouped_split.read_text())
    role_by_group = {
        group_id: role
        for role, groups in split["groups_by_role"].items()
        for group_id in groups
    }
    branches = []
    for receipt, step in stalls:
        task_id = receipt["task_id"]
        group_id = goals["goals"][task_id]["asin"]
        candidate_indices = (
            range(len(step["candidates"]))
            if args.candidate_indices is None
            else args.candidate_indices
        )
        for candidate_index in candidate_indices:
            if not 0 <= candidate_index < len(step["candidates"]):
                raise SystemExit(
                    f"candidate index {candidate_index} is invalid for "
                    f"{task_id} step {step['step']}"
                )
            branch = _run_branch(
                receipt=receipt,
                step=step,
                candidate_index=candidate_index,
                wrapper_root=args.wrapper_root,
                expected_goal=goals["goals"][task_id]["instruction_text"],
                run_id=args.run_id,
            )
            branch["semantic_group"] = group_id
            branch["group_role"] = role_by_group[group_id]
            branches.append(branch)
            print(json.dumps({
                key: branch[key] for key in (
                    "task_id", "step", "candidate_index", "action", "state_match",
                    "actual_state_changed", "actual_reward", "actual_terminated", "failure",
                )
            }), flush=True)
    valid = [row for row in branches if row["failure"] is None]
    report = {
        "schema_version": 1,
        "experiment": "webshop_candidate_interventions_v8_diagnostic",
        "claim_limit": (
            "Consumed grouped-development roles only; not held-out evidence."
        ),
        "opportunity_kind": args.opportunity_kind,
        "minimum_no_effect_steps": args.minimum_no_effect_steps,
        "opportunities": len(stalls),
        "branches": branches,
        "metrics": {
            "branches": len(branches),
            "failures": len(branches) - len(valid),
            "all_state_matches": all(row["state_match"] for row in valid),
            "state_changed_mse": float(np.mean([
                (row["predicted_state_changed"] - float(row["actual_state_changed"])) ** 2
                for row in valid
            ])) if valid else None,
            "terminated_mse": float(np.mean([
                (row["predicted_terminated"] - float(row["actual_terminated"])) ** 2
                for row in valid
            ])) if valid else None,
        },
        "runtime_hashes": {
            "collector": file_sha256(Path(__file__)),
            "goal_manifest": file_sha256(args.goal_manifest),
            "grouped_split": file_sha256(args.grouped_split),
        },
        "held_out_read_or_run": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["metrics"]))


if __name__ == "__main__":
    main()
