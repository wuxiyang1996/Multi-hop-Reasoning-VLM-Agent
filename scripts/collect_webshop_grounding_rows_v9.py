#!/usr/bin/env python3
"""Replay WebShop adaptation receipts into target-native neural labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.webshop_applicability_v8 import candidate_semantics  # noqa: E402
from motif_transfer.webshop_neural_symbolic_v9 import (  # noqa: E402
    target_features,
    visible_goal_constraint_status,
)
from scripts.run_webshop_transfer_qualification_v5 import (  # noqa: E402
    _canonicalize_session_text,
)


def _flatten(observation: dict, actual_session: str, canonical_session: str) -> tuple[str, str]:
    from browsergym.utils.obs import flatten_axtree_to_str

    axtree = flatten_axtree_to_str(
        observation["axtree_object"],
        extra_properties=observation.get("extra_element_properties", {}),
    )
    return (
        _canonicalize_session_text(axtree, actual_session, canonical_session),
        _canonicalize_session_text(
            observation.get("url"), actual_session, canonical_session,
        ),
    )


def _run_sequence(
    *,
    receipt: dict,
    prefix_length: int,
    actions: list[str],
    sequence_id: str,
    wrapper_root: Path,
    run_id: str,
) -> dict:
    os.environ["WEBSHOP_BASE_URL"] = "http://127.0.0.1:3000"
    os.environ["WEBSHOP_NUM_GOALS"] = "50"
    task_id = receipt["task_id"]
    task_index = int(task_id.split(".")[1])
    namespace = f"{run_id}.t{task_index}.{stable_hash(sequence_id)[:10]}"
    os.environ["WEBSHOP_SESSION_NAMESPACE"] = namespace
    if str(wrapper_root) not in sys.path:
        sys.path.insert(0, str(wrapper_root))
    from webshop_wrapper import register_webshop_tasks

    register_webshop_tasks(50)
    import gymnasium as gym

    actual_session = f"{namespace}_fixed_{task_index}"
    canonical_session = f"fixed_{task_index}"
    env = gym.make(f"browsergym/{task_id}", headless=True)
    rows = []
    failure = None
    failure_traceback = None
    prefix_state_match = False
    try:
        observation, _ = env.reset(seed=receipt["seed"])
        goal = str(observation.get("goal") or observation.get("goal_object") or "")
        if goal != receipt["goal"]:
            raise RuntimeError("live goal does not match source receipt")
        for prefix in receipt["steps"][:prefix_length]:
            observation, reward, terminated, truncated, _ = env.step(
                prefix["selected_action"]
            )
            if abs(float(reward) - float(prefix["reward"])) > 1e-9:
                raise RuntimeError("prefix reward mismatch")
            if terminated or truncated:
                raise RuntimeError("prefix terminated early")
        axtree, url = _flatten(observation, actual_session, canonical_session)
        before_hash = stable_hash({"axtree": axtree, "url": url})
        expected_hash = receipt["steps"][prefix_length]["before_hash"]
        prefix_state_match = before_hash == expected_hash
        if not prefix_state_match:
            raise RuntimeError("prefix state hash mismatch")
        previous_before_hash = (
            receipt["steps"][prefix_length - 1]["before_hash"]
            if prefix_length else None
        )
        previous_after_hash = (
            receipt["steps"][prefix_length - 1]["after_hash"]
            if prefix_length else None
        )
        for offset, action in enumerate(actions):
            step_index = prefix_length + offset
            before_hash = stable_hash({"axtree": axtree, "url": url})
            semantics = candidate_semantics(
                observation_text=axtree, url=url, goal=goal, action=action,
            )
            satisfied, unsatisfied = visible_goal_constraint_status(axtree, goal)
            prior_no_effect = bool(
                previous_before_hash
                and previous_after_hash
                and previous_before_hash == previous_after_hash
            )
            features = target_features(
                semantics,
                visible_satisfied=satisfied,
                visible_unsatisfied=unsatisfied,
                prior_no_effect=prior_no_effect,
                step_index=step_index,
                maximum_steps=receipt["maximum_steps"],
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            after_tree, after_url = _flatten(
                observation, actual_session, canonical_session,
            )
            after_hash = stable_hash({"axtree": after_tree, "url": after_url})
            changed = after_hash != before_hash
            rows.append({
                "sequence_id": sequence_id,
                "task_id": task_id,
                "step": step_index,
                "action": action,
                "semantics": semantics,
                "state_context": {
                    "visible_goal_constraint_satisfied": satisfied,
                    "visible_goal_constraint_unsatisfied": unsatisfied,
                    "prior_action_had_no_effect": prior_no_effect,
                },
                "features": list(features),
                "outcomes": [
                    float(changed),
                    float(terminated),
                    float(reward),
                    float(changed and semantics["is_goal_constraint"]),
                ],
                "before_hash": before_hash,
                "after_hash": after_hash,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            previous_before_hash = before_hash
            previous_after_hash = after_hash
            axtree, url = after_tree, after_url
            if terminated or truncated:
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}:{exc}"
        failure_traceback = traceback.format_exc(limit=8)
    finally:
        env.close()
    return {
        "sequence_id": sequence_id,
        "task_id": task_id,
        "prefix_length": prefix_length,
        "actions": actions,
        "prefix_state_match": prefix_state_match,
        "rows": rows,
        "failure": failure,
        "failure_traceback": failure_traceback,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/webshop_neural_symbolic_v9_adaptation.json",
    )
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--run-id", default="webshop-v9-grounding")
    parser.add_argument(
        "--role", choices=("adaptation", "calibration"), default="adaptation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    specs = []
    for raw_path in config["adaptation_receipts"]:
        path = REPO / raw_path
        receipt = json.loads(path.read_text())
        specs.append({
            "sequence_id": f"{receipt['task_id']}:target-on-policy",
            "target_receipt": raw_path,
            "prefix_length": 0,
            "actions": [row["selected_action"] for row in receipt["steps"]],
        })
    specs.extend(config["adaptation_sequences"])
    sequences = []
    for spec in specs:
        receipt_path = REPO / spec["target_receipt"]
        receipt = json.loads(receipt_path.read_text())
        row = _run_sequence(
            receipt=receipt,
            prefix_length=int(spec["prefix_length"]),
            actions=list(spec["actions"]),
            sequence_id=str(spec["sequence_id"]),
            wrapper_root=args.wrapper_root,
            run_id=args.run_id,
        )
        row["target_receipt"] = str(receipt_path)
        row["target_receipt_sha256"] = file_sha256(receipt_path)
        sequences.append(row)
        print(json.dumps({
            "sequence_id": row["sequence_id"],
            "rows": len(row["rows"]),
            "prefix_state_match": row["prefix_state_match"],
            "failure": row["failure"],
        }), flush=True)
    report = {
        "schema_version": 1,
        "experiment": f"webshop_v9_target_native_grounding_{args.role}",
        "claim_limit": (
            f"WebShop {args.role} groups only; not confirmation evidence."
        ),
        "sequences": sequences,
        "metrics": {
            "sequences": len(sequences),
            "rows": sum(len(row["rows"]) for row in sequences),
            "failures": sum(row["failure"] is not None for row in sequences),
            "all_prefix_states_match": all(
                row["prefix_state_match"] for row in sequences
            ),
        },
        "runtime_hashes": {
            "collector": file_sha256(Path(__file__)),
            "config": file_sha256(args.config),
        },
        "confirmation_read_or_run": False,
        "held_out_read_or_run": False,
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["metrics"]), flush=True)


if __name__ == "__main__":
    main()
