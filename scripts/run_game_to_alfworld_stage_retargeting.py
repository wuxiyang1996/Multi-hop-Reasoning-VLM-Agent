#!/usr/bin/env python3
"""Run the permission-bounded Sokoban effect program on consumed ALFWorld tasks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_masked_effect_grounder import (  # noqa: E402
    score_actions,
    validate_artifact as validate_target_artifact,
)
from motif_transfer.alfworld_stage_retargeting import (  # noqa: E402
    CONDITIONS,
    choose_action,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.sokoban_effect_program import validate_effect_program  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound(receipt: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(receipt["path"]))
    if _sha256(path) != str(receipt["file_sha256"]):
        raise RuntimeError(f"bound artifact changed: {path}")
    return _read(path)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [step["decision"] for row in rows for step in row["records"]]
    total_steps = sum(row["steps"] for row in rows)
    return {
        "tasks": len(rows),
        "successes": sum(row["official_success"] for row in rows),
        "success_rate": sum(row["official_success"] for row in rows) / len(rows),
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
        "source_admission_rate": (
            sum(row["source_admissions"] for row in rows) / total_steps
        ),
        "changed_action_rate": (
            sum(row["changed_actions"] for row in rows) / total_steps
        ),
        "changed_option_rate": (
            sum(row["changed_options"] for row in rows) / total_steps
        ),
        "positive_effect_rate": (
            sum(bool(row["effect_predicates"]["direct_progress_available"])
                for row in decisions) / len(decisions)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite report: {args.output}")
    config_path = args.config.resolve()
    config = _read(config_path)
    if config.get("claim_boundary") != "CONSUMED_ALFWORLD_DIAGNOSTIC_ONLY":
        raise SystemExit("this runner is restricted to consumed ALFWorld data")

    harness_path = (REPO / str(config["harness"])).resolve()
    harness = _read(harness_path)
    harness_body = dict(harness)
    claimed_harness_hash = str(harness_body.pop("harness_sha256", ""))
    if stable_hash(harness_body) != claimed_harness_hash:
        raise SystemExit("legacy frozen Harness hash mismatch")
    source = _bound(harness["source_artifact"])
    confirmation = _bound(harness["fresh_source_confirmation"])
    target = _bound(harness["masked_target_grounder"])
    validate_effect_program(source)
    validate_target_artifact(target)
    if not confirmation.get("source_gate_passed"):
        raise SystemExit("source confirmation gate no longer passes")

    manifest_path = (REPO / str(config["manifest"])).resolve()
    manifest = _read(manifest_path)
    manifest_body = dict(manifest)
    claimed_manifest_hash = str(manifest_body.pop("manifest_sha256", ""))
    if stable_hash(manifest_body) != claimed_manifest_hash:
        raise SystemExit("target manifest hash mismatch")
    split = str(config["split"])
    if split != "qualification":
        raise SystemExit("consumed diagnostic may only reuse qualification")
    task_ids = tuple(map(str, manifest["splits"][split]))
    task_limit = int(config.get("task_limit", len(task_ids)))
    task_ids = task_ids[:task_limit]
    target_config = config["target"]
    data_root = (
        Path(str(target_config["alfworld_data"])) / "json_2.1.1" / "valid_unseen"
    ).resolve()
    maximum_steps = int(target_config["maximum_steps"])
    threshold = float(config["effect_threshold"])
    episodes: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    initial_hashes: dict[str, dict[str, str]] = {
        condition: {} for condition in CONDITIONS
    }

    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(Path(str(target_config["alfworld_config"])).resolve()),
            data_path=str(Path(str(target_config["alfworld_data"])).resolve()),
            split="eval_out_of_distribution",
            seed=int(target_config["seed"]),
            game_ids=task_ids,
            max_steps=maximum_steps,
        )
        seen: set[str] = set()
        try:
            for task_index in range(len(task_ids)):
                observation = environment.reset()
                task_id = (
                    Path(environment.resolved_game_file).resolve()
                    .relative_to(data_root).as_posix()
                )
                if task_id not in task_ids or task_id in seen:
                    raise RuntimeError(f"paired target identity violation: {task_id}")
                seen.add(task_id)
                initial_hashes[condition][task_id] = stable_hash({
                    "task_id": task_id,
                    "state": dict(observation.state),
                    "native_actions": list(observation.native_actions),
                })
                history: list[str] = []
                records: list[dict[str, Any]] = []
                for step in range(maximum_steps):
                    goal = str(observation.state.get("task_goal", ""))
                    grounded = score_actions(
                        goal=goal,
                        observation=str(observation.state.get("observation", "")),
                        native_actions=observation.native_actions,
                        step=step,
                        action_history=history,
                        artifact=target,
                    )
                    if not grounded:
                        raise RuntimeError("target neural grounder excluded every action")
                    decision = choose_action(
                        condition=condition,
                        grounded=grounded,
                        history=history,
                        source_artifact=source,
                        effect_threshold=threshold,
                    )
                    selected = str(decision["action"])
                    before_state = dict(observation.state)
                    before_actions = tuple(observation.native_actions)
                    after, reward = environment.step(selected)
                    body = {
                        "task_id": task_id,
                        "condition": condition,
                        "step": step,
                        "before_state": before_state,
                        "before_native_actions": before_actions,
                        "grounding_sha256": stable_hash(grounded),
                        "decision": decision,
                        "selected_native_action": selected,
                        "after_state": dict(after.state),
                        "after_native_actions": list(after.native_actions),
                        "official_reward_evaluator_only": float(reward),
                        "official_success_evaluator_only": bool(after.official_success),
                    }
                    records.append(body | {"receipt_sha256": stable_hash(body)})
                    history.append(selected)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                episodes[condition].append({
                    "task_id": task_id,
                    "official_success": bool(
                        records and records[-1]["official_success_evaluator_only"]
                    ),
                    "steps": len(records),
                    "source_admissions": sum(
                        row["decision"]["source_admitted"] for row in records
                    ),
                    "changed_actions": sum(
                        row["decision"]["changed_action"] for row in records
                    ),
                    "changed_options": sum(
                        row["decision"]["changed_option"] for row in records
                    ),
                    "diagnostics": dict(Counter(
                        row["decision"]["diagnostic"] for row in records
                    )),
                    "records": records,
                })
                print(json.dumps({
                    "condition": condition,
                    "task": task_index + 1,
                    "tasks": len(task_ids),
                    "success": episodes[condition][-1]["official_success"],
                    "steps": len(records),
                }), flush=True)
        finally:
            environment.close()
        if seen != set(task_ids):
            raise RuntimeError(f"condition {condition} missed paired tasks")

    summaries = {name: _summary(rows) for name, rows in episodes.items()}
    reference_initial = initial_hashes["null_skill_same_harness"]
    matched_initial = all(
        values == reference_initial for values in initial_hashes.values()
    )
    authentic = summaries["authentic_source_skill"]
    controls = (
        "null_skill_same_harness",
        "commit_availability_control",
        "inverted_effect_control",
        "position_prior_control",
    )
    paired: dict[str, Any] = {}
    authentic_rows = {row["task_id"]: row for row in episodes["authentic_source_skill"]}
    for name in controls:
        comparator = {row["task_id"]: row for row in episodes[name]}
        delta = [
            int(authentic_rows[task]["official_success"])
            - int(comparator[task]["official_success"])
            for task in task_ids
        ]
        paired[name] = {
            "wins": sum(value > 0 for value in delta),
            "losses": sum(value < 0 for value in delta),
            "ties": sum(value == 0 for value in delta),
            "net_wins": sum(delta),
        }
    gates = {
        "source_qualified": True,
        "matched_initial_states": matched_initial,
        "authentic_nontrivial": authentic["changed_option_rate"] >= float(
            config["gates"]["minimum_changed_option_rate"]
        ),
        "authentic_zero_negative_transfer": paired[
            "null_skill_same_harness"
        ]["losses"] == 0,
        "authentic_success_gain_over_null": paired[
            "null_skill_same_harness"
        ]["net_wins"] > 0,
        "authentic_strictly_beats_source_controls": all(
            authentic["successes"] > summaries[name]["successes"]
            for name in controls[1:]
        ),
        "oracle_not_below_authentic": summaries["target_oracle_skill"][
            "successes"
        ] >= authentic["successes"],
    }
    passed = all(gates.values())
    body = {
        "schema_version": "game-to-alfworld-stage-retargeting-v14",
        "status": (
            "CONSUMED_MECHANISM_GATE_PASSED" if passed
            else "CONSUMED_MECHANISM_GATE_FAILED"
        ),
        "claim_boundary": config["claim_boundary"],
        "config_path": str(config_path),
        "config_file_sha256": _sha256(config_path),
        "harness_path": str(harness_path),
        "harness_file_sha256": _sha256(harness_path),
        "source_artifact_sha256": source["artifact_sha256"],
        "source_confirmation_sha256": confirmation["report_sha256"],
        "target_grounder_sha256": target["artifact_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": list(task_ids),
        "conditions": list(CONDITIONS),
        "authority_contract": {
            "source_selects": "POSITION_OR_COMMIT_ONLY",
            "target_neural_policy_selects": "NATIVE_ACTION_WITHIN_SELECTED_OPTION",
            "official_outcome_visible_to": "EVALUATOR_ONLY",
        },
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "episodes": episodes,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": body["status"],
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
