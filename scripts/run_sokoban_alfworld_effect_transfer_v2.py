#!/usr/bin/env python3
"""Run paired V2 effect-program transfer on a frozen ALFWorld split."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment
from motif_transfer.alfworld_masked_effect_grounder import (
    score_actions,
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.sokoban_alfworld_effect_harness import CONDITIONS, choose_action
from motif_transfer.sokoban_effect_program import validate_effect_program


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_bound(receipt: dict) -> dict:
    path = Path(receipt["path"])
    if _sha256(path) != receipt["file_sha256"]:
        raise SystemExit(f"frozen dependency changed: {path}")
    return _read(path)


def _summaries(episodes: dict[str, list[dict]]) -> dict[str, dict]:
    result = {}
    for condition, rows in episodes.items():
        steps = sum(row["steps"] for row in rows)
        decisions = [record["decision"] for row in rows for record in row["records"]]
        result[condition] = {
            "tasks": len(rows),
            "successes": sum(row["official_success"] for row in rows),
            "success_rate": sum(row["official_success"] for row in rows) / len(rows),
            "mean_steps": sum(row["steps"] for row in rows) / len(rows),
            "mean_return": sum(row["return"] for row in rows) / len(rows),
            "source_admission_rate": (
                sum(row["source_admissions"] for row in rows) / steps if steps else 0.0
            ),
            "changed_action_rate": (
                sum(row["changed_actions"] for row in rows) / steps if steps else 0.0
            ),
            "changed_option_rate": (
                sum(row["changed_options"] for row in rows) / steps if steps else 0.0
            ),
            "positive_effect_predicate_rate": (
                sum(bool(row["effect_predicates"]["direct_progress_available"])
                    for row in decisions) / len(decisions)
                if decisions else 0.0
            ),
            "required_option_invariance_rate": (
                sum(row.get("required_option_invariant", False) for row in decisions)
                / len(decisions) if decisions else 0.0
            ),
        }
    return result


def _required_mutation(grounded: dict[str, dict]) -> dict[str, dict]:
    required = {
        "SEARCH": "PLACE", "ACQUIRE": "SEARCH", "TRANSFORM": "SEARCH",
        "PLACE": "SEARCH", "VERIFY": "SEARCH",
    }
    return {
        action: row | {
            "required_option": required.get(str(row["required_option"]), "SEARCH")
        }
        for action, row in grounded.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--split", choices=("qualification", "held_out"), required=True)
    parser.add_argument("--seed", type=int, default=96401)
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--minimum-changed-option-rate", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite target report: {args.output}")
    harness = _read(args.harness)
    harness_body = dict(harness)
    harness_hash = str(harness_body.pop("harness_sha256", ""))
    if stable_hash(harness_body) != harness_hash:
        raise SystemExit("frozen Harness hash mismatch")
    if harness.get("status") != "QUALIFICATION_AUTHORIZED":
        raise SystemExit("frozen Harness did not authorize target qualification")
    source = _validate_bound(harness["source_artifact"])
    source_confirmation = _validate_bound(harness["fresh_source_confirmation"])
    target = _validate_bound(harness["masked_target_grounder"])
    validate_effect_program(source)
    validate_target_artifact(target)
    if not source_confirmation.get("source_gate_passed"):
        raise SystemExit("fresh source confirmation no longer passes")
    manifest = _read(args.manifest)
    manifest_body = dict(manifest)
    manifest_hash = str(manifest_body.pop("manifest_sha256", ""))
    if stable_hash(manifest_body) != manifest_hash:
        raise SystemExit("target manifest hash mismatch")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET":
        raise SystemExit("target manifest was not frozen before reset")
    if args.split == "held_out":
        raise SystemExit(
            "held_out is fail-closed; freeze the passed qualification and exact "
            "runner/config hashes in a separate final protocol first"
        )
    threshold = float(harness["effect_threshold"]["selected"])
    task_ids = tuple(map(str, manifest["splits"][args.split]))
    data_root = (args.alfworld_data / "json_2.1.1" / "valid_unseen").resolve()
    episodes: dict[str, list[dict]] = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(args.alfworld_config.resolve()),
            data_path=str(args.alfworld_data.resolve()),
            split="eval_out_of_distribution",
            seed=args.seed,
            game_ids=task_ids,
            max_steps=args.max_steps,
        )
        seen = set()
        try:
            for task_index in range(len(task_ids)):
                observation = environment.reset()
                actual_task_id = (
                    Path(environment.resolved_game_file).resolve()
                    .relative_to(data_root).as_posix()
                )
                if actual_task_id not in task_ids or actual_task_id in seen:
                    raise RuntimeError(
                        f"target pairing/identity violation: {actual_task_id}"
                    )
                seen.add(actual_task_id)
                history: list[str] = []
                records = []
                for step in range(args.max_steps):
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
                        raise RuntimeError("masked target grounder excluded every action")
                    decision = choose_action(
                        condition=condition,
                        grounded=grounded,
                        history=history,
                        source_artifact=source,
                        effect_threshold=threshold,
                    )
                    if condition == "authentic_source_effect_harness":
                        counterfactual = choose_action(
                            condition=condition,
                            grounded=_required_mutation(grounded),
                            history=history,
                            source_artifact=source,
                            effect_threshold=threshold,
                        )
                        decision["required_option_invariant"] = bool(
                            counterfactual["source_selected_option"]
                            == decision["source_selected_option"]
                            and counterfactual["action"] == decision["action"]
                        )
                    else:
                        decision["required_option_invariant"] = True
                    selected = str(decision["action"])
                    before = dict(observation.state)
                    after, reward = environment.step(selected)
                    record_body = {
                        "task_id": actual_task_id,
                        "condition": condition,
                        "step": step,
                        "goal": goal,
                        "before": before,
                        "native_actions": list(observation.native_actions),
                        "grounded": grounded,
                        "decision": decision,
                        "after": dict(after.state),
                        "reward": float(reward),
                        "official_success_after": bool(after.official_success),
                    }
                    records.append(record_body | {
                        "receipt_sha256": stable_hash(record_body),
                    })
                    history.append(selected)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                success = bool(records and records[-1]["official_success_after"])
                episodes[condition].append({
                    "task_index": task_index,
                    "task_id": actual_task_id,
                    "official_success": success,
                    "steps": len(records),
                    "return": sum(row["reward"] for row in records),
                    "source_admissions": sum(
                        bool(row["decision"]["source_admitted"]) for row in records
                    ),
                    "changed_actions": sum(
                        bool(row["decision"]["changed_action"]) for row in records
                    ),
                    "changed_options": sum(
                        bool(row["decision"]["changed_option"]) for row in records
                    ),
                    "diagnostics": dict(Counter(
                        str(row["decision"]["diagnostic"]) for row in records
                    )),
                    "records": records,
                })
                print(json.dumps({
                    "condition": condition,
                    "task_index": task_index,
                    "task_id": actual_task_id,
                    "success": success,
                    "steps": len(records),
                }), flush=True)
        finally:
            environment.close()
        if seen != set(task_ids):
            raise RuntimeError(f"condition {condition} did not cover frozen split")

    summaries = _summaries(episodes)
    authentic_name = "authentic_source_effect_harness"
    controls = (
        "target_only",
        "commit_availability_control",
        "inverted_effect_control",
        "position_occupancy_control",
        "effect_group_permuted_control",
    )
    authentic = summaries[authentic_name]
    nontrivial = authentic["changed_option_rate"] >= args.minimum_changed_option_rate
    predicate_coverage = 0.0 < authentic["positive_effect_predicate_rate"] < 1.0
    invariance = authentic["required_option_invariance_rate"] == 1.0
    superiority = all(
        authentic["successes"] > summaries[name]["successes"] for name in controls
    )
    reference = summaries["target_native_stage_reference"]
    target_capability = reference["successes"] > 0
    passed = bool(
        nontrivial and predicate_coverage and invariance and superiority
        and target_capability
    )
    authentic_by_task = {
        row["task_id"]: row["official_success"] for row in episodes[authentic_name]
    }
    paired = {}
    for name in (*controls, "target_native_stage_reference"):
        other = {row["task_id"]: row["official_success"] for row in episodes[name]}
        diffs = [
            int(authentic_by_task[task]) - int(other[task]) for task in task_ids
        ]
        paired[name] = {
            "wins": sum(value > 0 for value in diffs),
            "ties": sum(value == 0 for value in diffs),
            "losses": sum(value < 0 for value in diffs),
            "net_wins": sum(diffs),
        }
    body = {
        "schema_version": "sokoban-alfworld-effect-transfer-qualification-v2",
        "status": (
            "QUALIFICATION_PASSED_AWAITING_FROZEN_HELDOUT_PROTOCOL"
            if passed else "QUALIFICATION_FAILED_STOP_BEFORE_HELDOUT"
        ),
        "claim_boundary": (
            "FRESH_TARGET_QUALIFICATION_ONLY; HELDOUT_UNREAD; AUTHENTIC_PATH_"
            "USES_STAGE_MASKED_TARGET_NEURAL_GROUNDING_AND_FROZEN_SOURCE_"
            "EFFECT_CONTROL_FLOW"
        ),
        "harness_path": str(args.harness.resolve()),
        "harness_file_sha256": _sha256(args.harness),
        "harness_sha256": harness["harness_sha256"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_file_sha256": _sha256(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_split": args.split,
        "heldout_read": False,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "conditions": list(CONDITIONS),
        "summaries": summaries,
        "gates": {
            "target_capability": target_capability,
            "nontrivial_changed_option_rate": nontrivial,
            "both_effect_predicate_values_observed": predicate_coverage,
            "required_option_invariance": invariance,
            "strict_success_superiority_to_null_and_source_controls": superiority,
        },
        "thresholds": {
            "minimum_changed_option_rate": args.minimum_changed_option_rate,
            "authentic_successes_strictly_greater_than_each_control": True,
            "target_native_stage_reference_is_reported_not_a_source_control": True,
        },
        "paired_official_success": paired,
        "qualification_passed": passed,
        "next_step": (
            "FREEZE_RUNNER_CONFIG_AND_RUN_HELDOUT_ONCE"
            if passed else "STOP_AND_DIAGNOSE_WITHOUT_REUSING_THIS_SPLIT"
        ),
        "episodes": episodes,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": report["status"],
        "report_sha256": report["report_sha256"],
        "summaries": summaries,
        "gates": report["gates"],
        "paired_official_success": paired,
        "next_step": report["next_step"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
