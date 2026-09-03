#!/usr/bin/env python3
"""Run typed real-source IR on the consumed ALFWorld qualification split."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment
from motif_transfer.alfworld_masked_effect_grounder import (
    score_actions,
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.typed_alfworld_harness import (
    CONDITIONS,
    choose_typed_action,
    validate_typed_effect_ir,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_bound(receipt: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(receipt["path"]))
    if _sha256(path) != receipt["file_sha256"]:
        raise SystemExit(f"frozen dependency changed: {path}")
    return _read(path)


def _mutate_required_diagnostic(
    grounded: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    replacements = {
        "SEARCH": "PLACE",
        "ACQUIRE": "SEARCH",
        "TRANSFORM": "SEARCH",
        "PLACE": "SEARCH",
        "VERIFY": "SEARCH",
    }
    return {
        action: dict(row) | {
            "required_option": replacements.get(
                str(row.get("required_option", "SEARCH")), "SEARCH"
            )
        }
        for action, row in grounded.items()
    }


def _summaries(episodes: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    result = {}
    for condition, rows in episodes.items():
        steps = sum(int(row["steps"]) for row in rows)
        result[condition] = {
            "tasks": len(rows),
            "successes": sum(bool(row["official_success"]) for row in rows),
            "success_rate": sum(bool(row["official_success"]) for row in rows)
            / len(rows),
            "mean_steps": steps / len(rows),
            "mean_return": sum(float(row["return"]) for row in rows) / len(rows),
            "source_admission_rate": (
                sum(int(row["source_admissions"]) for row in rows) / steps
                if steps else 0.0
            ),
            "changed_action_rate": (
                sum(int(row["changed_actions"]) for row in rows) / steps
                if steps else 0.0
            ),
            "changed_effect_rate": (
                sum(int(row["changed_effects"]) for row in rows) / steps
                if steps else 0.0
            ),
            "required_option_invariance_rate": (
                sum(int(row["invariant_decisions"]) for row in rows) / steps
                if steps else 0.0
            ),
            "repeated_nonposition_effects": sum(
                int(row["repeated_nonposition_effects"]) for row in rows
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=96501)
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--minimum-changed-effect-rate", type=float, default=0.02)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite target report: {args.output}")
    harness = _read(args.harness)
    harness_body = dict(harness)
    claimed_harness_hash = str(harness_body.pop("harness_sha256", ""))
    if stable_hash(harness_body) != claimed_harness_hash:
        raise SystemExit("typed target Harness hash mismatch")
    if harness.get("status") != "DEVELOPMENT_DIAGNOSTIC_AUTHORIZED":
        raise SystemExit("typed target Harness did not authorize online diagnostic")
    source_report = _validate_bound(harness["source_report"])
    source_ir = harness["source_effect_ir"]
    if source_report.get("effect_ir", {}).get("ir_sha256") != source_ir.get("ir_sha256"):
        raise SystemExit("source report/embedded IR mismatch")
    validate_typed_effect_ir(source_ir)
    target = _validate_bound(harness["target_grounder"])
    validate_target_artifact(target)
    manifest = _read(args.manifest)
    manifest_body = dict(manifest)
    claimed_manifest_hash = str(manifest_body.pop("manifest_sha256", ""))
    if stable_hash(manifest_body) != claimed_manifest_hash:
        raise SystemExit("target manifest hash mismatch")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET":
        raise SystemExit("target manifest was not frozen before reset")
    task_ids = tuple(map(str, manifest["splits"]["qualification"]))
    data_root = (args.alfworld_data / "json_2.1.1" / "valid_unseen").resolve()
    threshold = float(harness["minimum_realization_score"]["selected"])
    concrete_action_ranking = str(
        harness.get("concrete_action_ranking", "realization_first")
    )
    minimum_target_policy_ratio = float(
        harness.get("minimum_target_policy_ratio", {}).get("selected", 0.0)
    )
    episodes: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(args.alfworld_config.resolve()),
            data_path=str(args.alfworld_data.resolve()),
            split="eval_out_of_distribution",
            seed=args.seed,
            game_ids=task_ids,
            max_steps=args.max_steps,
        )
        seen: set[str] = set()
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
                history_options: list[str] = []
                records: list[dict[str, Any]] = []
                previous_effect = ""
                repeated_nonposition = 0
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
                        raise RuntimeError("target grounder excluded every action")
                    decision = choose_typed_action(
                        condition=condition,
                        grounded=grounded,
                        history=history,
                        grounded_history_options=history_options,
                        source_ir=source_ir,
                        minimum_realization_score=threshold,
                        concrete_action_ranking=concrete_action_ranking,
                        minimum_target_policy_ratio=minimum_target_policy_ratio,
                    )
                    if condition == "authentic_typed_ir":
                        counterfactual = choose_typed_action(
                            condition=condition,
                            grounded=_mutate_required_diagnostic(grounded),
                            history=history,
                            grounded_history_options=history_options,
                            source_ir=source_ir,
                            minimum_realization_score=threshold,
                            concrete_action_ranking=concrete_action_ranking,
                            minimum_target_policy_ratio=minimum_target_policy_ratio,
                        )
                        decision["required_option_invariant"] = bool(
                            counterfactual["action"] == decision["action"]
                            and counterfactual["source_selected_effect"]
                            == decision["source_selected_effect"]
                        )
                    else:
                        decision["required_option_invariant"] = True
                    selected = str(decision["action"])
                    selected_option = str(grounded[selected]["option"])
                    realized_effect = str(decision["target_realized_effect"])
                    if realized_effect != "POSITION" and realized_effect == previous_effect:
                        repeated_nonposition += 1
                    previous_effect = realized_effect
                    before = dict(observation.state)
                    after, reward = environment.step(selected)
                    record_body = {
                        "task_id": actual_task_id,
                        "condition": condition,
                        "step": step,
                        "goal": goal,
                        "before": before,
                        "native_actions": list(observation.native_actions),
                        "selected_grounding": grounded[selected],
                        "fallback_grounding": grounded[decision["fallback_action"]],
                        "decision": decision,
                        "after": dict(after.state),
                        "reward": float(reward),
                        "official_success_after": bool(after.official_success),
                    }
                    records.append(record_body | {
                        "receipt_sha256": stable_hash(record_body)
                    })
                    history.append(selected)
                    history_options.append(selected_option)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                success = bool(records and records[-1]["official_success_after"])
                episodes[condition].append({
                    "task_index": task_index,
                    "task_id": actual_task_id,
                    "official_success": success,
                    "steps": len(records),
                    "return": sum(float(row["reward"]) for row in records),
                    "source_admissions": sum(
                        bool(row["decision"]["source_admitted"]) for row in records
                    ),
                    "changed_actions": sum(
                        bool(row["decision"]["changed_action"]) for row in records
                    ),
                    "changed_effects": sum(
                        bool(row["decision"]["changed_effect"]) for row in records
                    ),
                    "invariant_decisions": sum(
                        bool(row["decision"]["required_option_invariant"])
                        for row in records
                    ),
                    "repeated_nonposition_effects": repeated_nonposition,
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
    authentic_name = "authentic_typed_ir"
    authentic = summaries[authentic_name]
    controls = ("target_only", "edge_permuted_ir", "wrong_guard_ir")
    superiority = all(
        authentic["successes"] > summaries[condition]["successes"]
        for condition in controls
    )
    nontrivial = (
        authentic["changed_effect_rate"] >= args.minimum_changed_effect_rate
    )
    invariance = authentic["required_option_invariance_rate"] == 1.0
    target_capability = summaries["target_only"]["successes"] > 0
    passed = bool(superiority and nontrivial and invariance and target_capability)
    authentic_by_task = {
        str(row["task_id"]): bool(row["official_success"])
        for row in episodes[authentic_name]
    }
    paired = {}
    for condition in controls:
        other = {
            str(row["task_id"]): bool(row["official_success"])
            for row in episodes[condition]
        }
        deltas = [
            int(authentic_by_task[task]) - int(other[task]) for task in task_ids
        ]
        paired[condition] = {
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "net_wins": sum(deltas),
        }
    body = {
        "schema_version": (
            "typed-multisource-alfworld-diagnostic-v6"
            if concrete_action_ranking == "target_policy_within_effect"
            else "typed-multisource-alfworld-diagnostic-v5"
        ),
        "status": (
            "DEVELOPMENT_DIAGNOSTIC_POSITIVE_CANDIDATE"
            if passed else "DEVELOPMENT_DIAGNOSTIC_NEGATIVE_STOP"
        ),
        "claim_boundary": (
            "ALREADY_CONSUMED_ALFWORLD_QUALIFICATION_DIAGNOSTIC_ONLY;_"
            "TARGET_HELDOUT_UNREAD_AND_FORBIDDEN"
        ),
        "harness_path": str(args.harness.resolve()),
        "harness_file_sha256": _sha256(args.harness),
        "harness_sha256": harness["harness_sha256"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_file_sha256": _sha256(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_split": "qualification",
        "heldout_read": False,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "conditions": list(CONDITIONS),
        "summaries": summaries,
        "paired_official_success": paired,
        "gates": {
            "target_capability": target_capability,
            "nontrivial_changed_effect_rate": nontrivial,
            "required_option_invariance": invariance,
            "strict_success_superiority_to_all_controls": superiority,
        },
        "thresholds": {
            "minimum_changed_effect_rate": args.minimum_changed_effect_rate,
            "authentic_successes_strictly_greater_than_each_control": True,
        },
        "diagnostic_passed": passed,
        "next_step": (
            "FREEZE_EXACT_ARTIFACT_RUNNER_AND_CONTROLS_BEFORE_HELDOUT"
            if passed else "STOP_BEFORE_HELDOUT_AND_REPORT_NEGATIVE_TRANSFER"
        ),
        "episodes": episodes,
    }
    if concrete_action_ranking == "target_policy_within_effect":
        body["target_native_admission"] = {
            "concrete_action_ranking": concrete_action_ranking,
            "minimum_target_policy_ratio": minimum_target_policy_ratio,
            "selection_partition": "adaptation_train",
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
