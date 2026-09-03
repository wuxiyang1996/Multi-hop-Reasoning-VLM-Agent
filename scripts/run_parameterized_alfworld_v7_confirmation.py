#!/usr/bin/env python3
"""Run the frozen parameterized real-source Harness on fresh ALFWorld tasks."""

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
from motif_transfer.parameterized_alfworld_harness import (
    CONDITIONS,
    PROPERTY_CLASSES,
    choose_parameterized_action,
    property_router_probabilities,
    target_effect_receipt,
    validate_parameterized_source_ir,
    validate_property_router,
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
        changed_tasks = sum(int(row["changed_effects"] > 0) for row in rows)
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
            "changed_effect_count": sum(
                int(row["changed_effects"]) for row in rows
            ),
            "changed_task_count": changed_tasks,
            "required_option_invariance_rate": (
                sum(int(row["invariant_decisions"]) for row in rows) / steps
                if steps else 0.0
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
    parser.add_argument("--seed", type=int, default=97501)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite confirmation report: {args.output}")
    harness = _read(args.harness)
    harness_body = dict(harness)
    claimed_harness_hash = str(harness_body.pop("harness_sha256", ""))
    if stable_hash(harness_body) != claimed_harness_hash:
        raise SystemExit("V7 Harness hash mismatch")
    if harness.get("status") != "FRESH_CONFIRMATION_AUTHORIZED":
        raise SystemExit("V7 Harness did not authorize fresh confirmation")
    source_ir = harness["parameterized_source_ir"]
    validate_parameterized_source_ir(source_ir)
    target = _validate_bound(harness["target_grounder"])
    validate_target_artifact(target)
    router = harness["property_router"]
    validate_property_router(router)
    manifest = _read(args.manifest)
    manifest_body = dict(manifest)
    claimed_manifest_hash = str(manifest_body.pop("manifest_sha256", ""))
    if stable_hash(manifest_body) != claimed_manifest_hash:
        raise SystemExit("fresh confirmation manifest hash mismatch")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET":
        raise SystemExit("fresh confirmation manifest was not frozen before reset")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("fresh confirmation selection used target outcomes")
    if set(manifest.get("splits", {})) != {"fresh_confirmation"}:
        raise SystemExit("runner is restricted to fresh_confirmation")
    bound_manifest = harness["confirmation_manifest"]
    if (
        _sha256(args.manifest) != bound_manifest["file_sha256"]
        or manifest["manifest_sha256"] != bound_manifest["manifest_sha256"]
    ):
        raise SystemExit("runner manifest differs from frozen Harness manifest")
    task_ids = tuple(map(str, manifest["splits"]["fresh_confirmation"]))
    train_root = Path(str(manifest["train_root"])).resolve()
    thresholds = harness["thresholds"]
    minimum_property_confidence = float(
        thresholds["minimum_property_confidence"]
    )
    minimum_role_binding = float(thresholds["minimum_role_binding"])
    minimum_realization_score = float(
        thresholds["selected_minimum_realization_score"]
    )
    minimum_target_policy_ratio = float(
        thresholds["selected_minimum_target_policy_ratio"]
    )
    episodes: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(args.alfworld_config.resolve()),
            data_path=str(args.alfworld_data.resolve()),
            split="train",
            seed=args.seed,
            game_ids=task_ids,
            max_steps=args.max_steps,
        )
        seen: set[str] = set()
        try:
            for task_index in range(len(task_ids)):
                observation = environment.reset()
                task_id = (
                    Path(environment.resolved_game_file).resolve()
                    .relative_to(train_root).as_posix()
                )
                if task_id not in task_ids or task_id in seen:
                    raise RuntimeError(f"confirmation identity violation: {task_id}")
                seen.add(task_id)
                history: list[str] = []
                effect_receipts: list[str] = []
                records = []
                for step in range(args.max_steps):
                    goal = str(observation.state.get("task_goal", ""))
                    probabilities = property_router_probabilities(goal, router)
                    grounded = score_actions(
                        goal=goal,
                        observation=str(
                            observation.state.get("observation", "")
                        ),
                        native_actions=observation.native_actions,
                        step=step,
                        action_history=history,
                        artifact=target,
                    )
                    if not grounded:
                        raise RuntimeError("target grounder excluded every action")
                    decision = choose_parameterized_action(
                        condition=condition,
                        grounded=grounded,
                        history=history,
                        effect_receipts=effect_receipts,
                        source_ir=source_ir,
                        property_probabilities=probabilities,
                        minimum_property_confidence=minimum_property_confidence,
                        minimum_role_binding=minimum_role_binding,
                        minimum_realization_score=minimum_realization_score,
                        minimum_target_policy_ratio=minimum_target_policy_ratio,
                    )
                    if condition == "authentic_parameterized_ir":
                        counterfactual = choose_parameterized_action(
                            condition=condition,
                            grounded=_mutate_required_diagnostic(grounded),
                            history=history,
                            effect_receipts=effect_receipts,
                            source_ir=source_ir,
                            property_probabilities=probabilities,
                            minimum_property_confidence=(
                                minimum_property_confidence
                            ),
                            minimum_role_binding=minimum_role_binding,
                            minimum_realization_score=minimum_realization_score,
                            minimum_target_policy_ratio=(
                                minimum_target_policy_ratio
                            ),
                        )
                        decision["required_option_invariant"] = bool(
                            counterfactual["action"] == decision["action"]
                            and counterfactual["source_selected_effect"]
                            == decision["source_selected_effect"]
                        )
                    else:
                        decision["required_option_invariant"] = True
                    selected = str(decision["action"])
                    required_property = str(decision.get(
                        "required_property",
                        max(
                            PROPERTY_CLASSES,
                            key=lambda name: (
                                float(probabilities.get(name, 0.0)), name
                            ),
                        ),
                    ))
                    before = dict(observation.state)
                    after, reward = environment.step(selected)
                    effect_receipt = target_effect_receipt(
                        action=selected,
                        grounding=grounded[selected],
                        required_property=required_property,
                        minimum_role_binding=minimum_role_binding,
                    )
                    record_body = {
                        "task_id": task_id,
                        "condition": condition,
                        "step": step,
                        "goal": goal,
                        "before": before,
                        "native_actions": list(observation.native_actions),
                        "property_probabilities": probabilities,
                        "selected_grounding": grounded[selected],
                        "fallback_grounding": grounded[
                            decision["fallback_action"]
                        ],
                        "decision": decision,
                        "target_effect_receipt": effect_receipt,
                        "after": dict(after.state),
                        "reward": float(reward),
                        "official_success_after": bool(after.official_success),
                    }
                    records.append(record_body | {
                        "receipt_sha256": stable_hash(record_body)
                    })
                    history.append(selected)
                    effect_receipts.append(effect_receipt)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                success = bool(records and records[-1]["official_success_after"])
                episodes[condition].append({
                    "task_index": task_index,
                    "task_id": task_id,
                    "task_family": task_id.split("-", 1)[0],
                    "official_success": success,
                    "steps": len(records),
                    "return": sum(float(row["reward"]) for row in records),
                    "source_admissions": sum(
                        bool(row["decision"]["source_admitted"])
                        for row in records
                    ),
                    "changed_actions": sum(
                        bool(row["decision"]["changed_action"])
                        for row in records
                    ),
                    "changed_effects": sum(
                        bool(row["decision"]["changed_effect"])
                        for row in records
                    ),
                    "invariant_decisions": sum(
                        bool(row["decision"]["required_option_invariant"])
                        for row in records
                    ),
                    "diagnostics": dict(Counter(
                        str(row["decision"]["diagnostic"])
                        for row in records
                    )),
                    "effect_receipts": dict(Counter(
                        str(row["target_effect_receipt"]) for row in records
                    )),
                    "records": records,
                })
                print(json.dumps({
                    "condition": condition,
                    "task_index": task_index,
                    "task_id": task_id,
                    "success": success,
                    "steps": len(records),
                }), flush=True)
        finally:
            environment.close()
        if seen != set(task_ids):
            raise RuntimeError(f"condition {condition} missed frozen tasks")

    summaries = _summaries(episodes)
    authentic_name = "authentic_parameterized_ir"
    authentic = summaries[authentic_name]
    controls = (
        "target_only",
        "edge_permuted_ir",
        "property_permuted_router",
    )
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
    superiority = all(
        authentic["successes"] > summaries[condition]["successes"]
        for condition in controls
    )
    gates = {
        "target_capability": summaries["target_only"]["successes"] > 0,
        "authentic_nontrivial": (
            authentic["changed_effect_count"] >= 2
            and authentic["changed_task_count"] >= 2
        ),
        "required_option_invariance": (
            authentic["required_option_invariance_rate"] == 1.0
        ),
        "strict_success_superiority_to_all_controls": superiority,
        "paired_net_win_over_target_only": (
            paired["target_only"]["net_wins"] > 0
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "parameterized-real-source-alfworld-confirmation-v7",
        "status": (
            "FRESH_CONFIRMATION_POSITIVE"
            if passed else "FRESH_CONFIRMATION_NEGATIVE_STOP"
        ),
        "claim_boundary": (
            "FRESH_UNSEEN_TRAIN_INSTANCE_CONFIRMATION_ONLY; NOT VALID_UNSEEN OOD; "
            "EXISTING_VALID_UNSEEN_HELDOUT_UNREAD"
        ),
        "harness_path": str(args.harness.resolve()),
        "harness_file_sha256": _sha256(args.harness),
        "harness_sha256": harness["harness_sha256"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_file_sha256": _sha256(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_split": "fresh_confirmation",
        "existing_valid_unseen_heldout_read": False,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "conditions": list(CONDITIONS),
        "summaries": summaries,
        "paired_official_success": paired,
        "gates": gates,
        "confirmation_passed": passed,
        "next_step": (
            "REPORT_POSITIVE_IN_DISTRIBUTION_CONFIRMATION_AND_RESERVE_OOD_HELDOUT"
            if passed else "REPORT_NEGATIVE_AND_DO_NOT_READ_EXISTING_OOD_HELDOUT"
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
        "paired_official_success": paired,
        "gates": gates,
        "next_step": report["next_step"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
