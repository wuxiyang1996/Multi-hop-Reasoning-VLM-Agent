#!/usr/bin/env python3
"""Run V8 closed-loop adaptation or hash-authorized fresh confirmation."""

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
    property_router_probabilities,
    validate_property_router,
)
from motif_transfer.slot_aware_alfworld_harness import (
    CONDITIONS,
    choose_slot_aware_action,
    condition_required_property,
    initialize_slot_ledger,
    observe_target_transition,
    reconcile_visible_target_objects,
    slot_state,
    validate_slot_source_ir,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_hash(value: dict[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise SystemExit(f"invalid frozen artifact hash: {field}")


def _validate_dependency(receipt: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(receipt["path"]))
    if _sha256(path) != receipt["file_sha256"]:
        raise SystemExit(f"frozen dependency changed: {path}")
    return _read(path)


def _validate_file_receipt(receipt: Mapping[str, Any]) -> None:
    path = Path(str(receipt["path"]))
    if _sha256(path) != receipt["file_sha256"]:
        raise SystemExit(f"frozen implementation changed: {path}")


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


def _summaries(
    episodes: Mapping[str, list[Mapping[str, Any]]]
) -> dict[str, Any]:
    result = {}
    for condition, rows in episodes.items():
        steps = sum(int(row["steps"]) for row in rows)
        changed_by_effect: Counter[str] = Counter()
        admitted_by_effect: Counter[str] = Counter()
        receipts: Counter[str] = Counter()
        for episode in rows:
            changed_by_effect.update(episode["changed_by_effect"])
            admitted_by_effect.update(episode["admitted_by_effect"])
            receipts.update(episode["effect_receipts"])
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
            "changed_effect_count": sum(
                int(row["changed_effects"]) for row in rows
            ),
            "changed_action_count": sum(
                int(row["changed_actions"]) for row in rows
            ),
            "changed_task_count": sum(
                int(row["changed_effects"] > 0) for row in rows
            ),
            "slot_safety_shield_count": sum(
                int(row["slot_safety_shields"]) for row in rows
            ),
            "slot_safety_shield_task_count": sum(
                int(row["slot_safety_shields"] > 0) for row in rows
            ),
            "changed_by_effect": {
                effect: int(changed_by_effect[effect])
                for effect in ("BIND", "MUTATE", "RELATE")
            },
            "admitted_by_effect": {
                effect: int(admitted_by_effect[effect])
                for effect in ("BIND", "MUTATE", "RELATE")
            },
            "effect_receipts": dict(sorted(receipts.items())),
            "reopened_completed_slots": sum(
                int(row["reopened_completed_slots"]) for row in rows
            ),
            "selected_postcondition_failures": sum(
                int(row["selected_postcondition_failures"]) for row in rows
            ),
            "required_option_invariance_rate": (
                sum(int(row["invariant_decisions"]) for row in rows) / steps
                if steps else 0.0
            ),
        }
    return result


def _paired(
    episodes: Mapping[str, list[Mapping[str, Any]]],
    task_ids: tuple[str, ...],
) -> dict[str, Any]:
    authentic = {
        str(row["task_id"]): bool(row["official_success"])
        for row in episodes["authentic_slot_ir"]
    }
    result = {}
    for condition in CONDITIONS:
        if condition == "authentic_slot_ir":
            continue
        other = {
            str(row["task_id"]): bool(row["official_success"])
            for row in episodes[condition]
        }
        deltas = [
            int(authentic[task_id]) - int(other[task_id])
            for task_id in task_ids
        ]
        result[condition] = {
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "net_wins": sum(deltas),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("adaptation_gate", "fresh_confirmation"),
        required=True,
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=98301)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V8 report: {args.output}")
    artifact = _read(args.artifact)
    if args.phase == "adaptation_gate":
        _validate_hash(artifact, "candidate_sha256")
        if artifact.get("status") != "ADAPTATION_GATE_ONLY":
            raise SystemExit("adaptation gate requires the frozen V8 candidate")
        artifact_hash_field = "candidate_sha256"
        expected_split = "adaptation_gate"
    else:
        _validate_hash(artifact, "harness_sha256")
        if artifact.get("status") != "FRESH_CONFIRMATION_AUTHORIZED":
            raise SystemExit("fresh confirmation requires an authorized Harness")
        gate = _validate_dependency(artifact["adaptation_gate_report"])
        if gate.get("status") != "ADAPTATION_GATE_PASSED":
            raise SystemExit("bound V8 adaptation gate did not pass")
        artifact_hash_field = "harness_sha256"
        expected_split = "fresh_confirmation"
    implementation = artifact.get("implementation")
    if not isinstance(implementation, dict):
        raise SystemExit("V8 artifact does not bind its implementation")
    _validate_file_receipt(implementation["slot_harness"])
    _validate_file_receipt(implementation["runner"])
    manifest = _read(args.manifest)
    _validate_hash(manifest, "manifest_sha256")
    if manifest.get("status") not in {
        "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET",
        "FROZEN_BEFORE_ANY_REVISED_ADAPTATION_RESET",
    }:
        raise SystemExit("V8 manifest was not frozen before reset")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("V8 manifest selection used target outcomes")
    bound_manifest = artifact["manifest"]
    if (
        _sha256(args.manifest) != bound_manifest["file_sha256"]
        or manifest["manifest_sha256"] != bound_manifest["manifest_sha256"]
    ):
        raise SystemExit("runner manifest differs from frozen V8 artifact")
    task_ids = tuple(map(str, manifest["splits"][expected_split]))
    other_split = (
        "fresh_confirmation"
        if expected_split == "adaptation_gate"
        else "adaptation_gate"
    )
    if set(task_ids) & set(map(str, manifest["splits"][other_split])):
        raise SystemExit("V8 adaptation and confirmation task overlap")
    train_root = Path(str(manifest["train_root"])).resolve()
    target = _validate_dependency(artifact["target_grounder"])
    validate_target_artifact(target)
    router = dict(artifact["property_router"])
    validate_property_router(router)
    source_ir = dict(artifact["slot_source_ir"])
    validate_slot_source_ir(source_ir)
    thresholds = artifact["thresholds"]
    transfer_scope = artifact.get("transfer_scope", {
        "allowed_source_effects": ["BIND", "MUTATE", "RELATE"],
        "active_required_properties": list(
            ("NONE", "CLEAN", "HEAT", "COOL", "LIGHT")
        ),
        "claimed_changed_effects": ["BIND", "MUTATE", "RELATE"],
    })
    allowed_source_effects = tuple(map(
        str, transfer_scope["allowed_source_effects"]
    ))
    active_required_properties = tuple(map(
        str, transfer_scope["active_required_properties"]
    ))
    claimed_changed_effects = tuple(map(
        str, transfer_scope["claimed_changed_effects"]
    ))
    minimum_property_confidence = float(
        thresholds["minimum_property_confidence"]
    )
    minimum_role_binding = float(thresholds["minimum_role_binding"])
    minimum_realization_score = float(
        thresholds["minimum_realization_score"]
    )
    minimum_target_policy_ratio = float(
        thresholds["minimum_target_policy_ratio"]
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
                    raise RuntimeError(f"V8 identity violation: {task_id}")
                seen.add(task_id)
                goal = str(observation.state.get("task_goal", ""))
                probabilities = property_router_probabilities(goal, router)
                required_property, _, _ = condition_required_property(
                    probabilities, condition
                )
                ledger = initialize_slot_ledger(
                    goal,
                    required_property=required_property,
                    initial_observation=str(
                        observation.state.get("observation", "")
                    ),
                )
                history: list[str] = []
                records = []
                for step in range(args.max_steps):
                    ledger = reconcile_visible_target_objects(
                        ledger,
                        str(observation.state.get("observation", "")),
                    )
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
                    decision = choose_slot_aware_action(
                        condition=condition,
                        grounded=grounded,
                        history=history,
                        ledger=ledger,
                        source_ir=source_ir,
                        property_probabilities=probabilities,
                        minimum_property_confidence=minimum_property_confidence,
                        minimum_role_binding=minimum_role_binding,
                        minimum_realization_score=minimum_realization_score,
                        minimum_target_policy_ratio=minimum_target_policy_ratio,
                        allowed_source_effects=allowed_source_effects,
                        active_required_properties=active_required_properties,
                    )
                    if condition == "authentic_slot_ir":
                        counterfactual = choose_slot_aware_action(
                            condition=condition,
                            grounded=_mutate_required_diagnostic(grounded),
                            history=history,
                            ledger=ledger,
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
                            allowed_source_effects=allowed_source_effects,
                            active_required_properties=(
                                active_required_properties
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
                    before_state = dict(observation.state)
                    ledger_before = deepcopy_json(ledger)
                    after, reward = environment.step(selected)
                    ledger, receipt = observe_target_transition(
                        ledger,
                        action=selected,
                        after_observation=str(
                            after.state.get("observation", "")
                        ),
                    )
                    expected_receipts = {
                        "BIND": {"BIND_INSTANCE"},
                        "MUTATE": {
                            "MUTATE_REQUIRED_PROPERTY",
                            "LIGHT_SLOT_CLOSED",
                        },
                        "RELATE": {"RELATE_SLOT_CLOSED"},
                    }
                    requested = decision.get("requested_source_effect")
                    selected_postcondition_failure = bool(
                        decision["source_admitted"]
                        and requested in expected_receipts
                        and receipt not in expected_receipts[requested]
                    )
                    record_body = {
                        "task_id": task_id,
                        "condition": condition,
                        "step": step,
                        "goal": goal,
                        "before": before_state,
                        "native_actions": list(observation.native_actions),
                        "property_probabilities": probabilities,
                        "decision": decision,
                        "selected_grounding": grounded[selected],
                        "fallback_grounding": grounded[
                            decision["fallback_action"]
                        ],
                        "ledger_before": ledger_before,
                        "target_effect_receipt": receipt,
                        "ledger_after": deepcopy_json(ledger),
                        "selected_postcondition_failure": (
                            selected_postcondition_failure
                        ),
                        "after": dict(after.state),
                        "reward": float(reward),
                        "official_success_after": bool(after.official_success),
                    }
                    records.append(record_body | {
                        "receipt_sha256": stable_hash(record_body)
                    })
                    history.append(selected)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                success = bool(records and records[-1]["official_success_after"])
                changed_by_effect = Counter(
                    str(row["decision"].get("requested_source_effect"))
                    for row in records
                    if row["decision"]["changed_effect"]
                )
                admitted_by_effect = Counter(
                    str(row["decision"].get("requested_source_effect"))
                    for row in records
                    if row["decision"]["source_admitted"]
                )
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
                    "slot_safety_shields": sum(
                        bool(row["decision"]["slot_safety_shielded"])
                        for row in records
                    ),
                    "changed_by_effect": dict(changed_by_effect),
                    "admitted_by_effect": dict(admitted_by_effect),
                    "reopened_completed_slots": int(
                        ledger["reopened_completed_slots"]
                    ),
                    "selected_postcondition_failures": sum(
                        bool(row["selected_postcondition_failure"])
                        for row in records
                    ),
                    "invariant_decisions": sum(
                        bool(row["decision"]["required_option_invariant"])
                        for row in records
                    ),
                    "effect_receipts": dict(Counter(
                        str(row["target_effect_receipt"]) for row in records
                    )),
                    "final_slot_state": slot_state(ledger),
                    "records": records,
                })
                print(json.dumps({
                    "phase": args.phase,
                    "condition": condition,
                    "task_index": task_index,
                    "task_id": task_id,
                    "success": success,
                    "steps": len(records),
                    "changes": sum(
                        bool(row["decision"]["changed_effect"])
                        for row in records
                    ),
                }), flush=True)
        finally:
            environment.close()
        if seen != set(task_ids):
            raise RuntimeError(f"condition {condition} missed frozen tasks")
    summaries = _summaries(episodes)
    paired = _paired(episodes, task_ids)
    authentic = summaries["authentic_slot_ir"]
    target_only = summaries["target_only"]
    if args.phase == "adaptation_gate":
        requirements = artifact["adaptation_gates"]
        lower, upper = map(
            float, requirements["source_admission_rate_range"]
        )
        gates = {
            "authentic_success_noninferior_to_target_only": (
                authentic["successes"] >= target_only["successes"]
            ),
            "paired_net_win_nonnegative": (
                paired["target_only"]["net_wins"] >= 0
            ),
            "changed_effects_each_claimed_effect": all(
                authentic["changed_by_effect"][effect]
                >= int(requirements[
                    "changed_effects_each_claimed_effect"
                ])
                for effect in claimed_changed_effects
            ),
            "changed_tasks": (
                authentic["changed_task_count"]
                >= int(requirements["changed_tasks"])
            ),
            "source_admission_rate_range": (
                lower <= authentic["source_admission_rate"] <= upper
            ),
            "reopened_completed_slots": (
                authentic["reopened_completed_slots"]
                == int(requirements["reopened_completed_slots"])
            ),
            "failed_selected_postconditions": (
                authentic["selected_postcondition_failures"]
                == int(requirements["failed_selected_postconditions"])
            ),
            "required_option_invariance": (
                authentic["required_option_invariance_rate"] == 1.0
            ),
        }
        passed = all(gates.values())
        status = (
            "ADAPTATION_GATE_PASSED"
            if passed else "ADAPTATION_GATE_FAILED_STOP"
        )
    else:
        gates = {
            "authentic_nontrivial": (
                authentic["changed_effect_count"] >= 3
                and authentic["changed_task_count"] >= 3
            ),
            "all_claimed_effects_intervened": all(
                authentic["changed_by_effect"][effect] >= 1
                for effect in claimed_changed_effects
            ),
            "no_reopened_completed_slots": (
                authentic["reopened_completed_slots"] == 0
            ),
            "no_failed_selected_postconditions": (
                authentic["selected_postcondition_failures"] == 0
            ),
            "required_option_invariance": (
                authentic["required_option_invariance_rate"] == 1.0
            ),
            "paired_net_win_over_target_only": (
                paired["target_only"]["net_wins"] > 0
            ),
            "strict_success_superiority_to_all_controls": all(
                authentic["successes"] > summaries[condition]["successes"]
                for condition in (
                    "target_only",
                    "edge_permuted_ir",
                    "property_permuted_router",
                )
            ),
        }
        passed = all(gates.values())
        status = (
            "FRESH_CONFIRMATION_POSITIVE"
            if passed else "FRESH_CONFIRMATION_NEGATIVE_STOP"
        )
    body = {
        "schema_version": f"slot-aware-alfworld-{args.phase}-v8",
        "status": status,
        "claim_boundary": (
            "FRESH_UNSEEN_TRAIN_INSTANCES_ONLY; EXISTING_VALID_UNSEEN_"
            "HELDOUT_UNREAD; OFFICIAL_OUTCOME_NEVER_USED_FOR_ACTION_SELECTION"
        ),
        "phase": args.phase,
        "artifact_path": str(args.artifact.resolve()),
        "artifact_file_sha256": _sha256(args.artifact),
        "artifact_hash_field": artifact_hash_field,
        "artifact_sha256": artifact[artifact_hash_field],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_file_sha256": _sha256(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_split": expected_split,
        "existing_valid_unseen_heldout_read": False,
        "official_outcome_used_for_action_selection": False,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "conditions": list(CONDITIONS),
        "transfer_scope": transfer_scope,
        "episodes": episodes,
        "summaries": summaries,
        "paired_official_success": paired,
        "gates": gates,
        "passed": passed,
        "next_step": (
            "FINALIZE_HASH_BOUND_HARNESS_AND_RUN_FRESH_CONFIRMATION"
            if args.phase == "adaptation_gate" and passed
            else "REPORT_AND_STOP_WITHOUT_READING_RESERVED_HELDOUT"
        ),
    }
    result = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": status,
        "report_sha256": result["report_sha256"],
        "summaries": summaries,
        "paired_official_success": paired,
        "gates": gates,
        "existing_valid_unseen_heldout_read": False,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


def deepcopy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the JSON-only ledger while keeping report receipts independent."""
    return json.loads(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
