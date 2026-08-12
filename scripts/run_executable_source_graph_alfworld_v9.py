#!/usr/bin/env python3
"""Run hash-bound V9 source-graph adaptation or fresh confirmation."""

from __future__ import annotations

import argparse
from collections import Counter
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
    initialize_slot_ledger,
    observe_target_transition,
    reconcile_visible_target_objects,
    slot_state,
    validate_slot_source_ir,
)
from motif_transfer.slot_aware_alfworld_harness_v9 import (
    CONDITIONS,
    choose_slot_aware_action,
    condition_required_property,
)

from run_slot_aware_alfworld_v8 import (
    _mutate_required_diagnostic,
    _paired,
    _read,
    _sha256,
    _summaries,
    _validate_dependency,
    _validate_file_receipt,
    _validate_hash,
    deepcopy_json,
)


def _source_transition_summary(
    episodes: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    result = {}
    for condition, rows in episodes.items():
        changed = Counter()
        admitted = Counter()
        graph_hashes: set[str] = set()
        changed_tasks = 0
        for episode in rows:
            task_changed = False
            for record in episode["records"]:
                decision = record["decision"]
                transition = decision.get("source_transition")
                if not isinstance(transition, dict):
                    continue
                kind = str(transition.get("kind", "NONE"))
                graph_hash = transition.get("graph_sha256")
                if graph_hash:
                    graph_hashes.add(str(graph_hash))
                if decision["source_admitted"]:
                    admitted[kind] += 1
                if decision["changed_effect"]:
                    changed[kind] += 1
                    task_changed = True
            changed_tasks += int(task_changed)
        result[condition] = {
            "admitted_by_source_transition_kind": dict(
                sorted(admitted.items())
            ),
            "changed_by_source_transition_kind": dict(
                sorted(changed.items())
            ),
            "source_edge_changed_count": int(changed["EDGE"]),
            "source_node_changed_count": int(changed["NODE"]),
            "source_transition_changed_task_count": changed_tasks,
            "compiled_graph_hashes": sorted(graph_hashes),
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
    parser.add_argument("--seed", type=int, default=99301)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V9 report: {args.output}")
    artifact = _read(args.artifact)
    experiment_version = str(artifact.get("experiment_version", "v9"))
    if experiment_version not in {"v9", "v10", "v11"}:
        raise SystemExit("unsupported executable-source-graph version")
    candidate_schemas = {
        "v9": "executable-source-graph-alfworld-candidate-v9",
        "v10": "budgeted-executable-source-graph-alfworld-candidate-v10",
        "v11": "budgeted-relation-edge-alfworld-candidate-v11",
    }
    harness_schemas = {
        "v9": "executable-source-graph-alfworld-harness-v9",
        "v10": "budgeted-executable-source-graph-alfworld-harness-v10",
        "v11": "budgeted-relation-edge-alfworld-harness-v11",
    }
    if args.phase == "adaptation_gate":
        _validate_hash(artifact, "candidate_sha256")
        if artifact.get("schema_version") != candidate_schemas[
            experiment_version
        ]:
            raise SystemExit("wrong executable-source-graph candidate schema")
        if artifact.get("status") != "ADAPTATION_GATE_ONLY":
            raise SystemExit("V9 adaptation requires a frozen candidate")
        artifact_hash_field = "candidate_sha256"
        expected_split = "adaptation_gate"
    else:
        _validate_hash(artifact, "harness_sha256")
        if artifact.get("schema_version") != harness_schemas[
            experiment_version
        ]:
            raise SystemExit("wrong executable-source-graph Harness schema")
        if artifact.get("status") != "FRESH_CONFIRMATION_AUTHORIZED":
            raise SystemExit("V9 confirmation requires authorization")
        gate = _validate_dependency(artifact["adaptation_gate_report"])
        if gate.get("status") != "ADAPTATION_GATE_PASSED":
            raise SystemExit("bound V9 adaptation gate did not pass")
        artifact_hash_field = "harness_sha256"
        expected_split = "fresh_confirmation"
    implementation = artifact.get("implementation")
    if not isinstance(implementation, dict):
        raise SystemExit("V9 artifact does not bind implementation")
    for receipt in implementation.values():
        _validate_file_receipt(receipt)
    parameters = artifact.get("experiment_parameters", {})
    if "max_steps" in parameters and args.max_steps != int(
        parameters["max_steps"]
    ):
        raise SystemExit("runner max_steps differs from frozen candidate")
    if "runner_seed" in parameters and args.seed != int(
        parameters["runner_seed"]
    ):
        raise SystemExit("runner seed differs from frozen candidate")
    manifest = _read(args.manifest)
    _validate_hash(manifest, "manifest_sha256")
    expected_manifest_schema = str(artifact.get(
        "manifest_schema",
        "executable-source-graph-alfworld-manifest-v9",
    ))
    expected_manifest_status = str(artifact.get(
        "manifest_status",
        "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET",
    ))
    if manifest.get("schema_version") != expected_manifest_schema:
        raise SystemExit("wrong executable-source-graph manifest schema")
    if manifest.get("status") != expected_manifest_status:
        raise SystemExit("V9 manifest was not frozen before reset")
    if manifest.get("selection_used_target_rollout_outcomes"):
        raise SystemExit("V9 manifest selection used target outcomes")
    bound_manifest = artifact["manifest"]
    if (
        _sha256(args.manifest) != bound_manifest["file_sha256"]
        or manifest["manifest_sha256"]
        != bound_manifest["manifest_sha256"]
    ):
        raise SystemExit("V9 runner manifest differs from candidate")
    task_ids = tuple(map(str, manifest["splits"][expected_split]))
    other_split = (
        "fresh_confirmation"
        if expected_split == "adaptation_gate"
        else "adaptation_gate"
    )
    if set(task_ids) & set(map(str, manifest["splits"][other_split])):
        raise SystemExit("V9 adaptation and confirmation overlap")
    train_root = Path(str(manifest["train_root"])).resolve()
    target = _validate_dependency(artifact["target_grounder"])
    validate_target_artifact(target)
    router = dict(artifact["property_router"])
    validate_property_router(router)
    source_ir = dict(artifact["slot_source_ir"])
    validate_slot_source_ir(source_ir)
    thresholds = artifact["thresholds"]
    transfer_scope = artifact["transfer_scope"]
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
        thresholds["minimum_property_confidence_diagnostic_only"]
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
                    raise RuntimeError(f"V9 identity violation: {task_id}")
                seen.add(task_id)
                goal = str(observation.state.get("task_goal", ""))
                probabilities = property_router_probabilities(goal, router)
                required_property, _, _ = condition_required_property(
                    goal, probabilities, condition
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
                        raise RuntimeError(
                            "target grounder excluded every action"
                        )
                    choose_kwargs = {
                        "condition": condition,
                        "grounded": grounded,
                        "history": history,
                        "ledger": ledger,
                        "source_ir": source_ir,
                        "property_probabilities": probabilities,
                        "minimum_property_confidence": (
                            minimum_property_confidence
                        ),
                        "minimum_role_binding": minimum_role_binding,
                        "minimum_realization_score": (
                            minimum_realization_score
                        ),
                        "minimum_target_policy_ratio": (
                            minimum_target_policy_ratio
                        ),
                        "allowed_source_effects": allowed_source_effects,
                        "active_required_properties": (
                            active_required_properties
                        ),
                    }
                    decision = choose_slot_aware_action(**choose_kwargs)
                    if condition == "authentic_slot_ir":
                        counterfactual = choose_slot_aware_action(
                            **(choose_kwargs | {
                                "grounded": _mutate_required_diagnostic(
                                    grounded
                                )
                            })
                        )
                        decision["required_option_invariant"] = bool(
                            counterfactual["action"] == decision["action"]
                            and counterfactual["source_transition"]
                            == decision["source_transition"]
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
                        "official_success_after": bool(
                            after.official_success
                        ),
                    }
                    records.append(record_body | {
                        "receipt_sha256": stable_hash(record_body)
                    })
                    history.append(selected)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                success = bool(
                    records and records[-1]["official_success_after"]
                )
                changed_by_effect = Counter(
                    str(row["decision"].get(
                        "requested_source_effect"
                    ))
                    for row in records
                    if row["decision"]["changed_effect"]
                )
                admitted_by_effect = Counter(
                    str(row["decision"].get(
                        "requested_source_effect"
                    ))
                    for row in records
                    if row["decision"]["source_admitted"]
                )
                episodes[condition].append({
                    "task_index": task_index,
                    "task_id": task_id,
                    "task_family": task_id.split("-", 1)[0],
                    "official_success": success,
                    "steps": len(records),
                    "return": sum(
                        float(row["reward"]) for row in records
                    ),
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
                        bool(row["decision"][
                            "required_option_invariant"
                        ])
                        for row in records
                    ),
                    "effect_receipts": dict(Counter(
                        str(row["target_effect_receipt"])
                        for row in records
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
            raise RuntimeError(f"condition {condition} missed tasks")
    summaries = _summaries(episodes)
    paired = _paired(episodes, task_ids)
    transitions = _source_transition_summary(episodes)
    authentic = summaries["authentic_slot_ir"]
    target_only = summaries["target_only"]
    edge_control = summaries["edge_permuted_ir"]
    authentic_transitions = transitions["authentic_slot_ir"]
    if args.phase == "adaptation_gate":
        requirements = artifact["adaptation_gates"]
        lower, upper = map(
            float, requirements["source_admission_rate_range"]
        )
        gates = {
            "authentic_success_noninferior_to_target_only": (
                authentic["successes"] >= target_only["successes"]
            ),
            "authentic_success_superior_to_edge_control": (
                authentic["successes"] > edge_control["successes"]
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
            "changed_source_edges": (
                authentic_transitions["source_edge_changed_count"]
                >= int(requirements["changed_source_edges"])
            ),
            "authentic_changes_exceed_edge_control": (
                authentic["changed_effect_count"]
                > edge_control["changed_effect_count"]
            ),
            "source_admission_rate_range": (
                lower <= authentic["source_admission_rate"] <= upper
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
        }
        passed = all(gates.values())
        status = (
            "ADAPTATION_GATE_PASSED"
            if passed else "ADAPTATION_GATE_FAILED_STOP"
        )
    else:
        gates = {
            "authentic_nontrivial": (
                authentic["changed_effect_count"] >= 4
                and authentic["changed_task_count"] >= 4
            ),
            "bind_and_source_edge_intervened": (
                authentic["changed_by_effect"]["BIND"] >= 1
                and authentic_transitions[
                    "source_edge_changed_count"
                ] >= 2
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
        "schema_version": (
            f"executable-source-graph-{args.phase}-{experiment_version}"
        ),
        "status": status,
        "claim_boundary": (
            "FRESH_RELATION_ONLY_TRAIN_INSTANCES; SOURCE_GRAPH_EXECUTED; "
            "EXISTING_VALID_UNSEEN_HELDOUT_UNREAD; OFFICIAL_OUTCOME_"
            "NEVER_USED_FOR_ACTION_SELECTION"
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
        "condition_semantics": artifact.get("condition_semantics"),
        "experiment_parameters": parameters,
        "transfer_scope": transfer_scope,
        "episodes": episodes,
        "summaries": summaries,
        "source_transition_summaries": transitions,
        "paired_official_success": paired,
        "gates": gates,
        "passed": passed,
        "next_step": (
            "FINALIZE_HASH_BOUND_V9_AND_RUN_CONFIRMATION_ONCE"
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
        "source_transition_summaries": transitions,
        "paired_official_success": paired,
        "gates": gates,
        "existing_valid_unseen_heldout_read": False,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
