#!/usr/bin/env python3
"""Outcome-blind first-opportunity enumeration for a frozen V20 eval split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_masked_effect_grounder import (  # noqa: E402
    score_actions,
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.parameterized_alfworld_harness import (  # noqa: E402
    property_router_probabilities,
    validate_property_router,
)
from motif_transfer.real_source_relation_causal_v20 import (  # noqa: E402
    score_relation_decision,
)
from motif_transfer.slot_aware_alfworld_harness import (  # noqa: E402
    initialize_slot_ledger,
    observe_target_transition,
    reconcile_visible_target_objects,
    validate_slot_source_ir,
)
from motif_transfer.slot_aware_alfworld_harness_v10 import (  # noqa: E402
    choose_slot_aware_action,
    condition_required_property,
)
from run_slot_aware_alfworld_v8 import (  # noqa: E402
    _read,
    _sha256,
    _validate_dependency,
    _validate_hash,
    deepcopy_json,
)


ALLOWED_ROLES = (
    "utility_requalification",
    "development_gate",
    "sealed_confirmation",
)


def _relative_game_matches(actual: str, expected: str) -> bool:
    normalized = str(actual).replace("\\", "/")
    target = str(expected).replace("\\", "/").lstrip("/")
    return normalized == target or normalized.endswith("/" + target)


def _choose(
    *,
    condition: str,
    grounded: Mapping[str, Mapping[str, Any]],
    history: list[str],
    ledger: Mapping[str, Any],
    source_ir: Mapping[str, Any],
    probabilities: Mapping[str, float],
    thresholds: Mapping[str, Any],
    allowed_source_effects: tuple[str, ...],
    active_required_properties: tuple[str, ...],
) -> dict[str, Any]:
    return choose_slot_aware_action(
        condition=condition,
        grounded=grounded,
        history=history,
        ledger=ledger,
        source_ir=source_ir,
        property_probabilities=probabilities,
        minimum_property_confidence=float(
            thresholds["minimum_property_confidence_diagnostic_only"]
        ),
        minimum_role_binding=float(thresholds["minimum_role_binding"]),
        minimum_realization_score=float(
            thresholds["minimum_realization_score"]
        ),
        minimum_target_policy_ratio=float(
            thresholds["minimum_target_policy_ratio"]
        ),
        allowed_source_effects=allowed_source_effects,
        active_required_properties=active_required_properties,
    )


def _edge(decision: Mapping[str, Any]) -> Mapping[str, Any] | None:
    transition = decision.get("source_transition")
    if isinstance(transition, dict) and transition.get("kind") == "EDGE":
        return transition
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--role", choices=ALLOWED_ROLES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V20 eval enumeration: {args.output}")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    candidate = _read(args.candidate)
    candidate_hash = _validate_hash(candidate, "candidate_sha256")
    expected_status = {
        "utility_requalification": (
            "PROSPECTIVE_UTILITY_REQUALIFICATION_AUTHORIZED"
        ),
        "development_gate": "TARGET_CAUSAL_AND_UTILITY_GATE_PASSED",
        "sealed_confirmation": (
            "DEVELOPMENT_TRANSFER_GATE_PASSED_CONFIRMATION_AUTHORIZED"
        ),
    }[args.role]
    if candidate.get("status") != expected_status:
        raise SystemExit(f"V20 candidate has no authority for {args.role}")
    authority_field = {
        "utility_requalification": "utility_requalification_authorized",
        "development_gate": "development_authorized",
        "sealed_confirmation": "confirmation_authorized",
    }[args.role]
    if not candidate.get(authority_field):
        raise SystemExit(f"V20 candidate did not authorize {args.role}")
    parent_receipt = manifest["parent_candidate"]
    parent = _read(Path(str(parent_receipt["path"])))
    _validate_hash(parent, "candidate_sha256")
    if _sha256(Path(str(parent_receipt["path"]))) != parent_receipt["file_sha256"]:
        raise SystemExit("V20 parent candidate file changed")
    target = _validate_dependency(parent["target_grounder"])
    validate_target_artifact(target)
    router = dict(parent["property_router"])
    validate_property_router(router)
    source_ir = dict(parent["slot_source_ir"])
    validate_slot_source_ir(source_ir)
    thresholds = dict(parent["thresholds"])
    allowed_source_effects = tuple(map(str, manifest["allowed_source_effects"]))
    active_required_properties = tuple(map(
        str, manifest["active_required_properties"]
    ))
    task_ids = tuple(map(str, manifest["splits"][args.role]))
    max_steps = int(manifest["max_steps"])
    seed = int(manifest["seed"]) + ALLOWED_ROLES.index(args.role) + 2
    environment = ALFWorldTextBatchEnvironment(
        config_path=str(args.alfworld_config.resolve()),
        data_path=str(args.alfworld_data.resolve()),
        split="train",
        seed=seed,
        game_ids=task_ids,
        max_steps=max_steps,
    )
    seen: set[str] = set()
    tasks = []
    try:
        for task_index in range(len(task_ids)):
            observation = environment.reset()
            matches = [
                task_id for task_id in task_ids
                if _relative_game_matches(environment.resolved_game_file, task_id)
            ]
            if len(matches) != 1:
                raise RuntimeError("V20 eval reset identity mismatch")
            task_id = matches[0]
            if task_id in seen:
                raise RuntimeError("V20 eval repeated a frozen identity")
            seen.add(task_id)
            goal = str(observation.state.get("task_goal", ""))
            probabilities = property_router_probabilities(goal, router)
            required_property, _, _ = condition_required_property(
                goal, probabilities, "authentic_slot_ir"
            )
            ledger = initialize_slot_ledger(
                goal,
                required_property=required_property,
                initial_observation=str(observation.state.get("observation", "")),
            )
            history: list[str] = []
            opportunity = None
            actionable_edges = 0
            mismatched_fallbacks = 0
            for step in range(max_steps):
                ledger = reconcile_visible_target_objects(
                    ledger, str(observation.state.get("observation", ""))
                )
                grounded = score_actions(
                    goal=goal,
                    observation=str(observation.state.get("observation", "")),
                    native_actions=observation.native_actions,
                    step=step,
                    action_history=history,
                    artifact=target,
                )
                if not grounded:
                    raise RuntimeError("V20 eval target grounder excluded every action")
                baseline = _choose(
                    condition="property_permuted_router",
                    grounded=grounded,
                    history=history,
                    ledger=ledger,
                    source_ir=source_ir,
                    probabilities=probabilities,
                    thresholds=thresholds,
                    allowed_source_effects=allowed_source_effects,
                    active_required_properties=active_required_properties,
                )
                authentic = _choose(
                    condition="authentic_slot_ir",
                    grounded=grounded,
                    history=history,
                    ledger=ledger,
                    source_ir=source_ir,
                    probabilities=probabilities,
                    thresholds=thresholds,
                    allowed_source_effects=allowed_source_effects,
                    active_required_properties=active_required_properties,
                )
                transition = _edge(authentic)
                actionable = bool(
                    transition is not None
                    and authentic.get("source_admitted")
                    and "best_realization_score" in authentic
                    and "target_policy_ratio" in authentic
                    and str(authentic["action"]) != str(authentic["fallback_action"])
                )
                actionable_edges += int(actionable)
                baseline_matches_fallback = (
                    str(baseline["action"]) == str(authentic["fallback_action"])
                )
                mismatched_fallbacks += int(actionable and not baseline_matches_fallback)
                if actionable and baseline_matches_fallback:
                    score = score_relation_decision(
                        candidate=candidate,
                        decision=authentic,
                        grounded=grounded,
                        ledger=ledger,
                        history=history,
                        step=step,
                        max_steps=max_steps,
                        native_action_count=len(observation.native_actions),
                    )
                    state_body = {
                        "task_id": task_id,
                        "step": step,
                        "goal": goal,
                        "before": dict(observation.state),
                        "native_actions": list(map(str, observation.native_actions)),
                        "ledger_before": deepcopy_json(ledger),
                        "history": list(history),
                        "property_probabilities": probabilities,
                    }
                    row_body = {
                        "role": args.role,
                        "task_id": task_id,
                        "task_family": task_id.split("-", 1)[0],
                        "fork_step": step,
                        "prefix_actions": list(history),
                        "expected_fork_state_sha256": stable_hash(state_body),
                        "expected_fallback_action": str(authentic["fallback_action"]),
                        "expected_source_action": str(authentic["action"]),
                        "expected_source_graph_sha256": str(
                            transition["graph_sha256"]
                        ),
                        "expected_source_edge": {
                            key: transition[key]
                            for key in ("from", "to", "guard", "kind")
                        },
                        "candidate_score": score,
                        "policy_admissions": {
                            "v20_selective": bool(score["admitted"]),
                            "always_source_edge": True,
                            "causal_effect_only": bool(
                                score["source_causal_effect_probability"] >= 0.5
                                and score["causal_effect_margin"] > 0.0
                            ),
                            "lexical_move_relation": bool(
                                score["source_action_features"]["verb_move"]
                                and score["source_action_features"][
                                    "goal_object_token_match"
                                ]
                                and score["source_action_features"][
                                    "goal_receptacle_token_match"
                                ]
                            ),
                            "late_step_heuristic": step >= 9,
                            "target_only_graph_erased": False,
                        },
                        "selection_authority": (
                            "FIRST_AUTHENTIC_SOURCE_EDGE_ACTION_CONTRAST_ON_"
                            "SOURCE_DISABLED_TARGET_TRAJECTORY"
                        ),
                    }
                    opportunity = row_body | {"fork_id": stable_hash(row_body)}
                    break
                selected = str(baseline["action"])
                observation, _discarded_reward = environment.step(selected)
                ledger, _receipt = observe_target_transition(
                    ledger,
                    action=selected,
                    after_observation=str(observation.state.get("observation", "")),
                )
                history.append(selected)
                if observation.terminal:
                    break
            task_body = {
                "task_index": task_index,
                "task_id": task_id,
                "task_family": task_id.split("-", 1)[0],
                "prefix_steps_executed": len(history),
                "actionable_edge_states_before_stop": actionable_edges,
                "actionable_edges_with_baseline_fallback_mismatch": (
                    mismatched_fallbacks
                ),
                "first_action_contrast": opportunity,
            }
            tasks.append(task_body | {"task_receipt_sha256": stable_hash(task_body)})
            print(json.dumps({
                "role": args.role,
                "task_index": task_index,
                "task_count": len(task_ids),
                "task_id": task_id,
                "found_contrast": opportunity is not None,
                "v20_admitted": bool(
                    opportunity
                    and opportunity["policy_admissions"]["v20_selective"]
                ),
                "outcomes_recorded": False,
            }), flush=True)
    finally:
        environment.close()
    if seen != set(task_ids):
        raise RuntimeError("V20 eval did not enumerate every frozen task")
    opportunities = [
        row["first_action_contrast"] for row in tasks
        if row["first_action_contrast"] is not None
    ]
    policy_counts = {
        policy: sum(bool(row["policy_admissions"][policy]) for row in opportunities)
        for policy in opportunities[0]["policy_admissions"]
    } if opportunities else {}
    body = {
        "schema_version": "real-source-relation-eval-enumeration-v20",
        "status": "OUTCOME_BLIND_EVAL_OPPORTUNITIES_COMPLETE",
        "claim_boundary": (
            "FROZEN_FIRST_ACTION_CONTRAST_AND_PREACTION_POLICY_ADMISSIONS_"
            "ONLY; REWARD_DISCARDED; OFFICIAL_SUCCESS_NOT_READ_OR_SERIALIZED"
        ),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "file_sha256": _sha256(args.manifest),
            "manifest_sha256": manifest_hash,
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "file_sha256": _sha256(args.candidate),
            "candidate_sha256": candidate_hash,
        },
        "role": args.role,
        "seed": seed,
        "max_steps": max_steps,
        "task_count": len(tasks),
        "tasks": tasks,
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "opportunities_by_family": dict(Counter(
            str(row["task_family"]) for row in opportunities
        )),
        "policy_admission_counts": policy_counts,
        "outcomes_recorded": False,
        "rewards_recorded": False,
        "existing_valid_unseen_read_or_run": False,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_sha256": report["report_sha256"],
        "role": args.role,
        "task_count": len(tasks),
        "opportunity_count": len(opportunities),
        "policy_admission_counts": policy_counts,
        "outcomes_recorded": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
