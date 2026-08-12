#!/usr/bin/env python3
"""Run frozen V20 matched forks and end-to-end policy accounting."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
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
from motif_transfer.relation_edge_value_v13 import fork_utility  # noqa: E402
from motif_transfer.slot_aware_alfworld_harness import (  # noqa: E402
    initialize_slot_ledger,
    observe_target_transition,
    reconcile_visible_target_objects,
    slot_state,
    validate_slot_source_ir,
)
from motif_transfer.slot_aware_alfworld_harness_v10 import (  # noqa: E402
    condition_required_property,
)
from enumerate_real_source_relation_eval_v20 import (  # noqa: E402
    _choose,
    _relative_game_matches,
)
from run_relation_edge_intervention_forks_v13 import (  # noqa: E402
    TREATMENTS,
    _run_branch,
)
from run_slot_aware_alfworld_v8 import (  # noqa: E402
    _read,
    _sha256,
    _validate_dependency,
    _validate_file_receipt,
    _validate_hash,
)


def _run_target_only(
    *,
    environment: ALFWorldTextBatchEnvironment,
    observation: Any,
    task_id: str,
    target: Mapping[str, Any],
    router: Mapping[str, Any],
    source_ir: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    allowed_source_effects: tuple[str, ...],
    active_required_properties: tuple[str, ...],
    max_steps: int,
) -> dict[str, Any]:
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
    total_return = 0.0
    while (
        len(history) < max_steps
        and not observation.terminal
        and not observation.official_success
    ):
        ledger = reconcile_visible_target_objects(
            ledger, str(observation.state.get("observation", ""))
        )
        grounded = score_actions(
            goal=goal,
            observation=str(observation.state.get("observation", "")),
            native_actions=observation.native_actions,
            step=len(history),
            action_history=history,
            artifact=target,
        )
        if not grounded:
            raise RuntimeError("V20 eval target-only grounder excluded every action")
        decision = _choose(
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
        action = str(decision["action"])
        observation, reward = environment.step(action)
        total_return += float(reward)
        ledger, _receipt = observe_target_transition(
            ledger,
            action=action,
            after_observation=str(observation.state.get("observation", "")),
        )
        history.append(action)
    final = slot_state(ledger)
    required = max(int(final["required_count"]), 1)
    body = {
        "task_id": task_id,
        "official_success": bool(observation.official_success),
        "steps": len(history),
        "return": total_return,
        "completed_fraction": float(final["completed_count"]) / required,
        "final_slot_state": final,
    }
    return body | {"baseline_sha256": stable_hash(body)}


def _exact_sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return float(sum(
        math.comb(discordant, count) for count in range(wins, discordant + 1)
    ) / (2 ** discordant))


def _policy_metrics(
    *,
    policy: str,
    task_count: int,
    opportunities: list[Mapping[str, Any]],
    forks: list[Mapping[str, Any]],
    no_contrast: list[Mapping[str, Any]],
    max_steps: int,
) -> dict[str, Any]:
    by_fork = {str(row["fork_id"]): row for row in forks}
    wins = losses = source_successes = fallback_successes = 0
    selected = 0
    utilities = []
    event_hits = 0
    selected_by_family: dict[str, int] = defaultdict(int)
    wins_by_family: dict[str, int] = defaultdict(int)
    losses_by_family: dict[str, int] = defaultdict(int)
    for opportunity in opportunities:
        fork = by_fork[str(opportunity["fork_id"])]
        source = fork["branches"]["SOURCE_EDGE"]
        fallback = fork["branches"]["TARGET_ABSTAIN"]
        admit = bool(opportunity["policy_admissions"][policy])
        fallback_successes += int(fallback["official_success"])
        if not admit:
            source_successes += int(fallback["official_success"])
            continue
        selected += 1
        family = str(opportunity["task_family"])
        selected_by_family[family] += 1
        source_successes += int(source["official_success"])
        win = bool(source["official_success"] and not fallback["official_success"])
        loss = bool(fallback["official_success"] and not source["official_success"])
        wins += int(win)
        losses += int(loss)
        wins_by_family[family] += int(win)
        losses_by_family[family] += int(loss)
        event_hits += int(
            source["fork_target_effect_receipt"] == "RELATE_SLOT_CLOSED"
        )
        utilities.append(fork_utility(
            source_success=bool(source["official_success"]),
            control_success=bool(fallback["official_success"]),
            source_steps=int(source["steps"]),
            control_steps=int(fallback["steps"]),
            source_completed_fraction=float(source["completed_fraction"]),
            control_completed_fraction=float(fallback["completed_fraction"]),
            max_steps=max_steps,
        ))
    unchanged_successes = sum(int(row["official_success"]) for row in no_contrast)
    source_successes += unchanged_successes
    fallback_successes += unchanged_successes
    return {
        "policy": policy,
        "task_count": task_count,
        "opportunity_count": len(opportunities),
        "selected": selected,
        "selected_by_family": dict(selected_by_family),
        "success_wins": wins,
        "success_losses": losses,
        "success_ties": selected - wins - losses,
        "success_delta": wins - losses,
        "one_sided_exact_sign_p": _exact_sign_p(wins, losses),
        "policy_successes": source_successes,
        "target_baseline_successes": fallback_successes,
        "policy_success_rate": source_successes / task_count,
        "target_baseline_success_rate": fallback_successes / task_count,
        "success_rate_delta": (source_successes - fallback_successes) / task_count,
        "selected_incremental_utility": float(sum(utilities)),
        "selected_positive_utility": sum(value > 1e-12 for value in utilities),
        "selected_negative_utility": sum(value < -1e-12 for value in utilities),
        "source_event_recall": event_hits / selected if selected else 0.0,
        "wins_by_family": dict(wins_by_family),
        "losses_by_family": dict(losses_by_family),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V20 eval report: {args.output}")
    plan = _read(args.plan)
    plan_hash = _validate_hash(plan, "plan_sha256")
    expected_status = {
        "utility_requalification": (
            "FROZEN_BEFORE_ANY_UTILITY_REQUALIFICATION_OUTCOME"
        ),
        "development_gate": "FROZEN_BEFORE_ANY_DEVELOPMENT_OUTCOME",
        "sealed_confirmation": "FROZEN_BEFORE_ANY_SEALED_CONFIRMATION_OUTCOME",
    }[str(plan["role"])]
    if plan.get("status") != expected_status:
        raise SystemExit("V20 eval plan has unexpected authority")
    for receipt in plan["implementation"].values():
        _validate_file_receipt(receipt)
    for name in ("manifest", "candidate", "enumeration"):
        _validate_file_receipt(plan[name])
    parent_receipt = plan["parent_candidate"]
    parent = _read(Path(str(parent_receipt["path"])))
    _validate_hash(parent, "candidate_sha256")
    if _sha256(Path(str(parent_receipt["path"]))) != parent_receipt["file_sha256"]:
        raise SystemExit("V20 eval parent candidate file changed")
    target = _validate_dependency(parent["target_grounder"])
    validate_target_artifact(target)
    router = dict(parent["property_router"])
    validate_property_router(router)
    source_ir = dict(parent["slot_source_ir"])
    validate_slot_source_ir(source_ir)
    thresholds = dict(parent["thresholds"])
    allowed_source_effects = tuple(map(
        str, parent["transfer_scope"]["allowed_source_effects"]
    ))
    active_required_properties = tuple(map(
        str, parent["transfer_scope"]["active_required_properties"]
    ))
    opportunities = list(plan["opportunities"])
    task_ids = tuple(map(str, plan["task_ids"]))
    max_steps = int(plan["max_steps"])
    result_by_fork: dict[str, dict[str, Any]] = {}
    for treatment in TREATMENTS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(args.alfworld_config.resolve()),
            data_path=str(args.alfworld_data.resolve()),
            split="train",
            seed=int(plan["seed"]),
            game_ids=tuple(str(row["task_id"]) for row in opportunities),
            max_steps=max_steps,
        )
        seen: set[str] = set()
        try:
            for index in range(len(opportunities)):
                observation = environment.reset()
                matches = [
                    row for row in opportunities
                    if _relative_game_matches(
                        environment.resolved_game_file, str(row["task_id"])
                    )
                ]
                if len(matches) != 1:
                    raise RuntimeError("V20 eval fork reset identity mismatch")
                opportunity = matches[0]
                task_id = str(opportunity["task_id"])
                if task_id in seen:
                    raise RuntimeError("V20 eval fork repeated a task identity")
                seen.add(task_id)
                branch = _run_branch(
                    environment=environment,
                    observation=observation,
                    opportunity=opportunity,
                    treatment=treatment,
                    target=target,
                    router=router,
                    source_ir=source_ir,
                    thresholds=thresholds,
                    allowed_source_effects=allowed_source_effects,
                    active_required_properties=active_required_properties,
                    max_steps=max_steps,
                )
                fork_id = str(opportunity["fork_id"])
                result_by_fork.setdefault(fork_id, dict(opportunity) | {
                    "branches": {}
                })["branches"][treatment] = branch
                print(json.dumps({
                    "role": plan["role"],
                    "treatment": treatment,
                    "branch_index": index,
                    "branch_count": len(opportunities),
                    "task_id": task_id,
                    "success": branch["official_success"],
                }), flush=True)
        finally:
            environment.close()
    forks = []
    for opportunity in opportunities:
        row = result_by_fork[str(opportunity["fork_id"])]
        source = row["branches"]["SOURCE_EDGE"]
        fallback = row["branches"]["TARGET_ABSTAIN"]
        invariants = {
            "fork_state_match": (
                source["fork_state_sha256"]
                == fallback["fork_state_sha256"]
                == opportunity["expected_fork_state_sha256"]
            ),
            "source_action_match": (
                source["source_action"] == fallback["source_action"]
                == opportunity["expected_source_action"]
            ),
            "fallback_action_match": (
                source["control_action"] == fallback["control_action"]
                == opportunity["expected_fallback_action"]
            ),
            "feature_match": source["features_sha256"] == fallback["features_sha256"],
            "action_contrast": source["source_action"] != source["control_action"],
        }
        if not all(invariants.values()):
            raise RuntimeError("V20 eval matched-fork invariant failed")
        row["invariants"] = invariants
        row["fork_sha256"] = stable_hash(row)
        forks.append(row)
    contrast_tasks = {str(row["task_id"]) for row in opportunities}
    no_contrast_ids = tuple(task_id for task_id in task_ids if task_id not in contrast_tasks)
    no_contrast = []
    if no_contrast_ids:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(args.alfworld_config.resolve()),
            data_path=str(args.alfworld_data.resolve()),
            split="train",
            seed=int(plan["seed"]),
            game_ids=no_contrast_ids,
            max_steps=max_steps,
        )
        seen: set[str] = set()
        try:
            for index in range(len(no_contrast_ids)):
                observation = environment.reset()
                matches = [
                    task_id for task_id in no_contrast_ids
                    if _relative_game_matches(environment.resolved_game_file, task_id)
                ]
                if len(matches) != 1:
                    raise RuntimeError("V20 eval no-contrast reset identity mismatch")
                task_id = matches[0]
                if task_id in seen:
                    raise RuntimeError("V20 eval no-contrast task repeated")
                seen.add(task_id)
                row = _run_target_only(
                    environment=environment,
                    observation=observation,
                    task_id=task_id,
                    target=target,
                    router=router,
                    source_ir=source_ir,
                    thresholds=thresholds,
                    allowed_source_effects=allowed_source_effects,
                    active_required_properties=active_required_properties,
                    max_steps=max_steps,
                )
                no_contrast.append(row)
                print(json.dumps({
                    "role": plan["role"],
                    "treatment": "NO_CONTRAST_TARGET_ONLY",
                    "task_index": index,
                    "task_count": len(no_contrast_ids),
                    "task_id": task_id,
                    "success": row["official_success"],
                }), flush=True)
        finally:
            environment.close()
    policies = list(opportunities[0]["policy_admissions"]) if opportunities else []
    metrics = {
        policy: _policy_metrics(
            policy=policy,
            task_count=len(task_ids),
            opportunities=opportunities,
            forks=forks,
            no_contrast=no_contrast,
            max_steps=max_steps,
        )
        for policy in policies
    }
    primary = metrics[str(plan["primary_policy"])]
    always = metrics["always_source_edge"]
    lexical = metrics["lexical_move_relation"]
    gate_spec = plan["gates"]
    invariants_passed = all(
        all(row["invariants"].values()) for row in forks
    )
    gates = {
        "minimum_opportunities": len(opportunities) >= int(
            gate_spec["minimum_opportunities"]
        ),
        "minimum_primary_admissions": primary["selected"] >= int(
            gate_spec["minimum_primary_admissions"]
        ),
        "minimum_primary_success_wins": primary["success_wins"] >= int(
            gate_spec["minimum_primary_success_wins"]
        ),
        "primary_success_delta_strictly_positive": primary["success_delta"] > 0,
        "primary_exact_sign_test_passed": primary["one_sided_exact_sign_p"] <= float(
            gate_spec["primary_one_sided_exact_sign_test_alpha"]
        ),
        "primary_selected_utility_strictly_positive": (
            primary["selected_incremental_utility"] > 0.0
        ),
        "primary_loss_count_strictly_less_than_always_source": (
            primary["success_losses"] < always["success_losses"]
        ),
        "primary_net_delta_strictly_greater_than_lexical_move_heuristic": (
            primary["success_delta"] > lexical["success_delta"]
        ),
        "source_event_recall_passed": primary["source_event_recall"] >= float(
            gate_spec["source_event_recall_at_least"]
        ),
        "all_exact_state_fork_invariants": invariants_passed,
    }
    passed = all(gates.values())
    role = str(plan["role"])
    status = {
        ("utility_requalification", True): (
            "UTILITY_REQUALIFICATION_PASSED_DEVELOPMENT_AUTHORIZED"
        ),
        ("utility_requalification", False): (
            "UTILITY_REQUALIFICATION_FAILED_STOP"
        ),
        ("development_gate", True): (
            "DEVELOPMENT_TRANSFER_GATE_PASSED_CONFIRMATION_AUTHORIZED"
        ),
        ("development_gate", False): "DEVELOPMENT_TRANSFER_GATE_FAILED_STOP",
        ("sealed_confirmation", True): "SEALED_CROSS_DOMAIN_TRANSFER_VALIDATED",
        ("sealed_confirmation", False): "SEALED_CROSS_DOMAIN_TRANSFER_NOT_VALIDATED",
    }[(role, passed)]
    body = {
        "schema_version": "real-source-relation-eval-report-v20",
        "status": status,
        "claim_boundary": (
            "REAL_SOURCE_GAME_BIND_TO_RELATE_GRAPH; TARGET_NATIVE_CAUSAL_"
            "GROUNDING_AND_INCREMENTAL_UTILITY; FROZEN_FIRST_OPPORTUNITY_"
            "PAIRED_SUCCESS_EVALUATION ON_DISJOINT_ALFWORLD_TRAIN_IDENTITIES"
        ),
        "role": role,
        "plan": {
            "path": str(args.plan.resolve()),
            "file_sha256": _sha256(args.plan),
            "plan_sha256": plan_hash,
        },
        "task_count": len(task_ids),
        "opportunity_count": len(opportunities),
        "no_contrast_task_count": len(no_contrast),
        "forks": forks,
        "no_contrast_target_only": no_contrast,
        "policy_metrics": metrics,
        "primary_policy": str(plan["primary_policy"]),
        "gates": gates,
        "all_gates_passed": passed,
        "development_authorized": (
            role == "utility_requalification" and passed
        ),
        "confirmation_authorized": role == "development_gate" and passed,
        "cross_domain_transfer_validated": role == "sealed_confirmation" and passed,
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
        "status": status,
        "role": role,
        "task_count": len(task_ids),
        "opportunity_count": len(opportunities),
        "policy_metrics": metrics,
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
