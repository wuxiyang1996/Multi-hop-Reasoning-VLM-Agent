#!/usr/bin/env python3
"""Enumerate first typed source-edge action contrasts without reading outcomes."""

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
    _validate_file_receipt,
    _validate_hash,
    deepcopy_json,
)


ALLOWED_ROLES = (
    "causal_adaptation",
    "causal_calibration",
    "prospective_requalification",
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
        minimum_realization_score=float(thresholds["minimum_realization_score"]),
        minimum_target_policy_ratio=float(thresholds["minimum_target_policy_ratio"]),
        allowed_source_effects=allowed_source_effects,
        active_required_properties=active_required_properties,
    )


def _edge(decision: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = decision.get("source_transition")
    if isinstance(value, dict) and value.get("kind") == "EDGE":
        return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=ALLOWED_ROLES, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V22 contrast report: {args.output}")
    manifest = _read(args.manifest)
    manifest_hash = _validate_hash(manifest, "manifest_sha256")
    if manifest.get("status") != "FROZEN_BEFORE_ANY_V22_SELECTED_TASK_RESET":
        raise SystemExit("V22 manifest has unexpected authority")
    if args.role == "prospective_requalification" and args.candidate is None:
        raise SystemExit("V22 requalification requires a frozen candidate")
    if args.role != "prospective_requalification" and args.candidate is not None:
        raise SystemExit("V22 adaptation/calibration enumeration cannot use candidate")
    for key in (
        "outcome_blind_enumerator", "fork_plan_freezer", "fork_runner",
        "target_native_model", "candidate_trainer",
    ):
        _validate_file_receipt(manifest["implementation"][key])
    candidate_receipt = None
    if args.candidate is not None:
        candidate = _read(args.candidate)
        candidate_hash = _validate_hash(candidate, "candidate_sha256")
        if candidate.get("status") != "PROSPECTIVE_REQUALIFICATION_AUTHORIZED":
            raise SystemExit("V22 candidate lacks requalification authority")
        if candidate["manifest"]["manifest_sha256"] != manifest_hash:
            raise SystemExit("V22 candidate belongs to another manifest")
        candidate_receipt = {
            "path": str(args.candidate.resolve()),
            "file_sha256": _sha256(args.candidate),
            "candidate_sha256": candidate_hash,
        }
    parent_receipt = manifest["parent_candidate"]
    parent = _read(Path(str(parent_receipt["path"])))
    _validate_hash(parent, "candidate_sha256")
    if _sha256(Path(str(parent_receipt["path"]))) != parent_receipt["file_sha256"]:
        raise SystemExit("V22 parent candidate file changed")
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
    seed = int(manifest["seed"]) + ALLOWED_ROLES.index(args.role)
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
                raise RuntimeError("V22 reset did not map to one frozen identity")
            task_id = matches[0]
            if task_id in seen:
                raise RuntimeError("V22 reset repeated a frozen identity")
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
            effect_opportunities = Counter()
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
                    raise RuntimeError("V22 target grounder excluded every action")
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
                shadow = _choose(
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
                transition = _edge(shadow)
                actionable = bool(
                    transition is not None
                    and "best_realization_score" in shadow
                    and "target_policy_ratio" in shadow
                )
                if actionable:
                    effect_opportunities[str(shadow["requested_source_effect"])] += 1
                if actionable and bool(shadow["source_admitted"]) and (
                    str(shadow["action"]) != str(baseline["action"])
                ):
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
                        "required_property": required_property,
                        "requested_source_effect": str(
                            shadow["requested_source_effect"]
                        ),
                        "fork_step": step,
                        "prefix_actions": list(history),
                        "expected_fork_state_sha256": stable_hash(state_body),
                        "expected_fallback_action": str(shadow["fallback_action"]),
                        "expected_source_action": str(shadow["action"]),
                        "expected_source_graph_sha256": str(
                            transition["graph_sha256"]
                        ),
                        "expected_source_edge": {
                            key: transition[key]
                            for key in ("from", "to", "guard", "kind")
                        },
                        "source_target_policy_ratio": float(
                            shadow["target_policy_ratio"]
                        ),
                        "source_realization_score": float(
                            shadow["best_realization_score"]
                        ),
                        "selection_authority": (
                            "FIRST_AUTHENTIC_TYPED_SOURCE_EDGE_ACTION_CONTRAST_"
                            "ON_SOURCE_DISABLED_TARGET_TRAJECTORY"
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
                "required_property": required_property,
                "prefix_steps_executed": len(history),
                "actionable_edge_states_by_effect": dict(effect_opportunities),
                "first_action_contrast": opportunity,
            }
            tasks.append(task_body | {
                "task_receipt_sha256": stable_hash(task_body)
            })
            print(json.dumps({
                "role": args.role,
                "task_index": task_index,
                "task_count": len(task_ids),
                "task_id": task_id,
                "found_contrast": opportunity is not None,
                "effect": None if opportunity is None else opportunity[
                    "requested_source_effect"
                ],
                "outcomes_recorded": False,
            }), flush=True)
    finally:
        environment.close()
    if seen != set(task_ids):
        raise RuntimeError("V22 did not enumerate every frozen task")
    opportunities = [
        row["first_action_contrast"] for row in tasks
        if row["first_action_contrast"] is not None
    ]
    body = {
        "schema_version": "real-source-multiskill-contrast-report-v22",
        "status": "OUTCOME_BLIND_TYPED_CONTRASTS_COMPLETE",
        "claim_boundary": (
            "FIRST_TYPED_ACTION_CONTRAST_SELECTION_ONLY; REWARD_DISCARDED; "
            "OFFICIAL_SUCCESS_NOT_READ_OR_SERIALIZED; FUTURE_DEVELOPMENT_"
            "CONFIRMATION_AND_VALID_UNSEEN_UNREAD"
        ),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "file_sha256": _sha256(args.manifest),
            "manifest_sha256": manifest_hash,
        },
        "candidate": candidate_receipt,
        "role": args.role,
        "seed": seed,
        "max_steps": max_steps,
        "task_count": len(tasks),
        "tasks": tasks,
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "opportunities_by_effect": dict(Counter(
            str(row["requested_source_effect"]) for row in opportunities
        )),
        "opportunities_by_family": dict(Counter(
            str(row["task_family"]) for row in opportunities
        )),
        "outcomes_recorded": False,
        "rewards_recorded": False,
        "future_development_confirmation_read_or_run": False,
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
        "opportunities_by_effect": report["opportunities_by_effect"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
