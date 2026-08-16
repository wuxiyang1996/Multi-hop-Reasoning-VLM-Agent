"""Prospective DiscoveryWorld Phase-3 fork selection and outcome analysis."""

from __future__ import annotations

from math import comb
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .phase3_discoveryworld_transfer import (
    CONDITIONS,
    extract_phase3_acquisition_evidence,
    phase3_acquisition_outlier_candidates,
)


EFFECT_HORIZON = {
    "EFFECT_BY_TRANSITION_1": 1,
    "EFFECT_BY_TRANSITION_4": 4,
    "EFFECT_BY_TRANSITION_8": 8,
    "EXECUTABLE_TRANSITION_PERSISTENCE": 8,
}


def _held_flag(facts: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row, Mapping)
        and "flag" in str(row.get("name") or "").lower()
        for row in facts.get("inventory") or ()
    )


def _named_statue_visible(
    facts: Mapping[str, Any], subject: str,
) -> bool:
    for key in ("accessible_objects", "salient_relative_objects"):
        for row in facts.get(key) or ():
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name") or "").lower()
            if "statue" in name and subject.lower() in name:
                return True
    return False


def select_outcome_blind_formal_fork(
    reference: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select the first structurally ready state without target outcomes.

    Only pre-fork policy-visible instrument messages and sanitized native facts
    participate.  Transition success, terminal flags, evaluation, scorecard,
    and the recorded next action are deliberately outside the selection view.
    """

    steps = reference.get("steps") or ()
    for fork_step in range(1, len(steps) + 1):
        row = steps[fork_step - 1]
        if not isinstance(row, Mapping):
            continue
        after = row.get("after_target_native_facts") or {}
        if not isinstance(after, Mapping):
            continue
        # Reconstruct just the fact categories used by the target grounding
        # contract, then strip all evaluator/completion fields locally.
        facts = {
            "inventory": list(after.get("inventory") or ()),
            "accessible_objects": list(after.get("accessible_objects") or ()),
            "salient_relative_objects": list(
                after.get("salient_relative_objects") or ()
            ),
        }
        evidence = extract_phase3_acquisition_evidence(reference, fork_step)
        outliers = phase3_acquisition_outlier_candidates(evidence)
        if len(evidence) != 3 or len(outliers) != 1:
            continue
        if not _held_flag(facts):
            continue
        if not _named_statue_visible(facts, outliers[0]):
            continue
        body = {
            "schema_version": "phase3-discoveryworld-outcome-blind-fork-v1",
            "fork_after_episode_step": fork_step,
            "acquisition_evidence_count": len(evidence),
            "acquisition_evidence_sha256": stable_hash(list(evidence)),
            "acquisition_outlier_candidates": list(outliers),
            "held_flag": True,
            "derived_target_statue_visible": True,
            "selection_fields": [
                "policy_visible_instrument_messages",
                "inventory_names",
                "visible_object_names",
            ],
            "forbidden_fields_read": False,
        }
        return body | {"fork_receipt_sha256": stable_hash(body)}
    raise ValueError("no outcome-blind structurally ready formal fork")


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _first_success_step(arm: Mapping[str, Any]) -> int | None:
    for index, step in enumerate(arm.get("recovery") or (), start=1):
        transition = step.get("transition") or {}
        if bool(transition.get("official_success")):
            return index
    return None


def _program_horizon(result: Mapping[str, Any]) -> tuple[int, str | None]:
    recovery = (
        (result.get("conditions") or {}).get("source_induced") or {}
    ).get("recovery") or ()
    if not recovery:
        return 8, None
    selection = recovery[0].get("selection") or {}
    receipt = selection.get("portfolio_selection_receipt") or {}
    effect = receipt.get("selected_effect_type")
    return EFFECT_HORIZON.get(str(effect), 8), str(effect) if effect else None


def analyze_formal_results(
    *, manifest: Mapping[str, Any], acquisition_summary: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Aggregate the frozen ITT and program-aligned Phase-3 gates."""

    per_task = []
    successes = {condition: 0 for condition in CONDITIONS}
    source_wins = source_losses = 0
    permuted_wins = permuted_losses = 0
    generic_wins = generic_losses = 0
    applicable = contrasts = 0
    selected_program_counts: dict[str, int] = {}
    binder_repairs = grounder_repairs = grounder_calls = 0
    receipts_valid = True
    no_outcome_exposure = True
    for result in results:
        arms = result.get("conditions") or {}
        horizon, effect = _program_horizon(result)
        outcomes = {}
        steps_to_success = {}
        arm_map = {"neural_only": arms.get("target_only_recorded") or {}}
        arm_map.update({
            condition: arms.get(condition) or {}
            for condition in CONDITIONS if condition != "neural_only"
        })
        for condition, arm in arm_map.items():
            step = _first_success_step(arm)
            success = step is not None and step <= horizon
            outcomes[condition] = success
            steps_to_success[condition] = step
            successes[condition] += int(success)
        source_wins += int(outcomes["source_induced"] and not outcomes["neural_only"])
        source_losses += int(outcomes["neural_only"] and not outcomes["source_induced"])
        permuted_wins += int(
            outcomes["source_induced"] and not outcomes["source_permuted"]
        )
        permuted_losses += int(
            outcomes["source_permuted"] and not outcomes["source_induced"]
        )
        generic_wins += int(
            outcomes["source_induced"] and not outcomes["generic_scaffold"]
        )
        generic_losses += int(
            outcomes["generic_scaffold"] and not outcomes["source_induced"]
        )
        source_recovery = (arms.get("source_induced") or {}).get("recovery") or ()
        permuted_recovery = (arms.get("source_permuted") or {}).get("recovery") or ()
        first_source = source_recovery[0].get("selection") if source_recovery else {}
        first_permuted = (
            permuted_recovery[0].get("selection") if permuted_recovery else {}
        )
        admitted = bool((first_source or {}).get("source_admitted"))
        applicable += int(admitted)
        contrast = bool(
            source_recovery and permuted_recovery
            and (first_source or {}).get("selected_candidate_sha256")
            != (first_permuted or {}).get("selected_candidate_sha256")
        )
        contrasts += int(admitted and contrast)
        program_sha = str((first_source or {}).get("source_profile_sha256") or "")
        if program_sha:
            selected_program_counts[program_sha] = (
                selected_program_counts.get(program_sha, 0) + 1
            )
        attempts = result.get("target_binding_schema_attempts") or ()
        binder_repairs += int(any(not row.get("accepted") for row in attempts))
        no_outcome_exposure &= bool(attempts) and all(
            row.get("formal_outcome_fields_visible") is False for row in attempts
        )
        for condition in (
            "source_induced", "source_permuted", "generic_scaffold",
            "target_native_ceiling",
        ):
            for step in (arms.get(condition) or {}).get("recovery") or ():
                grounder_calls += 1
                attempts = step.get("grounder_schema_attempts") or ()
                grounder_repairs += int(any(
                    not row.get("accepted") for row in attempts
                ))
                no_outcome_exposure &= bool(attempts) and all(
                    row.get("formal_outcome_fields_visible") is False
                    for row in attempts
                )
        receipts_valid &= bool(result.get("all_matched_forks"))
        receipts_valid &= bool(result.get("all_selection_receipts_valid"))
        receipts_valid &= result.get("policy_runtime_saw_oracle_scorecard") is False
        receipts_valid &= all(
            arm.get("runtime_error") is None
            for name, arm in arms.items() if name != "target_only_recorded"
        )
        task = result.get("task") or {}
        per_task.append({
            "task_id": (
                f"{str(task.get('scenario') or '').lower()}."
                f"{str(task.get('difficulty') or '').lower()}."
                f"seed{task.get('seed')}"
            ),
            "program_aligned_horizon": horizon,
            "selected_effect_type": effect,
            "source_initially_admitted": admitted,
            "source_permuted_first_selection_contrast": contrast,
            "outcomes": outcomes,
            "steps_to_success": steps_to_success,
        })

    task_count = int(manifest["task_count"])
    acquisition_steps = int(acquisition_summary.get("steps") or 0)
    fallback_rate = (
        int(acquisition_summary.get("schema_fallback_steps") or 0)
        / acquisition_steps if acquisition_steps else 1.0
    )
    negative_rate = (
        source_losses / (source_wins + source_losses)
        if source_wins + source_losses else 0.0
    )
    gates_cfg = manifest["frozen_gates"]
    source_sign_p = exact_two_sided_sign_p(source_wins, source_losses)
    gates = {
        "exact_formal_tasks_complete": len(results) == task_count,
        "acquisition_complete": int(acquisition_summary.get("complete_tasks", 0)) == task_count,
        "acquisition_schema_fallback_rate": fallback_rate <= float(
            gates_cfg["maximum_acquisition_schema_fallback_rate"]
        ),
        "minimum_applicable_tasks": applicable >= int(
            gates_cfg["minimum_applicable_tasks"]
        ),
        "all_receipts_valid": receipts_valid,
        "source_improves_neural": successes["source_induced"] > successes["neural_only"],
        "source_vs_neural_sign_test": source_sign_p <= float(
            gates_cfg["source_vs_neural_exact_sign_p_max"]
        ),
        "negative_transfer": negative_rate <= float(
            gates_cfg["source_vs_neural_negative_transfer_rate_max"]
        ),
        "source_beats_permuted": successes["source_induced"] > successes["source_permuted"],
        "source_beats_generic": successes["source_induced"] > successes["generic_scaffold"],
        "ceiling_not_below_source": successes["target_native_ceiling"] >= successes["source_induced"],
        "source_permuted_behaviorally_distinct": contrasts >= int(
            gates_cfg["minimum_admitted_source_permuted_first_selection_contrasts"]
        ),
        "formal_binder_repair_rate": (
            binder_repairs / len(results) if results else 1.0
        ) <= float(gates_cfg["maximum_formal_binder_repair_rate"]),
        "formal_grounder_repair_rate": (
            grounder_repairs / grounder_calls if grounder_calls else 1.0
        ) <= float(gates_cfg["maximum_formal_grounder_repair_rate"]),
        "zero_outcome_exposure_to_grounding": no_outcome_exposure,
    }
    body = {
        "schema_version": "phase3-discoveryworld-formal-report-v1",
        "status": (
            "DISCOVERYWORLD_PHASE3_CROSS_DOMAIN_TRANSFER_VALIDATED"
            if all(gates.values()) else
            "DISCOVERYWORLD_PHASE3_CROSS_DOMAIN_TRANSFER_NOT_VALIDATED"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": task_count,
        "program_aligned_successes": successes,
        "source_vs_neural": {
            "wins": source_wins, "losses": source_losses,
            "ties": task_count - source_wins - source_losses,
            "exact_two_sided_sign_p": source_sign_p,
            "negative_transfer_rate": negative_rate,
        },
        "source_vs_permuted": {
            "wins": permuted_wins, "losses": permuted_losses,
            "ties": task_count - permuted_wins - permuted_losses,
            "exact_two_sided_sign_p": exact_two_sided_sign_p(
                permuted_wins, permuted_losses
            ),
        },
        "source_vs_generic": {
            "wins": generic_wins, "losses": generic_losses,
            "ties": task_count - generic_wins - generic_losses,
        },
        "applicable_tasks": applicable,
        "admitted_source_permuted_first_selection_contrasts": contrasts,
        "selected_program_counts": selected_program_counts,
        "acquisition_schema_fallback_rate": fallback_rate,
        "formal_binder_repair_rate": binder_repairs / len(results) if results else 1.0,
        "formal_grounder_repair_rate": grounder_repairs / grounder_calls if grounder_calls else 1.0,
        "gates": gates,
        "per_task": per_task,
        "formal_results_used_to_change_protocol": False,
        "source_identity_used_as_runtime_feature": False,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "EFFECT_HORIZON", "analyze_formal_results", "exact_two_sided_sign_p",
    "select_outcome_blind_formal_fork",
]
