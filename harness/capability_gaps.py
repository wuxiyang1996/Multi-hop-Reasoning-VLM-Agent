"""Machine-readable claim boundary for the game-skill transfer Harness.

Capability records are audit metadata only.  They never rank candidates,
change admission, or turn an Agent statement into evidence.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Tuple

from harness.skill_admission import (
    AdmissionArtifact,
    BindingCandidate,
    TargetDemoReceipt,
)
from skill_bank.program_ir import CanonicalSkillProgram, ProgramStatus


class CapabilityState(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    UNVERIFIABLE_WITH_CURRENT_DATA = "UNVERIFIABLE_WITH_CURRENT_DATA"


def _record(
    capability_id: str,
    state: CapabilityState,
    *,
    evidence: Mapping[str, Any],
    claim_limit: str,
) -> Dict[str, Any]:
    return {
        "capability_id": capability_id,
        "state": state.value,
        "evidence": dict(evidence),
        "claim_limit": claim_limit,
        "affects_admission_verdict": False,
    }


def build_capability_report(
    *,
    proposal_metadata: Mapping[str, Any],
    selections: Sequence[
        Tuple[CanonicalSkillProgram, Sequence[BindingCandidate], AdmissionArtifact]
    ],
    demo: TargetDemoReceipt,
) -> Dict[str, Any]:
    """Describe exactly what this Harness run did and did not implement."""
    programs = [item[0] for item in selections]
    candidates = [candidate for _, group, _ in selections for candidate in group]
    artifacts = [item[2] for item in selections]
    selected = []
    for program, group, artifact in selections:
        selected.append({
            "source_program_id": program.program_id,
            "source_program_hash": program.content_hash(),
            "source_game": list(program.source_games),
            # Legacy labels are retained only for traceability.  They are not
            # semantic proof and were hidden from the v2 Agent prompt.
            "legacy_source_skill_name": program.name,
            "source_step_count": len(program.steps),
            "candidate_ids": [item.candidate_id for item in group],
            "candidate_target_operators": sorted({item.target_operator for item in group}),
            "proposal_sources": sorted({item.proposal_source for item in group}),
            "admission_status": artifact.status.value,
            "admitted_candidate_id": artifact.admitted_candidate_id,
            "semantic_alignment_claimed": bool(
                artifact.verified_scope
                and artifact.verified_scope.semantic_alignment_claimed
            ),
        })

    all_source_verified = bool(programs) and all(
        item.status == ProgramStatus.SOURCE_VERIFIED for item in programs
    )
    all_multistep = bool(programs) and all(len(item.steps) > 1 for item in programs)
    any_multistep = any(len(item.steps) > 1 for item in programs)
    candidate_source = str(proposal_metadata.get("candidate_source") or "")
    agent_selected = candidate_source == "independent_untrusted_agents"
    verdict_counts = Counter(item.status.value for item in artifacts)

    capabilities = [
        _record(
            "agent_game_skill_selection",
            CapabilityState.IMPLEMENTED if agent_selected else CapabilityState.NOT_IMPLEMENTED,
            evidence={
                "selection_rule": "valid_non_abstain_closed_schema_candidate",
                "candidate_source": candidate_source,
                "n_selected_programs": len(programs),
                "n_candidates": len(candidates),
                "ranking": "none",
                "voting": "none",
            },
            claim_limit=(
                "Agent non-ABSTAIN selects proposals only; it does not prove that a game skill "
                "is relevant or semantically aligned."
            ),
        ),
        _record(
            "source_program_receipt_identity",
            CapabilityState.IMPLEMENTED if all_source_verified else CapabilityState.PARTIAL,
            evidence={
                "n_source_programs": len(programs),
                "all_program_status_source_verified": all_source_verified,
                "program_hashes_checked_by_admission": True,
            },
            claim_limit=(
                "This run checks immutable program identity; full raw source replay is a separate "
                "pre-run obligation."
            ),
        ),
        _record(
            "target_native_one_shot_admission",
            CapabilityState.IMPLEMENTED,
            evidence={
                "demo_id": demo.demo_id,
                "demo_hash": demo.content_hash(),
                "native_evidence_version": demo.native_evidence_version,
                "official_success": demo.official_success,
                "target_gradient_updates": 0,
                "verdict_counts": dict(sorted(verdict_counts.items())),
            },
            claim_limit="Admission is bounded to operators and transitions covered by one demo.",
        ),
        _record(
            "source_multistep_control_program",
            (
                CapabilityState.IMPLEMENTED
                if all_multistep
                else CapabilityState.PARTIAL
                if any_multistep
                else CapabilityState.NOT_IMPLEMENTED
            ),
            evidence={
                "source_step_counts": [len(item.steps) for item in programs],
                "all_multistep": all_multistep,
            },
            claim_limit=(
                "Current CanonicalSkillPrograms are single-step; full-episode TracePrograms exist "
                "but are not yet bound into this admission path."
            ),
        ),
        _record(
            "source_skill_boundary_proof",
            CapabilityState.UNVERIFIABLE_WITH_CURRENT_DATA,
            evidence={
                "legacy_labels_used_as_proof": False,
                "segmentation_heuristic_used": False,
            },
            claim_limit="Legacy skill/intention labels are Agent-generated and cannot prove boundaries.",
        ),
        _record(
            "source_reasoning_event_instrumentation",
            CapabilityState.NOT_IMPLEMENTED,
            evidence={
                "proposal_events_receipted": False,
                "rejected_candidates_receipted": False,
                "continue_abstain_decisions_receipted": False,
            },
            claim_limit="Old episodes cannot prove an internal reasoning backbone.",
        ),
        _record(
            "counterfactual_intervention_verification",
            CapabilityState.NOT_IMPLEMENTED,
            evidence={"snapshot_alternative_rollouts": 0},
            claim_limit="Guard, branch, retry and termination hypotheses remain unverified.",
        ),
        _record(
            "multistep_target_binding",
            CapabilityState.NOT_IMPLEMENTED,
            evidence={
                "binding_granularity": "one_source_step_to_one_target_operator",
                "full_target_control_graph": False,
            },
            claim_limit="Current target artifact gates local actions, not a full transferred program.",
        ),
        _record(
            "runtime_proposal_set_consensus",
            CapabilityState.NOT_IMPLEMENTED,
            evidence={
                "admission_ambiguity_check": True,
                "per_state_all_survivors_action_consensus": False,
            },
            claim_limit="Runtime uses frozen admitted artifacts; it does not execute a full candidate set.",
        ),
        _record(
            "target_native_same_demo_baseline",
            CapabilityState.NOT_IMPLEMENTED,
            evidence={"registered_condition": False},
            claim_limit="Source contribution cannot be isolated until this baseline is run.",
        ),
    ]
    return {
        "schema_version": 1,
        "report_role": "claim_boundary_only_never_verdict",
        "selection_semantics": (
            "Every valid Agent non-ABSTAIN candidate is retained; no heuristic ranking or vote."
        ),
        "selected_game_skills": selected,
        "capabilities": capabilities,
        "implemented": [
            item["capability_id"] for item in capabilities
            if item["state"] == CapabilityState.IMPLEMENTED.value
        ],
        "gaps": [
            item["capability_id"] for item in capabilities
            if item["state"] != CapabilityState.IMPLEMENTED.value
        ],
    }


def build_v3_implementation_report() -> Dict[str, Any]:
    """Static code-level readiness report; it is not an experiment result."""
    capabilities = [
        _record(
            "content_addressed_agent_evidence_query",
            CapabilityState.IMPLEMENTED,
            evidence={"lookup": "exact_transition_id", "semantic_search": False},
            claim_limit="Reference integrity does not make an Agent hypothesis true.",
        ),
        _record(
            "agent_control_hypothesis_validation",
            CapabilityState.IMPLEMENTED,
            evidence={
                "merge": "content_hash_set_union",
                "ranking": False,
                "voting": False,
                "real_35b_agent_calls": 3,
                "real_qualified_hypotheses": 2,
                "real_agent_abstentions": 1,
                "instrumented_2048_agent_calls": 12,
                "instrumented_2048_full_path_qualified_hypotheses": 6,
                "instrumented_2048_single_node_rejections": 4,
                "instrumented_2048_detached_receipt_rejections": 2,
                "instrumented_2048_receipt_attached_candidates": 2,
                "formal_six_fresh_agent_calls": 18,
                "formal_six_fresh_programs": 6,
                "formal_six_fresh_qualified_hypotheses": 17,
                "formal_six_fresh_invalid_outputs": 1,
                "formal_six_merged_artifact_sha256": (
                    "016a9c354b6cb40c74d5ac33cfd43ca6ec6039d09d06d61ecca0fa54096d425a"
                ),
                "semantic_control_claims_verified": 0,
            },
            claim_limit=(
                "Accepted structures remain AGENT_HYPOTHESIS. Exact fork identity and edge "
                "attachment do not verify the Agent-authored semantic control claim."
            ),
        ),
        _record(
            "reasoning_event_schema",
            CapabilityState.PARTIAL,
            evidence={
                "tamper_evident_chain": True,
                "production_source_rollouts_recorded": 12,
                "production_source_agent_decisions_recorded": 144,
                "agent_origin_decisions": 137,
                "parser_fallback_decisions": 0,
                "policy_override_decisions": 7,
                "policy_override_transitions_excluded_from_programs": 7,
                "agent_span_segmentation_content_dependent": False,
                "explicit_replan_abstain_protocol": False,
                "target_development_episodes_recorded": 30,
            },
            claim_limit=(
                "The source execution path now emits complete attribution events, but the "
                "current Actor protocol does not natively propose REPLAN or ABSTAIN. Old traces "
                "cannot be retroactively upgraded."
            ),
        ),
        _record(
            "seeded_source_replay_to_fork",
            CapabilityState.PARTIAL,
            evidence={
                "gymv_subprocess_seed_rpc": True,
                "fresh_env_per_branch_required": True,
                "gamingagent_2048_observable_intervention_receipts": 13,
                "new_seeded_smoke_episodes": 12,
                "formal_six_seed_base": 242,
                "new_replay_forks_observed": 166,
                "new_replay_mismatches": 0,
                "orak_seed_supported": False,
                "old_trace_seed_receipts": 0,
            },
            claim_limit=(
                "GamingAgent/Gym-V can be probed after environment bootstrap; Orak and old "
                "unseeded traces must emit a gap, not an intervention verdict."
            ),
        ),
        _record(
            "multistep_target_binding_v3",
            CapabilityState.IMPLEMENTED,
            evidence={
                "linear_demo_edges_only": True,
                "semantic_predicates": False,
                "node_binding_version": 3,
                "one_source_node_to_ordered_target_span": True,
                "source_node_receipts_frozen_in_artifact": True,
                "source_conditioning_exact_identity_checked": True,
            },
            claim_limit="One successful demo cannot verify branches, loops, retries, or guards.",
        ),
        _record(
            "runtime_all_candidate_exact_action_consensus",
            CapabilityState.IMPLEMENTED,
            evidence={
                "runtime_library": True,
                "per_candidate_native_action_sets": True,
                "common_set_operation": "exact_intersection",
                "single_actor_call_on_common_set": True,
                "all_active_source_candidates_shown_without_ranking": True,
                "development_eval_driver_integration": True,
                "held_out_results": 0,
            },
            claim_limit=(
                "Mechanics and development integration are verified; no held-out transfer result "
                "or 2x4 evaluation has been run."
            ),
        ),
        _record(
            "online_target_source_control",
            CapabilityState.PARTIAL,
            evidence={
                "episode_local_state_machine": True,
                "native_transition_receipts": True,
                "tamper_evident_online_event_chain": True,
                "bounded_rebind_requests": True,
                "live_target_only_fallback_from_current_state": True,
                "same_demo_prefix_required_for_fallback": False,
                "real_development_smoke_episodes": 4,
                "real_source_disabled_events": 2,
                "online_binding_agent": True,
                "closed_schema_identity_and_action_admission": True,
                "admitted_rebind_receipts": 2,
                "executed_rebind_receipts": 2,
                "verified_expected_evidence_contracts": 2,
                "predeclared_contract_required_for_every_source_action": True,
                "generic_observation_delta_counts_as_transfer_verification": False,
                "candidate_independent_cursor_and_status": True,
                "per_candidate_preaction_evidence_contract": True,
                "per_candidate_transition_verdict": True,
                "partial_survivor_advancement": True,
                "agent_selects_action_contract_predicates": False,
                "harness_compiles_contract_from_demo_transition_receipt": True,
                "uncited_contract_failure_is_inconclusive_not_refutation": True,
                "receipt_backed_candidate_refutation": True,
                "compute_matched_shadow_contract_mode": True,
                "development_action_contract_agent_calls": 0,
                "action_contract_compiler": (
                    "exact_one_shot_transition_receipt_signature_v1"
                ),
                "development_receipt_contract_compilations": 21,
                "development_prebinding_factorial_pairs": 1,
                "development_prebinding_factorial_successes": 0,
                "development_source_candidates_fail_closed_to_target_only": True,
                "binding_receipt_registry": True,
                "decompose_recompose": False,
                "experimental_negative_transfer_verdict_at_runtime": False,
            },
            claim_limit=(
                "Per-candidate contracts and live fallback were exercised in a one-pair "
                "development factorial. A satisfied native evidence contract is not semantic "
                "equivalence or task value; decomposition remains unimplemented and there is "
                "no held-out transfer result. Contracts are grounded to exact one-shot "
                "transition receipts, but a single observed delta does not establish a "
                "semantic necessity; the current local compatibility test can therefore "
                "still be overly conservative."
            ),
        ),
        _record(
            "prebinding_source_dependence_controls",
            CapabilityState.IMPLEMENTED,
            evidence={
                "treatments": ["empty", "correct", "wrong", "renamed"],
                "paper_labels": {
                    "correct": "designated_source_not_semantically_prevalidated",
                    "wrong": "cross_game_source_control_not_semantically_prevalidated",
                },
                "six_game_sources_are_anonymous_treatments": True,
                "control_stage": "before_binding_agent_call",
                "content_independent_renaming": True,
                "cross_game_wrong_source_requires_frozen_artifact": True,
                "control_receipt_frozen_into_admission_artifact": True,
                "target_semantic_scoring": False,
            },
            claim_limit=(
                "The controls isolate source exposure mechanically. A one-pair development "
                "E/S/W/R smoke found no positive ordering; source dependence is not established."
            ),
        ),
        _record(
            "online_negative_transfer_development_matrix",
            CapabilityState.PARTIAL,
            evidence={
                "paired_development_episodes": 4,
                "initial_observation_hash_matches": 4,
                "target_only_successes": 1,
                "naive_source_successes": 1,
                "online_harness_successes": 1,
                "rotated_conditioning_successes": 1,
                "actual_token_consumption_matched": False,
                "online_harness_total_tokens": 102854,
                "target_only_total_tokens": 8119,
                "authorizes_large_scale_2x4": False,
            },
            claim_limit=(
                "This four-pair development matrix has no inferential power and unequal actual "
                "token consumption. It shows no positive source or mitigation signal."
            ),
        ),
        _record(
            "formal_six_source_observational_programs",
            CapabilityState.PARTIAL,
            evidence={
                "source_games": 6,
                "fixed_episode_000_per_game": True,
                "full_episode_trace_programs": 6,
                "observed_transition_receipts": 409,
                "source_file_replay_pass": 6,
                "source_file_replay_fail": 0,
                "reasoning_claim": "none_observational_trace_only",
                "fresh_instrumented_reasoning_rollouts": 0,
            },
            claim_limit=(
                "Exact replay against an old JSON source file proves observational integrity "
                "only. It does not recover missing Agent decisions, seed receipts, interventions "
                "or official success."
            ),
        ),
        _record(
            "formal_six_fresh_instrumented_programs",
            CapabilityState.IMPLEMENTED,
            evidence={
                "source_games": 6,
                "fresh_episodes": 12,
                "reasoning_events": 1332,
                "agent_origin_decisions": 137,
                "excluded_policy_postprocessor_decisions": 7,
                "replay_fork_receipts": 166,
                "protocol_or_replay_failures": 0,
                "frozen_programs": 6,
                "qualified_agent_hypotheses": 17,
                "merged_artifact_sha256": (
                    "016a9c354b6cb40c74d5ac33cfd43ca6ec6039d09d06d61ecca0fa54096d425a"
                ),
                "source_selection_uses_target_results": False,
            },
            claim_limit=(
                "Fresh receipts justify source-side observed programs and Agent hypotheses, "
                "not source-to-target semantic equivalence or positive transfer."
            ),
        ),
        _record(
            "target_native_same_demo_baseline",
            CapabilityState.IMPLEMENTED,
            evidence={
                "origin_isolation_schema": True,
                "matched_agent_proposal_run": True,
                "matched_roles": 3,
                "development_episode_pairs": 8,
                "paired_initial_observation_hash_matches": 8,
                "source_actor_calls_conditioned": 39,
                "target_only_actor_calls_conditioned": 0,
            },
            claim_limit=(
                "The matched development baseline is complete, but it is not held-out evidence; "
                "copying the expert trace as an oracle policy remains forbidden."
            ),
        ),
        _record(
            "large_scale_2x4_v3_experiment",
            CapabilityState.NOT_IMPLEMENTED,
            evidence={
                "preregistered_v3_config": False,
                "paired_development_pilot_completed": True,
                "source_successes": 2,
                "target_only_successes": 2,
                "episode_pairs": 8,
                "source_reported_cost": 0.02737905,
                "target_only_reported_cost": 0.0027876775,
                "source_treatment_active": True,
                "authorizes_large_scale_2x4": False,
            },
            claim_limit=(
                "Do not launch 2x4: verified source conditioning reached every source Actor "
                "call but the matched development pilot has no positive source signal. Test a "
                "different source program family or a general Agent-side receipt-use protocol "
                "without adapting mappings to ALFWorld development rewards."
            ),
        ),
    ]
    return {
        "schema_version": 1,
        "report_role": "code_readiness_claim_boundary_never_verdict",
        "capabilities": capabilities,
        "implemented": [
            item["capability_id"] for item in capabilities
            if item["state"] == CapabilityState.IMPLEMENTED.value
        ],
        "gaps": [
            item["capability_id"] for item in capabilities
            if item["state"] != CapabilityState.IMPLEMENTED.value
        ],
    }


__all__ = ["CapabilityState", "build_capability_report", "build_v3_implementation_report"]
