from __future__ import annotations

import hashlib

import pytest

from harness.causal_reasoning_motif import (
    LegacyMegaSkillLineage,
    MatchedEnvironmentOutcome,
    audit_motif_anti_collapse,
    compile_causal_motif,
    evaluate_matched_environment_contrasts,
    motif_conditioning_view,
)


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _transition(index: int, *, verdict: str, decision: str) -> dict:
    return {
        "transition_id": _h(f"t{index}"),
        "action": f"native-{index}",
        "state_sha256": _h(f"s{index}"),
        "next_state_sha256": _h(f"s{index + 1}"),
        "reward": float(index),
        "done": index == 1,
        "action_proposal_event_sha256": _h(f"proposal-event-{index}"),
        "action_proposal_receipt": {
            "proposal_set": {
                "proposals": [
                    {"proposal_id": "a"}, {"proposal_id": "b"},
                ],
                "selected_proposal_id": "b",
                "decision": "EXECUTE",
            },
        },
        "post_transition_verdict_event_sha256": _h(f"verdict-event-{index}"),
        "post_transition_verdict_receipt": {
            "verdict": {"verdict": verdict, "decision": decision},
        },
    }


def _graph() -> dict:
    return {
        "source_hypothesis_hash": _h("hypothesis"),
        "source_program_hash": _h("program"),
        "source_reasoning_trace_sha256": _h("reasoning-trace"),
        "nodes": [
            {"node_id": "n0", "observed_transitions": [
                _transition(0, verdict="REFUTED", decision="REPLAN"),
            ]},
            {"node_id": "n1", "observed_transitions": [
                _transition(1, verdict="SUPPORTED", decision="CONTINUE"),
            ]},
        ],
        "edges": [{
            "source_node_id": "n0", "target_node_id": "n1",
            "kind": "BRANCH", "agent_claim": {"meaning": "untrusted"},
            "intervention_receipt_sha256s": [_h("fork")],
        }],
    }


def test_protocol_is_not_the_skill_and_motif_retains_decision_structure() -> None:
    motif = compile_causal_motif(_graph())
    assert motif.protocol_projection().count("PROPOSE") == 2
    view = motif.transferable_view()
    assert view["nodes"][0]["steps"][0]["proposal_count"] == 2
    assert view["nodes"][0]["steps"][0]["continuation_decision"] == "REPLAN"
    audit = audit_motif_anti_collapse(motif)
    assert audit.status == "STRUCTURALLY_SPECIFIC_CANDIDATE"
    assert not audit.checks["source_attribution_requires_matched_intervention"]
    assert not audit.checks["target_incremental_value_requires_matched_forks"]


def test_agent_prose_and_edge_label_do_not_change_verified_fingerprint() -> None:
    graph = _graph()
    first = compile_causal_motif(graph)
    graph["edges"][0]["kind"] = "LOOP"
    graph["edges"][0]["agent_claim"] = {"invented": "different semantics"}
    second = compile_causal_motif(graph)
    assert first.causal_fingerprint() == second.causal_fingerprint()
    assert first.content_hash() != second.content_hash()


def test_registered_controls_remove_or_break_source_specific_content() -> None:
    motif = compile_causal_motif(_graph())
    authentic = motif_conditioning_view(motif, treatment="authentic")
    generic = motif_conditioning_view(motif, treatment="generic_protocol")
    null = motif_conditioning_view(motif, treatment="receipt_null")
    shuffled = motif_conditioning_view(motif, treatment="shuffled_topology", seed=3)
    assert len({
        authentic["conditioning_sha256"], generic["conditioning_sha256"],
        null["conditioning_sha256"], shuffled["conditioning_sha256"],
    }) == 4
    assert "nodes" not in generic["conditioning"]
    assert null["conditioning"]["source_receipt_refs"] is None
    assert shuffled["conditioning"]["edges"] != authentic["conditioning"]["edges"]


def test_legacy_megaskill_cannot_gain_execution_authority() -> None:
    lineage = LegacyMegaSkillLineage.from_record({
        "mega_skill_id": "mega.0", "template_signature": "ACT → VERIFY",
        "members": [{"task": "game", "skill_id": "skill"}],
    }, source_artifact_sha256=_h("legacy"))
    lineage.validate()
    assert lineage.authority == "LINEAGE_RETRIEVAL_ONLY"
    with pytest.raises(ValueError, match="execution authority"):
        LegacyMegaSkillLineage(
            lineage.mega_skill_id, lineage.source_artifact_sha256,
            lineage.member_refs, lineage.legacy_template_signature,
            authority="EXECUTABLE",
        ).validate()


def _matched(comparison: str, treatment: str, success: bool) -> MatchedEnvironmentOutcome:
    return MatchedEnvironmentOutcome(
        comparison_id=comparison, treatment=treatment,
        initial_state_sha256=_h("state"), prefix_sha256=_h("prefix"),
        policy_identity_sha256=_h("policy"), budget_sha256=_h("budget"),
        official_success=success, official_score=float(success),
    )


def test_matched_contrast_uses_environment_outcomes_not_agent_verdicts() -> None:
    controls = ("target_only", "generic_protocol", "shuffled_topology", "other_source")
    rows = [_matched("episode-0", "authentic", True)]
    rows.extend(_matched("episode-0", treatment, False) for treatment in controls)
    report = evaluate_matched_environment_contrasts(
        rows, claim="target_incremental_value",
    )
    assert report.status == "PILOT_SUPPORTED"
    assert all(value == 1 for value in report.authentic_wins.values())


def test_incomplete_matched_set_is_inconclusive() -> None:
    report = evaluate_matched_environment_contrasts(
        [_matched("episode-0", "authentic", True)],
        claim="source_attribution",
    )
    assert report.status == "INCONCLUSIVE"
    assert report.failure_codes[0].startswith("INCOMPLETE_MATCHED_SET")
