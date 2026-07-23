from __future__ import annotations

import hashlib
import json

from harness.online_reasoning_backbone import (
    FrozenBackboneConditioning,
    admit_online_backbone_plan,
    close_online_backbone_cycle,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _conditioning(treatment="correct") -> FrozenBackboneConditioning:
    return FrozenBackboneConditioning(
        treatment=treatment,
        source_artifact_sha256=(None if treatment == "target_only" else _digest("source")),
        adaptation_demo_sha256=_digest("demo"),
        prompt_template_sha256=_digest("prompt"),
    )


def _plan() -> str:
    return json.dumps({
        "proposals": [{
            "proposal_id": "p0", "action_number": 2,
            "predicted_observable_delta": "observable door state may change",
            "rationale": "test the live target state",
        }],
        "selected_proposal_id": "p0", "decision": "EXECUTE",
    })


def test_online_backbone_executes_only_target_native_admitted_action() -> None:
    admission = admit_online_backbone_plan(
        _plan(), conditioning=_conditioning(), target_state="before",
        native_actions=["look", "open door"], agent_prompt_sha256=_digest("agent"),
    )
    assert admission.status == "ADMITTED"
    assert admission.admitted_native_action == "open door"
    cycle = close_online_backbone_cycle(
        admission, executed_action="open door", before_state="before",
        after_state="after", reward=0.0, done=False,
        raw_verdict=json.dumps({
            "proposal_id": "p0", "verdict": "INCONCLUSIVE",
            "decision": "REPLAN", "evidence_claim": "more target evidence is needed",
        }), verifier_prompt_sha256=_digest("verify"),
    )
    assert cycle.status == "VERIFIED_AGENT_CYCLE"
    assert cycle.runtime_directive == "AGENT_CYCLE_REPLAN"


def test_online_backbone_falls_back_when_execution_or_verdict_is_unbound() -> None:
    admission = admit_online_backbone_plan(
        _plan(), conditioning=_conditioning(), target_state="before",
        native_actions=["look", "open door"], agent_prompt_sha256=_digest("agent"),
    )
    cycle = close_online_backbone_cycle(
        admission, executed_action="look", before_state="before", after_state="after",
        reward=0.0, done=False, raw_verdict="{}",
        verifier_prompt_sha256=_digest("verify"),
    )
    assert cycle.status == "REJECTED"
    assert cycle.runtime_directive == "FALLBACK_TARGET_ONLY"
    assert cycle.failure_code == "EXECUTED_ACTION_DIFFERS_FROM_ADMISSION"


def test_target_only_condition_has_no_source_artifact() -> None:
    conditioning = _conditioning("target_only")
    assert conditioning.content_hash()
