"""Deterministic vertical-slice smoke test for the neural-symbolic monitor."""

from __future__ import annotations

import json

from motif_transfer.contracts import (
    DecisionProposal,
    Observation,
    TransitionReceipt,
    stable_hash,
)
from motif_transfer.neural_event_probes import (
    FrozenTableProbeBackend,
    before_probe_payload,
    transition_probe_payload,
)
from motif_transfer.neurosymbolic_ir import (
    ControlRoute,
    NeuralProbeSpec,
    NeuroSymbolicNode,
    NeuroSymbolicProgram,
    ProbeInputKind,
    RouteKind,
)
from motif_transfer.program_monitor import NeuroSymbolicMonitor


def main() -> None:
    model_hash = stable_hash("smoke-frozen-probe-model")
    guard = NeuralProbeSpec.create(
        input_kind=ProbeInputKind.BEFORE,
        model_artifact_sha256=model_hash,
        source_receipt_ids=(stable_hash("source-before-receipt"),),
        untrusted_description="there appears to be enough evidence to act",
    )
    effect = NeuralProbeSpec.create(
        input_kind=ProbeInputKind.TRANSITION,
        model_artifact_sha256=model_hash,
        source_receipt_ids=(stable_hash("source-transition-receipt"),),
        untrusted_description="the attempted transition appears complete",
    )
    program = NeuroSymbolicProgram.create(
        entry_node_id="verify_once",
        probes=(guard, effect),
        nodes=(NeuroSymbolicNode(
            "verify_once",
            guard.probe_id,
            effect.probe_id,
            ControlRoute(RouteKind.REPLAN),
            ControlRoute(RouteKind.TERMINATE),
            ControlRoute(RouteKind.REPLAN),
        ),),
        source_lineage=(stable_hash("source-episode"),),
    )
    before = Observation({"step": 0}, ("native-action",))
    after = Observation(
        {"step": 1}, (), terminal=True, official_success=True, score=1.0,
    )
    proposal = DecisionProposal("decision-proposal", "native-action")
    receipt = TransitionReceipt.create(before, proposal, after, reward=1.0)
    scores = {
        (guard.probe_id, stable_hash(before_probe_payload(before))): 0.95,
        (
            effect.probe_id,
            stable_hash(transition_probe_payload(
                before, proposal.action, after, receipt,
            )),
        ): 0.97,
    }
    monitor = NeuroSymbolicMonitor(
        program, FrozenTableProbeBackend(model_hash, scores),
    )
    before_decision = monitor.review_before_action(before)
    after_decision = monitor.observe_transition(
        before=before,
        action=proposal.action,
        after=after,
        receipt=receipt,
    )
    print(json.dumps({
        "program_id": program.program_id,
        "before_verdict": before_decision.verdict.value,
        "after_verdict": after_decision.verdict.value,
        "official_success": after.official_success,
        "terminated": monitor.terminated,
        "action_authority": proposal.agent_id,
        "probe_authority": before_decision.evaluation.authority,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
