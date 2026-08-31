from dataclasses import replace

import pytest

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
from motif_transfer.program_monitor import MonitorVerdict, NeuroSymbolicMonitor


MODEL_HASH = stable_hash("frozen-neural-probe-checkpoint")


def _probe(kind, description):
    return NeuralProbeSpec.create(
        input_kind=kind,
        model_artifact_sha256=MODEL_HASH,
        source_receipt_ids=(stable_hash(f"source-receipt-{description}"),),
        untrusted_description=description,
    )


def _program():
    guard_0 = _probe(ProbeInputKind.BEFORE, "untrusted guard zero")
    effect_0 = _probe(ProbeInputKind.TRANSITION, "untrusted effect zero")
    guard_1 = _probe(ProbeInputKind.BEFORE, "untrusted guard one")
    effect_1 = _probe(ProbeInputKind.TRANSITION, "untrusted effect one")
    replan = ControlRoute(RouteKind.REPLAN)
    program = NeuroSymbolicProgram.create(
        entry_node_id="n0",
        probes=(guard_0, effect_0, guard_1, effect_1),
        nodes=(
            NeuroSymbolicNode(
                "n0", guard_0.probe_id, effect_0.probe_id, replan,
                ControlRoute(RouteKind.NEXT_NODE, "n1"), replan,
                "untrusted gather role",
            ),
            NeuroSymbolicNode(
                "n1", guard_1.probe_id, effect_1.probe_id, replan,
                ControlRoute(RouteKind.TERMINATE), replan,
                "untrusted verify role",
            ),
        ),
        source_lineage=("source-episode-a", "source-episode-b"),
        untrusted_description="labels have no runtime authority",
    )
    return program, (guard_0, effect_0, guard_1, effect_1)


def _transition(before, after, action="native"):
    proposal = DecisionProposal("p", action)
    return TransitionReceipt.create(before, proposal, after, reward=after.score)


def _score(scores, probe, payload, value):
    scores[(probe.probe_id, stable_hash(payload))] = value


def test_supported_guards_and_effects_execute_symbolic_control_flow():
    program, (guard_0, effect_0, guard_1, effect_1) = _program()
    before_0 = Observation({"step": 0}, ("native",))
    after_0 = Observation({"step": 1}, ("native",))
    receipt_0 = _transition(before_0, after_0)
    before_1 = after_0
    after_1 = Observation(
        {"step": 2}, (), terminal=True, official_success=True, score=1.0,
    )
    receipt_1 = _transition(before_1, after_1)
    scores = {}
    _score(scores, guard_0, before_probe_payload(before_0), 0.95)
    _score(
        scores, effect_0,
        transition_probe_payload(before_0, "native", after_0, receipt_0), 0.9,
    )
    _score(scores, guard_1, before_probe_payload(before_1), 0.85)
    _score(
        scores, effect_1,
        transition_probe_payload(before_1, "native", after_1, receipt_1), 0.99,
    )
    monitor = NeuroSymbolicMonitor(
        program, FrozenTableProbeBackend(MODEL_HASH, scores),
    )

    first = monitor.review_before_action(before_0)
    assert first.verdict == MonitorVerdict.ADMIT
    assert "action" not in first.__dataclass_fields__
    advanced = monitor.observe_transition(
        before=before_0, action="native", after=after_0, receipt=receipt_0,
    )
    assert advanced.verdict == MonitorVerdict.CONTINUE
    assert advanced.next_node_id == "n1"
    assert monitor.review_before_action(before_1).verdict == MonitorVerdict.ADMIT
    terminal = monitor.observe_transition(
        before=before_1, action="native", after=after_1, receipt=receipt_1,
    )
    assert terminal.verdict == MonitorVerdict.TERMINATE
    assert monitor.terminated
    assert all(row.validate() for row in (
        first.evaluation, advanced.evaluation, terminal.evaluation,
    ))


def test_missing_guard_score_is_unknown_and_abstains_without_execution():
    program, _ = _program()
    monitor = NeuroSymbolicMonitor(
        program, FrozenTableProbeBackend(MODEL_HASH, {}),
    )
    decision = monitor.review_before_action(
        Observation({"out_of_scope": True}, ("native",)),
    )
    assert decision.verdict == MonitorVerdict.ABSTAIN
    assert decision.evaluation.score is None
    assert decision.evaluation.verdict.value == "UNKNOWN"
    assert monitor.current_node_id == "n0"
    assert monitor.suspended


def test_refuted_guard_requests_replan_and_does_not_admit_transition():
    program, (guard_0, _, _, _) = _program()
    before = Observation({"step": 0}, ("native",))
    scores = {}
    _score(scores, guard_0, before_probe_payload(before), 0.1)
    monitor = NeuroSymbolicMonitor(
        program, FrozenTableProbeBackend(MODEL_HASH, scores),
    )
    assert monitor.review_before_action(before).verdict == MonitorVerdict.REPLAN
    with pytest.raises(RuntimeError, match="without an admitted"):
        monitor.observe_transition(
            before=before,
            action="native",
            after=Observation({"step": 1}, ()),
            receipt=_transition(before, Observation({"step": 1}, ())),
        )


def test_unknown_effect_abstains_and_invalid_receipt_fails_closed():
    program, (guard_0, _, _, _) = _program()
    before = Observation({"step": 0}, ("native",))
    after = Observation({"step": 1}, ("native",))
    receipt = _transition(before, after)
    scores = {}
    _score(scores, guard_0, before_probe_payload(before), 0.9)
    monitor = NeuroSymbolicMonitor(
        program, FrozenTableProbeBackend(MODEL_HASH, scores),
    )
    assert monitor.review_before_action(before).verdict == MonitorVerdict.ADMIT
    invalid = replace(receipt, action="fabricated")
    with pytest.raises(ValueError, match="hash mismatch"):
        monitor.observe_transition(
            before=before, action="native", after=after, receipt=invalid,
        )

    # A fresh monitor with a valid receipt reaches the missing table entry,
    # which deterministically becomes UNKNOWN -> ABSTAIN.
    monitor = NeuroSymbolicMonitor(
        program, FrozenTableProbeBackend(MODEL_HASH, scores),
    )
    monitor.review_before_action(before)
    decision = monitor.observe_transition(
        before=before, action="native", after=after, receipt=receipt,
    )
    assert decision.verdict == MonitorVerdict.ABSTAIN
    assert decision.evaluation.verdict.value == "UNKNOWN"
    assert monitor.suspended


def test_neural_effect_cannot_replace_official_success():
    program, (guard_0, effect_0, guard_1, effect_1) = _program()
    before_0 = Observation({"step": 0}, ("native",))
    after_0 = Observation({"step": 1}, ("native",))
    receipt_0 = _transition(before_0, after_0)
    before_1 = after_0
    after_1 = Observation({"step": 2}, (), terminal=True, official_success=False)
    receipt_1 = _transition(before_1, after_1)
    scores = {}
    _score(scores, guard_0, before_probe_payload(before_0), 0.9)
    _score(
        scores, effect_0,
        transition_probe_payload(before_0, "native", after_0, receipt_0), 0.9,
    )
    _score(scores, guard_1, before_probe_payload(before_1), 0.9)
    _score(
        scores, effect_1,
        transition_probe_payload(before_1, "native", after_1, receipt_1), 0.99,
    )
    monitor = NeuroSymbolicMonitor(
        program, FrozenTableProbeBackend(MODEL_HASH, scores),
    )
    monitor.review_before_action(before_0)
    monitor.observe_transition(
        before=before_0, action="native", after=after_0, receipt=receipt_0,
    )
    monitor.review_before_action(before_1)
    decision = monitor.observe_transition(
        before=before_1, action="native", after=after_1, receipt=receipt_1,
    )
    assert decision.verdict == MonitorVerdict.ABSTAIN
    assert not monitor.terminated
    assert monitor.suspended
    assert "official success" in decision.reason


def test_probe_backend_is_bound_to_frozen_model_artifact():
    program, (guard_0, _, _, _) = _program()
    before = Observation({"step": 0}, ("native",))
    scores = {}
    _score(scores, guard_0, before_probe_payload(before), 0.9)
    monitor = NeuroSymbolicMonitor(
        program,
        FrozenTableProbeBackend(stable_hash("different-checkpoint"), scores),
    )
    with pytest.raises(ValueError, match="frozen model artifact"):
        monitor.review_before_action(before)
