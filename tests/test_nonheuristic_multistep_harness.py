from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from harness.capability_gaps import build_v3_implementation_report
from harness.binding_source_controls import apply_binding_source_control
from harness.candidate_set_runtime import (
    CandidateActionProposal,
    FrozenCandidateSetRuntime,
)
from harness.multistep_binding import (
    FrozenMultiStepArtifactStore,
    MultiStepBindingCandidate,
    MultiStepTargetAdmission,
    ProgramOrigin,
    TargetNodeBinding,
    TargetStepBinding,
    multistep_artifact_from_dict,
)
from harness.online_transfer_runtime import (
    NativeTransitionEvidence,
    OnlineTransferController,
    OnlineTransferState,
    OnlineTransferVerdict,
    online_transfer_log_from_dict,
)
from harness.online_rebinding import (
    ActionEvidenceContractProposal,
    CandidateActionEvidenceContract,
    CandidateRebinding,
    EvidenceQueryKind,
    OnlineRebindProposal,
    OnlineRebindingAdmission,
    build_action_contract_scope,
    compile_receipt_grounded_action_contract,
    build_rebind_scope,
    parse_online_rebind_reply,
    qualify_action_evidence_contract,
    qualified_online_rebind_from_dict,
    rebind_evidence_verification_from_dict,
    verify_action_evidence_contract,
    verify_rebind_evidence,
)
from harness.reasoning_event_log import (
    ReasoningEventKind,
    ReasoningEventRecorder,
    reasoning_event_log_from_dict,
    validate_reasoning_protocol,
)
from harness.source_conditioning_controls import (
    conditioning_control_receipt_from_dict,
    rotate_source_conditioning,
)
from harness.replay_fork import ReplayForkVerifier
from harness.skill_admission import TargetActionEvidence, TargetDemoReceipt
from skill_agents.control_hypotheses import (
    AgentControlHypothesis,
    ControlHypothesisValidator,
    HypothesisEdge,
    HypothesisNode,
    union_qualified_hypotheses,
)
from skill_agents.evidence_query import ContentAddressedEvidenceSession, EvidenceQuery
from skill_bank.trace_program_ir import (
    BackboneCoverage,
    ControlClaimKind,
    NativeTransitionReceipt,
    ObservedOrderEdge,
    TraceProgram,
)
from skill_bank.trace_program_validator import compile_observed_episode
from scripts.propose_alfworld_multistep_bindings_35b import (
    _parse as _parse_v3_binding,
    _source_graphs,
)


def _hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _trace() -> TraceProgram:
    transitions = [
        NativeTransitionReceipt(_hash("t0"), 0, _hash("s0"), _hash("s1"), _hash(["a"]), "a", 0, False),
        NativeTransitionReceipt(_hash("t1"), 1, _hash("s1"), _hash("s2"), _hash(["b"]), "b", 1, True),
    ]
    return TraceProgram(
        program_id="trace.p",
        game="g",
        episode_id="e",
        source_file_sha256=_hash("file"),
        transitions=transitions,
        observed_order=[ObservedOrderEdge(transitions[0].transition_id, transitions[1].transition_id)],
        coverage=BackboneCoverage(True, True, True, True, False, False, False),
    )


def _target_action(index, command, operator, types, before, after, success=False):
    return TargetActionEvidence(
        transition_index=index,
        action=command,
        operator=operator,
        arguments={},
        argument_types=types,
        before_admissible_actions=before,
        after_admissible_actions=after,
        admissible_actions_sha256=_hash(before),
        next_admissible_actions_sha256=_hash(after),
        state_sha256=_hash(f"s{index}"),
        next_state_sha256=_hash(f"s{index + 1}"),
        reward=float(success),
        terminated=success,
        truncated=False,
        official_success_after=success,
    )


def _demo() -> TargetDemoReceipt:
    a0 = ["go to fridge 1", "go to table 1"]
    a1 = ["open fridge 1", "look"]
    a2 = ["look"]
    return TargetDemoReceipt(
        demo_id="d", target_domain="alfworld", task_family="pick_and_place",
        split="train", episode_id="e", source_file_sha256=_hash("demo"),
        executor_kind="real", evaluator="alfworld_official", official_success=True,
        official_score=1.0,
        actions=[
            _target_action(0, "go to fridge 1", "GOTO", {"location": "location"}, a0, a1),
            _target_action(1, "open fridge 1", "OPEN", {"receptacle": "receptacle"}, a1, a2, True),
        ],
    )


def _binding(candidate_id="c1", origin=ProgramOrigin.SOURCE_HYPOTHESIS):
    source_contexts = (
        {
            "n0": {"observed_transitions": [{"action": "left"}], "incident_edges": []},
            "n1": {"observed_transitions": [{"action": "up"}], "incident_edges": []},
        }
        if origin == ProgramOrigin.SOURCE_HYPOTHESIS else {"n0": {}, "n1": {}}
    )
    return MultiStepBindingCandidate(
        candidate_id=candidate_id,
        origin=origin,
        proposal_source="agent-a",
        proposal_receipt_sha256=_hash("proposal-" + candidate_id),
        source_hypothesis_hash=(
            _hash("source-" + candidate_id)
            if origin == ProgramOrigin.SOURCE_HYPOTHESIS else None
        ),
        nodes=[
            TargetNodeBinding("n0", [
                TargetStepBinding(0, "GOTO", {"location": "location"}),
            ], source_contexts["n0"]),
            TargetNodeBinding("n1", [
                TargetStepBinding(1, "OPEN", {"receptacle": "receptacle"}),
            ], source_contexts["n1"]),
        ],
    )


def _admit(bindings):
    return MultiStepTargetAdmission().admit(
        candidates=bindings,
        demo=_demo(),
        known_proposal_receipt_hashes=[item.proposal_receipt_sha256 for item in bindings],
        known_source_hypothesis_nodes={
            item.source_hypothesis_hash: [node.node_id for node in item.nodes]
            for item in bindings if item.source_hypothesis_hash is not None
        },
        known_source_node_conditioning={
            item.source_hypothesis_hash: {
                node.node_id: dict(node.source_conditioning) for node in item.nodes
            }
            for item in bindings if item.source_hypothesis_hash is not None
        },
    )


def test_content_addressed_query_and_hypothesis_are_reference_only() -> None:
    program = _trace()
    session = ContentAddressedEvidenceSession(program)
    response = session.query(EvidenceQuery(
        "q", program.program_id, program.content_hash(),
        [item.transition_id for item in program.transitions],
    ))
    hypothesis = AgentControlHypothesis(
        hypothesis_id="h", program_id=program.program_id,
        program_hash=program.content_hash(), proposal_source="agent-a",
        evidence_response_hashes=[response.response_sha256],
        nodes=[HypothesisNode("n0", [program.transitions[0].transition_id]),
               HypothesisNode("n1", [program.transitions[1].transition_id])],
        edges=[HypothesisEdge("e0", "n0", "n1", ControlClaimKind.VERIFY, {"free": "untrusted"})],
    )
    qualified = ControlHypothesisValidator().validate(
        hypothesis, program=program, evidence_responses=[response]
    )
    assert qualified.status == "AGENT_HYPOTHESIS"
    assert union_qualified_hypotheses([qualified, qualified]) == (qualified,)

    partial_response = session.query(EvidenceQuery(
        "partial", program.program_id, program.content_hash(),
        [program.transitions[0].transition_id],
    ))
    not_shown = replace(
        hypothesis, evidence_response_hashes=[partial_response.response_sha256]
    )
    rejected = ControlHypothesisValidator().validate(
        not_shown, program=program, evidence_responses=[partial_response]
    )
    assert rejected.status == "REJECTED"
    assert "TRANSITION_REFERENCES_WERE_SHOWN" in rejected.failure_codes


def test_full_path_hypothesis_mechanically_binds_observed_fork_receipt() -> None:
    program = _trace()
    session = ContentAddressedEvidenceSession(program, native_evidence_by_transition_id={
        program.transitions[0].transition_id: {"available_actions": ["a"]},
        program.transitions[1].transition_id: {"available_actions": ["b", "c"]},
    })
    response = session.query(EvidenceQuery(
        "q", program.program_id, program.content_hash(),
        [item.transition_id for item in program.transitions],
    ))
    receipt_payload = {
        "intervention_id": "e.fork_step_1.alt_0",
        "seed": 2,
        "prefix_actions": ["a"],
        "expected_fork_state_sha256": program.transitions[1].state_sha256,
        "replayed_fork_state_sha256": program.transitions[1].state_sha256,
        "alternative_action": "c",
        "admissible_actions_sha256": _hash(["b", "c"]),
        "alternative_next_state_sha256": _hash("alternative"),
        "status": "INTERVENTION_OBSERVED",
        "failure_codes": [],
    }
    receipt = {**receipt_payload, "receipt_sha256": _hash(receipt_payload)}
    hypothesis = AgentControlHypothesis(
        hypothesis_id="full", program_id=program.program_id,
        program_hash=program.content_hash(), proposal_source="agent",
        evidence_response_hashes=[response.response_sha256],
        nodes=[
            HypothesisNode("n0", [program.transitions[0].transition_id]),
            HypothesisNode("n1", [program.transitions[1].transition_id]),
        ],
        edges=[HypothesisEdge(
            "e", "n0", "n1", ControlClaimKind.BRANCH, {},
            [receipt["receipt_sha256"]],
        )],
    )
    qualified = ControlHypothesisValidator().validate(
        hypothesis, program=program, evidence_responses=[response],
        intervention_receipts=[receipt], require_full_partition=True,
        require_multinode=True,
    )
    assert qualified.status == "AGENT_HYPOTHESIS"
    tampered = {**receipt, "alternative_action": "invented"}
    rejected = ControlHypothesisValidator().validate(
        hypothesis, program=program, evidence_responses=[response],
        intervention_receipts=[tampered], require_full_partition=True,
        require_multinode=True,
    )
    assert rejected.status == "REJECTED"
    assert "INTERVENTION_RECEIPT_HASHES_VALID" in rejected.failure_codes

    detached = AgentControlHypothesis(
        hypothesis_id="detached", program_id=program.program_id,
        program_hash=program.content_hash(), proposal_source="agent",
        evidence_response_hashes=[response.response_sha256],
        nodes=[
            HypothesisNode("n0", [program.transitions[0].transition_id]),
            HypothesisNode("n1", [program.transitions[1].transition_id]),
        ],
        edges=[HypothesisEdge(
            "e", "n0", "n0", ControlClaimKind.LOOP, {},
            [receipt["receipt_sha256"]],
        )],
    )
    # The fork is at transition 1, but this edge is anchored only to node 0.
    rejected = ControlHypothesisValidator().validate(
        detached, program=program, evidence_responses=[response],
        intervention_receipts=[receipt], require_full_partition=True,
        require_multinode=True,
    )
    assert "INTERVENTION_RECEIPTS_EDGE_ATTACHED" in rejected.failure_codes


def test_source_episode_query_exposes_only_hash_verified_native_evidence(tmp_path) -> None:
    source = tmp_path / "episode.json"
    payload = {
        "game_name": "opaque-game", "episode_id": "e",
        "experiences": [
            {"idx": 0, "raw_state": "s0", "raw_next_state": "s1",
             "available_actions": ["a"], "action": "a", "reward": 0, "done": False},
            {"idx": 1, "raw_state": "s1", "raw_next_state": "s2",
             "available_actions": ["b"], "action": "b", "reward": 1, "done": True},
        ],
    }
    source.write_text(json.dumps(payload))
    program = compile_observed_episode(source)
    session = ContentAddressedEvidenceSession.from_source_episode(program, source)
    response = session.query(EvidenceQuery(
        "q", program.program_id, program.content_hash(),
        [program.transitions[0].transition_id],
    ))
    assert response.transitions[0]["native_evidence"]["state"] == "s0"
    assert "legacy_skill_label" not in response.transitions[0]
    source.write_text(source.read_text().replace('"s0"', '"tampered"', 1))
    with pytest.raises(ValueError, match="source episode hash mismatch"):
        ContentAddressedEvidenceSession.from_source_episode(program, source)


def test_reasoning_event_log_is_hash_chained_and_requires_full_protocol() -> None:
    recorder = ReasoningEventRecorder("e")
    for kind in ReasoningEventKind:
        recorder.append(kind, {"kind": kind.value})
    recorder.validate_chain()
    assert validate_reasoning_protocol(recorder.events) == ()
    tampered = list(recorder.events)
    tampered[2] = replace(tampered[2], payload={"changed": True})
    with pytest.raises(ValueError, match="hash mismatch"):
        tampered[2].validate_hash()


def test_source_reasoning_protocol_requires_decision_attribution_events() -> None:
    recorder = ReasoningEventRecorder("source-e")
    core = (
        ReasoningEventKind.RESET,
        ReasoningEventKind.OBSERVATION,
        ReasoningEventKind.AGENT_PROPOSAL_SET,
        ReasoningEventKind.NATIVE_ADMISSIBILITY,
        ReasoningEventKind.AGENT_DECISION,
        ReasoningEventKind.ENVIRONMENT_STEP,
        ReasoningEventKind.NATIVE_DELTA,
        ReasoningEventKind.OFFICIAL_STOP,
    )
    for kind in core:
        recorder.append(kind, {})
    failures = validate_reasoning_protocol(recorder.events, profile="source_agent")
    assert failures == (
        "MISSING_EVENT_KIND:AGENT_RESPONSE",
        "MISSING_EVENT_KIND:PARSED_DECISION",
        "MISSING_EVENT_KIND:POLICY_TRANSFORM",
    )
    loaded = reasoning_event_log_from_dict(recorder.to_dict())
    assert loaded == recorder.events
    payload = recorder.to_dict()
    payload["events"][0]["payload"] = {"tampered": True}
    with pytest.raises(ValueError, match="log hash mismatch"):
        reasoning_event_log_from_dict(payload)


class _ReplayEnv:
    def __init__(self):
        self.value = 0

    def reset(self, *, seed: int):
        self.value = seed

    def state_receipt(self):
        return {"value": self.value}

    def admissible_actions(self):
        return ["inc", "double"]

    def step(self, action: str):
        self.value = self.value + 1 if action == "inc" else self.value * 2


def test_replay_to_fork_requires_exact_fork_state_before_intervention() -> None:
    env = _ReplayEnv()
    env.reset(seed=2)
    env.step("inc")
    expected = _hash(env.state_receipt())
    receipt = ReplayForkVerifier().run(
        _ReplayEnv(), intervention_id="i", seed=2, prefix_actions=["inc"],
        expected_fork_state_sha256=expected, alternative_action="double",
    )
    assert receipt.status == "INTERVENTION_OBSERVED"
    mismatch = ReplayForkVerifier().run(
        _ReplayEnv(), intervention_id="j", seed=2, prefix_actions=["inc"],
        expected_fork_state_sha256=_hash("wrong"), alternative_action="double",
    )
    assert mismatch.status == "REPLAY_MISMATCH"


def test_multistep_admission_retains_set_and_runtime_uses_common_action_set() -> None:
    artifact = _admit([_binding("a"), _binding("b")])
    assert len(artifact.candidates) == 2
    assert len(artifact.demo_transition_contract_receipts) == 2
    assert artifact.demo_transition_contract_receipts[0].supported_evidence == (
        "COMMAND_WAS_ADMISSIBLE",
        "OBSERVATION_CHANGED",
        "ADMISSIBLE_SET_CHANGED",
        "EXECUTED_ACTION_DISAPPEARED",
    )
    runtime = FrozenCandidateSetRuntime(artifact)
    active = runtime.active_source_conditioning()
    assert len(active) == 2
    assert {item["node_id"] for item in active} == {"n0"}
    assert all(item["demo_transition_receipt_sha256"] for item in active)
    assert all(item["target_transition_index"] == 0 for item in active)

    calls = []

    def same(candidates, allowed, scope_hash):
        calls.append((len(candidates), tuple(allowed)))
        return CandidateActionProposal(scope_hash, allowed[0])

    decision = runtime.choose(
        admissible=["go to fridge 1", "go to table 1"], actor=same
    )
    assert decision.status == "EXECUTE"
    assert calls == [(2, ("go to fridge 1", "go to table 1"))]
    runtime.observe_executed(decision, executed_command="go to fridge 1")
    assert set(runtime.cursors.values()) == {1}


def test_same_demo_fallback_alignment_requires_exact_demo_identity() -> None:
    runtime = FrozenCandidateSetRuntime(_admit([_binding("target")]))
    runtime.align_to_same_demo_prefix(
        demo_hash=runtime.artifact.demo_hash, completed_steps=1,
    )
    assert set(runtime.cursors.values()) == {1}
    with pytest.raises(ValueError, match="different demos"):
        runtime.align_to_same_demo_prefix(
            demo_hash=_hash("other-demo"), completed_steps=1,
        )
    with pytest.raises(ValueError, match="exceeds"):
        runtime.align_to_same_demo_prefix(
            demo_hash=runtime.artifact.demo_hash, completed_steps=3,
        )


def test_target_only_shadow_cursor_advances_only_on_compatible_exact_command() -> None:
    runtime = FrozenCandidateSetRuntime(_admit([_binding("target")]))
    assert runtime.observe_external_command_if_compatible(
        admissible=["go to fridge 1", "look"],
        executed_command="go to fridge 1",
    )
    assert set(runtime.cursors.values()) == {1}
    assert not runtime.observe_external_command_if_compatible(
        admissible=["look", "open fridge 1"],
        executed_command="look",
    )
    assert set(runtime.cursors.values()) == {1}


def _native_evidence(*, step=0, changed=False, command_was_admissible=True):
    before_observation = _hash("before")
    after_observation = _hash("after") if changed else before_observation
    return NativeTransitionEvidence.build(
        step=step,
        command="go to fridge 1",
        before_observation_sha256=before_observation,
        after_observation_sha256=after_observation,
        before_actions_sha256=_hash(["go to fridge 1"]),
        after_actions_sha256=_hash(["open fridge 1"]) if changed else _hash(["go to fridge 1"]),
        reward=0.0,
        official_success=False,
        command_was_admissible=command_was_admissible,
        executed_action_admissible_after=not changed,
        terminated=False,
        truncated=False,
    )


def test_online_controller_uses_native_receipts_and_never_claims_negative_transfer() -> None:
    controller = OnlineTransferController(
        max_rebind_requests=1, max_consecutive_no_delta=2,
    )
    first = controller.observe_source_transition(_native_evidence(step=0))
    assert first.verdict == OnlineTransferVerdict.REBIND
    assert first.reason == "MISSING_PREDECLARED_EVIDENCE_CONTRACT"
    assert controller.state == OnlineTransferState.REBIND_REQUIRED
    disabled = controller.fallback_to_target_only(
        step=2, reason="ONLINE_REBINDER_NOT_CONFIGURED",
    )
    assert disabled.verdict == OnlineTransferVerdict.SOURCE_DISABLED
    assert controller.state == OnlineTransferState.TARGET_ONLY
    payload = controller.to_dict()
    assert payload["claim_scope"] == "operational_online_evidence_not_negative_transfer"
    assert "NEGATIVE_TRANSFER" not in json.dumps(payload)
    assert online_transfer_log_from_dict(payload) == controller.events
    payload["state"] = OnlineTransferState.SOURCE_ACTIVE.value
    with pytest.raises(ValueError, match="log hash mismatch"):
        online_transfer_log_from_dict(payload)


def test_online_controller_requires_external_rebind_receipt_and_is_tamper_evident() -> None:
    controller = OnlineTransferController(max_rebind_requests=1)
    requested = controller.observe_source_abstention(
        step=0, reason="NO_COMMON_EXACT_COMMAND",
    )
    assert requested.verdict == OnlineTransferVerdict.REBIND
    with pytest.raises(ValueError, match="sha256"):
        controller.accept_rebind(
            step=0, binding_receipt_sha256="agent-said-so",
            known_binding_receipt_sha256s=[],
        )
    with pytest.raises(ValueError, match="sha256"):
        controller.accept_rebind(
            step=0, binding_receipt_sha256="z" * 64,
            known_binding_receipt_sha256s=[],
        )
    receipt_hash = _hash("binding-receipt")
    with pytest.raises(ValueError, match="admission registry"):
        controller.accept_rebind(
            step=0, binding_receipt_sha256=receipt_hash,
            known_binding_receipt_sha256s=[],
        )
    accepted = controller.accept_rebind(
        step=0, binding_receipt_sha256=receipt_hash,
        known_binding_receipt_sha256s=[receipt_hash],
    )
    assert accepted.verdict == OnlineTransferVerdict.REBIND_ACCEPTED
    continued = controller.observe_contract_transition(
        _native_evidence(step=0, changed=True),
        evidence_contract_satisfied=True,
        contract_kind="SOURCE_ACTION",
    )
    assert continued.verdict == OnlineTransferVerdict.NOT_REFUTED_LOCALLY
    controller.validate_chain()
    controller._events[0] = replace(  # noqa: SLF001 - intentional tamper test
        controller.events[0], reason="tampered",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        controller.validate_chain()


def _active_rebind_contexts():
    return (
        {
            "candidate_hash": _hash("candidate-a"),
            "source_hypothesis_hash": _hash("hypothesis-a"),
            "node_id": "n1",
            "target_transition_index": 0,
            "demo_transition_receipt_sha256": _hash("demo-transition-a"),
            "demo_supported_evidence": [
                "COMMAND_WAS_ADMISSIBLE", "OBSERVATION_CHANGED",
            ],
            "source_conditioning": {"observed_transitions": [{"action": "left"}]},
        },
        {
            "candidate_hash": _hash("candidate-b"),
            "source_hypothesis_hash": _hash("hypothesis-b"),
            "node_id": "n1",
            "target_transition_index": 0,
            "demo_transition_receipt_sha256": _hash("demo-transition-b"),
            "demo_supported_evidence": [
                "COMMAND_WAS_ADMISSIBLE", "OBSERVATION_CHANGED",
            ],
            "source_conditioning": {"observed_transitions": [{"action": "up"}]},
        },
    )


def test_randomized_conditioning_control_rotates_payload_only_and_is_receipted() -> None:
    contexts = _active_rebind_contexts()
    controlled, receipt = rotate_source_conditioning(contexts, seed=7, step=3)
    assert [row["candidate_hash"] for row in controlled] == [
        row["candidate_hash"] for row in contexts
    ]
    assert [row["source_hypothesis_hash"] for row in controlled] == [
        row["source_hypothesis_hash"] for row in contexts
    ]
    assert [row["node_id"] for row in controlled] == [row["node_id"] for row in contexts]
    assert controlled[0]["source_conditioning"] == contexts[1]["source_conditioning"]
    assert controlled[1]["source_conditioning"] == contexts[0]["source_conditioning"]
    assert receipt.original_contexts_sha256 != receipt.controlled_contexts_sha256
    receipt.validate_hash()
    assert conditioning_control_receipt_from_dict(receipt.to_dict()) == receipt
    with pytest.raises(ValueError, match="at least two"):
        rotate_source_conditioning(contexts[:1], seed=7, step=3)


def test_binding_source_control_runs_before_agent_and_renames_only_identities() -> None:
    graphs = ({
        "source_hypothesis_hash": _hash("hypothesis"),
        "nodes": [
            {"node_id": "n0", "observed_transitions": [{"action": "left"}]},
            {"node_id": "n1", "observed_transitions": [{"action": "up"}]},
        ],
        "edges": [{"source_node_id": "n0", "target_node_id": "n1", "kind": "SEQUENCE"}],
    },)
    renamed, receipt = apply_binding_source_control(
        graphs, treatment="renamed", seed=11,
    )
    assert renamed[0]["source_hypothesis_hash"] != graphs[0]["source_hypothesis_hash"]
    assert [node["observed_transitions"] for node in renamed[0]["nodes"]] == [
        node["observed_transitions"] for node in graphs[0]["nodes"]
    ]
    renamed_ids = {node["node_id"] for node in renamed[0]["nodes"]}
    assert renamed[0]["edges"][0]["source_node_id"] in renamed_ids
    assert renamed[0]["edges"][0]["target_node_id"] in renamed_ids
    assert receipt.input_graphs_sha256 != receipt.output_graphs_sha256
    receipt.validate_hash()

    empty, empty_receipt = apply_binding_source_control(
        (), treatment="empty", seed=11,
    )
    assert empty == ()
    empty_receipt.validate_hash()


def _online_rebind_proposal(*, action_numbers=(1, 2)):
    actions = ("go to drawer 1", "look")
    contexts = _active_rebind_contexts()
    scope = build_rebind_scope(
        artifact_hash=_hash("artifact"), demo_hash=_hash("demo"), step=3,
        observation_sha256=_hash("observation"), admissible_actions=actions,
        active_contexts=contexts,
    )
    proposal = OnlineRebindProposal(
        proposal_scope_hash=scope["proposal_scope_hash"],
        candidate_bindings=tuple(CandidateRebinding(
            candidate_hash=row["candidate_hash"],
            source_hypothesis_hash=row["source_hypothesis_hash"],
            node_id=row["node_id"],
            demo_transition_receipt_sha256=row[
                "demo_transition_receipt_sha256"
            ],
            allowed_action_numbers=action_numbers,
            expected_evidence=(
                EvidenceQueryKind.COMMAND_WAS_ADMISSIBLE,
                EvidenceQueryKind.OBSERVATION_CHANGED,
            ),
        ) for row in contexts),
        abstain=False,
    )
    return proposal, contexts, actions


def test_online_rebind_parser_and_admission_are_closed_schema() -> None:
    proposal, contexts, actions = _online_rebind_proposal()
    raw = json.dumps({
        "proposal_scope_hash": proposal.proposal_scope_hash,
        "candidate_bindings": [{
            **asdict(row),
            "expected_evidence": [item.value for item in row.expected_evidence],
        } for row in proposal.candidate_bindings],
        "abstain": False,
    })
    parsed = parse_online_rebind_reply(raw)
    admission = OnlineRebindingAdmission()
    qualified, failures = admission.admit(
        proposal=parsed, proposal_source="online-agent",
        proposal_receipt_sha256=_hash("raw-agent-reply"),
        artifact_hash=_hash("artifact"), demo_hash=_hash("demo"), step=3,
        observation_sha256=_hash("observation"), admissible_actions=actions,
        active_contexts=contexts,
    )
    assert failures == () and qualified is not None
    assert qualified.common_actions == actions
    assert qualified.status == "AGENT_HYPOTHESIS"
    qualified.validate_hash()
    assert admission.known_receipt_sha256s == (qualified.receipt_sha256,)
    assert qualified_online_rebind_from_dict(qualified.to_dict()) == qualified

    invented = replace(
        parsed,
        candidate_bindings=(
            replace(parsed.candidate_bindings[0], node_id="invented"),
            parsed.candidate_bindings[1],
        ),
    )
    rejected, failures = OnlineRebindingAdmission().admit(
        proposal=invented, proposal_source="online-agent",
        proposal_receipt_sha256=_hash("raw-agent-reply"),
        artifact_hash=_hash("artifact"), demo_hash=_hash("demo"), step=3,
        observation_sha256=_hash("observation"), admissible_actions=actions,
        active_contexts=contexts,
    )
    assert rejected is None and "CANDIDATE_IDENTITY_EXACT" in failures


def test_online_rebind_requires_common_action_and_checks_declared_evidence() -> None:
    proposal, contexts, actions = _online_rebind_proposal()
    disjoint = replace(
        proposal,
        candidate_bindings=(
            replace(proposal.candidate_bindings[0], allowed_action_numbers=(1,)),
            replace(proposal.candidate_bindings[1], allowed_action_numbers=(2,)),
        ),
    )
    rejected, failures = OnlineRebindingAdmission().admit(
        proposal=disjoint, proposal_source="online-agent",
        proposal_receipt_sha256=_hash("raw"), artifact_hash=_hash("artifact"),
        demo_hash=_hash("demo"), step=3, observation_sha256=_hash("observation"),
        admissible_actions=actions, active_contexts=contexts,
    )
    assert rejected is None and "COMMON_EXACT_ACTION_NONEMPTY" in failures

    qualified, failures = OnlineRebindingAdmission().admit(
        proposal=proposal, proposal_source="online-agent",
        proposal_receipt_sha256=_hash("raw"), artifact_hash=_hash("artifact"),
        demo_hash=_hash("demo"), step=3, observation_sha256=_hash("observation"),
        admissible_actions=actions, active_contexts=contexts,
    )
    assert qualified is not None and not failures
    verification = verify_rebind_evidence(
        binding=qualified, transition=_native_evidence(step=3, changed=True),
    )
    assert verification.all_satisfied
    verification.validate_hash()
    assert rebind_evidence_verification_from_dict(verification.to_dict()) == verification
    failed = verify_rebind_evidence(
        binding=qualified, transition=_native_evidence(step=3, changed=False),
    )
    assert not failed.all_satisfied

    tampered_proposal = replace(
        proposal,
        candidate_bindings=(
            proposal.candidate_bindings[0],
            replace(
                proposal.candidate_bindings[1],
                expected_evidence=(EvidenceQueryKind.POSITIVE_NATIVE_REWARD,),
            ),
        ),
    )
    tampered_binding, failures = OnlineRebindingAdmission().admit(
        proposal=tampered_proposal, proposal_source="online-agent",
        proposal_receipt_sha256=_hash("partial"), artifact_hash=_hash("artifact"),
        demo_hash=_hash("demo"), step=3, observation_sha256=_hash("observation"),
        admissible_actions=actions, active_contexts=contexts,
    )
    assert tampered_binding is None
    assert "CANDIDATE_IDENTITY_EXACT" in failures


def test_every_source_action_can_freeze_and_verify_a_preaction_contract() -> None:
    contexts = _active_rebind_contexts()
    actions = ("go to fridge 1", "look")
    scope = build_action_contract_scope(
        artifact_hash=_hash("artifact"), step=0, command=actions[0],
        observation_sha256=_hash("before"), admissible_actions=actions,
        active_contexts=contexts,
    )
    proposal = ActionEvidenceContractProposal(
        proposal_scope_hash=scope["proposal_scope_hash"],
        candidate_contracts=tuple(CandidateActionEvidenceContract(
            candidate_hash=row["candidate_hash"],
            source_hypothesis_hash=row["source_hypothesis_hash"],
            node_id=row["node_id"],
            demo_transition_receipt_sha256=row[
                "demo_transition_receipt_sha256"
            ],
            expected_evidence=tuple(
                EvidenceQueryKind(item)
                for item in row["demo_supported_evidence"]
            ),
        ) for row in contexts),
        abstain=False,
    )
    contract = qualify_action_evidence_contract(
        proposal=proposal, proposal_receipt_sha256=_hash("proposal"), scope=scope,
    )
    verification = verify_action_evidence_contract(
        contract=contract, transition=_native_evidence(step=0, changed=True),
    )
    assert verification.any_satisfied and verification.all_satisfied
    assert [row.all_satisfied for row in verification.candidate_results] == [True, True]
    contract.validate_hash()
    verification.validate_hash()

    compiled = compile_receipt_grounded_action_contract(scope=scope)
    assert compiled.proposal == proposal
    assert compiled.proposal_receipt_sha256 != _hash("proposal")
    compiled.validate_hash()

    ungrounded_scope = dict(scope)
    ungrounded_identities = [dict(row) for row in scope["active_candidate_identities"]]
    ungrounded_identities[0]["demo_transition_receipt_sha256"] = ""
    ungrounded_scope["active_candidate_identities"] = ungrounded_identities
    with pytest.raises(ValueError, match="GROUNDING_INCOMPLETE"):
        compile_receipt_grounded_action_contract(scope=ungrounded_scope)

    tampered = replace(
        proposal,
        candidate_contracts=(
            proposal.candidate_contracts[0],
            replace(
                proposal.candidate_contracts[1],
                expected_evidence=(EvidenceQueryKind.POSITIVE_NATIVE_REWARD,),
            ),
        ),
    )
    with pytest.raises(ValueError, match="IDENTITIES_NOT_EXACT"):
        qualify_action_evidence_contract(
            proposal=tampered, proposal_receipt_sha256=_hash("tampered"), scope=scope,
        )


def test_candidate_cursor_advances_only_after_contract_and_refutation_is_retained() -> None:
    runtime = FrozenCandidateSetRuntime(_admit([_binding("a"), _binding("b")]))

    def first(_candidates, allowed, scope_hash):
        return CandidateActionProposal(scope_hash, allowed[0])

    decision = runtime.choose(
        admissible=["go to fridge 1", "go to table 1"], actor=first,
    )
    runtime.observe_evidence_contract(
        decision, executed_command=decision.command,
        candidate_results={key: False for key in runtime.cursors},
        verification_receipt_sha256=_hash("verification"),
    )
    assert set(runtime.cursors.values()) == {0}
    assert set(runtime.statuses.values()) == {"REFUTED"}
    assert set(runtime.status_receipts.values()) == {_hash("verification")}
    after = runtime.choose(
        admissible=["go to fridge 1"],
        actor=lambda *_: pytest.fail("refuted candidates cannot reach Actor"),
    )
    assert after.reason == "NO_ACTIVE_CANDIDATES"


def test_candidate_contract_results_advance_and_refute_independently() -> None:
    runtime = FrozenCandidateSetRuntime(_admit([_binding("a"), _binding("b")]))

    def first(_candidates, allowed, scope_hash):
        return CandidateActionProposal(scope_hash, allowed[0])

    decision = runtime.choose(
        admissible=["go to fridge 1", "go to table 1"], actor=first,
    )
    hashes = list(runtime.cursors)
    runtime.observe_evidence_contract(
        decision, executed_command=decision.command,
        candidate_results={hashes[0]: True, hashes[1]: False},
        verification_receipt_sha256=_hash("partial-verification"),
    )
    assert runtime.cursors[hashes[0]] == 1
    assert runtime.cursors[hashes[1]] == 0
    assert runtime.statuses[hashes[0]] == "ACTIVE"
    assert runtime.statuses[hashes[1]] == "REFUTED"
    assert runtime.status_receipts[hashes[1]] == _hash("partial-verification")
    assert len(runtime.active_source_conditioning()) == 1


def test_runtime_rejects_empty_intersection_or_invalid_common_action() -> None:
    artifact = _admit([_binding("a"), _binding("b")])
    first, second = artifact.candidates
    incompatible_candidate = replace(
        second.candidate,
        nodes=(
            TargetNodeBinding("n0", [
                TargetStepBinding(0, "OPEN", {"receptacle": "receptacle"}),
            ]),
            *second.candidate.nodes[1:],
        ),
    )
    incompatible = replace(
        artifact,
        candidates=(first, replace(second, candidate=incompatible_candidate)),
    )
    runtime = FrozenCandidateSetRuntime(incompatible)

    decision = runtime.choose(
        admissible=["go to fridge 1", "open fridge 1"],
        actor=lambda *_: pytest.fail("empty intersection must abstain before Actor"),
    )
    assert decision.status == "ABSTAIN"
    assert decision.reason == "NO_COMMON_EXACT_COMMAND"

    runtime = FrozenCandidateSetRuntime(artifact)

    def invalid(_candidates, _allowed, scope_hash):
        return CandidateActionProposal(scope_hash, "invented")

    decision = runtime.choose(
        admissible=["go to fridge 1", "go to table 1"], actor=invalid
    )
    assert decision.reason == "INVALID_COMMON_COMMAND"
    assert set(runtime.cursors.values()) == {0}


def test_target_native_same_demo_has_no_source_identity() -> None:
    artifact = _admit([_binding("target", ProgramOrigin.TARGET_NATIVE_SAME_DEMO)])
    assert len(artifact.candidates) == 1
    assert artifact.candidates[0].candidate.source_hypothesis_hash is None
    assert artifact.semantic_alignment_claimed is False
    assert FrozenCandidateSetRuntime(artifact).active_source_conditioning() == ()


def test_v3_admission_rejects_tampered_source_conditioning() -> None:
    binding = _binding()
    tampered = replace(
        binding,
        nodes=(
            replace(
                binding.nodes[0],
                source_conditioning={
                    "observed_transitions": [{"action": "right"}],
                    "incident_edges": [],
                },
            ),
            binding.nodes[1],
        ),
    )
    artifact = MultiStepTargetAdmission().admit(
        candidates=[tampered], demo=_demo(),
        known_proposal_receipt_hashes=[binding.proposal_receipt_sha256],
        known_source_hypothesis_nodes={
            binding.source_hypothesis_hash: [node.node_id for node in binding.nodes]
        },
        known_source_node_conditioning={
            binding.source_hypothesis_hash: {
                node.node_id: dict(node.source_conditioning) for node in binding.nodes
            }
        },
    )
    assert "SOURCE_CONDITIONING_EXACT" in artifact.rejected_candidates[0]["failure_codes"]


def test_v3_artifact_is_immutable_and_tamper_evident(tmp_path) -> None:
    artifact = _admit([_binding()])
    path = FrozenMultiStepArtifactStore(tmp_path).freeze(artifact)
    payload = json.loads(path.read_text())
    assert multistep_artifact_from_dict(payload).artifact_hash == artifact.artifact_hash
    payload["task_family"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        multistep_artifact_from_dict(payload)


def test_v3_artifact_carries_prebinding_source_control_provenance() -> None:
    binding = _binding()
    artifact = MultiStepTargetAdmission().admit(
        candidates=[binding], demo=_demo(),
        known_proposal_receipt_hashes=[binding.proposal_receipt_sha256],
        known_source_hypothesis_nodes={
            binding.source_hypothesis_hash: [node.node_id for node in binding.nodes]
        },
        known_source_node_conditioning={
            binding.source_hypothesis_hash: {
                node.node_id: dict(node.source_conditioning) for node in binding.nodes
            }
        },
        source_treatment="correct",
        source_control_receipt_sha256=_hash("source-control"),
    )
    loaded = multistep_artifact_from_dict(artifact.to_dict())
    assert loaded.source_treatment == "correct"
    assert loaded.source_control_receipt_sha256 == _hash("source-control")


def test_v3_readiness_does_not_claim_unrun_production_work() -> None:
    report = build_v3_implementation_report()
    assert "multistep_target_binding_v3" in report["implemented"]
    assert "runtime_all_candidate_exact_action_consensus" in report["implemented"]
    assert "target_native_same_demo_baseline" in report["implemented"]
    assert "large_scale_2x4_v3_experiment" in report["gaps"]
    large_scale = next(
        item for item in report["capabilities"]
        if item["capability_id"] == "large_scale_2x4_v3_experiment"
    )
    assert large_scale["evidence"]["paired_development_pilot_completed"] is True
    assert large_scale["evidence"]["authorizes_large_scale_2x4"] is False
    assert large_scale["evidence"]["source_successes"] == 2
    assert large_scale["evidence"]["target_only_successes"] == 2
    assert large_scale["evidence"]["source_treatment_active"] is True


def test_v3_admission_rejects_fabricated_receipt_identity() -> None:
    binding = _binding()
    artifact = MultiStepTargetAdmission().admit(
        candidates=[binding], demo=_demo(),
        known_proposal_receipt_hashes=[],
        known_source_hypothesis_nodes={
            binding.source_hypothesis_hash: [node.node_id for node in binding.nodes]
        },
        known_source_node_conditioning={
            binding.source_hypothesis_hash: {
                node.node_id: dict(node.source_conditioning) for node in binding.nodes
            }
        },
    )
    assert artifact.candidates == ()
    assert artifact.rejected_candidates[0]["failure_codes"] == ["PROPOSAL_RECEIPT_KNOWN"]


def test_v3_admission_rejects_partial_target_demo_partition() -> None:
    binding = _binding()
    partial = replace(binding, nodes=(binding.nodes[0],))
    artifact = MultiStepTargetAdmission().admit(
        candidates=[partial], demo=_demo(),
        known_proposal_receipt_hashes=[partial.proposal_receipt_sha256],
        known_source_hypothesis_nodes={partial.source_hypothesis_hash: ["n0"]},
        known_source_node_conditioning={
            partial.source_hypothesis_hash: {
                "n0": dict(partial.nodes[0].source_conditioning),
            }
        },
    )
    assert artifact.candidates == ()
    assert "TARGET_INDICES_PARTITION_FULL_DEMO" in (
        artifact.rejected_candidates[0]["failure_codes"]
    )


def test_source_binding_parser_requires_exact_hypothesis_nodes() -> None:
    graphs = [{
        "source_hypothesis_hash": _hash("h"),
        "nodes": [{"node_id": "a"}, {"node_id": "b"}],
        "edges": [],
    }]
    raw = json.dumps({
        "source_hypothesis_hash": _hash("h"),
        "nodes": [
            {"node_id": "a", "target_transition_ids": ["target_t0"]},
            {"node_id": "b", "target_transition_ids": ["target_t1"]},
        ],
        "abstain": False,
    })
    parsed, abstained = _parse_v3_binding(
        raw, condition="source", graphs=graphs, demo=_demo()
    )
    assert not abstained and parsed["nodes"] == [("a", [0]), ("b", [1])]
    bad = raw.replace('"node_id": "b"', '"node_id": "invented"')
    with pytest.raises(ValueError, match="SOURCE_NODE_SEQUENCE_MISMATCH"):
        _parse_v3_binding(bad, condition="source", graphs=graphs, demo=_demo())


def _instrumented_source_artifact() -> dict:
    transition_ids = [_hash("source-t0"), _hash("source-t1")]
    hypothesis_hash = _hash("source-hypothesis")
    payload = {
        "schema_version": 1,
        "candidate_source": "independent_untrusted_agents",
        "full_observed_path_partition_required": True,
        "semantic_scoring": False,
        "ranking": False,
        "voting": False,
        "programs": [{
            "episode_id": "source-episode",
            "n_qualified": 1,
            "program": {
                "program_hash": _hash("source-program"),
                "transitions": [{
                    "transition_id": transition_ids[index],
                    "step_index": index,
                    "action": action,
                    "reward": float(index),
                    "done": index == 1,
                    "state_sha256": _hash(f"source-s{index}"),
                    "next_state_sha256": _hash(f"source-s{index + 1}"),
                } for index, action in enumerate(("left", "up"))],
            },
            "qualified_hypotheses": [{
                "hypothesis_hash": hypothesis_hash,
                "checks": {"full_partition": True, "agent_generated": True},
                "hypothesis": {
                    "nodes": [
                        {"node_id": "n0", "transition_ids": [transition_ids[0]]},
                        {"node_id": "n1", "transition_ids": [transition_ids[1]]},
                    ],
                    "edges": [{
                        "source_node_id": "n0", "target_node_id": "n1",
                        "kind": "SEQUENCE", "agent_claim": {"claim": "untrusted"},
                        "intervention_receipt_sha256s": [],
                    }],
                },
            }],
        }],
    }
    payload["artifact_sha256"] = _hash(payload)
    return payload


def test_new_instrumented_source_artifact_is_loaded_with_native_evidence() -> None:
    graphs = _source_graphs(_instrumented_source_artifact())
    assert len(graphs) == 1
    assert [node["node_id"] for node in graphs[0]["nodes"]] == ["n0", "n1"]
    first = graphs[0]["nodes"][0]["observed_transitions"][0]
    assert first["action"] == "left"
    assert "step_index" not in first
    assert "source_step_index" not in first
    assert graphs[0]["edges"][0]["status"] == "AGENT_HYPOTHESIS"


def test_source_artifact_loader_fails_closed_on_tamper_or_empty() -> None:
    tampered = _instrumented_source_artifact()
    tampered["programs"][0]["n_qualified"] = 0
    with pytest.raises(ValueError, match="SOURCE_ARTIFACT_HASH_MISMATCH"):
        _source_graphs(tampered)
    empty = _instrumented_source_artifact()
    empty["programs"] = []
    empty["artifact_sha256"] = _hash({key: value for key, value in empty.items() if key != "artifact_sha256"})
    with pytest.raises(ValueError, match="SOURCE_ARTIFACT_HAS_NO_QUALIFIED_HYPOTHESES"):
        _source_graphs(empty)
