"""Closed-schema Agent proposals for episode-local target rebinding.

The Agent may propose a tentative relation between currently active source
nodes and exact target-native actions.  The Harness validates identities and
later checks a predeclared native evidence contract.  Neither admission nor a
satisfied contract proves semantic equivalence or positive transfer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Sequence

from harness.online_transfer_runtime import NativeTransitionEvidence


_JSON_ONLY = re.compile(r"\A\s*\{.*\}\s*\Z", re.DOTALL)


def _hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _valid_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


class EvidenceQueryKind(str, Enum):
    COMMAND_WAS_ADMISSIBLE = "COMMAND_WAS_ADMISSIBLE"
    OBSERVATION_CHANGED = "OBSERVATION_CHANGED"
    ADMISSIBLE_SET_CHANGED = "ADMISSIBLE_SET_CHANGED"
    EXECUTED_ACTION_DISAPPEARED = "EXECUTED_ACTION_DISAPPEARED"
    POSITIVE_NATIVE_REWARD = "POSITIVE_NATIVE_REWARD"
    OFFICIAL_SUCCESS = "OFFICIAL_SUCCESS"


_INFORMATIVE_QUERIES = {
    EvidenceQueryKind.OBSERVATION_CHANGED,
    EvidenceQueryKind.ADMISSIBLE_SET_CHANGED,
    EvidenceQueryKind.EXECUTED_ACTION_DISAPPEARED,
    EvidenceQueryKind.POSITIVE_NATIVE_REWARD,
    EvidenceQueryKind.OFFICIAL_SUCCESS,
}


@dataclass(frozen=True)
class CandidateRebinding:
    candidate_hash: str
    source_hypothesis_hash: str
    node_id: str
    allowed_action_numbers: Sequence[int]
    expected_evidence: Sequence[EvidenceQueryKind]


@dataclass(frozen=True)
class OnlineRebindProposal:
    proposal_scope_hash: str
    candidate_bindings: Sequence[CandidateRebinding]
    abstain: bool


@dataclass(frozen=True)
class QualifiedOnlineRebind:
    proposal: OnlineRebindProposal
    proposal_source: str
    proposal_receipt_sha256: str
    artifact_hash: str
    demo_hash: str
    step: int
    observation_sha256: str
    native_actions_sha256: str
    common_actions: Sequence[str]
    checks: Mapping[str, bool]
    status: str
    receipt_sha256: str

    def unsigned_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        for row, binding in zip(
            payload["proposal"]["candidate_bindings"],
            self.proposal.candidate_bindings,
        ):
            row["expected_evidence"] = [
                item.value for item in binding.expected_evidence
            ]
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("online rebind receipt hash mismatch")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate_hash()
        payload = self.unsigned_payload()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class EvidenceQueryResult:
    kind: EvidenceQueryKind
    satisfied: bool


@dataclass(frozen=True)
class CandidateEvidenceVerification:
    candidate_hash: str
    source_hypothesis_hash: str
    node_id: str
    results: Sequence[EvidenceQueryResult]
    all_satisfied: bool


@dataclass(frozen=True)
class RebindEvidenceVerification:
    binding_receipt_sha256: str
    transition_receipt_sha256: str
    candidate_results: Sequence[CandidateEvidenceVerification]
    any_satisfied: bool
    all_satisfied: bool
    receipt_sha256: str

    def unsigned_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        for candidate_row, candidate in zip(
            payload["candidate_results"], self.candidate_results,
        ):
            for row, item in zip(candidate_row["results"], candidate.results):
                row["kind"] = item.kind.value
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("rebind evidence verification hash mismatch")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate_hash()
        payload = self.unsigned_payload()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class CandidateActionEvidenceContract:
    candidate_hash: str
    source_hypothesis_hash: str
    node_id: str
    expected_evidence: Sequence[EvidenceQueryKind]


@dataclass(frozen=True)
class ActionEvidenceContractProposal:
    proposal_scope_hash: str
    candidate_contracts: Sequence[CandidateActionEvidenceContract]
    abstain: bool


@dataclass(frozen=True)
class QualifiedActionEvidenceContract:
    proposal: ActionEvidenceContractProposal
    artifact_hash: str
    step: int
    command: str
    observation_sha256: str
    native_actions_sha256: str
    active_contexts_sha256: str
    proposal_receipt_sha256: str
    receipt_sha256: str

    def unsigned_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        for row, contract in zip(
            payload["proposal"]["candidate_contracts"],
            self.proposal.candidate_contracts,
        ):
            row["expected_evidence"] = [
                item.value for item in contract.expected_evidence
            ]
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("action evidence contract receipt hash mismatch")

    def to_dict(self) -> Mapping[str, Any]:
        payload = self.unsigned_payload()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class ActionEvidenceVerification:
    contract_receipt_sha256: str
    transition_receipt_sha256: str
    candidate_results: Sequence[CandidateEvidenceVerification]
    any_satisfied: bool
    all_satisfied: bool
    receipt_sha256: str

    def unsigned_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        for candidate_row, candidate in zip(
            payload["candidate_results"], self.candidate_results,
        ):
            for row, item in zip(candidate_row["results"], candidate.results):
                row["kind"] = item.kind.value
        payload.pop("receipt_sha256")
        return payload

    def validate_hash(self) -> None:
        if _hash(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("action evidence verification hash mismatch")

    def to_dict(self) -> Mapping[str, Any]:
        self.validate_hash()
        payload = self.unsigned_payload()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


def build_rebind_scope(
    *,
    artifact_hash: str,
    demo_hash: str,
    step: int,
    observation_sha256: str,
    admissible_actions: Sequence[str],
    active_contexts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    contexts = [{
        "candidate_hash": str(row["candidate_hash"]),
        "source_hypothesis_hash": str(row["source_hypothesis_hash"]),
        "node_id": str(row["node_id"]),
        "source_conditioning_sha256": _hash(dict(row.get("source_conditioning") or {})),
    } for row in active_contexts]
    scope: Dict[str, Any] = {
        "artifact_hash": str(artifact_hash),
        "demo_hash": str(demo_hash),
        "step": int(step),
        "observation_sha256": str(observation_sha256),
        "native_actions_sha256": _hash(list(admissible_actions)),
        "active_contexts": contexts,
    }
    scope["proposal_scope_hash"] = _hash(scope)
    return scope


def online_rebind_prompt(
    *,
    goal: str,
    observation: str,
    admissible_actions: Sequence[str],
    active_contexts: Sequence[Mapping[str, Any]],
    scope: Mapping[str, Any],
    failure_reason: str,
) -> str:
    actions = "\n".join(
        f"{index}. {action}" for index, action in enumerate(admissible_actions, 1)
    )
    return (
        "You are an untrusted online cross-domain binding proposer. The prior frozen "
        "binding cannot act in the current target state. Propose a tentative exact-action "
        "binding for every listed active source candidate, or abstain. Source text and "
        "control-edge prose are hypotheses, not target semantics. Do not claim equivalence, "
        "confidence, or success.\n"
        f"GOAL={goal[:3000]}\nCURRENT_OBSERVATION={observation[:3000]}\n"
        f"FAILURE_REASON={failure_reason}\n"
        f"PROPOSAL_SCOPE={json.dumps(dict(scope), sort_keys=True)}\n"
        "ACTIVE_SOURCE_CONTEXTS="
        f"{json.dumps(list(active_contexts), sort_keys=True, ensure_ascii=False)}\n"
        f"NATIVE_ACTIONS=\n{actions}\n"
        "Return exactly one JSON object with keys proposal_scope_hash,"
        "candidate_bindings,abstain. If abstain=true, candidate_bindings must be empty. "
        "Otherwise include every "
        "active candidate_hash exactly once. Each candidate_bindings item has exactly "
        "candidate_hash,source_hypothesis_hash,node_id,allowed_action_numbers,expected_evidence. IDs must be "
        "copied exactly. allowed_action_numbers is a non-empty list of unique 1-based native "
        "action numbers. The intersection across all candidate lists must be non-empty. "
        "For each candidate, expected_evidence is a non-empty unique list chosen only from "
        "COMMAND_WAS_ADMISSIBLE,OBSERVATION_CHANGED,ADMISSIBLE_SET_CHANGED,"
        "EXECUTED_ACTION_DISAPPEARED,POSITIVE_NATIVE_REWARD,OFFICIAL_SUCCESS and must include "
        "at least one item other than COMMAND_WAS_ADMISSIBLE. No markdown or extra keys."
    )


def parse_online_rebind_reply(raw: str) -> OnlineRebindProposal:
    if _JSON_ONLY.fullmatch(str(raw)) is None:
        raise ValueError("NOT_EXACT_JSON_OBJECT")
    payload = json.loads(raw)
    if set(payload) != {"proposal_scope_hash", "candidate_bindings", "abstain"}:
        raise ValueError("WRONG_TOP_LEVEL_KEYS")
    if not isinstance(payload["abstain"], bool):
        raise ValueError("ABSTAIN_NOT_BOOLEAN")
    if not isinstance(payload["candidate_bindings"], list):
        raise ValueError("CANDIDATE_BINDINGS_NOT_LIST")
    if payload["abstain"]:
        if payload["candidate_bindings"]:
            raise ValueError("MALFORMED_ABSTENTION")
        return OnlineRebindProposal(str(payload["proposal_scope_hash"]), (), True)
    bindings = []
    for row in payload["candidate_bindings"]:
        if not isinstance(row, dict) or set(row) != {
            "candidate_hash", "source_hypothesis_hash", "node_id", "allowed_action_numbers",
            "expected_evidence",
        }:
            raise ValueError("WRONG_CANDIDATE_BINDING_KEYS")
        numbers = row["allowed_action_numbers"]
        if (
            not isinstance(numbers, list)
            or not numbers
            or any(not isinstance(item, int) or isinstance(item, bool) for item in numbers)
            or len(numbers) != len(set(numbers))
        ):
            raise ValueError("INVALID_ALLOWED_ACTION_NUMBERS")
        if not isinstance(row["expected_evidence"], list):
            raise ValueError("EXPECTED_EVIDENCE_NOT_LIST")
        try:
            queries = tuple(
                EvidenceQueryKind(str(item)) for item in row["expected_evidence"]
            )
        except ValueError as exc:
            raise ValueError("UNKNOWN_EVIDENCE_QUERY") from exc
        if not queries or len(queries) != len(set(queries)):
            raise ValueError("EXPECTED_EVIDENCE_EMPTY_OR_DUPLICATE")
        if not set(queries).intersection(_INFORMATIVE_QUERIES):
            raise ValueError("EXPECTED_EVIDENCE_NOT_INFORMATIVE")
        bindings.append(CandidateRebinding(
            candidate_hash=str(row["candidate_hash"]),
            source_hypothesis_hash=str(row["source_hypothesis_hash"]),
            node_id=str(row["node_id"]),
            allowed_action_numbers=tuple(numbers),
            expected_evidence=queries,
        ))
    return OnlineRebindProposal(
        proposal_scope_hash=str(payload["proposal_scope_hash"]),
        candidate_bindings=tuple(bindings),
        abstain=False,
    )


class OnlineRebindingAdmission:
    """Exact identity/action checks over an Agent-proposed tentative rebind."""

    def __init__(self) -> None:
        self._known_receipt_sha256s: set[str] = set()

    @property
    def known_receipt_sha256s(self) -> Sequence[str]:
        return tuple(sorted(self._known_receipt_sha256s))

    def admit(
        self,
        *,
        proposal: OnlineRebindProposal,
        proposal_source: str,
        proposal_receipt_sha256: str,
        artifact_hash: str,
        demo_hash: str,
        step: int,
        observation_sha256: str,
        admissible_actions: Sequence[str],
        active_contexts: Sequence[Mapping[str, Any]],
    ) -> tuple[QualifiedOnlineRebind | None, Sequence[str]]:
        scope = build_rebind_scope(
            artifact_hash=artifact_hash,
            demo_hash=demo_hash,
            step=step,
            observation_sha256=observation_sha256,
            admissible_actions=admissible_actions,
            active_contexts=active_contexts,
        )
        expected = {
            str(row["candidate_hash"]): (
                str(row["source_hypothesis_hash"]), str(row["node_id"]),
            )
            for row in active_contexts
        }
        proposed = {row.candidate_hash: row for row in proposal.candidate_bindings}
        numbers_in_range = all(
            all(1 <= number <= len(admissible_actions) for number in row.allowed_action_numbers)
            for row in proposal.candidate_bindings
        )
        identity_exact = (
            set(proposed) == set(expected)
            and len(proposed) == len(proposal.candidate_bindings)
            and all(
                (row.source_hypothesis_hash, row.node_id) == expected.get(row.candidate_hash)
                for row in proposal.candidate_bindings
            )
        )
        action_sets = [
            {admissible_actions[number - 1] for number in row.allowed_action_numbers}
            for row in proposal.candidate_bindings
        ] if numbers_in_range else []
        common = set.intersection(*action_sets) if action_sets else set()
        common_ordered = tuple(action for action in admissible_actions if action in common)
        checks = {
            "not_abstain": not proposal.abstain,
            "proposal_scope_exact": proposal.proposal_scope_hash == scope["proposal_scope_hash"],
            "proposal_receipt_sha256_valid": _valid_sha256(proposal_receipt_sha256),
            "active_contexts_nonempty": bool(active_contexts),
            "active_candidate_identities_unique": len(expected) == len(active_contexts),
            "candidate_identity_exact": identity_exact,
            "allowed_action_numbers_in_range": numbers_in_range,
            "common_exact_action_nonempty": bool(common_ordered),
            "per_candidate_expected_evidence_nonempty": all(
                row.expected_evidence for row in proposal.candidate_bindings
            ),
            "per_candidate_expected_evidence_informative": all(
                set(row.expected_evidence).intersection(_INFORMATIVE_QUERIES)
                for row in proposal.candidate_bindings
            ),
        }
        failures = tuple(name.upper() for name, passed in checks.items() if not passed)
        if failures:
            return None, failures
        unsigned = {
            "proposal": asdict(proposal),
            "proposal_source": str(proposal_source),
            "proposal_receipt_sha256": str(proposal_receipt_sha256),
            "artifact_hash": str(artifact_hash),
            "demo_hash": str(demo_hash),
            "step": int(step),
            "observation_sha256": str(observation_sha256),
            "native_actions_sha256": str(scope["native_actions_sha256"]),
            "common_actions": common_ordered,
            "checks": checks,
            "status": "AGENT_HYPOTHESIS",
        }
        for row, binding in zip(
            unsigned["proposal"]["candidate_bindings"],
            proposal.candidate_bindings,
        ):
            row["expected_evidence"] = [
                item.value for item in binding.expected_evidence
            ]
        qualified = QualifiedOnlineRebind(
            proposal=proposal,
            proposal_source=unsigned["proposal_source"],
            proposal_receipt_sha256=unsigned["proposal_receipt_sha256"],
            artifact_hash=unsigned["artifact_hash"],
            demo_hash=unsigned["demo_hash"],
            step=unsigned["step"],
            observation_sha256=unsigned["observation_sha256"],
            native_actions_sha256=unsigned["native_actions_sha256"],
            common_actions=common_ordered,
            checks=checks,
            status="AGENT_HYPOTHESIS",
            receipt_sha256=_hash(unsigned),
        )
        qualified.validate_hash()
        self._known_receipt_sha256s.add(qualified.receipt_sha256)
        return qualified, ()


def qualified_online_rebind_from_dict(
    payload: Mapping[str, Any],
) -> QualifiedOnlineRebind:
    proposal_row = dict(payload["proposal"])
    proposal = OnlineRebindProposal(
        proposal_scope_hash=str(proposal_row["proposal_scope_hash"]),
        candidate_bindings=tuple(CandidateRebinding(
            candidate_hash=str(row["candidate_hash"]),
            source_hypothesis_hash=str(row["source_hypothesis_hash"]),
            node_id=str(row["node_id"]),
            allowed_action_numbers=tuple(int(item) for item in row["allowed_action_numbers"]),
            expected_evidence=tuple(
                EvidenceQueryKind(str(item)) for item in row["expected_evidence"]
            ),
        ) for row in proposal_row["candidate_bindings"]),
        abstain=bool(proposal_row["abstain"]),
    )
    qualified = QualifiedOnlineRebind(
        proposal=proposal,
        proposal_source=str(payload["proposal_source"]),
        proposal_receipt_sha256=str(payload["proposal_receipt_sha256"]),
        artifact_hash=str(payload["artifact_hash"]),
        demo_hash=str(payload["demo_hash"]),
        step=int(payload["step"]),
        observation_sha256=str(payload["observation_sha256"]),
        native_actions_sha256=str(payload["native_actions_sha256"]),
        common_actions=tuple(str(item) for item in payload["common_actions"]),
        checks={str(key): bool(value) for key, value in payload["checks"].items()},
        status=str(payload["status"]),
        receipt_sha256=str(payload["receipt_sha256"]),
    )
    qualified.validate_hash()
    return qualified


def _evaluate_evidence_queries(
    queries: Sequence[EvidenceQueryKind], transition: NativeTransitionEvidence,
) -> tuple[EvidenceQueryResult, ...]:
    results = []
    for kind in queries:
        if kind == EvidenceQueryKind.COMMAND_WAS_ADMISSIBLE:
            satisfied = transition.command_was_admissible
        elif kind == EvidenceQueryKind.OBSERVATION_CHANGED:
            satisfied = (
                transition.before_observation_sha256 != transition.after_observation_sha256
            )
        elif kind == EvidenceQueryKind.ADMISSIBLE_SET_CHANGED:
            satisfied = transition.before_actions_sha256 != transition.after_actions_sha256
        elif kind == EvidenceQueryKind.EXECUTED_ACTION_DISAPPEARED:
            satisfied = not transition.executed_action_admissible_after
        elif kind == EvidenceQueryKind.POSITIVE_NATIVE_REWARD:
            satisfied = transition.reward > 0.0
        elif kind == EvidenceQueryKind.OFFICIAL_SUCCESS:
            satisfied = transition.official_success
        else:  # pragma: no cover - exhaustive Enum guard
            raise ValueError(f"unsupported evidence query: {kind}")
        results.append(EvidenceQueryResult(kind, satisfied))
    return tuple(results)


def _candidate_verification(
    *,
    candidate_hash: str,
    source_hypothesis_hash: str,
    node_id: str,
    queries: Sequence[EvidenceQueryKind],
    transition: NativeTransitionEvidence,
) -> CandidateEvidenceVerification:
    results = _evaluate_evidence_queries(queries, transition)
    return CandidateEvidenceVerification(
        candidate_hash=str(candidate_hash),
        source_hypothesis_hash=str(source_hypothesis_hash),
        node_id=str(node_id),
        results=results,
        all_satisfied=bool(results) and all(item.satisfied for item in results),
    )


def _candidate_verification_dict(
    item: CandidateEvidenceVerification,
) -> Mapping[str, Any]:
    return {
        "candidate_hash": item.candidate_hash,
        "source_hypothesis_hash": item.source_hypothesis_hash,
        "node_id": item.node_id,
        "results": [
            {"kind": row.kind.value, "satisfied": row.satisfied}
            for row in item.results
        ],
        "all_satisfied": item.all_satisfied,
    }


def verify_rebind_evidence(
    *,
    binding: QualifiedOnlineRebind,
    transition: NativeTransitionEvidence,
) -> RebindEvidenceVerification:
    binding.validate_hash()
    transition.validate_hash()
    candidate_results = tuple(
        _candidate_verification(
            candidate_hash=item.candidate_hash,
            source_hypothesis_hash=item.source_hypothesis_hash,
            node_id=item.node_id,
            queries=item.expected_evidence,
            transition=transition,
        )
        for item in binding.proposal.candidate_bindings
    )
    any_satisfied = bool(candidate_results) and any(
        item.all_satisfied for item in candidate_results
    )
    all_satisfied = bool(candidate_results) and all(
        item.all_satisfied for item in candidate_results
    )
    unsigned = {
        "binding_receipt_sha256": binding.receipt_sha256,
        "transition_receipt_sha256": transition.receipt_sha256,
        "candidate_results": [_candidate_verification_dict(item) for item in candidate_results],
        "any_satisfied": any_satisfied,
        "all_satisfied": all_satisfied,
    }
    receipt = RebindEvidenceVerification(
        binding_receipt_sha256=binding.receipt_sha256,
        transition_receipt_sha256=transition.receipt_sha256,
        candidate_results=candidate_results,
        any_satisfied=any_satisfied,
        all_satisfied=all_satisfied,
        receipt_sha256=_hash(unsigned),
    )
    receipt.validate_hash()
    return receipt


def build_action_contract_scope(
    *,
    artifact_hash: str,
    step: int,
    command: str,
    observation_sha256: str,
    admissible_actions: Sequence[str],
    active_contexts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    identities = [{
        "candidate_hash": str(row["candidate_hash"]),
        "source_hypothesis_hash": str(row["source_hypothesis_hash"]),
        "node_id": str(row["node_id"]),
    } for row in active_contexts]
    scope: Dict[str, Any] = {
        "artifact_hash": str(artifact_hash),
        "step": int(step),
        "command": str(command),
        "observation_sha256": str(observation_sha256),
        "native_actions_sha256": _hash(list(admissible_actions)),
        "active_contexts_sha256": _hash(list(active_contexts)),
        "active_candidate_identities": identities,
    }
    scope["proposal_scope_hash"] = _hash(scope)
    return scope


def action_evidence_contract_prompt(
    *,
    goal: str,
    observation: str,
    command: str,
    admissible_actions: Sequence[str],
    active_contexts: Sequence[Mapping[str, Any]],
    scope: Mapping[str, Any],
) -> str:
    return (
        "You are an untrusted evidence-contract proposer. Before the chosen target-native "
        "action executes, declare only mechanically observable evidence whose absence would "
        "refute continued use of the active source hypotheses. Source prose is not proof. "
        "Do not claim semantic equivalence, progress, confidence, or success. Abstain when no "
        "listed query is justified.\n"
        f"GOAL={goal[:3000]}\nCURRENT_OBSERVATION={observation[:3000]}\n"
        f"CHOSEN_COMMAND={command}\n"
        f"NATIVE_ACTIONS={json.dumps(list(admissible_actions), ensure_ascii=False)}\n"
        f"ACTIVE_SOURCE_CONTEXTS={json.dumps(list(active_contexts), sort_keys=True, ensure_ascii=False)}\n"
        f"PROPOSAL_SCOPE={json.dumps(dict(scope), sort_keys=True)}\n"
        "Return exactly one JSON object with keys proposal_scope_hash,candidate_contracts,"
        "abstain. Include every active candidate exactly once. Each candidate_contracts item "
        "has exactly candidate_hash,source_hypothesis_hash,node_id,expected_evidence; copy all "
        "identities exactly. Each expected_evidence must be a non-empty unique list chosen only from "
        "COMMAND_WAS_ADMISSIBLE,OBSERVATION_CHANGED,ADMISSIBLE_SET_CHANGED,"
        "EXECUTED_ACTION_DISAPPEARED,POSITIVE_NATIVE_REWARD,OFFICIAL_SUCCESS and include at "
        "least one item other than COMMAND_WAS_ADMISSIBLE. If abstain=true, candidate_contracts "
        "must be empty. No markdown or extra keys."
    )


def parse_action_evidence_contract_reply(raw: str) -> ActionEvidenceContractProposal:
    if _JSON_ONLY.fullmatch(str(raw)) is None:
        raise ValueError("NOT_EXACT_JSON_OBJECT")
    payload = json.loads(raw)
    if set(payload) != {"proposal_scope_hash", "candidate_contracts", "abstain"}:
        raise ValueError("WRONG_TOP_LEVEL_KEYS")
    if not isinstance(payload["abstain"], bool):
        raise ValueError("ABSTAIN_NOT_BOOLEAN")
    if not isinstance(payload["candidate_contracts"], list):
        raise ValueError("CANDIDATE_CONTRACTS_NOT_LIST")
    if payload["abstain"]:
        if payload["candidate_contracts"]:
            raise ValueError("MALFORMED_ABSTENTION")
        return ActionEvidenceContractProposal(
            str(payload["proposal_scope_hash"]), (), True,
        )
    contracts = []
    for row in payload["candidate_contracts"]:
        if not isinstance(row, dict) or set(row) != {
            "candidate_hash", "source_hypothesis_hash", "node_id", "expected_evidence",
        }:
            raise ValueError("WRONG_CANDIDATE_CONTRACT_KEYS")
        if not isinstance(row["expected_evidence"], list):
            raise ValueError("EXPECTED_EVIDENCE_NOT_LIST")
        try:
            queries = tuple(
                EvidenceQueryKind(str(item)) for item in row["expected_evidence"]
            )
        except ValueError as exc:
            raise ValueError("UNKNOWN_EVIDENCE_QUERY") from exc
        if not queries or len(queries) != len(set(queries)):
            raise ValueError("EXPECTED_EVIDENCE_EMPTY_OR_DUPLICATE")
        if not set(queries).intersection(_INFORMATIVE_QUERIES):
            raise ValueError("EXPECTED_EVIDENCE_NOT_INFORMATIVE")
        contracts.append(CandidateActionEvidenceContract(
            candidate_hash=str(row["candidate_hash"]),
            source_hypothesis_hash=str(row["source_hypothesis_hash"]),
            node_id=str(row["node_id"]),
            expected_evidence=queries,
        ))
    return ActionEvidenceContractProposal(
        str(payload["proposal_scope_hash"]), tuple(contracts), False,
    )


def qualify_action_evidence_contract(
    *,
    proposal: ActionEvidenceContractProposal,
    proposal_receipt_sha256: str,
    scope: Mapping[str, Any],
) -> QualifiedActionEvidenceContract:
    if proposal.abstain:
        raise ValueError("ACTION_EVIDENCE_CONTRACT_ABSTAINED")
    if proposal.proposal_scope_hash != scope["proposal_scope_hash"]:
        raise ValueError("PROPOSAL_SCOPE_MISMATCH")
    if not _valid_sha256(proposal_receipt_sha256):
        raise ValueError("PROPOSAL_RECEIPT_NOT_SHA256")
    expected = {
        str(row["candidate_hash"]): (
            str(row["source_hypothesis_hash"]), str(row["node_id"]),
        )
        for row in scope["active_candidate_identities"]
    }
    proposed = {row.candidate_hash: row for row in proposal.candidate_contracts}
    if (
        set(proposed) != set(expected)
        or len(proposed) != len(proposal.candidate_contracts)
        or any(
            (row.source_hypothesis_hash, row.node_id) != expected.get(row.candidate_hash)
            for row in proposal.candidate_contracts
        )
    ):
        raise ValueError("CANDIDATE_CONTRACT_IDENTITIES_NOT_EXACT")
    if any(
        not row.expected_evidence
        or not set(row.expected_evidence).intersection(_INFORMATIVE_QUERIES)
        for row in proposal.candidate_contracts
    ):
        raise ValueError("EXPECTED_EVIDENCE_NOT_INFORMATIVE")
    unsigned = {
        "proposal": {
            "proposal_scope_hash": proposal.proposal_scope_hash,
            "candidate_contracts": [{
                "candidate_hash": row.candidate_hash,
                "source_hypothesis_hash": row.source_hypothesis_hash,
                "node_id": row.node_id,
                "expected_evidence": [item.value for item in row.expected_evidence],
            } for row in proposal.candidate_contracts],
            "abstain": proposal.abstain,
        },
        "artifact_hash": str(scope["artifact_hash"]),
        "step": int(scope["step"]),
        "command": str(scope["command"]),
        "observation_sha256": str(scope["observation_sha256"]),
        "native_actions_sha256": str(scope["native_actions_sha256"]),
        "active_contexts_sha256": str(scope["active_contexts_sha256"]),
        "proposal_receipt_sha256": str(proposal_receipt_sha256),
    }
    contract = QualifiedActionEvidenceContract(
        proposal=proposal,
        artifact_hash=unsigned["artifact_hash"],
        step=unsigned["step"],
        command=unsigned["command"],
        observation_sha256=unsigned["observation_sha256"],
        native_actions_sha256=unsigned["native_actions_sha256"],
        active_contexts_sha256=unsigned["active_contexts_sha256"],
        proposal_receipt_sha256=unsigned["proposal_receipt_sha256"],
        receipt_sha256=_hash(unsigned),
    )
    contract.validate_hash()
    return contract


def verify_action_evidence_contract(
    *,
    contract: QualifiedActionEvidenceContract,
    transition: NativeTransitionEvidence,
) -> ActionEvidenceVerification:
    contract.validate_hash()
    transition.validate_hash()
    if contract.command != transition.command or contract.step != transition.step:
        raise ValueError("contract/transition identity mismatch")
    candidate_results = tuple(
        _candidate_verification(
            candidate_hash=item.candidate_hash,
            source_hypothesis_hash=item.source_hypothesis_hash,
            node_id=item.node_id,
            queries=item.expected_evidence,
            transition=transition,
        )
        for item in contract.proposal.candidate_contracts
    )
    any_satisfied = bool(candidate_results) and any(
        item.all_satisfied for item in candidate_results
    )
    all_satisfied = bool(candidate_results) and all(
        item.all_satisfied for item in candidate_results
    )
    unsigned = {
        "contract_receipt_sha256": contract.receipt_sha256,
        "transition_receipt_sha256": transition.receipt_sha256,
        "candidate_results": [_candidate_verification_dict(item) for item in candidate_results],
        "any_satisfied": any_satisfied,
        "all_satisfied": all_satisfied,
    }
    verification = ActionEvidenceVerification(
        contract_receipt_sha256=contract.receipt_sha256,
        transition_receipt_sha256=transition.receipt_sha256,
        candidate_results=candidate_results,
        any_satisfied=any_satisfied,
        all_satisfied=all_satisfied,
        receipt_sha256=_hash(unsigned),
    )
    verification.validate_hash()
    return verification


def rebind_evidence_verification_from_dict(
    payload: Mapping[str, Any],
) -> RebindEvidenceVerification:
    candidate_results = tuple(CandidateEvidenceVerification(
        candidate_hash=str(candidate["candidate_hash"]),
        source_hypothesis_hash=str(candidate["source_hypothesis_hash"]),
        node_id=str(candidate["node_id"]),
        results=tuple(EvidenceQueryResult(
            EvidenceQueryKind(str(row["kind"])), bool(row["satisfied"]),
        ) for row in candidate["results"]),
        all_satisfied=bool(candidate["all_satisfied"]),
    ) for candidate in payload["candidate_results"])
    verification = RebindEvidenceVerification(
        binding_receipt_sha256=str(payload["binding_receipt_sha256"]),
        transition_receipt_sha256=str(payload["transition_receipt_sha256"]),
        candidate_results=candidate_results,
        any_satisfied=bool(payload["any_satisfied"]),
        all_satisfied=bool(payload["all_satisfied"]),
        receipt_sha256=str(payload["receipt_sha256"]),
    )
    verification.validate_hash()
    expected_any = bool(candidate_results) and any(
        row.all_satisfied for row in candidate_results
    )
    expected_all = bool(candidate_results) and all(
        row.all_satisfied for row in candidate_results
    )
    if (
        verification.any_satisfied != expected_any
        or verification.all_satisfied != expected_all
        or any(
            row.all_satisfied != (
                bool(row.results) and all(item.satisfied for item in row.results)
            )
            for row in candidate_results
        )
    ):
        raise ValueError("rebind evidence aggregate verdict mismatch")
    return verification


__all__ = [
    "ActionEvidenceContractProposal",
    "ActionEvidenceVerification",
    "CandidateActionEvidenceContract",
    "CandidateEvidenceVerification",
    "CandidateRebinding",
    "EvidenceQueryKind",
    "EvidenceQueryResult",
    "OnlineRebindProposal",
    "OnlineRebindingAdmission",
    "QualifiedActionEvidenceContract",
    "QualifiedOnlineRebind",
    "RebindEvidenceVerification",
    "action_evidence_contract_prompt",
    "build_action_contract_scope",
    "build_rebind_scope",
    "online_rebind_prompt",
    "parse_action_evidence_contract_reply",
    "parse_online_rebind_reply",
    "qualify_action_evidence_contract",
    "qualified_online_rebind_from_dict",
    "rebind_evidence_verification_from_dict",
    "verify_action_evidence_contract",
    "verify_rebind_evidence",
]
