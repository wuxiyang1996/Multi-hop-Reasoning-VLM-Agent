from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .contracts import stable_hash


class VTBReviewVerdict(str, Enum):
    ADMIT = "ADMIT"
    REPLAN = "REPLAN"
    ABSTAIN = "ABSTAIN"


class VTBVerificationVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class VTBToolProposal:
    proposal_id: str
    round_index: int
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    agent_id: str = "decision-agent"

    @classmethod
    def create(
        cls, round_index: int, call_id: str, tool_name: str, arguments: Mapping[str, Any]
    ) -> "VTBToolProposal":
        body = {
            "round_index": round_index,
            "call_id": call_id,
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "agent_id": "decision-agent",
        }
        return cls(proposal_id=stable_hash(body), **body)

    def validate_hash(self) -> bool:
        body = asdict(self)
        proposal_id = body.pop("proposal_id")
        return stable_hash(body) == proposal_id


@dataclass(frozen=True)
class VTBHarnessReview:
    verdict: VTBReviewVerdict
    reason: str
    expected_transition: str
    termination_test: str
    source_receipt_ids: tuple[str, ...] = ()
    live_receipt_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class VTBToolReceipt:
    receipt_id: str
    proposal_id: str
    round_index: int
    tool_contract_sha256: str
    observation_sha256: str
    success: bool
    output_paths_sha256: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        proposal: VTBToolProposal,
        *,
        tool_contract_sha256: str,
        observation: Mapping[str, Any],
        success: bool,
        output_paths_sha256: Sequence[str] = (),
    ) -> "VTBToolReceipt":
        body = {
            "proposal_id": proposal.proposal_id,
            "round_index": proposal.round_index,
            "tool_contract_sha256": tool_contract_sha256,
            "observation_sha256": stable_hash(observation),
            "success": bool(success),
            "output_paths_sha256": tuple(output_paths_sha256),
        }
        return cls(receipt_id=stable_hash(body), **body)

    def validate(self) -> bool:
        body = asdict(self)
        receipt_id = body.pop("receipt_id")
        return stable_hash(body) == receipt_id


@dataclass(frozen=True)
class VTBHarnessVerification:
    verdict: VTBVerificationVerdict
    reason: str
    receipt_id: str


class VTBInterpositionHarness:
    """Deterministic authority boundary for the online VTB Harness Agent."""

    def __init__(self, tools: Sequence[Mapping[str, Any]], tool_contract_sha256: str) -> None:
        self.tool_contract_sha256 = tool_contract_sha256
        self.schemas = {
            str(row["function"]["name"]): dict(row["function"]["parameters"])
            for row in tools
        }
        if not self.schemas:
            raise ValueError("VTB tool contract is empty")

    def validate_proposal(self, proposal: VTBToolProposal) -> None:
        if proposal.agent_id != "decision-agent":
            raise ValueError("only Decision Agent may construct target tool calls")
        if not proposal.validate_hash():
            raise ValueError("tool proposal hash mismatch")
        if proposal.tool_name not in self.schemas:
            raise ValueError(f"unknown official tool: {proposal.tool_name}")
        errors = sorted(
            Draft202012Validator(self.schemas[proposal.tool_name]).iter_errors(
                dict(proposal.arguments)
            ),
            key=lambda row: tuple(str(item) for item in row.path),
        )
        if errors:
            raise ValueError(f"invalid official tool arguments: {errors[0].message}")

    def validate_review(
        self,
        review: VTBHarnessReview,
        *,
        condition: str,
        known_source_receipts: set[str],
        known_live_receipts: set[str],
    ) -> None:
        if not review.reason.strip():
            raise ValueError("Harness review has no reason")
        if not review.expected_transition.strip() or not review.termination_test.strip():
            raise ValueError("Harness review must make a live-testable prediction and stop test")
        if not set(review.source_receipt_ids) <= known_source_receipts:
            raise ValueError("Harness review fabricated a source receipt")
        if not set(review.live_receipt_ids) <= known_live_receipts:
            raise ValueError("Harness review fabricated a live receipt")
        if condition in {"target_only", "generic_reasoning"} and review.source_receipt_ids:
            raise ValueError("non-source condition may not cite source evidence")
        if condition not in {"target_only", "generic_reasoning"} and not review.source_receipt_ids:
            raise ValueError("source treatment review must cite a frozen source receipt")

    def validate_receipt(self, proposal: VTBToolProposal, receipt: VTBToolReceipt) -> None:
        if not receipt.validate():
            raise ValueError("live VTB receipt hash mismatch")
        if receipt.proposal_id != proposal.proposal_id:
            raise ValueError("live VTB receipt belongs to a different proposal")
        if receipt.tool_contract_sha256 != self.tool_contract_sha256:
            raise ValueError("live VTB receipt used a different tool contract")

    @staticmethod
    def validate_verification(
        verification: VTBHarnessVerification, known_live_receipts: set[str]
    ) -> None:
        if verification.receipt_id not in known_live_receipts:
            raise ValueError("Harness verification fabricated a live receipt")
        if not verification.reason.strip():
            raise ValueError("Harness verification has no reason")


def parse_harness_review(raw: str | Mapping[str, Any]) -> VTBHarnessReview:
    value = json.loads(raw) if isinstance(raw, str) else dict(raw)
    allowed = {
        "verdict", "reason", "expected_transition", "termination_test",
        "source_receipt_ids", "live_receipt_ids",
    }
    if set(value) - allowed:
        raise ValueError(f"Harness review contains forbidden fields: {sorted(set(value) - allowed)}")
    return VTBHarnessReview(
        verdict=VTBReviewVerdict(str(value["verdict"])),
        reason=str(value.get("reason") or ""),
        expected_transition=str(value.get("expected_transition") or ""),
        termination_test=str(value.get("termination_test") or ""),
        source_receipt_ids=tuple(str(row) for row in value.get("source_receipt_ids") or ()),
        live_receipt_ids=tuple(str(row) for row in value.get("live_receipt_ids") or ()),
    )


def parse_harness_verification(raw: str | Mapping[str, Any]) -> VTBHarnessVerification:
    value = json.loads(raw) if isinstance(raw, str) else dict(raw)
    allowed = {"verdict", "reason", "receipt_id"}
    if set(value) - allowed:
        raise ValueError(
            f"Harness verification contains forbidden fields: {sorted(set(value) - allowed)}"
        )
    return VTBHarnessVerification(
        verdict=VTBVerificationVerdict(str(value["verdict"])),
        reason=str(value.get("reason") or ""),
        receipt_id=str(value.get("receipt_id") or ""),
    )


def pad_json_to_exact_tokens(payload: Mapping[str, Any], target_tokens: int) -> str:
    import tiktoken

    if target_tokens <= 0:
        raise ValueError("Harness input token target must be positive")
    value = dict(payload)
    value["_matched_padding"] = ""
    encoding = tiktoken.get_encoding("o200k_base")
    for _ in range(target_tokens * 3):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        count = len(encoding.encode(text))
        if count == target_tokens:
            return text
        if count > target_tokens:
            raise ValueError(
                f"Harness payload exceeds or skipped exact token target: {count}>{target_tokens}"
            )
        value["_matched_padding"] += " p"
    raise RuntimeError("could not construct exact-token Harness payload")
