"""Matched video-transfer measurement with an explicit grounding boundary.

The primary purpose of this module is to keep two scientifically different
questions separate:

* ``ORACLE_EVENT_GRAPH`` measures controller/skill transfer conditional on a
  shared, target-native state abstraction; and
* ``MODEL_TOOL_EVENT_GRAPH`` measures the complete pixel-to-answer pipeline.

Oracle state is allowed to read an official scene graph, but never an answer,
functional program, or current-task outcome.  Every controller arm is bound to
the exact same content-addressed grounding receipt.  Consequently a matched
uplift cannot be attributed to one arm receiving better perceptual evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import re
from typing import Any, Callable, Mapping, Sequence

from .contracts import stable_hash


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GroundingMode(str, Enum):
    ORACLE_EVENT_GRAPH = "ORACLE_EVENT_GRAPH"
    MODEL_TOOL_EVENT_GRAPH = "MODEL_TOOL_EVENT_GRAPH"


class VideoTransferClaim(str, Enum):
    CONDITIONAL_SKILL_TRANSFER = "CONDITIONAL_SKILL_TRANSFER"
    END_TO_END_VIDEO_TRANSFER = "END_TO_END_VIDEO_TRANSFER"


BENCHMARKS = ("clevrer", "agqa2")
REQUIRED_MATCHED_ARMS = (
    "neural_only",
    "source_induced",
    "source_permuted",
    "generic_scaffold",
)
LAYER_B_REQUIRED_MATCHED_ARMS = REQUIRED_MATCHED_ARMS + (
    "target_written_isomorphic",
)
FORBIDDEN_STATE_KEYS = {
    "answer",
    "answer_slot",
    "correct_answer",
    "correct_option",
    "functional_program",
    "gold_answer",
    "question_program",
    "target_outcome",
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in FORBIDDEN_STATE_KEYS:
                paths.append(path)
            paths.extend(_forbidden_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


@dataclass(frozen=True)
class GroundingToolBudget:
    """Maximum evidence-acquisition resources available to every arm."""

    max_tool_calls: int
    max_frames: int
    max_provider_calls: int

    def validate(self) -> None:
        values = (self.max_tool_calls, self.max_frames, self.max_provider_calls)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("grounding-tool budgets must be integers")
        if any(value < 0 for value in values):
            raise ValueError("grounding-tool budgets must be nonnegative")


@dataclass(frozen=True)
class GroundingToolCall:
    tool: str
    arguments: Mapping[str, Any]
    information_need: str


@dataclass(frozen=True)
class SharedGroundingAcquisition:
    """Frozen result of one arm-independent target-native tool pass."""

    grounding: "SharedVideoGroundingReceipt"
    calls: tuple[Mapping[str, Any], ...]
    tool_calls: int
    frames_observed: int
    provider_calls: int
    source_controller_read: bool
    answer_or_program_read: bool
    acquisition_sha256: str

    def validate(self) -> None:
        self.grounding.validate()
        if self.source_controller_read or self.answer_or_program_read:
            raise ValueError("shared grounding acquisition crossed an authority boundary")
        body = asdict(self)
        claimed = body.pop("acquisition_sha256")
        # asdict serializes the enum members as their str-enum values through
        # json.dumps, matching the construction-time nested grounding payload.
        if stable_hash(body) != claimed:
            raise ValueError("shared grounding acquisition hash mismatch")


@dataclass(frozen=True)
class SharedVideoGroundingReceipt:
    """One answer-blind state receipt shared verbatim by all controller arms."""

    benchmark: str
    task_id: str
    split: str
    mode: GroundingMode
    claim: VideoTransferClaim
    target_state_sha256: str
    evidence_source_sha256: str
    tool_backend_sha256: str | None
    allowed_tools: tuple[str, ...]
    tool_budget: GroundingToolBudget
    official_scene_graph_read: bool
    functional_program_read: bool
    gold_answer_read: bool
    target_outcome_read: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        benchmark: str,
        task_id: str,
        split: str,
        mode: GroundingMode,
        state: Mapping[str, Any],
        evidence_source_sha256: str,
        tool_budget: GroundingToolBudget,
        allowed_tools: Sequence[str] = (),
        tool_backend_sha256: str | None = None,
        official_scene_graph_read: bool = False,
        functional_program_read: bool = False,
        gold_answer_read: bool = False,
        target_outcome_read: bool = False,
    ) -> "SharedVideoGroundingReceipt":
        benchmark = str(benchmark).casefold()
        if benchmark not in BENCHMARKS:
            raise ValueError(f"unsupported video benchmark: {benchmark}")
        forbidden = _forbidden_paths(state)
        if forbidden:
            raise ValueError(
                "grounding state contains answer/program fields: "
                + ", ".join(forbidden)
            )
        tool_budget.validate()
        tools = tuple(sorted({str(value) for value in allowed_tools}))
        if mode == GroundingMode.ORACLE_EVENT_GRAPH:
            claim = VideoTransferClaim.CONDITIONAL_SKILL_TRANSFER
            if not official_scene_graph_read:
                raise ValueError("oracle grounding must disclose scene-graph access")
            if tools or tool_backend_sha256 is not None:
                raise ValueError("oracle grounding cannot masquerade as model tools")
        elif mode == GroundingMode.MODEL_TOOL_EVENT_GRAPH:
            claim = VideoTransferClaim.END_TO_END_VIDEO_TRANSFER
            if official_scene_graph_read:
                raise ValueError("model/tool grounding cannot read an official scene graph")
            if not tools or not _is_sha256(tool_backend_sha256):
                raise ValueError("model/tool grounding needs tools and a pinned backend")
        else:  # pragma: no cover - protects callers passing an untyped string.
            raise ValueError("invalid grounding mode")
        if functional_program_read or gold_answer_read or target_outcome_read:
            raise ValueError("grounding may not read program, answer, or target outcome")
        body = {
            "benchmark": benchmark,
            "task_id": str(task_id),
            "split": str(split),
            "mode": mode.value,
            "claim": claim.value,
            "target_state_sha256": stable_hash(state),
            "evidence_source_sha256": str(evidence_source_sha256),
            "tool_backend_sha256": tool_backend_sha256,
            "allowed_tools": tools,
            "tool_budget": asdict(tool_budget),
            "official_scene_graph_read": bool(official_scene_graph_read),
            "functional_program_read": False,
            "gold_answer_read": False,
            "target_outcome_read": False,
        }
        if not _is_sha256(body["evidence_source_sha256"]):
            raise ValueError("evidence source must be content-addressed")
        return cls(
            benchmark=benchmark,
            task_id=body["task_id"],
            split=body["split"],
            mode=mode,
            claim=claim,
            target_state_sha256=body["target_state_sha256"],
            evidence_source_sha256=body["evidence_source_sha256"],
            tool_backend_sha256=tool_backend_sha256,
            allowed_tools=tools,
            tool_budget=tool_budget,
            official_scene_graph_read=body["official_scene_graph_read"],
            functional_program_read=False,
            gold_answer_read=False,
            target_outcome_read=False,
            receipt_sha256=stable_hash(body),
        )

    def validate(self) -> None:
        self.tool_budget.validate()
        if not all(_is_sha256(value) for value in (
            self.target_state_sha256,
            self.evidence_source_sha256,
            self.receipt_sha256,
        )):
            raise ValueError("grounding receipt contains an invalid sha256")
        if self.functional_program_read or self.gold_answer_read or self.target_outcome_read:
            raise ValueError("grounding receipt crossed an outcome authority boundary")
        if self.mode == GroundingMode.ORACLE_EVENT_GRAPH:
            if self.claim != VideoTransferClaim.CONDITIONAL_SKILL_TRANSFER:
                raise ValueError("oracle result must use the conditional-transfer claim")
            if not self.official_scene_graph_read:
                raise ValueError("oracle receipt failed to disclose scene-graph access")
        else:
            if self.claim != VideoTransferClaim.END_TO_END_VIDEO_TRANSFER:
                raise ValueError("model grounding must use the end-to-end claim")
            if self.official_scene_graph_read:
                raise ValueError("end-to-end receipt read an official scene graph")
        body = {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "split": self.split,
            "mode": self.mode.value,
            "claim": self.claim.value,
            "target_state_sha256": self.target_state_sha256,
            "evidence_source_sha256": self.evidence_source_sha256,
            "tool_backend_sha256": self.tool_backend_sha256,
            "allowed_tools": self.allowed_tools,
            "tool_budget": asdict(self.tool_budget),
            "official_scene_graph_read": self.official_scene_graph_read,
            "functional_program_read": self.functional_program_read,
            "gold_answer_read": self.gold_answer_read,
            "target_outcome_read": self.target_outcome_read,
        }
        if stable_hash(body) != self.receipt_sha256:
            raise ValueError("shared grounding receipt hash mismatch")


def acquire_shared_model_grounding(
    *,
    benchmark: str,
    task_id: str,
    split: str,
    public_state: Mapping[str, Any],
    evidence_source_sha256: str,
    tool_backend_sha256: str,
    tool_budget: GroundingToolBudget,
    plan: Sequence[GroundingToolCall],
    dispatch: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
) -> SharedGroundingAcquisition:
    """Execute one fixed grounding plan before any controller arm runs.

    A dispatcher may wrap local vision tools or a VLM.  Its payload must be
    evidence-only.  ``_usage`` is removed before hashing into target state and
    is used solely for enforcing the preregistered resource cap.
    """

    tool_budget.validate()
    if len(plan) > tool_budget.max_tool_calls:
        raise ValueError("shared grounding plan exceeds tool-call budget")
    calls: list[dict[str, Any]] = []
    total_frames = 0
    total_provider_calls = 0
    for index, call in enumerate(plan):
        if not call.tool or not call.information_need:
            raise ValueError("grounding tool call is incomplete")
        payload = dict(dispatch(call.tool, dict(call.arguments)))
        usage = dict(payload.pop("_usage", {}) or {})
        frames = int(usage.get("frames_observed", 0))
        provider = int(usage.get("provider_calls", 0))
        if frames < 0 or provider < 0:
            raise ValueError("grounding tool returned invalid usage")
        forbidden = _forbidden_paths(payload)
        if forbidden:
            raise ValueError(
                "grounding tool returned answer/program fields: "
                + ", ".join(forbidden)
            )
        total_frames += frames
        total_provider_calls += provider
        if total_frames > tool_budget.max_frames:
            raise ValueError("shared grounding exceeded frame budget")
        if total_provider_calls > tool_budget.max_provider_calls:
            raise ValueError("shared grounding exceeded provider-call budget")
        calls.append({
            "call_id": f"G{index}",
            "tool": call.tool,
            "arguments": dict(call.arguments),
            "information_need": call.information_need,
            "evidence": payload,
            "usage": {"frames_observed": frames, "provider_calls": provider},
        })
    state = {
        "public_state": dict(public_state),
        "grounding_tool_receipts": calls,
        "source_controller_read": False,
        "answer_or_program_read": False,
    }
    grounding = SharedVideoGroundingReceipt.create(
        benchmark=benchmark,
        task_id=task_id,
        split=split,
        mode=GroundingMode.MODEL_TOOL_EVENT_GRAPH,
        state=state,
        evidence_source_sha256=evidence_source_sha256,
        tool_backend_sha256=tool_backend_sha256,
        allowed_tools=tuple(call.tool for call in plan),
        tool_budget=tool_budget,
    )
    body = {
        "grounding": asdict(grounding),
        "calls": tuple(calls),
        "tool_calls": len(calls),
        "frames_observed": total_frames,
        "provider_calls": total_provider_calls,
        "source_controller_read": False,
        "answer_or_program_read": False,
    }
    result = SharedGroundingAcquisition(
        grounding=grounding,
        calls=tuple(calls),
        tool_calls=len(calls),
        frames_observed=total_frames,
        provider_calls=total_provider_calls,
        source_controller_read=False,
        answer_or_program_read=False,
        acquisition_sha256=stable_hash(body),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class VideoTransferDecision:
    """A prediction frozen before an evaluator opens the gold answer."""

    task_id: str
    arm: str
    prediction: str
    controller_sha256: str
    source_program_sha256: str | None
    grounding_receipt_sha256: str
    tool_calls: int
    frames_observed: int
    provider_calls: int
    gold_answer_read: bool
    target_outcome_read: bool
    decision_sha256: str

    def validate(self, grounding: SharedVideoGroundingReceipt) -> None:
        grounding.validate()
        if self.task_id != grounding.task_id:
            raise ValueError("decision/grounding task mismatch")
        if self.grounding_receipt_sha256 != grounding.receipt_sha256:
            raise ValueError("controller arms did not consume identical grounding")
        if self.gold_answer_read or self.target_outcome_read:
            raise ValueError("decision crossed an outcome authority boundary")
        cap = grounding.tool_budget
        if self.tool_calls > cap.max_tool_calls:
            raise ValueError("tool-call budget exceeded")
        if self.frames_observed > cap.max_frames:
            raise ValueError("frame budget exceeded")
        if self.provider_calls > cap.max_provider_calls:
            raise ValueError("provider-call budget exceeded")
        body = asdict(self)
        claimed = body.pop("decision_sha256")
        if stable_hash(body) != claimed:
            raise ValueError("video transfer decision hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        grounding: SharedVideoGroundingReceipt,
        arm: str,
        prediction: str,
        controller_sha256: str,
        source_program_sha256: str | None = None,
        tool_calls: int = 0,
        frames_observed: int = 0,
        provider_calls: int = 0,
        gold_answer_read: bool = False,
        target_outcome_read: bool = False,
    ) -> "VideoTransferDecision":
        grounding.validate()
        arm = str(arm)
        if arm not in LAYER_B_REQUIRED_MATCHED_ARMS and arm != "target_native_ceiling":
            raise ValueError(f"unsupported matched arm: {arm}")
        if not _is_sha256(controller_sha256):
            raise ValueError("controller must be content-addressed")
        if arm in {"source_induced", "source_permuted"}:
            if not _is_sha256(source_program_sha256):
                raise ValueError("source arm needs a content-addressed program")
        elif source_program_sha256 is not None:
            raise ValueError("target-only arm cannot claim a source program")
        usage = (tool_calls, frames_observed, provider_calls)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in usage):
            raise ValueError("tool usage must be integer-valued")
        if any(value < 0 for value in usage):
            raise ValueError("tool usage must be nonnegative")
        cap = grounding.tool_budget
        if tool_calls > cap.max_tool_calls:
            raise ValueError("tool-call budget exceeded")
        if frames_observed > cap.max_frames:
            raise ValueError("frame budget exceeded")
        if provider_calls > cap.max_provider_calls:
            raise ValueError("provider-call budget exceeded")
        if gold_answer_read or target_outcome_read:
            raise ValueError("decision must freeze before outcome evaluation")
        body = {
            "task_id": grounding.task_id,
            "arm": arm,
            "prediction": str(prediction),
            "controller_sha256": controller_sha256,
            "source_program_sha256": source_program_sha256,
            "grounding_receipt_sha256": grounding.receipt_sha256,
            "tool_calls": tool_calls,
            "frames_observed": frames_observed,
            "provider_calls": provider_calls,
            "gold_answer_read": False,
            "target_outcome_read": False,
        }
        return cls(**body, decision_sha256=stable_hash(body))


def _exact_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def evaluate_matched_transfer(
    *,
    groundings: Sequence[SharedVideoGroundingReceipt],
    decisions: Sequence[VideoTransferDecision],
    gold_answers: Mapping[str, str],
    answer_equivalence: Mapping[str, Callable[[Any, Any], bool]] | None = None,
) -> dict[str, Any]:
    """Score frozen predictions without mixing grounding modes or evidence."""

    grounding_by_task = {row.task_id: row for row in groundings}
    if len(grounding_by_task) != len(groundings):
        raise ValueError("grounding task IDs must be unique")
    if set(grounding_by_task) != set(gold_answers):
        raise ValueError("evaluator gold IDs do not match the frozen grounding cohort")
    by_task: dict[str, dict[str, VideoTransferDecision]] = {}
    for decision in decisions:
        grounding = grounding_by_task.get(decision.task_id)
        if grounding is None:
            raise ValueError("decision has no shared grounding receipt")
        decision.validate(grounding)
        arms = by_task.setdefault(decision.task_id, {})
        if decision.arm in arms:
            raise ValueError("duplicate controller arm for one task")
        arms[decision.arm] = decision
    rows = []
    for task_id, grounding in grounding_by_task.items():
        arms = by_task.get(task_id, {})
        missing = sorted(set(REQUIRED_MATCHED_ARMS) - set(arms))
        if missing:
            raise ValueError(f"matched task is missing arms: {missing}")
        gold = str(gold_answers[task_id])
        scorer = (answer_equivalence or {}).get(
            grounding.benchmark, lambda prediction, expected: prediction == expected,
        )
        correct = {
            arm: bool(scorer(decision.prediction, gold))
            for arm, decision in arms.items()
        }
        rows.append({
            "task_id": task_id,
            "benchmark": grounding.benchmark,
            "mode": grounding.mode.value,
            "claim": grounding.claim.value,
            "grounding_receipt_sha256": grounding.receipt_sha256,
            "correct": correct,
        })
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["benchmark"]), str(row["mode"])), []).append(row)
    summaries = []
    for (benchmark, mode), group in sorted(groups.items()):
        arm_names = sorted(group[0]["correct"])
        arm_correct = {
            arm: sum(bool(row["correct"][arm]) for row in group)
            for arm in arm_names
        }
        comparisons = {}
        for baseline in (
            "neural_only", "source_permuted", "generic_scaffold",
            "target_written_isomorphic", "target_native_ceiling",
        ):
            if baseline not in arm_correct:
                continue
            wins = sum(
                bool(row["correct"]["source_induced"])
                and not bool(row["correct"][baseline])
                for row in group
            )
            losses = sum(
                bool(row["correct"][baseline])
                and not bool(row["correct"]["source_induced"])
                for row in group
            )
            comparisons[f"source_vs_{baseline}"] = {
                "wins": wins,
                "losses": losses,
                "ties": len(group) - wins - losses,
                "exact_two_sided_p": _exact_two_sided(wins, losses),
            }
        primary = comparisons["source_vs_neural_only"]
        summaries.append({
            "benchmark": benchmark,
            "mode": mode,
            "claim": group[0]["claim"],
            "tasks": len(group),
            "source_induced_correct": arm_correct["source_induced"],
            "neural_only_correct": arm_correct["neural_only"],
            "arm_correct": arm_correct,
            "paired_comparisons": comparisons,
            "source_vs_neural_wins": primary["wins"],
            "source_vs_neural_losses": primary["losses"],
            "source_vs_neural_ties": primary["ties"],
            "source_vs_neural_exact_two_sided_p": primary["exact_two_sided_p"],
            "all_arms_shared_exact_grounding": True,
        })
    return {
        "schema_version": "matched-video-transfer-measurement-v1",
        "tasks": len(rows),
        "grounding_modes_combined": False,
        "summaries": summaries,
        "rows": rows,
    }


def evaluate_layer_b_matched_transfer(
    *, groundings: Sequence[SharedVideoGroundingReceipt],
    decisions: Sequence[VideoTransferDecision], gold_answers: Mapping[str, str],
    answer_equivalence: Mapping[str, Callable[[Any, Any], bool]] | None = None,
) -> dict[str, Any]:
    """Evaluate the five-arm raw-video protocol and reject missing/extra arms."""

    expected = set(LAYER_B_REQUIRED_MATCHED_ARMS)
    per_task: dict[str, set[str]] = {}
    for decision in decisions:
        per_task.setdefault(decision.task_id, set()).add(decision.arm)
    for task_id in {row.task_id for row in groundings}:
        observed = per_task.get(task_id, set())
        if observed != expected:
            raise ValueError(
                f"Layer-B task {task_id} must contain exactly the five frozen arms; "
                f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
            )
    result = evaluate_matched_transfer(
        groundings=groundings, decisions=decisions, gold_answers=gold_answers,
        answer_equivalence=answer_equivalence,
    )
    result["schema_version"] = "agqa-layer-b-five-arm-matched-transfer-v1"
    result["required_arms"] = list(LAYER_B_REQUIRED_MATCHED_ARMS)
    result["raw_video_end_to_end_only"] = all(
        row.mode == GroundingMode.MODEL_TOOL_EVENT_GRAPH for row in groundings
    )
    if not result["raw_video_end_to_end_only"]:
        raise ValueError("Layer-B evaluation cannot include oracle event graphs")
    return result


def assert_unified_target_uses_shared_grounding(
    grounding: SharedVideoGroundingReceipt, target: Any,
) -> None:
    """Check compatibility with ``UnifiedTargetGrounding`` without importing it."""

    grounding.validate()
    applicability = target.applicability
    if applicability.task_id != grounding.task_id:
        raise ValueError("unified target task differs from shared video grounding")
    if str(applicability.target_domain).casefold() != grounding.benchmark:
        raise ValueError("unified target domain differs from grounding benchmark")
    if applicability.target_state_sha256 != grounding.target_state_sha256:
        raise ValueError("unified target did not bind the shared grounding state")
    if applicability.formal_outcome_read:
        raise ValueError("unified target read formal outcome before decision")


__all__ = [
    "BENCHMARKS",
    "GroundingMode",
    "GroundingToolCall",
    "GroundingToolBudget",
    "LAYER_B_REQUIRED_MATCHED_ARMS",
    "REQUIRED_MATCHED_ARMS",
    "SharedVideoGroundingReceipt",
    "SharedGroundingAcquisition",
    "VideoTransferClaim",
    "VideoTransferDecision",
    "assert_unified_target_uses_shared_grounding",
    "acquire_shared_model_grounding",
    "evaluate_matched_transfer",
    "evaluate_layer_b_matched_transfer",
]
