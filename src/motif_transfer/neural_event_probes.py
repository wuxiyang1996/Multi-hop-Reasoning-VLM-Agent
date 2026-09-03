"""Probe scoring interfaces and a deterministic table backend for evaluation.

The table backend is intentionally simple: a future LoRA/VLM implementation
can implement the same protocol, while unit tests and matched controls can use
frozen input-hash-to-score tables without model sampling noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from .contracts import Observation, TransitionReceipt, stable_hash
from .neurosymbolic_ir import NeuralProbeSpec, ProbeVerdict


def before_probe_payload(observation: Observation) -> Mapping[str, Any]:
    """The pre-action interface cannot inspect a selected action or after-state."""

    return {
        "state": dict(observation.state),
        "native_actions": observation.native_actions,
        "terminal": observation.terminal,
        "official_success": observation.official_success,
        "score": observation.score,
    }


def transition_probe_payload(
    before: Observation,
    action: str,
    after: Observation,
    receipt: TransitionReceipt,
) -> Mapping[str, Any]:
    return {
        "before": dict(before.state),
        "before_native_actions": before.native_actions,
        "action": action,
        "after": dict(after.state),
        "after_native_actions": after.native_actions,
        "reward": receipt.reward,
        "terminal": after.terminal,
        "official_success": after.official_success,
        "transition_receipt_id": receipt.receipt_id,
    }


class NeuralProbeBackend(Protocol):
    @property
    def model_artifact_sha256(self) -> str: ...

    def score(
        self,
        probe: NeuralProbeSpec,
        input_payload: Mapping[str, Any],
    ) -> float | None: ...


@dataclass(frozen=True)
class ProbeEvaluationReceipt:
    evaluation_id: str
    program_id: str
    node_id: str
    probe_id: str
    model_artifact_sha256: str
    input_sha256: str
    score: float | None
    verdict: ProbeVerdict
    authority: str = "NEURAL_PROPOSAL_ONLY"

    @classmethod
    def create(
        cls,
        *,
        program_id: str,
        node_id: str,
        probe: NeuralProbeSpec,
        input_payload: Mapping[str, Any],
        score: float | None,
    ) -> "ProbeEvaluationReceipt":
        verdict = probe.verdict(score)
        body = {
            "program_id": program_id,
            "node_id": node_id,
            "probe_id": probe.probe_id,
            "model_artifact_sha256": probe.model_artifact_sha256,
            "input_sha256": stable_hash(input_payload),
            "score": score,
            "verdict": verdict.value,
            "authority": "NEURAL_PROPOSAL_ONLY",
        }
        return cls(stable_hash(body), program_id, node_id, probe.probe_id,
                   probe.model_artifact_sha256, body["input_sha256"], score,
                   verdict)

    def validate(self) -> bool:
        body = asdict(self)
        expected = body.pop("evaluation_id")
        body["verdict"] = self.verdict.value
        return expected == stable_hash(body)


class FrozenTableProbeBackend:
    """Frozen scores keyed by ``(probe_id, input_sha256)``.

    Missing entries return ``None`` and therefore become ``UNKNOWN``.  This is
    the desired fail-closed behavior for out-of-scope states.
    """

    def __init__(
        self,
        model_artifact_sha256: str,
        scores: Mapping[tuple[str, str], float],
    ) -> None:
        self._model_artifact_sha256 = model_artifact_sha256
        self._scores = dict(scores)

    @property
    def model_artifact_sha256(self) -> str:
        return self._model_artifact_sha256

    def score(
        self,
        probe: NeuralProbeSpec,
        input_payload: Mapping[str, Any],
    ) -> float | None:
        return self._scores.get((probe.probe_id, stable_hash(input_payload)))


def evaluate_probe(
    *,
    program_id: str,
    node_id: str,
    probe: NeuralProbeSpec,
    input_payload: Mapping[str, Any],
    backend: NeuralProbeBackend,
) -> ProbeEvaluationReceipt:
    if backend.model_artifact_sha256 != probe.model_artifact_sha256:
        raise ValueError("probe backend does not match the frozen model artifact")
    receipt = ProbeEvaluationReceipt.create(
        program_id=program_id,
        node_id=node_id,
        probe=probe,
        input_payload=input_payload,
        score=backend.score(probe, input_payload),
    )
    if not receipt.validate():
        raise ValueError("probe evaluation receipt failed self-validation")
    return receipt


__all__ = [
    "FrozenTableProbeBackend",
    "NeuralProbeBackend",
    "ProbeEvaluationReceipt",
    "before_probe_payload",
    "evaluate_probe",
    "transition_probe_payload",
]
