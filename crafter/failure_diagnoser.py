"""`FailureDiagnoser` — first-pass diagnosis of a single FailureTrace.

Spec: PLAN-SKILL-CRAFTER §6.4.

The diagnoser maps a `FailureTrace` to a `FailureDiagnosis` containing a
recommended `RecoveryStrategy`. The MVP uses a deterministic rule table;
a frozen-teacher LLM hook is provided for future replacement
(`set_llm_diagnoser`) but defaults to the rule path so the pipeline is
runnable without any external model.
"""

from __future__ import annotations

from typing import Callable, Optional

from common.enums import RecoveryStrategy
from data_structure.extensions.failure_trace import FailureDiagnosis, FailureTrace


_LLMDiagnoser = Callable[[FailureTrace], FailureDiagnosis]


class FailureDiagnoser:
    def __init__(self, llm: Optional[_LLMDiagnoser] = None) -> None:
        self._llm = llm

    def set_llm_diagnoser(self, llm: _LLMDiagnoser) -> None:
        self._llm = llm

    def diagnose(self, trace: FailureTrace) -> FailureDiagnosis:
        if self._llm is not None:
            try:
                return self._llm(trace)
            except Exception:                                  # noqa: BLE001
                # Fall through to the rule path on LLM failure.
                pass
        return self._rule_diagnose(trace)

    # -- rule path --------------------------------------------------------

    def _rule_diagnose(self, trace: FailureTrace) -> FailureDiagnosis:
        cls = trace.failure_class.upper()
        if cls == "PRECONDITION_VIOLATION":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="precondition",
                root_cause="precondition not satisfied at invocation",
                recommended_strategy=RecoveryStrategy.PRECONDITION_STRENGTHENING,
                confidence=0.7,
            )
        if cls == "INVARIANT_VIOLATION":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="effect_check",
                root_cause="claimed effect lacks evidence (G0)",
                recommended_strategy=RecoveryStrategy.HOP_INSERTION,
                confidence=0.6,
                notes="insert a VERIFY hop before COMMIT",
            )
        if cls == "BUDGET_EXCEEDED":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="protocol_step",
                root_cause="protocol exceeded its hop / token budget",
                recommended_strategy=RecoveryStrategy.SKILL_DECOMPOSITION,
                confidence=0.55,
            )
        if cls == "MISSING_ADAPTER":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="protocol_step",
                root_cause="no adapter registered for (domain, type)",
                recommended_strategy=RecoveryStrategy.SKILL_RETIREMENT,
                confidence=0.4,
            )
        if cls == "ADAPTER_EXCEPTION":
            return FailureDiagnosis(
                failure_id=trace.failure_id,
                locus="protocol_step",
                root_cause="adapter raised an exception during a hop",
                recommended_strategy=RecoveryStrategy.FALLBACK_INJECTION,
                confidence=0.5,
            )
        return FailureDiagnosis(
            failure_id=trace.failure_id,
            locus="unknown",
            root_cause=f"unclassified failure: {trace.failure_class}",
            recommended_strategy=RecoveryStrategy.PROTOCOL_PATCH,
            confidence=0.3,
        )


__all__ = ["FailureDiagnoser"]
