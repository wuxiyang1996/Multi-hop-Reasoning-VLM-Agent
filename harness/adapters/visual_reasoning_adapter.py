"""`visual_reasoning` adapter — image-QA / visual reasoning execution path.

Stub for the fifth transfer-target domain
(PLAN-COMPONENTS-IMPLEMENTATION Phase A.5 / Phase D). Real env
binding (single-image VLM tools, structured visual question answering)
is plugged in via `set_executor()`; the deterministic stub satisfies
the evidence-driven invariant so few-shot adapter binding attempts can
be exercised end-to-end at the gate.
"""

from __future__ import annotations

from common.enums import SkillType
from harness.adapters._stub_base import HopExecutor, StubTransferTargetAdapter


class VisualReasoningAdapter(StubTransferTargetAdapter):
    name = "visual_reasoning"
    supported_types = (
        SkillType.REASONING,
        SkillType.GROUNDING,
        SkillType.MIXED,
    )


__all__ = ["HopExecutor", "VisualReasoningAdapter"]
