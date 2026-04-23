"""`video` adapter — short-video evidence-grounded reasoning execution path.

Short-video is the **first transfer arena** under
PLAN-EVAL-FIRST-TARGET — game-learned protocols (e.g.
`collect_evidence_chain`, `disambiguate_target`) earn their first
non-game `verified_domains` entry here through the few-shot adapter
(PLAN-UNIFIED-SKILL-GATE Stage 3a). Real env binding (clip indexing,
frame loaders, multi-hop video evidence) is plugged in via
`set_executor()`; until then the deterministic stub keeps the
adapter-binding handshake testable.
"""

from __future__ import annotations

from common.enums import SkillType
from harness.adapters._stub_base import HopExecutor, StubTransferTargetAdapter


class VideoAdapter(StubTransferTargetAdapter):
    name = "video"
    supported_types = (
        SkillType.REASONING,
        SkillType.GROUNDING,
        SkillType.MIXED,
    )


__all__ = ["HopExecutor", "VideoAdapter"]
