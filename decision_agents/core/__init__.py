"""Shared scaffolding for the SFT / GRPO actor flavours.

The two specialised actors under :mod:`decision_agents.SFT` and
:mod:`decision_agents.grpo` differ only in (a) which LLM backend they
hit, (b) what artefact they record per step, and (c) whether the LLM
call is sync (GPT-4o) or async (vLLM).

Everything they share — schema parsing, skill interface, skill tracker,
inner-MDP loop, reward computation, and the action prompt builder —
already lives at :mod:`decision_agents.<module>` and is re-imported here
for convenience.  This sub-package adds only the multimodal scaffolding
that the legacy text-only :class:`~decision_agents.actor_agent.ActorAgent`
does not own.
"""

from __future__ import annotations

from decision_agents.core.multimodal import (
    VisualInput,
    build_qwen_vl_messages,
    build_openai_vision_messages,
    load_image_as_data_url,
)

__all__ = [
    "VisualInput",
    "build_openai_vision_messages",
    "build_qwen_vl_messages",
    "load_image_as_data_url",
]
