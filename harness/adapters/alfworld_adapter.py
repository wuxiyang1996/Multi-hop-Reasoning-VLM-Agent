"""ALFWorld transfer-target adapter.

ALFWorld is the embodied text/control target in the active transfer study.
It is deliberately separate from OSWorld: ALFWorld binds protocols to
admissible household commands, while OSWorld's desktop/GUI integration is
kept as an isolated legacy module.

The default executor is deterministic for gate dry-runs.  Runtime callers can
bind a live text environment with :func:`bind_alfworld_executor`.
"""

from __future__ import annotations

from common.enums import SkillType
from harness.adapters._stub_base import HopExecutor, StubTransferTargetAdapter


class AlfworldAdapter(StubTransferTargetAdapter):
    name = "alfworld"
    supported_types = (
        SkillType.ACTION,
        SkillType.MIXED,
        SkillType.GROUNDING,
        SkillType.REASONING,
    )


def bind_alfworld_executor(
    adapter: AlfworldAdapter,
    *,
    env,
    on_unresolved: str = "abort",
    reset_each_run: bool = True,
):
    """Bind an initialized ``ALFWorldNLWrapper`` to ``adapter``."""
    from harness.alfworld_executor import make_alfworld_executor

    executor, holder = make_alfworld_executor(
        env=env,
        on_unresolved=on_unresolved,
        reset_each_run=reset_each_run,
    )
    adapter.set_executor(executor)
    return executor, holder


__all__ = ["AlfworldAdapter", "HopExecutor", "bind_alfworld_executor"]
