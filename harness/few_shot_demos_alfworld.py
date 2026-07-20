"""ALFWorld evaluation probes for the Stage-3a few-shot gate.

ALFWorld chooses a concrete household game when the live environment resets.
The adapter executor performs that reset at the start of each probe, so these
records intentionally carry only the split/task label and the verifiable
completion-reward expectation; they do not pretend to be offline GUI traces.
"""

from __future__ import annotations

from typing import List

from common.state_schema import StateSchema
from harness.few_shot_adapter import FewShotDemo


def build_alfworld_probe_demos(
    *,
    split: str,
    max_demos: int,
    min_reward: float = 1.0,
) -> List[FewShotDemo]:
    """Create live-reset probes scored by ALFWorld completion reward."""
    if max_demos < 1:
        return []
    return [
        FewShotDemo(
            state=StateSchema(
                task=f"alfworld/{split}/probe_{index:03d}",
                domain="alfworld",
            ),
            expected={"min_reward": float(min_reward)},
            notes="live ALFWorld reset probe",
        )
        for index in range(max_demos)
    ]


__all__ = ["build_alfworld_probe_demos"]
