"""Concrete `SkillAdapter` implementations.

Phase A targets (PLAN-COMPONENTS-IMPLEMENTATION §4):
  - `gymv` — game-env (gym-v) adapter, hooks into `vlm_wrapper.gymv_adapter`.
  - `browser` — browser adapter, hooks into `vlm_wrapper.browser_adapter`.

Both adapters are deliberately thin: they translate `SkillRecord.protocol`
hops into adapter-native tool calls and return an `AdapterRunResult`. Real
env binding lives in `vlm_wrapper/`; we depend on it via late imports so
the harness package can be imported (and tested) without those heavy
deps installed.
"""

from harness.adapters.browser_adapter import BrowserAdapter
from harness.adapters.gymv_adapter import GymvAdapter

__all__ = ["BrowserAdapter", "GymvAdapter"]
