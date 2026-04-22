"""Single source of truth for the project's backbone LLM model.

Decision (2026-04-21): all live code paths in this project default to
**GPT-4o** as the backbone reasoning/policy/teacher model.

The 8B / 32B / 72B Qwen tracks (LoRA / GRPO / frozen-teacher) are
explicitly **deferred**: they remain reachable through dedicated
entrypoints under `scripts/qwen3_*.py`, `inference/run_qwen3_8b_eval.py`,
and `skill_agents/lora/`, but no library-level default points at them.

Rules
-----

1.  Every new module in `harness/`, `orchestrator/`, `crafter/`,
    `skill_bank/`, and `evaluation/` MUST import `BACKBONE_MODEL` from
    here when picking a default model.
2.  Existing modules (`decision_agents/`, `skill_agents/`,
    `vlm_wrapper/`, `inference/`) have been migrated where it changes a
    *runtime* default. Documentation strings retaining the old model
    names are kept as historical context.
3.  Routing is unchanged: the central LLM caller `API_func.ask_model`
    already routes `"gpt-4o"` to OpenAI / OpenRouter, so no code path
    needs to know about the underlying provider.

Switching the backbone in the future
------------------------------------

Set the environment variable `VLM_AGENT_BACKBONE_MODEL` to override the
default at process start. Programmatic overrides should be passed as
explicit `model=` arguments and never mutate `BACKBONE_MODEL`.
"""

from __future__ import annotations

import os

# ---- canonical defaults ------------------------------------------------

#: The default backbone reasoning / policy model for the entire project.
BACKBONE_MODEL: str = os.environ.get("VLM_AGENT_BACKBONE_MODEL", "gpt-4o")

#: The default *teacher* / Synthesis-Reflection-Agent model used by the
#: Skill Crafter (PLAN-SKILL-CRAFTER §3 "Frozen-first design"). For the
#: GPT-4o-only phase this is the same model. Keep them as separate
#: symbols so a later "frozen 72B teacher" rollout only flips one knob.
BACKBONE_TEACHER_MODEL: str = os.environ.get(
    "VLM_AGENT_BACKBONE_TEACHER_MODEL", BACKBONE_MODEL
)

#: The default *judge* model used by the eval driver (E0 / E1 / E2).
#: Same as backbone for now.
BACKBONE_JUDGE_MODEL: str = os.environ.get(
    "VLM_AGENT_BACKBONE_JUDGE_MODEL", BACKBONE_MODEL
)


# ---- deferred-track registry ------------------------------------------

#: Canonical model names that are *deferred* — mentioned in plans but
#: not part of the current default surface. Tests check that no live
#: code path defaults to one of these.
DEFERRED_MODELS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-VL-72B",
    }
)


def is_deferred(model: str) -> bool:
    """True if `model` is one of the deferred Qwen tracks."""
    return model in DEFERRED_MODELS


def assert_default_is_gpt4o() -> None:
    """Used by tests; raises if the backbone has been silently changed."""
    if not BACKBONE_MODEL.startswith("gpt-4o"):
        raise AssertionError(
            f"BACKBONE_MODEL must start with 'gpt-4o' for the current "
            f"phase; got {BACKBONE_MODEL!r}. Set VLM_AGENT_BACKBONE_MODEL "
            f"to override only when explicitly enabling a deferred track."
        )


__all__ = [
    "BACKBONE_JUDGE_MODEL",
    "BACKBONE_MODEL",
    "BACKBONE_TEACHER_MODEL",
    "DEFERRED_MODELS",
    "assert_default_is_gpt4o",
    "is_deferred",
]
