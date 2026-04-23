"""Single source of truth for the project's backbone LLM model.

Decision (2026-04-21): all live code paths in this project default to
**GPT-4o** as the backbone reasoning/policy/teacher model.

The 8B / 32B / 72B Qwen tracks (LoRA / GRPO / frozen-teacher) are
explicitly **deferred**: they remain reachable through dedicated
entrypoints under `scripts/qwen3_*.py`, `inference/run_qwen3_8b_eval.py`,
and `skill_agents/lora/`, but no library-level default points at them.

Phase-F frozen Qwen3-VL teachers
--------------------------------

Per the crafter README's Phase F entry, the Skill Crafter is designed
to swap its frozen teacher backbone from GPT-4o to a frozen Qwen3-VL
(``Qwen/Qwen3-VL-32B`` or the larger MoE ``Qwen/Qwen3-VL-235B-A22B``)
once the inference plumbing is in place.  We expose those names as
canonical constants and as a ``QWEN3_VL_TEACHERS`` registry so
``SkillCrafterService(teacher_model=...)`` can be wired without string
literals scattered across the codebase, *while keeping the project-wide
``BACKBONE_TEACHER_MODEL`` default at GPT-4o* (so the existing
test_backbone_model invariants still pass).

Two opt-in env-vars activate the Phase-F teacher without code edits:

* ``VLM_AGENT_BACKBONE_TEACHER_MODEL=Qwen/Qwen3-VL-32B`` — full override
  (works today for any phase).
* ``VLM_AGENT_PHASE_F_TEACHER=qwen3-vl-32b`` (or ``qwen3-vl-235b-a22b``)
  — Phase-F flag.  Read by ``crafter.SkillCrafterService.from_phase_f``
  (no implicit mutation of ``BACKBONE_TEACHER_MODEL``).

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
from typing import Mapping

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


# ---- Phase-F frozen Qwen3-VL teacher registry --------------------------

#: Frozen Qwen3-VL teacher model identifiers (PLAN-SKILL-CRAFTER §2,
#: "Frozen-first design").  Map of `size_key` → canonical HF model name.
#: New entries are additive; nothing here is a *default*.
QWEN3_VL_TEACHERS: Mapping[str, str] = {
    "32b": "Qwen/Qwen3-VL-32B",
    "235b-a22b": "Qwen/Qwen3-VL-235B-A22B",
}


def qwen3_vl_teacher(size: str = "32b") -> str:
    """Return the canonical HF id for a frozen Qwen3-VL teacher.

    ``size`` is matched case-insensitively against the keys of
    :data:`QWEN3_VL_TEACHERS`.  Raises :class:`ValueError` for unknown
    sizes so a typo can't silently fall back to GPT-4o.
    """
    key = size.lower().strip()
    if key not in QWEN3_VL_TEACHERS:
        raise ValueError(
            f"Unknown Qwen3-VL teacher size {size!r}; "
            f"valid options: {sorted(QWEN3_VL_TEACHERS)}."
        )
    return QWEN3_VL_TEACHERS[key]


def phase_f_teacher_from_env() -> str | None:
    """Resolve the Phase-F teacher from ``VLM_AGENT_PHASE_F_TEACHER``.

    Returns the canonical HF model id when the env-var is set to a
    recognized size, or ``None`` when unset.  Unknown sizes raise so
    misconfiguration surfaces at the entry point, not silently as a
    GPT-4o fallback.
    """
    raw = os.environ.get("VLM_AGENT_PHASE_F_TEACHER")
    if not raw:
        return None
    return qwen3_vl_teacher(raw)


# ---- deferred-track registry ------------------------------------------

#: Canonical model names that are *deferred* — mentioned in plans but
#: not part of the current default surface. Tests check that no live
#: code path defaults to one of these.  The Qwen3-VL Phase-F teachers
#: are deferred-by-default too: they're opt-in via
#: ``SkillCrafterService(teacher_model=qwen3_vl_teacher(...))``.
DEFERRED_MODELS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3-8B",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-VL-72B",
        *QWEN3_VL_TEACHERS.values(),
    }
)


def is_deferred(model: str) -> bool:
    """True if `model` is one of the deferred Qwen tracks."""
    return model in DEFERRED_MODELS


def is_frozen_qwen_teacher(model: str) -> bool:
    """True if `model` is one of the Phase-F frozen Qwen3-VL teachers."""
    return model in set(QWEN3_VL_TEACHERS.values())


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
    "QWEN3_VL_TEACHERS",
    "assert_default_is_gpt4o",
    "is_deferred",
    "is_frozen_qwen_teacher",
    "phase_f_teacher_from_env",
    "qwen3_vl_teacher",
]
