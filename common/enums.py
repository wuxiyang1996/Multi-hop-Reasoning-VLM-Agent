"""Canonical enums shared across the Harness / Orchestrator / Crafter / Bank.

All values are normative — see the cited canonical specs.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple


# The five general target domains every skill must be feasible in
# (PLAN-SKILL-BANK §0.1 general-protocol invariant).
DOMAINS: Tuple[str, ...] = (
    "gymv",                # game (gym-v adapter)
    "browser",             # webagent
    "osworld",             # os-agent
    "video",               # short-video evidence-grounded reasoning
    "visual_reasoning",    # image-QA / visual reasoning
)


# Evidence-role taxonomy (PLAN-SKILL-BANK §0.3 Clause B).
EVIDENCE_ROLES: Tuple[str, ...] = ("GATHER", "VERIFY", "REASON", "COMMIT")


class SkillType(str, Enum):
    """Skill categorical type used by the Harness for adapter dispatch.

    See PLAN-HARNESS §5.1 (`SkillEpisode.skill_type`) and §8 (unified skill
    interface).
    """

    REASONING = "reasoning"
    ACTION = "action"
    GROUNDING = "grounding"
    MIXED = "mixed"


class SkillStatus(str, Enum):
    """Canonical lifecycle (PLAN-UNIFIED-SKILL-GATE §2.1)."""

    DRAFT = "draft"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class SkillSourceType(str, Enum):
    """Where the skill came from (PLAN-UNIFIED-SKILL-GATE §2.2).

    All sources share the same gate; there is no fast path based on
    model size, lineage, or human authorship.
    """

    MINED = "mined_from_trace"
    CRAFTED = "crafted_by_composition"
    REPAIRED = "repaired_from_failure"
    TRANSFERRED = "transferred_from_other_domain"
    TEACHER = "teacher_proposed"
    SEEDED = "human_seeded"


class GateStage(str, Enum):
    """Unified gate stages (PLAN-UNIFIED-SKILL-GATE §7)."""

    STATIC = "static"
    REPLAY = "replay"
    SHADOW = "shadow"
    TRANSFER = "transfer"
    NON_REGRESSION = "non_regression"


class GateVerdict(str, Enum):
    """Per-stage and final verdicts (PLAN-UNIFIED-SKILL-GATE §3.3)."""

    PASS = "pass"
    FAIL = "fail"
    LIMITED_PASS = "limited_pass"


class InnerAction(str, Enum):
    """Inner-MDP action vocabulary (PLAN-ACTION-AGENT §5)."""

    GROUND = "GROUND"
    CHECK = "CHECK"
    RETRIEVE = "RETRIEVE"
    COMMIT = "COMMIT"
    EXECUTE = "EXECUTE"


class RecoveryStrategy(str, Enum):
    """Recovery strategies (PLAN-SKILL-CRAFTER §6.5)."""

    PROTOCOL_PATCH = "protocol_patch"
    PRECONDITION_STRENGTHENING = "precondition_strengthening"
    FALLBACK_INJECTION = "fallback_injection"
    HOP_INSERTION = "hop_insertion"
    SKILL_DECOMPOSITION = "skill_decomposition"
    REGROUNDING_TRIGGER = "regrounding_trigger"
    SKILL_RETIREMENT = "skill_retirement"


__all__ = [
    "DOMAINS",
    "EVIDENCE_ROLES",
    "GateStage",
    "GateVerdict",
    "InnerAction",
    "RecoveryStrategy",
    "SkillSourceType",
    "SkillStatus",
    "SkillType",
]
