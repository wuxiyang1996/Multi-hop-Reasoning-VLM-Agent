"""Outcome-blind AGQA 2.0 program compiler for source-IR preflight.

The compiler reads only the public functional program string.  It does not
accept the question answer, scene-graph grounding, reasoning labels, source
game identity, video frames, or a target action.  Its purpose is narrow: ask
whether an AGQA program exposes an exact anonymous obligation already present
in a source-induced IR contract.

Programs requiring both temporal and relational obligations are represented
as a composite type.  No atomic source program is allowed to partially match
that type; the selector must abstain.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .contracts import stable_hash
from .structural_ir_applicability import (
    OperatorSignature,
    SourceIRContract,
    TargetIRRequirement,
    select_source_contract,
)


TARGET_DOMAIN = "agqa2"
TARGET_INTERFACE = "functional_program_obligation_v1"
RELATION_IR = "RECURRENT_GOAL_RELATION_PROGRAM"
TEMPORAL_IR = "SPARSE_TEMPORAL_EFFECT_FUNCTION"
COMPOSITE_IR = "COMPOSITE_TEMPORAL_RELATION_PROGRAM"
UNSUPPORTED_IR = "UNSUPPORTED_AGQA_PROGRAM"

RELATION_OPERATOR = OperatorSignature(
    operation="UPDATE",
    predicate_family="ENTITY_GOAL_RELATION",
    arity=2,
    value_kind="RELATION_COVERAGE",
)
TEMPORAL_PAIR_OPERATOR = OperatorSignature(
    operation="SCORE",
    predicate_family="TEMPORAL_EFFECT_VECTOR",
    arity=2,
    value_kind="NORMALIZED_PROBABILITY",
)
TEMPORAL_SINGLE_OPERATOR = OperatorSignature(
    operation="SCORE",
    predicate_family="TEMPORAL_EFFECT_VECTOR",
    arity=1,
    value_kind="NORMALIZED_PROBABILITY",
)

FUNCTION_PATTERN = re.compile(r"([A-Za-z][A-Za-z_]*)\(")
TEMPORAL_FUNCTIONS = frozenset({
    "Localize", "IterateUntil", "Subtract", "Superlative",
})
PAIR_ROOTS = frozenset({"Choose", "Compare"})

RELATION_ROUTE = "RELATION_RECURRENT"
TEMPORAL_PAIR_ROUTE = "TEMPORAL_PAIR_RECURRENT"
TEMPORAL_SINGLE_ROUTE = "TEMPORAL_SINGLE_NONRECURRENT"
COMPOSITE_ROUTE = "COMPOSITE_ABSTAIN"
UNSUPPORTED_ROUTE = "NO_EXACT_SOURCE_TYPE"


@dataclass(frozen=True)
class AGQAProgramProfile:
    task_id: str
    root_function: str
    functions: tuple[str, ...]
    relation_obligation: bool
    temporal_obligation: bool
    recurrent_scan: bool
    route_kind: str
    answer_read: bool
    scene_graph_grounding_read: bool
    source_identity_read: bool
    profile_sha256: str


def profile_program(*, task_id: str, program: str) -> AGQAProgramProfile:
    """Compile a public AGQA functional program without target outcomes."""

    functions = tuple(FUNCTION_PATTERN.findall(str(program)))
    root = functions[0] if functions else ""
    function_set = set(functions)
    relation = "[relations" in program or "[relation," in program
    temporal = bool(function_set & TEMPORAL_FUNCTIONS)
    recurrent = bool(function_set & {"Iterate", "IterateUntil"})
    if relation and temporal:
        route = COMPOSITE_ROUTE
    elif relation and recurrent:
        route = RELATION_ROUTE
    elif temporal and recurrent and root in PAIR_ROOTS:
        route = TEMPORAL_PAIR_ROUTE
    elif (
        temporal and not recurrent and root in {"Superlative", "Equals"}
        and {"Superlative", "Subtract"} <= function_set
    ):
        # Superlative scores each candidate interval independently.  This is
        # an arity-one score operator even when the candidate list has two
        # activities.  AGQA also wraps this same typed obligation in Equals
        # for binary duration questions; Equals consumes the target-native
        # score and does not introduce a second source-IR obligation.
        route = TEMPORAL_SINGLE_ROUTE
    else:
        route = UNSUPPORTED_ROUTE
    body = {
        "task_id": str(task_id),
        "root_function": root,
        "functions": functions,
        "relation_obligation": relation,
        "temporal_obligation": temporal,
        "recurrent_scan": recurrent,
        "route_kind": route,
        "answer_read": False,
        "scene_graph_grounding_read": False,
        "source_identity_read": False,
    }
    return AGQAProgramProfile(**body, profile_sha256=stable_hash(body))


def target_requirement(
    profile: AGQAProgramProfile, *, target_grounder_sha256: str,
) -> TargetIRRequirement:
    """Translate program obligations into the shared anonymous typed IR."""

    if profile.route_kind == RELATION_ROUTE:
        ir_kind = RELATION_IR
        operators = (RELATION_OPERATOR,)
        recurrent = True
        terminal = ("ENTITY_GOAL_RELATION",)
    elif profile.route_kind == TEMPORAL_PAIR_ROUTE:
        ir_kind = TEMPORAL_IR
        operators = (TEMPORAL_PAIR_OPERATOR,)
        recurrent = True
        terminal = ()
    elif profile.route_kind == TEMPORAL_SINGLE_ROUTE:
        ir_kind = TEMPORAL_IR
        operators = (TEMPORAL_SINGLE_OPERATOR,)
        recurrent = False
        terminal = ()
    elif profile.route_kind == COMPOSITE_ROUTE:
        ir_kind = COMPOSITE_IR
        operators = (RELATION_OPERATOR, TEMPORAL_PAIR_OPERATOR)
        recurrent = True
        terminal = ("ENTITY_GOAL_RELATION",)
    else:
        ir_kind = UNSUPPORTED_IR
        operators = ()
        recurrent = False
        terminal = ()
    return TargetIRRequirement.create(
        task_id=profile.task_id,
        target_domain=TARGET_DOMAIN,
        target_interface=TARGET_INTERFACE,
        target_grounder_sha256=target_grounder_sha256,
        ir_kind=ir_kind,
        operator_sequence=operators,
        recurrent=recurrent,
        terminal_predicate_families=terminal,
        grounder_qualified=True,
        formal_outcome_read=False,
    )


def select_program(
    sources: Sequence[SourceIRContract], profile: AGQAProgramProfile, *,
    target_grounder_sha256: str,
) -> dict[str, object]:
    requirement = target_requirement(
        profile, target_grounder_sha256=target_grounder_sha256,
    )
    return select_source_contract(sources, requirement)


__all__ = [
    "AGQAProgramProfile", "COMPOSITE_ROUTE", "RELATION_ROUTE",
    "TARGET_DOMAIN", "TARGET_INTERFACE", "TEMPORAL_PAIR_ROUTE",
    "TEMPORAL_SINGLE_ROUTE", "UNSUPPORTED_ROUTE", "profile_program",
    "select_program", "target_requirement",
]
