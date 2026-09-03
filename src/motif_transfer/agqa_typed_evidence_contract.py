"""Typed evidence authorization for answer-blind AGQA candidate grounding.

TEMPURA exposes native Action Genome relation channels.  Several AGQA query
predicates are instead approximated by a disjunction of those channels (for
example, ``taking`` is approximated by ``carrying`` or ``holding``).  A proxy
relation is useful for proposing candidates, but is not exact evidence for the
requested event.  This module keeps that distinction explicit and requires an
independent exact-action view before a proxy proposal can be committed.

The contract depends only on the public target ontology, the parsed predicate,
and neural-view provenance.  It does not consume answers, programs, source
controllers, or target outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# These are semantic aliases/inverses of native Action Genome relation
# channels.  Spatial channels are emitted object->person by TEMPURA and are
# deliberately inverted by the target adapter to match person->object query
# wording.  This mapping is an interface declaration, not an outcome-fitted
# reliability list.
NATIVE_RELATION_PROJECTION = {
    "beneath": "inverse(spatial:above)",
    "below": "inverse(spatial:above)",
    "above": "inverse(spatial:beneath)",
    "in front of": "inverse(spatial:behind)",
    "behind": "inverse(spatial:in_front_of)",
    "on the side of": "spatial:on_the_side_of",
    "in": "spatial:in",
    "touching": "contact:touching",
    "carrying": "contact:carrying",
    "holding": "contact:holding",
    "wearing": "contact:wearing",
    "sitting on": "contact:sitting_on",
    "standing on": "contact:standing_on",
    "leaning on": "contact:leaning_on",
    "watching": "attention:looking_at",
    "looking at": "attention:looking_at",
}


@dataclass(frozen=True)
class TypedEvidenceDecision:
    predicate: str
    evidence_kind: str
    native_projection: str | None
    sgdet_supported: bool
    slowfast_exact_action_supported: bool
    same_entity_cross_view_agreement: bool
    authorized: bool


def canonical_predicate(value: str) -> str:
    return " ".join(str(value).casefold().replace("_", " ").split())


def authorize_typed_candidate(
    predicate: str,
    sources: Iterable[str],
) -> TypedEvidenceDecision:
    """Authorize direct relations or independently confirmed action proxies."""

    value = canonical_predicate(predicate)
    source_set = frozenset(str(source).casefold().strip() for source in sources)
    sgdet = "sgdet" in source_set
    slowfast = "slowfast" in source_set
    projection = NATIVE_RELATION_PROJECTION.get(value)
    if projection is not None:
        kind = "NATIVE_RELATION"
        authorized = sgdet
    else:
        # The upstream query planner only emits predicates admitted by the
        # frozen target adapter.  A non-native predicate is therefore an
        # action proxy and must agree with the independent exact-action view.
        kind = "ACTION_PROXY"
        authorized = sgdet and slowfast
    return TypedEvidenceDecision(
        predicate=value,
        evidence_kind=kind,
        native_projection=projection,
        sgdet_supported=sgdet,
        slowfast_exact_action_supported=slowfast,
        same_entity_cross_view_agreement=sgdet and slowfast,
        authorized=authorized,
    )


__all__ = [
    "NATIVE_RELATION_PROJECTION",
    "TypedEvidenceDecision",
    "authorize_typed_candidate",
    "canonical_predicate",
]
