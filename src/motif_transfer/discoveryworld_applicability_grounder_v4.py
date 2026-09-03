"""Fail-closed neural candidate-set completeness for DiscoveryWorld.

The neural grounder may emit syntactically valid actions that leave the source
policy with no safe branch.  This module rejects that *set* before selection:
at least one parser-validated, neural-typed reversible POSITION must exist, or
one COMMIT must have an exact positive-effect witness.  If the legacy selector
would fall back to an unwitnessed COMMIT, the module chooses a POSITION from
the same neural set.  It never creates a target-native action.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .discoveryworld_sokoban_transfer import (
    DiscoveryWorldGroundedCandidate,
    DiscoveryWorldSelectionReceipt,
    DiscoveryWorldTargetBinding,
    TARGET_GROUNDER_SYSTEM_PROMPT,
    commit_available,
    evidence_supported,
    grounder_prompt_payload,
    parse_grounded_candidates,
    positive_commit_effect_witnessed,
    positive_commit_effect_kind,
    select_candidate,
    target_bound_position,
)


TRANSPORT_SUFFIX = "\nReturn one valid json object."


def source_applicability_complete(
    candidates: tuple[DiscoveryWorldGroundedCandidate, ...],
    observation,
    target_binding: DiscoveryWorldTargetBinding,
) -> bool:
    # parse_grounded_candidates has already validated the native action and
    # its current API preconditions.  Neural evidence is explanatory metadata
    # for a reversible POSITION; stale extra evidence must not force an unsafe
    # COMMIT fallback.  Irreversible COMMIT remains witness-gated below.
    safe_position = any(candidate.target_role == "POSITION" for candidate in candidates)
    witnessed_commit = any(
        candidate.target_role == "COMMIT"
        and positive_commit_effect_witnessed(
            candidate, observation, target_binding,
        )
        for candidate in candidates
    )
    return safe_position or witnessed_commit


def select_source_safe_candidate(
    condition: str,
    candidates,
    observation,
    *,
    target_binding=None,
    prerequisite_threshold: float = 0.75,
    positive_effect_threshold: float = 0.65,
):
    """Prevent the legacy empty-pool fallback from choosing an unsafe COMMIT."""

    selected, receipt = select_candidate(
        condition,
        candidates,
        observation,
        target_binding=target_binding,
        prerequisite_threshold=prerequisite_threshold,
        positive_effect_threshold=positive_effect_threshold,
    )
    if (
        condition != "authentic_sokoban_effect_plus_target"
        or selected.target_role != "COMMIT"
        or receipt.positive_commit_effect_witnessed
    ):
        return selected, receipt
    positions = [row for row in candidates if row.target_role == "POSITION"]
    if not positions:
        return selected, receipt
    selected = max(
        positions,
        key=lambda row: (
            row.information_gain_probability,
            row.positive_effect_probability,
            row.prerequisite_probability,
            row.candidate_sha256,
        ),
    )
    receipt = DiscoveryWorldSelectionReceipt.create(
        condition=condition,
        candidates=candidates,
        selected=selected,
        evidence_supported=evidence_supported(
            selected, observation, target_binding,
        ),
        target_bound_position=target_bound_position(selected, target_binding),
        commit_available=commit_available(selected, target_binding),
        positive_commit_effect_witnessed=positive_commit_effect_witnessed(
            selected, observation, target_binding,
        ),
        positive_commit_effect_kind=positive_commit_effect_kind(
            selected, observation, target_binding,
        ),
        selection_reason=(
            "SOURCE_UNWITNESSED_COMMIT_REJECTED_USE_NEURAL_TYPED_NATIVE_VALID_POSITION"
        ),
    )
    return selected, receipt


def call_applicability_complete_grounder(
    backend,
    observation,
    *,
    memory: str,
    hypotheses: tuple[str, ...],
    recent: list[dict[str, Any]],
    target_binding: DiscoveryWorldTargetBinding,
    attempts: int,
):
    """Retry neural grounding until the symbolic source has a safe branch."""

    schema_error = None
    audit = []
    for attempt in range(attempts):
        payload = grounder_prompt_payload(
            observation,
            memory=memory,
            hypotheses=hypotheses,
            recent=recent,
            target_binding=target_binding,
            schema_error=schema_error,
        )
        raw = backend.complete(
            "grounder", TARGET_GROUNDER_SYSTEM_PROMPT + TRANSPORT_SUFFIX, payload,
        )
        request_usage = dict(backend.last_usage)
        try:
            bundle, candidates = parse_grounded_candidates(raw, observation)
            if not source_applicability_complete(
                candidates, observation, target_binding,
            ):
                raise ValueError(
                    "candidate set has neither an evidence-supported/bound "
                    "POSITION nor a symbolically witnessed COMMIT"
                )
            audit.append({
                "attempt": attempt + 1,
                "accepted": True,
                "applicability_complete": True,
                "cache_hit": bool(request_usage.get("cache_hit")),
            })
            return bundle, candidates, raw, audit
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            audit.append({
                "attempt": attempt + 1,
                "accepted": False,
                "applicability_complete": False,
                "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "cache_hit": bool(request_usage.get("cache_hit")),
            })
    raise RuntimeError(
        f"target grounder exhausted applicability-complete attempts: {audit}"
    )


__all__ = [
    "TRANSPORT_SUFFIX",
    "call_applicability_complete_grounder",
    "select_source_safe_candidate",
    "source_applicability_complete",
]
