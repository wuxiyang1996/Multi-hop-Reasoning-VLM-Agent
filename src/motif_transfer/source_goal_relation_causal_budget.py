"""Causal terminal-candidate restriction for source relation induction.

The V3 relation learner considers every scalar that happens to be constant at
successful terminals.  With a small prefix this can select a nuisance such as
entity cardinality even though no successful macro intervention changed that
feature.  This module applies one source-only identification rule before the
unchanged V3 learner runs: a terminal candidate must be a feature whose value
was actually changed by the successful intervention tuples.

No feature name, target observation, target action, or target outcome is
provided to the rule.  Ambiguous or absent intervention-linked features fail
closed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .contracts import stable_hash
from .source_goal_relation_induction import (
    induce_goal_relation_macro_program,
    validate_goal_relation_macro_dataset,
    validate_goal_relation_macro_program,
)


PROJECTION_VERSION = "SOURCE_INTERVENTION_LINKED_TERMINAL_CANDIDATES_V1"
ARTIFACT_REVISION = "SOURCE_GOAL_RELATION_CAUSAL_BUDGET_V1"


def _successful_candidates(dataset: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        candidate
        for episode in dataset.get("episodes") or ()
        for candidate in episode.get("candidates") or ()
        if candidate.get("success_from_state_only")
    ]


def intervention_linked_features(
    dataset: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return features increased in every successful source macro path."""

    validate_goal_relation_macro_dataset(dataset)
    successful = _successful_candidates(dataset)
    if not successful:
        return ()
    per_candidate = []
    for candidate in successful:
        features = {
            str(row["effect"]["changed_feature"])
            for row in candidate.get("macro_tuples") or ()
            if row["effect"].get("change_sign") == "INCREASE"
        }
        if not features:
            return ()
        per_candidate.append(features)
    shared = set.intersection(*per_candidate)
    return tuple(sorted(shared))


def project_intervention_linked_terminals(
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove terminal scalars unsupported by a successful intervention."""

    features = intervention_linked_features(dataset)
    if len(features) != 1:
        raise ValueError(
            "source interventions do not identify exactly one terminal feature"
        )
    selected = features[0]
    projected = deepcopy(dict(dataset))
    original_sha256 = str(projected.pop("dataset_sha256"))
    for episode in projected.get("episodes") or ():
        for candidate in episode.get("candidates") or ():
            terminal = dict(candidate.get("terminal_features") or {})
            if selected not in terminal:
                raise ValueError("intervention-linked feature is not observable")
            candidate["terminal_features"] = {selected: terminal[selected]}
    projected["terminal_candidate_projection"] = {
        "version": PROJECTION_VERSION,
        "authority": "SOURCE_SUCCESSFUL_STATE_ACTION_EFFECT_NEXT_STATE_ONLY",
        "selected_features": [selected],
        "selection_rule": (
            "FEATURE_CHANGED_WITH_POSITIVE_SIGN_IN_EVERY_SUCCESSFUL_MACRO_PATH"
        ),
        "unprojected_dataset_sha256": original_sha256,
        "target_data_read": False,
        "named_terminal_feature_provided": False,
    }
    projected["dataset_sha256"] = stable_hash(projected)
    validate_goal_relation_macro_dataset(projected)
    return projected


def induce_causal_goal_relation_program(
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Run V3 induction after the source-only causal candidate projection."""

    projected = project_intervention_linked_terminals(dataset)
    artifact = induce_goal_relation_macro_program(projected)
    body = dict(artifact)
    body.pop("artifact_sha256", None)
    body["induction_revision"] = ARTIFACT_REVISION
    body["terminal_candidate_authority"] = dict(
        projected["terminal_candidate_projection"]
    )
    result = body | {"artifact_sha256": stable_hash(body)}
    validate_causal_goal_relation_program(result)
    return result


def validate_causal_goal_relation_program(
    artifact: Mapping[str, Any],
) -> None:
    validate_goal_relation_macro_program(artifact)
    if artifact.get("induction_revision") != ARTIFACT_REVISION:
        raise ValueError("unsupported causal relation induction revision")
    authority = artifact.get("terminal_candidate_authority") or {}
    if authority.get("version") != PROJECTION_VERSION:
        raise ValueError("missing intervention-linked terminal authority")
    if authority.get("target_data_read") is not False:
        raise ValueError("target data leaked into causal terminal selection")
    if authority.get("named_terminal_feature_provided") is not False:
        raise ValueError("terminal feature was supplied by name")
    selected = list(authority.get("selected_features") or ())
    predicates = artifact.get("program", {}).get("terminal_predicates") or ()
    if len(selected) != 1 or {
        str(row.get("feature")) for row in predicates
    } != set(map(str, selected)):
        raise ValueError("terminal predicate escaped intervention-linked support")


__all__ = [
    "ARTIFACT_REVISION",
    "PROJECTION_VERSION",
    "induce_causal_goal_relation_program",
    "intervention_linked_features",
    "project_intervention_linked_terminals",
    "validate_causal_goal_relation_program",
]
