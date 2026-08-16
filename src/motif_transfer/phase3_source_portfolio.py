"""Outcome-blind applicability selection over source-induced typed programs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .phase3_typed_effect_induction import (
    target_trial_order,
    validate_typed_effect_program,
)


def _validate_artifact(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    body = dict(artifact)
    claimed = body.pop("artifact_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("source portfolio artifact hash mismatch")
    program = artifact.get("typed_effect_program")
    if not isinstance(program, Mapping):
        raise ValueError("source portfolio artifact omitted typed program")
    validate_typed_effect_program(program)
    return program


def _source_reliability(program: Mapping[str, Any]) -> float:
    calibration = program.get("cross_batch_calibration") or {}
    metrics = calibration.get("metrics") or {}
    authentic = metrics.get("authentic") or program.get("qualification_metrics") or {}
    value = authentic.get("accuracy")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("typed program omitted calibrated source accuracy")
    output = float(value)
    if not 0.0 <= output <= 1.0:
        raise ValueError("calibrated source accuracy is outside [0,1]")
    return output


def select_source_program_portfolio(
    artifacts: Sequence[Mapping[str, Any]], *,
    candidate_ids: Sequence[str],
    candidate_effects: Sequence[Mapping[str, Any]],
    target_grounding_sha256: str,
) -> dict[str, Any]:
    """Select a program by source reliability × target discrimination.

    The score is the calibrated probability that the source effect predicts a
    valuable intervention multiplied by the target-native neural margin
    between the first and second candidates for that effect.  It contains no
    source identity, target outcome, task reward, or named policy template.
    """

    ids = tuple(map(str, candidate_ids))
    rows = []
    for artifact in artifacts:
        program = _validate_artifact(artifact)
        program_sha = str(program["program_sha256"])
        effect_type = str(program["selected_effect_type"])
        order, reason = target_trial_order(program, candidate_effects)
        reliability = _source_reliability(program)
        if reason is None:
            values = [float(row[effect_type]) for row in candidate_effects]
            ordered_values = sorted(values, reverse=True)
            margin = ordered_values[0] - ordered_values[1]
            maximum = ordered_values[0]
            applicability = reliability * margin
        else:
            margin = maximum = applicability = 0.0
        rows.append({
            "artifact_sha256": str(artifact["artifact_sha256"]),
            "program_sha256": program_sha,
            "selected_effect_type": effect_type,
            "source_qualified": (
                program.get("status") == "SOURCE_TYPED_EFFECT_PROGRAM_QUALIFIED"
            ),
            "target_binding_admitted": reason is None,
            "target_binding_abstention_reason": reason,
            "source_calibrated_accuracy": reliability,
            "target_unique_argmax_margin": margin,
            "target_maximum_effect_probability": maximum,
            "applicability_score": applicability,
            "target_trial_order": list(order),
        })
    eligible = [
        row for row in rows
        if row["source_qualified"] and row["target_binding_admitted"]
    ]
    selected = max(eligible, key=lambda row: (
        row["applicability_score"],
        row["source_calibrated_accuracy"],
        row["target_unique_argmax_margin"],
        row["target_maximum_effect_probability"],
        row["program_sha256"],
    )) if eligible else None
    body = {
        "schema_version": "phase3-source-program-portfolio-selection-v1",
        "status": (
            "TARGET_APPLICABLE_SOURCE_PROGRAM_SELECTED" if selected else
            "SOURCE_PROGRAM_PORTFOLIO_ABSTAINED"
        ),
        "score_rule": (
            "SOURCE_CROSS_BATCH_CALIBRATED_ACCURACY_TIMES_"
            "TARGET_NATIVE_UNIQUE_ARGMAX_MARGIN"
        ),
        "candidate_ids": list(ids),
        "target_grounding_sha256": str(target_grounding_sha256),
        "target_typed_effects_sha256": stable_hash(list(candidate_effects)),
        "programs": sorted(rows, key=lambda row: row["program_sha256"]),
        "selected_artifact_sha256": (
            selected["artifact_sha256"] if selected else None
        ),
        "selected_program_sha256": (
            selected["program_sha256"] if selected else None
        ),
        "selected_effect_type": (
            selected["selected_effect_type"] if selected else None
        ),
        "source_identity_used_as_feature": False,
        "target_outcome_read": False,
    }
    return body | {"portfolio_receipt_sha256": stable_hash(body)}


def permute_selected_effect_binding(
    program: Mapping[str, Any], *, candidate_ids: Sequence[str],
    candidate_effects: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Create a deterministic, non-identity candidate/effect control."""

    validate_typed_effect_program(program)
    ids = tuple(map(str, candidate_ids))
    if len(ids) < 2 or len(ids) != len(candidate_effects):
        raise ValueError("effect permutation requires aligned multiple candidates")
    effect_type = str(program["selected_effect_type"])
    values = [float(row[effect_type]) for row in candidate_effects]
    offset = 1 + int(stable_hash({
        "program_sha256": program["program_sha256"],
        "candidate_ids": list(ids),
        "control": "TARGET_CANDIDATE_EFFECT_BINDING_PERMUTATION_V1",
    })[:8], 16) % (len(values) - 1)
    permuted = []
    for index, row in enumerate(candidate_effects):
        updated = dict(row)
        updated[effect_type] = values[(index + offset) % len(values)]
        permuted.append(updated)
    body = {
        "schema_version": "phase3-target-effect-binding-control-v1",
        "program_sha256": program["program_sha256"],
        "selected_effect_type": effect_type,
        "candidate_ids": list(ids),
        "offset": offset,
        "authentic_effect_values_sha256": stable_hash(values),
        "permuted_effect_values_sha256": stable_hash([
            row[effect_type] for row in permuted
        ]),
        "nonidentity": [row[effect_type] for row in permuted] != values,
        "target_outcome_read": False,
    }
    return tuple(permuted), body | {
        "effect_binding_control_receipt_sha256": stable_hash(body)
    }


__all__ = [
    "permute_selected_effect_binding", "select_source_program_portfolio",
]
