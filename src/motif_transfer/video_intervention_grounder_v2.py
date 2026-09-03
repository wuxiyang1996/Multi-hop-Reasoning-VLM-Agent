"""Fail-closed contract for intervention-grounded video effect ledgers."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Iterable, Mapping

from .phase3_typed_effect_induction import TYPED_EFFECTS


LEDGER_SCHEMA_VERSION = "video-intervention-ledger-v2"
REQUIRED_HORIZONS = (1, 4, 8)
REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "record_id",
    "benchmark",
    "video_id",
    "split",
    "belief_state_before",
    "intervention",
    "observations_by_horizon",
    "typed_effects",
    "effect_derivation",
    "belief_state_after",
    "transition",
    "executability_by_horizon",
    "blindness",
)
FORBIDDEN_READ_FLAGS = (
    "gold_answer_read",
    "formal_success_read",
    "official_scene_graph_read",
    "functional_program_read",
    "source_identity_read",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def validate_intervention_ledger(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return validation errors; an empty tuple means training-eligible."""

    errors: list[str] = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in record:
            errors.append(f"MISSING_TOP_LEVEL:{field}")
    if errors:
        return tuple(errors)

    if record["schema_version"] != LEDGER_SCHEMA_VERSION:
        errors.append("INVALID_SCHEMA_VERSION")
    if record["split"] != "development":
        errors.append("NON_DEVELOPMENT_SPLIT")
    if not all(
        isinstance(record.get(field), str) and bool(record[field])
        for field in ("record_id", "benchmark", "video_id")
    ):
        errors.append("INVALID_IDENTITY_FIELDS")

    before = record["belief_state_before"]
    after = record["belief_state_after"]
    if not isinstance(before, Mapping) or not _is_hash(before.get("state_sha256")):
        errors.append("INVALID_BELIEF_STATE_BEFORE")
    if not isinstance(after, Mapping) or not _is_hash(after.get("state_sha256")):
        errors.append("INVALID_BELIEF_STATE_AFTER")

    intervention = record["intervention"]
    if not isinstance(intervention, Mapping):
        errors.append("INVALID_INTERVENTION")
    else:
        for field in ("intervention_id", "operator_type", "candidate_id"):
            if not isinstance(intervention.get(field), str) or not intervention[field]:
                errors.append(f"INVALID_INTERVENTION:{field}")

    observations = record["observations_by_horizon"]
    if not isinstance(observations, Mapping):
        errors.append("INVALID_OBSERVATIONS_BY_HORIZON")
    else:
        for horizon in REQUIRED_HORIZONS:
            receipt = observations.get(str(horizon), observations.get(horizon))
            if not isinstance(receipt, Mapping):
                errors.append(f"MISSING_HORIZON_OBSERVATION:{horizon}")
                continue
            if not _is_hash(receipt.get("receipt_sha256")):
                errors.append(f"INVALID_HORIZON_RECEIPT:{horizon}")
            evidence = receipt.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"EMPTY_HORIZON_EVIDENCE:{horizon}")

    effects = record["typed_effects"]
    if not isinstance(effects, Mapping) or set(effects) != set(TYPED_EFFECTS):
        errors.append("INVALID_TYPED_EFFECT_SCHEMA")
    else:
        for effect in TYPED_EFFECTS:
            value = effects[effect]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                errors.append(f"INVALID_TYPED_EFFECT_VALUE:{effect}")

    derivation = record["effect_derivation"]
    if not isinstance(derivation, Mapping):
        errors.append("INVALID_EFFECT_DERIVATION")
    else:
        if derivation.get("kind") != "MEASURED_INTERVENTION_BELIEF_DELTA":
            errors.append("UNSUPPORTED_EFFECT_DERIVATION")
        if derivation.get("human_formula_used") is not False:
            errors.append("HUMAN_FORMULA_NOT_EXCLUDED")
        if derivation.get("target_outcome_or_gold_used") is not False:
            errors.append("TARGET_OUTCOME_NOT_EXCLUDED")
        if not _is_hash(derivation.get("derivation_receipt_sha256")):
            errors.append("INVALID_DERIVATION_RECEIPT")

    transition = record["transition"]
    if not isinstance(transition, Mapping):
        errors.append("INVALID_TRANSITION")
    elif isinstance(before, Mapping) and isinstance(after, Mapping) and isinstance(
        intervention, Mapping
    ):
        if transition.get("from_state_sha256") != before.get("state_sha256"):
            errors.append("TRANSITION_FROM_STATE_MISMATCH")
        if transition.get("to_state_sha256") != after.get("state_sha256"):
            errors.append("TRANSITION_TO_STATE_MISMATCH")
        if transition.get("intervention_id") != intervention.get("intervention_id"):
            errors.append("TRANSITION_INTERVENTION_MISMATCH")

    executable = record["executability_by_horizon"]
    if not isinstance(executable, Mapping):
        errors.append("INVALID_EXECUTABILITY_BY_HORIZON")
    else:
        for horizon in REQUIRED_HORIZONS:
            value = executable.get(str(horizon), executable.get(horizon))
            if not isinstance(value, bool):
                errors.append(f"INVALID_EXECUTABILITY:{horizon}")

    blindness = record["blindness"]
    if not isinstance(blindness, Mapping):
        errors.append("INVALID_BLINDNESS_RECEIPT")
    else:
        for flag in FORBIDDEN_READ_FLAGS:
            if blindness.get(flag) is not False:
                errors.append(f"FORBIDDEN_READ_NOT_FALSE:{flag}")

    return tuple(errors)


def summarize_ledger_readiness(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize eligibility without treating legacy/static receipts as tuples."""

    errors = Counter()
    eligible = 0
    total = 0
    eligible_videos: set[str] = set()
    for row in rows:
        total += 1
        row_errors = validate_intervention_ledger(row)
        if row_errors:
            errors.update(row_errors)
        else:
            eligible += 1
            eligible_videos.add(str(row["video_id"]))
    return {
        "rows_scanned": total,
        "eligible_records": eligible,
        "eligible_unique_videos": len(eligible_videos),
        "ineligibility_reasons": dict(sorted(errors.items())),
    }
