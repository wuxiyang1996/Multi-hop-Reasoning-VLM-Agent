"""Induce source-specific typed-effect programs from intervention traces.

The Phase-3 V1 AttemptLedger established that a controller can be induced from
source intervention receipts.  It did *not* establish source-specific
transfer: its candidate order was a reward-blind hash rank.  This module has a
narrower and stricter job.  It learns which temporally typed transition effect
is predictive of long-horizon source value, qualifies that choice on disjoint
source states, and otherwise induces abstention.

Only generic transition measurements are admitted.  Source-native action
tokens, game names, and target outcomes are never features of the learned
operator.  A target-native neural grounder may emit probabilities for the same
opaque effect types, allowing the unchanged symbolic operator to order target
candidates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping, Sequence

from .contracts import stable_hash
from .phase3_source_induction import validate_row_hashes


# These are measurement types, not policy templates.  The identifiers say
# when an effect was observed in a transition chain; they do not encode a game
# action or a target-domain decision.
IMMEDIATE_EFFECT = "EFFECT_BY_TRANSITION_1"
SHORT_EFFECT = "EFFECT_BY_TRANSITION_4"
MEDIUM_EFFECT = "EFFECT_BY_TRANSITION_8"
PERSISTENCE_EFFECT = "EXECUTABLE_TRANSITION_PERSISTENCE"
TYPED_EFFECTS = (
    IMMEDIATE_EFFECT,
    SHORT_EFFECT,
    MEDIUM_EFFECT,
    PERSISTENCE_EFFECT,
)

SOURCE_SPLIT_MAP = {
    "development": "discovery",
    "qualification": "qualification",
    "heldout": "heldout",
    "discovery": "discovery",
    "calibration": "qualification",
    "fresh": "heldout",
}


@dataclass(frozen=True)
class TypedCandidate:
    """One anonymized intervention and its typed transition measurements."""

    candidate_rank: int
    effect_values: tuple[tuple[str, float], ...]
    long_horizon_value: float
    transition_receipt_sha256: str

    def effect(self, effect_type: str) -> float:
        values = dict(self.effect_values)
        if effect_type not in values:
            raise ValueError(f"candidate omitted typed effect: {effect_type}")
        return values[effect_type]


@dataclass(frozen=True)
class TypedInterventionSet:
    snapshot_sha256: str
    source_split: str
    candidates: tuple[TypedCandidate, ...]
    verified_candidate_rank: int


def _endpoint(row: Mapping[str, Any], endpoint: int, horizon: int) -> float:
    cumulative = row.get("cumulative_returns") or {}
    effective = min(int(endpoint), int(horizon))
    key = f"h{effective}"
    if key in cumulative:
        return float(cumulative[key])
    # Old receipts stored only a fixed endpoint set.  The primary endpoint is
    # always present and is the conservative fallback when H < requested H.
    return float(cumulative[f"h{horizon}"])


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    low = min(values)
    high = max(values)
    if high == low:
        return tuple(0.0 for _ in values)
    scale = high - low
    return tuple((float(value) - low) / scale for value in values)


def typed_intervention_sets_from_rows(
    rows: Sequence[Mapping[str, Any]], *, primary_horizon: int,
) -> tuple[tuple[TypedInterventionSet, ...], dict[str, Any]]:
    """Convert stable matched rollouts into anonymized typed effect sets.

    Duplicate executions are required to agree byte-for-byte on official
    return, terminal state, transition receipts, and execution length.  The
    exported examples contain no source action or candidate ID.
    """

    if primary_horizon < 1:
        raise ValueError("primary_horizon must be positive")
    validate_row_hashes(rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["snapshot_id"])].append(row)

    output: list[TypedInterventionSet] = []
    exclusions: dict[str, int] = defaultdict(int)
    for snapshot_id, snapshot_rows in sorted(grouped.items()):
        by_rank: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in snapshot_rows:
            by_rank[int(row["candidate_rank"])].append(row)
        ranks = list(range(len(by_rank)))
        if sorted(by_rank) != ranks:
            exclusions["INCOMPLETE_CANDIDATE_SET"] += 1
            continue
        representatives: list[Mapping[str, Any]] = []
        unstable = False
        for rank in ranks:
            repeats = by_rank[rank]
            if len(repeats) < 2 or any(
                str(row.get("status")) != "INTERVENTION_OBSERVED"
                for row in repeats
            ):
                unstable = True
                break
            signatures = {
                stable_hash({
                    "cumulative_returns": row.get("cumulative_returns"),
                    "observed_actions": row.get("observed_actions"),
                    "transition_hashes": row.get("transition_hashes"),
                    "transition_effects": row.get("transition_effects"),
                    "terminated": row.get("terminated"),
                    "truncated": row.get("truncated"),
                })
                for row in repeats
            }
            if len(signatures) != 1:
                unstable = True
                break
            representatives.append(repeats[0])
        if unstable:
            exclusions["UNSTABLE_OR_FAILED_INTERVENTION"] += 1
            continue

        long_values = [
            _endpoint(row, primary_horizon, primary_horizon)
            for row in representatives
        ]
        maximum = max(long_values)
        best = [index for index, value in enumerate(long_values) if value == maximum]
        if len(best) != 1 or maximum <= min(long_values):
            exclusions["NO_UNIQUE_LONG_HORIZON_EFFECT"] += 1
            continue

        raw_effects = {
            IMMEDIATE_EFFECT: [
                _endpoint(row, 1, primary_horizon) for row in representatives
            ],
            SHORT_EFFECT: [
                _endpoint(row, 4, primary_horizon) for row in representatives
            ],
            MEDIUM_EFFECT: [
                _endpoint(row, 8, primary_horizon) for row in representatives
            ],
            PERSISTENCE_EFFECT: [
                float(row["observed_actions"]) / primary_horizon
                for row in representatives
            ],
        }
        normalized = {
            effect_type: _normalize(values)
            for effect_type, values in raw_effects.items()
        }
        candidates = []
        for rank, row in enumerate(representatives):
            receipt_body = {
                "transition_hashes": list(row.get("transition_hashes") or ()),
                "transition_effects": list(row.get("transition_effects") or ()),
                "observed_actions": int(row["observed_actions"]),
                "terminal": bool(row.get("terminated")),
                "truncated": bool(row.get("truncated")),
                "effect_values": {
                    effect_type: normalized[effect_type][rank]
                    for effect_type in TYPED_EFFECTS
                },
            }
            candidates.append(TypedCandidate(
                candidate_rank=rank,
                effect_values=tuple(
                    (effect_type, normalized[effect_type][rank])
                    for effect_type in TYPED_EFFECTS
                ),
                long_horizon_value=long_values[rank],
                transition_receipt_sha256=stable_hash(receipt_body),
            ))
        split_values = {str(row["source_split"]) for row in representatives}
        if len(split_values) != 1:
            raise ValueError("one snapshot spans multiple source splits")
        raw_split = next(iter(split_values))
        if raw_split not in SOURCE_SPLIT_MAP:
            raise ValueError(f"unsupported source split: {raw_split}")
        output.append(TypedInterventionSet(
            snapshot_sha256=stable_hash({
                "snapshot_id": snapshot_id,
                "transition_receipts": [
                    row.transition_receipt_sha256 for row in candidates
                ],
            }),
            source_split=SOURCE_SPLIT_MAP[raw_split],
            candidates=tuple(candidates),
            verified_candidate_rank=best[0],
        ))
    audit = {
        "snapshots": len(grouped),
        "eligible_intervention_sets": len(output),
        "exclusions": dict(sorted(exclusions.items())),
        "native_action_tokens_exported": False,
        "target_data_read": False,
        "typed_effects": list(TYPED_EFFECTS),
        "explicit_transition_tuple_receipts": sum(
            bool((row.get("transition_effects") or ())) for row in rows
        ),
    }
    return tuple(output), audit


def _permuted_values(
    row: TypedInterventionSet, effect_type: str,
) -> tuple[float, ...]:
    values = tuple(candidate.effect(effect_type) for candidate in row.candidates)
    count = len(values)
    if count < 2:
        return values
    offset = 1 + int(stable_hash({
        "snapshot_sha256": row.snapshot_sha256,
        "effect_type": effect_type,
        "control": "DETERMINISTIC_EFFECT_BINDING_SHUFFLE_V1",
    })[:8], 16) % (count - 1)
    return tuple(values[(index + offset) % count] for index in range(count))


def evaluate_effect_type(
    examples: Iterable[TypedInterventionSet], *, effect_type: str,
    source_split: str, shuffled_effects: bool = False,
) -> dict[str, Any]:
    if effect_type not in TYPED_EFFECTS:
        raise ValueError(f"unknown typed effect: {effect_type}")
    selected = [row for row in examples if row.source_split == source_split]
    correct = 0
    varying = 0
    unique_argmax = 0
    for row in selected:
        values = (
            _permuted_values(row, effect_type)
            if shuffled_effects else
            tuple(candidate.effect(effect_type) for candidate in row.candidates)
        )
        varying += int(max(values) > min(values))
        winners = [index for index, value in enumerate(values) if value == max(values)]
        unique_argmax += int(len(winners) == 1)
        # Stable rank tie-breaking is part of the metric, but target execution
        # will abstain on a tie.  Reporting both prevents hidden tie success.
        predicted = min(winners)
        correct += int(predicted == row.verified_candidate_rank)
    total = len(selected)
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "varying_effect_sets": varying,
        "varying_effect_fraction": varying / total if total else 0.0,
        "unique_argmax_sets": unique_argmax,
        "unique_argmax_fraction": unique_argmax / total if total else 0.0,
        "shuffled_effects": bool(shuffled_effects),
    }


def induce_typed_effect_program(
    examples: Sequence[TypedInterventionSet], *,
    source_receipts_sha256: str,
    minimum_discovery_examples: int = 8,
    minimum_qualification_examples: int = 8,
    minimum_discovery_accuracy: float = 0.50,
    minimum_qualification_accuracy: float = 0.50,
    minimum_qualification_varying_fraction: float = 0.50,
    minimum_authentic_minus_shuffled: float = 0.25,
) -> dict[str, Any]:
    """Induce one sparse typed operator or an explicit abstention program."""

    discovery_metrics = {
        effect_type: evaluate_effect_type(
            examples, effect_type=effect_type, source_split="discovery",
        )
        for effect_type in TYPED_EFFECTS
    }
    # Induction reads only discovery outcomes.  Tuple order is the declared,
    # source-independent complexity tie-break; qualification cannot choose a
    # different operator.
    effect_type = max(TYPED_EFFECTS, key=lambda name: (
        discovery_metrics[name]["correct"],
        discovery_metrics[name]["varying_effect_sets"],
        -TYPED_EFFECTS.index(name),
    ))
    qualification = evaluate_effect_type(
        examples, effect_type=effect_type, source_split="qualification",
    )
    qualification_shuffled = evaluate_effect_type(
        examples, effect_type=effect_type, source_split="qualification",
        shuffled_effects=True,
    )
    discovery = discovery_metrics[effect_type]
    gates = {
        "minimum_discovery_examples": (
            discovery["examples"] >= minimum_discovery_examples
        ),
        "minimum_qualification_examples": (
            qualification["examples"] >= minimum_qualification_examples
        ),
        "discovery_predictive_accuracy": (
            discovery["accuracy"] >= minimum_discovery_accuracy
        ),
        "qualification_predictive_accuracy": (
            qualification["accuracy"] >= minimum_qualification_accuracy
        ),
        "qualification_effect_is_observably_varying": (
            qualification["varying_effect_fraction"]
            >= minimum_qualification_varying_fraction
        ),
        "qualification_beats_shuffled_effect_binding": (
            qualification["accuracy"] - qualification_shuffled["accuracy"]
            >= minimum_authentic_minus_shuffled
        ),
    }
    qualified = all(gates.values())
    operator_core = {
        "preconditions": {
            "source_qualification_passed": qualified,
            "target_candidate_count_minimum": 2,
            "target_typed_effect_required": effect_type,
            "target_unique_argmax_required": True,
        },
        "score": {
            "kind": "ARGMAX_SINGLE_TYPED_EFFECT",
            "effect_type": effect_type,
            "coefficient": 1.0,
        },
        "state_delta": [
            ["active_candidate", "UNIQUE_TYPED_EFFECT_ARGMAX"],
            ["attempt_ledger", "MARK_ACTIVE_AS_TRIED"],
        ],
    }
    operator = {
        "operator_id": f"OP_{stable_hash(operator_core)[:16]}",
        **operator_core,
    }
    thresholds = {
        "minimum_discovery_examples": minimum_discovery_examples,
        "minimum_qualification_examples": minimum_qualification_examples,
        "minimum_discovery_accuracy": minimum_discovery_accuracy,
        "minimum_qualification_accuracy": minimum_qualification_accuracy,
        "minimum_qualification_varying_fraction": (
            minimum_qualification_varying_fraction
        ),
        "minimum_authentic_minus_shuffled": minimum_authentic_minus_shuffled,
    }
    body = {
        "schema_version": "phase3-source-induced-typed-effect-program-v2",
        "status": (
            "SOURCE_TYPED_EFFECT_PROGRAM_QUALIFIED" if qualified
            else "SOURCE_TYPED_EFFECT_ABSTENTION_INDUCED"
        ),
        "source_receipts_sha256": str(source_receipts_sha256),
        "induction_authority": (
            "SOURCE_DISCOVERY_TRANSITION_TUPLES_ONLY;SOURCE_QUALIFICATION_"
            "CALIBRATION;NO_SOURCE_IDENTITY_FEATURE;NO_TARGET_DATA"
        ),
        "type_system": {
            "effect_types": list(TYPED_EFFECTS),
            "value_domain": "NORMALIZED_SCALAR_[0,1]",
            "measurement": (
                "CONTENT_ADDRESSED_STATE_TRANSITIONS_WITH_OFFICIAL_EFFECT_"
                "AND_EXECUTABLE_PREFIX_LENGTH"
            ),
        },
        "operators": [operator] if qualified else [],
        "selected_effect_type": effect_type,
        "discovery_metrics": discovery_metrics,
        "qualification_metrics": qualification,
        "qualification_shuffled_effect_metrics": qualification_shuffled,
        "qualification_gates": gates,
        "abstention_rule": {
            "source_not_qualified": "ABSTAIN",
            "missing_target_typed_effect": "ABSTAIN",
            "nonfinite_target_typed_effect": "ABSTAIN",
            "nonunique_target_argmax": "ABSTAIN",
            "candidate_set_smaller_than_two": "ABSTAIN",
        },
        "thresholds": thresholds,
        "target_data_read": False,
        "native_action_tokens_exported": False,
        "source_identity_used_as_feature": False,
        "forbidden_named_policy_tokens": [
            "EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY",
        ],
    }
    return body | {"program_sha256": stable_hash(body)}


def validate_typed_effect_program(program: Mapping[str, Any]) -> None:
    body = dict(program)
    claimed = body.pop("program_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("typed-effect program hash mismatch")
    if program.get("schema_version") != (
        "phase3-source-induced-typed-effect-program-v2"
    ):
        raise ValueError("unsupported typed-effect program schema")
    if program.get("selected_effect_type") not in TYPED_EFFECTS:
        raise ValueError("typed-effect program selected an unknown type")
    if program.get("target_data_read") is not False:
        raise ValueError("typed-effect program does not attest target-free induction")
    serialized = json.dumps(program, sort_keys=True)
    for token in program.get("forbidden_named_policy_tokens") or ():
        if serialized.count(str(token)) != 1:
            raise ValueError("named policy token leaked outside forbidden audit list")


def target_trial_order(
    program: Mapping[str, Any],
    candidate_effects: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, ...], str | None]:
    """Apply a qualified source program to target-native neural effects."""

    validate_typed_effect_program(program)
    if program.get("status") != "SOURCE_TYPED_EFFECT_PROGRAM_QUALIFIED":
        return (), "SOURCE_TYPED_EFFECT_PROGRAM_NOT_QUALIFIED"
    if len(candidate_effects) < 2:
        return (), "TARGET_CANDIDATE_SET_HAS_FEWER_THAN_TWO"
    effect_type = str(program["selected_effect_type"])
    values = []
    for row in candidate_effects:
        value = row.get(effect_type)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return (), "TARGET_TYPED_EFFECT_MISSING_OR_NONNUMERIC"
        value = float(value)
        if not 0.0 <= value <= 1.0:
            return (), "TARGET_TYPED_EFFECT_OUT_OF_RANGE"
        values.append(value)
    maximum = max(values)
    if sum(value == maximum for value in values) != 1:
        return (), "TARGET_TYPED_EFFECT_ARGMAX_NOT_UNIQUE"
    return tuple(sorted(
        range(len(values)), key=lambda index: (-values[index], index),
    )), None


__all__ = [
    "IMMEDIATE_EFFECT", "MEDIUM_EFFECT", "PERSISTENCE_EFFECT",
    "SHORT_EFFECT", "TYPED_EFFECTS", "TypedCandidate",
    "TypedInterventionSet", "evaluate_effect_type",
    "induce_typed_effect_program", "target_trial_order",
    "typed_intervention_sets_from_rows", "validate_typed_effect_program",
]
