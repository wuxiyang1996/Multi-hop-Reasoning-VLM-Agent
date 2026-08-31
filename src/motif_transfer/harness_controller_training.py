"""Source-only supervision for a neural executor of the symbolic Harness.

The controller trained from these examples is deliberately narrower than a
domain actor.  It receives an anonymous, source-induced symbolic program and
target-native typed effects, then selects an operator binding, advances the
program state, or abstains.  It never receives native actions, source identity,
target outcomes, or a hand-authored EXPLORE/BACKTRACK/COMMIT label.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .phase3_source_function_induction import (
    QUALIFIED,
    function_trial_order,
    validate_source_function_program,
)
from .phase3_typed_effect_induction import TYPED_EFFECTS, TypedInterventionSet


ENTRY = "RANKED_CANDIDATE_ABSENT"
ACTIVE = "FUNCTION_CANDIDATE_ACTIVE"
TERMINAL = "TERMINAL"
ABSTAIN = "ABSTAIN"


CONTROLLER_SYSTEM_PROMPT = (
    "You are the neural executor of an anonymous typed symbolic controller. "
    "Use only the supplied program, symbolic state, and grounded typed effects. "
    "Select an operator binding, apply one declared transition, or abstain. "
    "Never invent a domain identity, native action, hidden predicate, or transition. "
    "Return the exact JSON object only; do not provide reasoning."
)


def format_controller_prompt(
    *, objective: str, input_payload: Mapping[str, Any],
) -> str:
    """Serialize the exact prompt contract shared by SFT and live inference."""

    if objective not in {"SELECT_OPERATOR", "APPLY_TRANSITION"}:
        raise ValueError(f"unsupported controller objective: {objective}")
    return (
        CONTROLLER_SYSTEM_PROMPT
        + "\n\nOBJECTIVE=" + objective
        + "\nCONTROLLER_INPUT="
        + json.dumps(
            dict(input_payload), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\nOUTPUT_JSON="
    )


@dataclass(frozen=True)
class ControllerSFTExample:
    """One auditable program-execution example.

    ``source_family`` and ``evidence_receipt_ids`` are audit metadata.  The SFT
    formatter must not place either field in the model prompt.
    """

    example_id: str
    source_family: str
    split: str
    objective: str
    control_variant: str
    input_payload: Mapping[str, Any]
    target_payload: Mapping[str, Any]
    evidence_receipt_ids: tuple[str, ...]
    derivation: str
    target_data_used: bool = False

    def validate(self) -> bool:
        body = asdict(self)
        claimed = body.pop("example_id")
        serialized = str(self.input_payload) + str(self.target_payload)
        return (
            claimed == stable_hash(body)
            and self.split in {"train", "validation", "source_held_out"}
            and self.objective in {"SELECT_OPERATOR", "APPLY_TRANSITION"}
            and bool(self.evidence_receipt_ids)
            and self.target_data_used is False
            and self.source_family not in serialized
            and all(token not in serialized for token in (
                "EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY",
            ))
        )


def anonymous_program_ir(program: Mapping[str, Any]) -> dict[str, Any]:
    """Export only the executable, domain-independent portion of a program."""

    validate_source_function_program(program)
    operators = []
    for index, operator in enumerate(program.get("operators") or ()):
        preconditions = dict(operator["preconditions"])
        if "source_qualification_passed" in preconditions:
            preconditions["program_qualification_passed"] = preconditions.pop(
                "source_qualification_passed"
            )
        operators.append({
            "operator_id": f"O{index}",
            "preconditions": preconditions,
            "score": dict(operator["score"]),
            "state_delta": list(operator["state_delta"]),
        })
    function = program["source_function"]
    return {
        "schema_version": "ANONYMOUS_TYPED_CONTROLLER_IR_V2",
        "qualification_status": (
            "QUALIFIED" if program["status"] == QUALIFIED else "ABSTENTION_INDUCED"
        ),
        "effect_vocabulary": list(program["shared_ir"]["effect_types"]),
        "operators": operators,
        "maximum_trials": int(function["maximum_trials"]),
        "required_observation_horizon": int(
            function["required_observation_horizon"]
        ),
        "transition_graph": {
            "entry_state": str(program["transition_graph"]["entry_state"]),
            "transitions": [dict(row) for row in program["transition_graph"]["transitions"]],
        },
        "abstention_rule": {
            ("program_not_qualified" if key == "source_not_qualified" else key): value
            for key, value in program["abstention_rule"].items()
        },
    }


def initial_controller_state() -> dict[str, Any]:
    return {
        "controller_state": ENTRY,
        "active_candidate": None,
        "attempted_candidates": [],
        "trials_used": 0,
    }


def _checked_state(state: Mapping[str, Any]) -> dict[str, Any]:
    controller_state = str(state.get("controller_state"))
    if controller_state not in {ENTRY, ACTIVE, TERMINAL, ABSTAIN}:
        raise ValueError("unsupported controller state")
    attempted = list(map(str, state.get("attempted_candidates") or ()))
    if len(set(attempted)) != len(attempted):
        raise ValueError("attempt ledger contains duplicates")
    trials = int(state.get("trials_used", 0))
    if trials < 0 or trials != len(attempted):
        raise ValueError("attempt ledger and trials_used disagree")
    active = state.get("active_candidate")
    if active is not None:
        active = str(active)
    if controller_state == ACTIVE and active not in attempted:
        raise ValueError("active candidate is absent from attempt ledger")
    return {
        "controller_state": controller_state,
        "active_candidate": active,
        "attempted_candidates": attempted,
        "trials_used": trials,
    }


def _abstention(state: Mapping[str, Any], reason: str) -> dict[str, Any]:
    next_state = dict(state)
    next_state["controller_state"] = ABSTAIN
    next_state["active_candidate"] = None
    return {
        "decision": ABSTAIN,
        "operator_id": None,
        "binding": None,
        "next_symbolic_state": next_state,
        "reason": str(reason),
    }


def execute_controller_step(
    program: Mapping[str, Any], *, state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]], observed_effect: str | None = None,
) -> dict[str, Any]:
    """Execute one symbolic controller step with fail-closed validation."""

    validate_source_function_program(program)
    current = _checked_state(state)
    controller_state = current["controller_state"]
    if controller_state == ENTRY:
        if observed_effect is not None:
            return _abstention(current, "OBSERVED_EFFECT_WITHOUT_ACTIVE_CANDIDATE")
        if current["trials_used"] >= int(program["source_function"]["maximum_trials"]):
            return _abstention(current, "PROGRAM_TRIAL_BUDGET_EXHAUSTED")
        ids = [str(row.get("candidate_id")) for row in candidates]
        if any(value in {"", "None"} for value in ids) or len(ids) != len(set(ids)):
            return _abstention(current, "TARGET_CANDIDATE_IDS_INVALID")
        effects = [row.get("effects") for row in candidates]
        order, reason = function_trial_order(program, effects)
        if reason is not None:
            if reason == "SOURCE_DOMAIN_FUNCTION_NOT_QUALIFIED":
                reason = "PROGRAM_NOT_QUALIFIED"
            return _abstention(current, reason)
        attempted = set(current["attempted_candidates"])
        selected = next((rank for rank in order if ids[rank] not in attempted), None)
        if selected is None:
            return _abstention(current, "PROGRAM_TRIAL_SET_EXHAUSTED")
        candidate_id = ids[selected]
        next_attempted = [*current["attempted_candidates"], candidate_id]
        next_state = {
            "controller_state": ACTIVE,
            "active_candidate": candidate_id,
            "attempted_candidates": next_attempted,
            "trials_used": len(next_attempted),
        }
        return {
            "decision": "EXECUTE_OPERATOR",
            "operator_id": "O0",
            "binding": {"active_candidate": candidate_id},
            "next_symbolic_state": next_state,
            "reason": "UNIQUE_FUNCTION_ARGMAX",
        }

    if controller_state != ACTIVE:
        return _abstention(current, "CONTROLLER_STATE_NOT_EXECUTABLE")
    guard = str(observed_effect or "OBSERVED_EFFECT_UNKNOWN")
    allowed = {
        "OBSERVED_EFFECT_HIGH", "OBSERVED_EFFECT_LOW", "OBSERVED_EFFECT_UNKNOWN",
    }
    if guard not in allowed:
        guard = "OBSERVED_EFFECT_UNKNOWN"
    edges = [
        row for row in program["transition_graph"]["transitions"]
        if row["from"] == ACTIVE and row["guard"] == guard
    ]
    if len(edges) != 1:
        return _abstention(current, "SOURCE_GRAPH_EDGE_MISSING_OR_NONUNIQUE")
    destination = str(edges[0]["to"])
    next_state = dict(current)
    next_state["controller_state"] = destination
    next_state["active_candidate"] = None
    if destination == ABSTAIN:
        return {
            "decision": ABSTAIN,
            "operator_id": None,
            "binding": None,
            "next_symbolic_state": next_state,
            "reason": guard,
        }
    return {
        "decision": "TERMINATE" if destination == TERMINAL else "ADVANCE_STATE",
        "operator_id": None,
        "binding": None,
        "next_symbolic_state": next_state,
        "reason": guard,
    }


def _candidate_payload(row: TypedInterventionSet) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"C{candidate.candidate_rank}",
            "effects": dict(candidate.effect_values),
        }
        for candidate in row.candidates
    ]


def _make_example(
    *, source_family: str, split: str, objective: str,
    control_variant: str,
    input_payload: Mapping[str, Any], target_payload: Mapping[str, Any],
    receipt_ids: Sequence[str], derivation: str,
) -> ControllerSFTExample:
    body = {
        "source_family": source_family,
        "split": split,
        "objective": objective,
        "control_variant": str(control_variant),
        "input_payload": dict(input_payload),
        "target_payload": dict(target_payload),
        "evidence_receipt_ids": tuple(map(str, receipt_ids)),
        "derivation": derivation,
        "target_data_used": False,
    }
    return ControllerSFTExample(example_id=stable_hash(body), **body)


def _rotated_effect_binding(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(candidates) < 2:
        return [dict(row) for row in candidates]
    return [
        {
            "candidate_id": str(row["candidate_id"]),
            "effects": dict(candidates[(index + 1) % len(candidates)]["effects"]),
        }
        for index, row in enumerate(candidates)
    ]


def _reindex_candidates(
    candidates: Sequence[Mapping[str, Any]], *, offset: int = 0,
) -> list[dict[str, Any]]:
    """Alpha-rename an anonymous candidate list without changing effects."""

    return [
        {
            "candidate_id": f"C{offset + index}",
            "effects": dict(candidate["effects"]),
        }
        for index, candidate in enumerate(candidates)
    ]


def _source_cardinality_variant(
    program: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    cardinality: int,
) -> list[dict[str, Any]]:
    """Construct a source-authorized subset/extension at one list cardinality.

    Subsets retain the source-program ranking prefix, while extensions add only
    zero-effect dominated candidates.  The frozen executor still recomputes the
    label; this helper never assigns a decision or target-domain meaning.
    """

    if cardinality < 2:
        raise ValueError("cardinality-equivariant variants start at two")
    effects = [candidate["effects"] for candidate in candidates]
    order, reason = function_trial_order(program, effects)
    if reason is not None:
        raise ValueError(f"source candidate set is not rankable: {reason}")
    if cardinality <= len(candidates):
        selected = [candidates[index] for index in order[:cardinality]]
    else:
        selected = list(candidates)
        selected.extend({
            "candidate_id": "PLACEHOLDER",
            "effects": {effect_type: 0.0 for effect_type in TYPED_EFFECTS},
        } for _ in range(cardinality - len(candidates)))
    return _reindex_candidates(selected)


def build_controller_sft_examples(
    *, source_family: str, split: str, program: Mapping[str, Any],
    intervention_sets: Sequence[TypedInterventionSet],
    augment_retry_equivariance: bool = False,
    augment_cardinality_equivariance: bool = False,
    augment_missing_schema_all_cardinalities: bool = False,
    cardinality_grid: Sequence[int] = (),
) -> tuple[ControllerSFTExample, ...]:
    """Create labels exclusively by executing a frozen source program."""

    ir = anonymous_program_ir(program)
    output = []
    for row in intervention_sets:
        receipts = tuple(candidate.transition_receipt_sha256 for candidate in row.candidates)
        candidates = _candidate_payload(row)
        state = initial_controller_state()

        def add_selection(
            variant: str, candidate_rows: Sequence[Mapping[str, Any]],
            candidate_state: Mapping[str, Any] = state,
        ) -> dict[str, Any]:
            payload = {
                "program": ir,
                "symbolic_state": dict(candidate_state),
                "candidate_effects": list(candidate_rows),
                "observed_effect": None,
            }
            target = execute_controller_step(
                program, state=candidate_state, candidates=candidate_rows,
            )
            output.append(_make_example(
                source_family=source_family, split=split,
                objective="SELECT_OPERATOR", control_variant=variant,
                input_payload=payload,
                target_payload=target, receipt_ids=receipts,
                derivation="FROZEN_SOURCE_PROGRAM_EXECUTION",
            ))
            return target

        authentic = add_selection("AUTHENTIC_TYPED_EFFECT_BINDING", candidates)
        # An abstention-induced program has no executable operator.  Repeating
        # every malformed-input control for it creates identical ABSTAIN labels
        # and teaches a harmful majority-class shortcut, so its authentic
        # fail-closed decision is the only admitted example for this receipt.
        if program["status"] != QUALIFIED:
            continue
        add_selection(
            "DETERMINISTIC_EFFECT_BINDING_SHUFFLE",
            _rotated_effect_binding(candidates),
        )
        if len(candidates) >= 2:
            control_index = int(row.snapshot_sha256[:8], 16) % 3
            if control_index == 0:
                tied = [
                    {
                        "candidate_id": row_["candidate_id"],
                        "effects": dict(candidates[0]["effects"]),
                    }
                    for row_ in candidates
                ]
                add_selection("NONUNIQUE_ARGMAX_CONTROL", tied)
            elif control_index == 1:
                add_selection("SINGLETON_CANDIDATE_CONTROL", candidates[:1])
            else:
                missing = [dict(item) for item in candidates]
                missing[0] = {
                    "candidate_id": missing[0]["candidate_id"],
                    "effects": {
                        key: value for key, value in missing[0]["effects"].items()
                        if key != TYPED_EFFECTS[-1]
                    },
                }
                add_selection("MISSING_TYPED_EFFECT_CONTROL", missing)

        if (
            augment_cardinality_equivariance
            and authentic["decision"] == "EXECUTE_OPERATOR"
        ):
            for cardinality in cardinality_grid:
                cardinality = int(cardinality)
                variant = _source_cardinality_variant(
                    program, candidates, cardinality,
                )
                add_selection(
                    "CARDINALITY_EQUIVARIANT_SOURCE_EXECUTION", variant,
                )
                add_selection(
                    "CARDINALITY_EQUIVARIANT_EFFECT_ROTATION",
                    _rotated_effect_binding(variant),
                )
                # Boundary-near matched controls make arity independent from
                # applicability: two candidates may execute, tie, or have an
                # invalid schema, whereas only the singleton must fail count.
                if cardinality in {2, 3}:
                    tied = [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "effects": dict(variant[0]["effects"]),
                        }
                        for candidate in variant
                    ]
                    add_selection(
                        "CARDINALITY_EQUIVARIANT_TIED_CONTROL", tied,
                    )
                if (
                    cardinality in {2, 3}
                    or augment_missing_schema_all_cardinalities
                ):
                    missing = [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "effects": dict(candidate["effects"]),
                        }
                        for candidate in variant
                    ]
                    missing[0]["effects"].pop(TYPED_EFFECTS[-1])
                    add_selection(
                        "CARDINALITY_EQUIVARIANT_MISSING_SCHEMA_CONTROL",
                        missing,
                    )
                if cardinality == 2:
                    add_selection(
                        "CARDINALITY_EQUIVARIANT_SINGLETON_BOUNDARY",
                        variant[:1],
                    )

        if authentic["decision"] != "EXECUTE_OPERATOR":
            # Qualification is population-level.  A particular candidate set
            # can still fail an applicability guard (most often a tied
            # function argmax), in which case no ACTIVE-state edge exists.
            continue
        active_state = authentic["next_symbolic_state"]
        for guard in (
            "OBSERVED_EFFECT_HIGH", "OBSERVED_EFFECT_LOW", "OBSERVED_EFFECT_UNKNOWN",
        ):
            transition_input = {
                "program": ir,
                "symbolic_state": active_state,
                "candidate_effects": candidates,
                "observed_effect": guard,
            }
            output.append(_make_example(
                source_family=source_family, split=split,
                objective="APPLY_TRANSITION", control_variant="PROGRAM_GRAPH_EDGE",
                input_payload=transition_input,
                target_payload=execute_controller_step(
                    program, state=active_state, candidates=candidates,
                    observed_effect=guard,
                ),
                receipt_ids=receipts,
                derivation="SOURCE_INDUCED_TRANSITION_GRAPH_EXECUTION",
            ))
        if bool(program["source_function"]["retry_after_low"]):
            def add_retry(candidate_rows, variant):
                first = execute_controller_step(
                    program, state=state, candidates=candidate_rows,
                )
                if first["decision"] != "EXECUTE_OPERATOR":
                    return
                low = execute_controller_step(
                    program, state=first["next_symbolic_state"],
                    candidates=candidate_rows,
                    observed_effect="OBSERVED_EFFECT_LOW",
                )
                if low["next_symbolic_state"]["controller_state"] == ENTRY:
                    add_selection(
                        variant, candidate_rows, low["next_symbolic_state"],
                    )

            add_retry(candidates, "PROGRAM_RETRY_LEDGER")
            if augment_retry_equivariance:
                # Candidate identifiers are opaque.  Alpha-renaming is an
                # exact program symmetry and prevents a positional C0..C3
                # shortcut without introducing a target-domain label.
                for offset in (4, 8, 12):
                    renamed = [
                        {
                            "candidate_id": f"C{offset + index}",
                            "effects": dict(candidate["effects"]),
                        }
                        for index, candidate in enumerate(candidates)
                    ]
                    add_retry(renamed, "ALPHA_RENAMED_RETRY_LEDGER")
                # Adding dominated zero-effect candidates preserves the
                # unique source-induced argmax while exercising generic list
                # multiplicity and ledger exclusion.  Counts are an even grid,
                # not selected from any held-out target observation.
                for total in (6, 8, 10, 12):
                    if total <= len(candidates):
                        continue
                    extended = [
                        {
                            "candidate_id": str(candidate["candidate_id"]),
                            "effects": dict(candidate["effects"]),
                        }
                        for candidate in candidates
                    ]
                    extended.extend({
                        "candidate_id": f"C{index}",
                        "effects": {effect_type: 0.0 for effect_type in TYPED_EFFECTS},
                    } for index in range(len(candidates), total))
                    add_retry(extended, "DOMINATED_EXTENSION_RETRY_LEDGER")
        exhausted = dict(state)
        maximum = int(program["source_function"]["maximum_trials"])
        exhausted["attempted_candidates"] = [
            f"C{index}" for index in range(min(maximum, len(candidates)))
        ]
        exhausted["trials_used"] = len(exhausted["attempted_candidates"])
        if (
            exhausted["trials_used"] == maximum
            and int(row.snapshot_sha256[8:16], 16) % 4 == 0
        ):
            add_selection("PROGRAM_TRIAL_BUDGET", candidates, exhausted)

    if not all(example.validate() for example in output):
        raise ValueError("controller SFT example failed validation")
    return tuple(output)


def summarize_controller_sft_examples(
    examples: Sequence[ControllerSFTExample],
) -> dict[str, Any]:
    prompt_text = "\n".join(str(row.input_payload) for row in examples)
    return {
        "examples": len(examples),
        "family_count": len({row.source_family for row in examples}),
        "split_counts": dict(sorted(Counter(row.split for row in examples).items())),
        "objective_counts": dict(sorted(Counter(row.objective for row in examples).items())),
        "decision_counts": dict(sorted(Counter(
            str(row.target_payload["decision"]) for row in examples
        ).items())),
        "derivation_counts": dict(sorted(Counter(row.derivation for row in examples).items())),
        "control_variant_counts": dict(sorted(Counter(
            row.control_variant for row in examples
        ).items())),
        "all_valid": all(row.validate() for row in examples),
        "source_identity_in_prompt": any(
            row.source_family in str(row.input_payload) for row in examples
        ),
        "native_action_tokens_exported": False,
        "target_data_used": False,
        "named_policy_templates_used": any(token in prompt_text for token in (
            "EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY",
        )),
    }


__all__ = [
    "CONTROLLER_SYSTEM_PROMPT", "ControllerSFTExample", "anonymous_program_ir",
    "build_controller_sft_examples", "execute_controller_step",
    "format_controller_prompt", "initial_controller_state",
    "summarize_controller_sft_examples",
]
