"""Source-only induction of anonymous typed symbolic operators.

Phase 2 transferred a useful search automaton, but its event and action
vocabulary was supplied by hand.  This module moves the learning boundary
downward.  Its only empirical input is a collection of matched source-game
intervention ledgers derived from ``(state, action, effect, next_state)``
receipts.  The learner:

* represents decisions with low-level typed ledger predicates;
* identifies operations only by their observed state delta;
* induces minimal precondition conjunctions on the discovery split;
* calibrates fail-closed, unique-rule applicability on qualification data; and
* evaluates unchanged rules on held-out source ledgers.

Names such as EXPLORE, BACKTRACK, or COMMIT never occur in an induced artifact
or in runtime routing.  A target adapter may implement the same typed state
deltas with native actions, but source-native action tokens and candidate IDs
are forbidden from the exported program.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import stable_hash
from .phase1_common_search_ir import option_rows_to_ledgers, read_jsonl


SOURCE_SPLITS = ("discovery", "calibration", "heldout")

# Predicate typing, not policy semantics: a value describing an active object
# is meaningful only together with the predicate that says whether that object
# exists.  Dependency closure prevents a short rule learned on observed states
# from admitting an unseen, type-inconsistent combination.
PREDICATE_DEPENDENCIES = {"active_effect": ("active_presence",)}


def _canonical(value: Any) -> Any:
    """Return a JSON-stable copy used in content-addressed records."""

    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PredicateLiteral:
    field: str
    value: str | bool

    def body(self) -> dict[str, Any]:
        return {"field": self.field, "value": self.value}


@dataclass(frozen=True)
class AnonymousOperator:
    operator_id: str
    preconditions: tuple[PredicateLiteral, ...]
    state_delta: tuple[tuple[str, str], ...]
    discovery_support: int
    discovery_precision: float
    qualification_support: int
    qualification_precision: float

    @classmethod
    def create(
        cls,
        *,
        preconditions: Sequence[PredicateLiteral],
        state_delta: Sequence[tuple[str, str]],
        discovery_support: int,
        discovery_precision: float,
        qualification_support: int,
        qualification_precision: float,
    ) -> "AnonymousOperator":
        canonical_preconditions = tuple(sorted(
            preconditions, key=lambda row: (row.field, str(row.value)),
        ))
        canonical_delta = tuple(sorted(
            (str(field), str(value)) for field, value in state_delta
        ))
        identity = {
            "preconditions": [row.body() for row in canonical_preconditions],
            "state_delta": [list(row) for row in canonical_delta],
        }
        return cls(
            operator_id=f"OP_{stable_hash(identity)[:16]}",
            preconditions=canonical_preconditions,
            state_delta=canonical_delta,
            discovery_support=int(discovery_support),
            discovery_precision=float(discovery_precision),
            qualification_support=int(qualification_support),
            qualification_precision=float(qualification_precision),
        )

    def applies(self, state: Mapping[str, Any]) -> bool:
        return all(state.get(row.field) == row.value for row in self.preconditions)

    def body(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "preconditions": [row.body() for row in self.preconditions],
            "state_delta": [list(row) for row in self.state_delta],
            "discovery_support": self.discovery_support,
            "discovery_precision": self.discovery_precision,
            "qualification_support": self.qualification_support,
            "qualification_precision": self.qualification_precision,
        }


@dataclass(frozen=True)
class DecisionExample:
    example_id: str
    source_game: str
    source_split: str
    snapshot_id: str
    state: Mapping[str, str | bool]
    selected_delta: tuple[tuple[str, str], ...]
    true_active_effect: str
    bound_active_effect: str

    @classmethod
    def create(
        cls,
        *,
        source_game: str,
        source_split: str,
        snapshot_id: str,
        state: Mapping[str, str | bool],
        selected_delta: Sequence[tuple[str, str]],
        true_active_effect: str,
        bound_active_effect: str,
    ) -> "DecisionExample":
        canonical_state = _canonical(dict(state))
        canonical_delta = tuple(sorted(
            (str(field), str(value)) for field, value in selected_delta
        ))
        body = {
            "source_game": source_game,
            "source_split": source_split,
            "snapshot_id": snapshot_id,
            "state": canonical_state,
            "selected_delta": [list(row) for row in canonical_delta],
            "true_active_effect": true_active_effect,
            "bound_active_effect": bound_active_effect,
        }
        return cls(
            example_id=stable_hash(body),
            source_game=source_game,
            source_split=source_split,
            snapshot_id=snapshot_id,
            state=canonical_state,
            selected_delta=canonical_delta,
            true_active_effect=true_active_effect,
            bound_active_effect=bound_active_effect,
        )


def validate_row_hashes(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject altered source rows before induction."""

    for index, row in enumerate(rows):
        body = dict(row)
        claimed = body.pop("row_sha256", None)
        if not claimed or claimed != stable_hash(body):
            raise ValueError(f"invalid source row hash at index {index}")


def load_source_ledgers(
    rows_path: str | Path, *, primary_horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(Path(rows_path))
    validate_row_hashes(rows)
    ledgers, audit = option_rows_to_ledgers(
        rows, primary_horizon=primary_horizon,
    )
    return ledgers, {
        **audit,
        "rows_path": str(Path(rows_path)),
        "rows_file_sha256": file_sha256(rows_path),
        "source_rows": len(rows),
    }


def _split_name(ledger: Mapping[str, Any]) -> str:
    raw = str(ledger.get("automaton_split") or "")
    if raw == "fresh":
        return "heldout"
    if raw not in SOURCE_SPLITS:
        raise ValueError(f"unsupported induced source split: {raw!r}")
    return raw


def _delta(*rows: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(rows))


# These are state-delta signatures, not policy labels.  Their content is the
# generic AttemptLedger executor contract.  Operator identity and routing are
# learned from source tuples and are never keyed by the constant names below.
ACTIVATE_DELTA = _delta(("active", "OBSERVED_EFFECT"), ("tried", "ADD_ONE"))
RELEASE_DELTA = _delta(("active", "ABSENT"))
TERMINATE_DELTA = _delta(("terminal", "TRUE"))
SUSPEND_DELTA = _delta(("suspended", "TRUE"))
DELTA_SPACE = (ACTIVATE_DELTA, RELEASE_DELTA, TERMINATE_DELTA, SUSPEND_DELTA)


def _effect_binding(
    ledger: Mapping[str, Any], *, shuffled: bool,
) -> dict[int, str]:
    count = int(ledger["candidate_count"])
    verified = int(ledger["verified_candidate_rank"])
    bound_verified = verified
    if shuffled and count > 1:
        # A deterministic non-identity permutation.  It changes only the
        # observed effect binding, never the true source continuation value.
        offset = 1 + int(stable_hash(str(ledger["snapshot_id"]))[:8], 16) % (count - 1)
        bound_verified = (verified + offset) % count
    return {
        rank: ("HIGH" if rank == bound_verified else "LOW")
        for rank in range(count)
    }


def decision_examples_from_ledgers(
    ledgers: Iterable[Mapping[str, Any]], *, source_game: str,
    shuffled_effect_binding: bool = False,
) -> list[DecisionExample]:
    """Create source-only policy examples from matched intervention effects.

    The desired state delta follows the true unique-best continuation.  The
    state predicate ``active_effect`` follows either the authentic binding or
    a deterministic shuffled control.  Consequently shuffled evidence cannot
    silently redefine which continuation was actually valuable.
    """

    examples: list[DecisionExample] = []
    for ledger in ledgers:
        split = _split_name(ledger)
        snapshot_id = str(ledger["snapshot_id"])
        count = int(ledger["candidate_count"])
        verified = int(ledger["verified_candidate_rank"])
        binding = _effect_binding(ledger, shuffled=shuffled_effect_binding)
        tried = 0
        for rank in range(verified + 1):
            examples.append(DecisionExample.create(
                source_game=source_game,
                source_split=split,
                snapshot_id=snapshot_id,
                state={
                    "active_presence": "ABSENT",
                    "active_effect": "UNKNOWN",
                    "has_untried": tried < count,
                    "terminal": False,
                    "suspended": False,
                },
                selected_delta=ACTIVATE_DELTA,
                true_active_effect="UNKNOWN",
                bound_active_effect="UNKNOWN",
            ))
            tried += 1
            true_effect = "HIGH" if rank == verified else "LOW"
            bound_effect = binding[rank]
            selected = TERMINATE_DELTA if true_effect == "HIGH" else (
                RELEASE_DELTA if tried < count else SUSPEND_DELTA
            )
            examples.append(DecisionExample.create(
                source_game=source_game,
                source_split=split,
                snapshot_id=snapshot_id,
                state={
                    "active_presence": "PRESENT",
                    "active_effect": bound_effect,
                    "has_untried": tried < count,
                    "terminal": False,
                    "suspended": False,
                },
                selected_delta=selected,
                true_active_effect=true_effect,
                bound_active_effect=bound_effect,
            ))
    return examples


def _candidate_literals(
    positives: Sequence[DecisionExample],
) -> tuple[PredicateLiteral, ...]:
    if not positives:
        return ()
    fields = sorted(set.intersection(*(set(row.state) for row in positives)))
    literals = []
    for field in fields:
        values = {row.state[field] for row in positives}
        if len(values) == 1:
            literals.append(PredicateLiteral(field, next(iter(values))))
    return tuple(literals)


def _matches(state: Mapping[str, Any], rule: Sequence[PredicateLiteral]) -> bool:
    return all(state.get(row.field) == row.value for row in rule)


def infer_minimal_preconditions(
    examples: Sequence[DecisionExample],
    selected_delta: Sequence[tuple[str, str]],
    *, maximum_literals: int,
) -> tuple[PredicateLiteral, ...]:
    """Find the smallest invariant conjunction separating one state delta."""

    target = tuple(selected_delta)
    positives = [row for row in examples if row.selected_delta == target]
    negatives = [row for row in examples if row.selected_delta != target]
    if not positives:
        raise ValueError("cannot induce operator without positive examples")
    candidates = _candidate_literals(positives)
    by_field = {row.field: row for row in candidates}

    def close_dependencies(
        rule: Sequence[PredicateLiteral],
    ) -> tuple[PredicateLiteral, ...]:
        closed = {row.field: row for row in rule}
        pending = list(closed)
        while pending:
            field = pending.pop()
            for dependency in PREDICATE_DEPENDENCIES.get(field, ()):
                literal = by_field.get(dependency)
                if literal is None:
                    raise ValueError(
                        f"positive examples lack carrier predicate {dependency!r}"
                    )
                if dependency not in closed:
                    closed[dependency] = literal
                    pending.append(dependency)
        return tuple(sorted(
            closed.values(), key=lambda row: (row.field, str(row.value)),
        ))

    for size in range(1, min(maximum_literals, len(candidates)) + 1):
        possible = {
            close_dependencies(rule) for rule in combinations(candidates, size)
        }
        valid = [
            rule for rule in possible
            if len(rule) <= maximum_literals
            and not any(_matches(row.state, rule) for row in negatives)
        ]
        if valid:
            return min(valid, key=lambda rule: (
                len(rule),
                tuple((row.field, str(row.value)) for row in rule),
            ))
    raise ValueError(
        "no bounded precondition conjunction separates an observed state delta"
    )


def _operator_metrics(
    rule: Sequence[PredicateLiteral],
    delta: Sequence[tuple[str, str]],
    examples: Sequence[DecisionExample],
) -> tuple[int, float]:
    admitted = [row for row in examples if _matches(row.state, rule)]
    correct = sum(row.selected_delta == tuple(delta) for row in admitted)
    return len(admitted), correct / len(admitted) if admitted else 0.0


def route_state(
    operators: Sequence[AnonymousOperator], state: Mapping[str, Any],
) -> AnonymousOperator | None:
    matched = [operator for operator in operators if operator.applies(state)]
    return matched[0] if len(matched) == 1 else None


def evaluate_examples(
    operators: Sequence[AnonymousOperator],
    examples: Sequence[DecisionExample],
) -> dict[str, Any]:
    correct = abstained = wrong = 0
    routes: dict[str, int] = {}
    for row in examples:
        operator = route_state(operators, row.state)
        if operator is None:
            abstained += 1
            continue
        routes[operator.operator_id] = routes.get(operator.operator_id, 0) + 1
        if operator.state_delta == row.selected_delta:
            correct += 1
        else:
            wrong += 1
    total = len(examples)
    admitted = correct + wrong
    return {
        "examples": total,
        "admitted": admitted,
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "coverage": admitted / total if total else 0.0,
        "selective_accuracy": correct / admitted if admitted else 0.0,
        "overall_accuracy": correct / total if total else 0.0,
        "operator_route_counts": dict(sorted(routes.items())),
    }


def _replace_operator_metrics(
    operator: AnonymousOperator,
    *,
    discovery: Sequence[DecisionExample],
    qualification: Sequence[DecisionExample],
) -> AnonymousOperator:
    discovery_support, discovery_precision = _operator_metrics(
        operator.preconditions, operator.state_delta, discovery,
    )
    qualification_support, qualification_precision = _operator_metrics(
        operator.preconditions, operator.state_delta, qualification,
    )
    return AnonymousOperator.create(
        preconditions=operator.preconditions,
        state_delta=operator.state_delta,
        discovery_support=discovery_support,
        discovery_precision=discovery_precision,
        qualification_support=qualification_support,
        qualification_precision=qualification_precision,
    )


def induce_program(
    examples: Sequence[DecisionExample],
    *,
    source_game: str,
    source_induction_receipts_sha256: str,
    maximum_literals: int = 3,
    minimum_discovery_support: int = 2,
    minimum_qualification_support: int = 2,
    minimum_qualification_precision: float = 1.0,
    minimum_qualification_coverage: float = 0.95,
) -> dict[str, Any]:
    """Induce and qualification-gate one anonymous source program."""

    discovery = [row for row in examples if row.source_split == "discovery"]
    qualification = [row for row in examples if row.source_split == "calibration"]
    if not discovery or not qualification:
        raise ValueError("program induction requires discovery and calibration splits")
    deltas = sorted({row.selected_delta for row in discovery})
    operators: list[AnonymousOperator] = []
    induction_errors = []
    for delta in deltas:
        try:
            rule = infer_minimal_preconditions(
                discovery, delta, maximum_literals=maximum_literals,
            )
            provisional = AnonymousOperator.create(
                preconditions=rule,
                state_delta=delta,
                discovery_support=0,
                discovery_precision=0.0,
                qualification_support=0,
                qualification_precision=0.0,
            )
            operators.append(_replace_operator_metrics(
                provisional,
                discovery=discovery,
                qualification=qualification,
            ))
        except ValueError as exc:
            induction_errors.append({
                "state_delta": [list(row) for row in delta],
                "error": str(exc),
            })
    operators.sort(key=lambda row: row.operator_id)
    discovery_metrics = evaluate_examples(operators, discovery)
    qualification_metrics = evaluate_examples(operators, qualification)
    operator_gates = {
        operator.operator_id: {
            "discovery_support": (
                operator.discovery_support >= minimum_discovery_support
            ),
            "discovery_precision": operator.discovery_precision == 1.0,
            "qualification_support": (
                operator.qualification_support >= minimum_qualification_support
            ),
            "qualification_precision": (
                operator.qualification_precision >= minimum_qualification_precision
            ),
        }
        for operator in operators
    }
    gates = {
        "at_least_three_anonymous_operators_induced": len(operators) >= 3,
        "all_operator_gates_pass": bool(operator_gates) and all(
            all(values.values()) for values in operator_gates.values()
        ),
        "qualification_coverage": (
            qualification_metrics["coverage"] >= minimum_qualification_coverage
        ),
        "qualification_selective_accuracy": (
            qualification_metrics["selective_accuracy"]
            >= minimum_qualification_precision
        ),
        "unknown_or_ambiguous_routes_fail_closed": True,
        "no_named_phase2_policy_tokens": True,
        "no_source_native_action_or_candidate_export": True,
        "no_induction_errors": not induction_errors,
    }
    body = {
        "schema_version": "phase3-source-induced-anonymous-program-v1",
        "status": (
            "SOURCE_INDUCED_PROGRAM_QUALIFIED"
            if all(gates.values()) else "SOURCE_INDUCED_PROGRAM_NOT_QUALIFIED"
        ),
        "source_game": source_game,
        "source_induction_receipts_sha256": source_induction_receipts_sha256,
        "induction_authority": (
            "SOURCE_DISCOVERY_INTERVENTION_TUPLES_ONLY;QUALIFICATION_FOR_"
            "FAIL_CLOSED_CALIBRATION;NO_TARGET_DATA"
        ),
        "operator_vocabulary": "CONTENT_ADDRESSED_STATE_DELTAS_ONLY",
        "operators": [operator.body() for operator in operators],
        "abstention_rule": {
            "kind": "NO_UNIQUE_QUALIFIED_OPERATOR",
            "unknown_predicate_value": "ABSTAIN",
            "zero_matching_operators": "ABSTAIN",
            "multiple_matching_operators": "ABSTAIN",
        },
        "discovery_metrics": discovery_metrics,
        "qualification_metrics": qualification_metrics,
        "operator_gates": operator_gates,
        "gates": gates,
        "induction_errors": induction_errors,
        "thresholds": {
            "maximum_literals": maximum_literals,
            "minimum_discovery_support": minimum_discovery_support,
            "minimum_qualification_support": minimum_qualification_support,
            "minimum_qualification_precision": minimum_qualification_precision,
            "minimum_qualification_coverage": minimum_qualification_coverage,
        },
        "forbidden_export_fields": [
            "candidate_action", "candidate_id", "actions", "option_template_id",
        ],
    }
    serialized = json.dumps(body, sort_keys=True)
    forbidden = ("EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY")
    if any(token in serialized for token in forbidden):
        raise ValueError("named Phase-2 policy token leaked into induced program")
    return body | {"program_sha256": stable_hash(body)}


def operators_from_program(program: Mapping[str, Any]) -> tuple[AnonymousOperator, ...]:
    operators = []
    for row in program.get("operators") or ():
        operator = AnonymousOperator(
            operator_id=str(row["operator_id"]),
            preconditions=tuple(
                PredicateLiteral(str(item["field"]), item["value"])
                for item in row["preconditions"]
            ),
            state_delta=tuple(
                (str(item[0]), str(item[1])) for item in row["state_delta"]
            ),
            discovery_support=int(row["discovery_support"]),
            discovery_precision=float(row["discovery_precision"]),
            qualification_support=int(row["qualification_support"]),
            qualification_precision=float(row["qualification_precision"]),
        )
        expected = AnonymousOperator.create(
            preconditions=operator.preconditions,
            state_delta=operator.state_delta,
            discovery_support=operator.discovery_support,
            discovery_precision=operator.discovery_precision,
            qualification_support=operator.qualification_support,
            qualification_precision=operator.qualification_precision,
        )
        if expected != operator:
            raise ValueError("induced operator content hash mismatch")
        operators.append(operator)
    return tuple(operators)


def validate_program(program: Mapping[str, Any]) -> None:
    body = dict(program)
    claimed = body.pop("program_sha256", None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError("source-induced program hash mismatch")
    if program.get("schema_version") != "phase3-source-induced-anonymous-program-v1":
        raise ValueError("unsupported source-induced program schema")
    operators_from_program(program)
    serialized = json.dumps(program, sort_keys=True)
    if any(token in serialized for token in (
        "EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY",
    )):
        raise ValueError("named Phase-2 policy token leaked into induced program")


def evaluate_program_on_split(
    program: Mapping[str, Any],
    examples: Sequence[DecisionExample],
    *, source_split: str,
) -> dict[str, Any]:
    validate_program(program)
    selected = [row for row in examples if row.source_split == source_split]
    metrics = evaluate_examples(operators_from_program(program), selected)
    return {"source_split": source_split, **metrics}


def execute_program_on_ledgers(
    program: Mapping[str, Any],
    ledgers: Sequence[Mapping[str, Any]],
    *,
    source_split: str,
    shuffled_runtime_effect_binding: bool = False,
) -> dict[str, Any]:
    """Execute induced state deltas over complete held-out attempt ledgers."""

    validate_program(program)
    operators = operators_from_program(program)
    selected_ledgers = [
        row for row in ledgers if _split_name(row) == source_split
    ]
    successes = abstentions = incorrect_terminations = 0
    route_counts: dict[str, int] = {}
    per_ledger = []
    for ledger in selected_ledgers:
        count = int(ledger["candidate_count"])
        verified = int(ledger["verified_candidate_rank"])
        binding = _effect_binding(
            ledger, shuffled=shuffled_runtime_effect_binding,
        )
        active: int | None = None
        tried: set[int] = set()
        success = False
        abstained = False
        terminated_wrong = False
        routes = []
        for _ in range(count * 3 + 3):
            state = {
                "active_presence": "ABSENT" if active is None else "PRESENT",
                "active_effect": "UNKNOWN" if active is None else binding[active],
                "has_untried": len(tried) < count,
                "terminal": False,
                "suspended": False,
            }
            operator = route_state(operators, state)
            if operator is None:
                abstained = True
                break
            routes.append(operator.operator_id)
            route_counts[operator.operator_id] = route_counts.get(operator.operator_id, 0) + 1
            delta = operator.state_delta
            if delta == ACTIVATE_DELTA:
                choices = [rank for rank in range(count) if rank not in tried]
                if active is not None or not choices:
                    abstained = True
                    break
                active = choices[0]
                tried.add(active)
            elif delta == RELEASE_DELTA:
                if active is None:
                    abstained = True
                    break
                active = None
            elif delta == TERMINATE_DELTA:
                success = active == verified
                terminated_wrong = not success
                break
            elif delta == SUSPEND_DELTA:
                abstained = True
                break
            else:
                abstained = True
                break
        else:
            abstained = True
        successes += int(success)
        abstentions += int(abstained)
        incorrect_terminations += int(terminated_wrong)
        per_ledger.append({
            "snapshot_id": ledger["snapshot_id"],
            "success": success,
            "abstained": abstained,
            "incorrect_termination": terminated_wrong,
            "route_operator_ids": routes,
        })
    total = len(selected_ledgers)
    return {
        "source_split": source_split,
        "ledgers": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "abstentions": abstentions,
        "incorrect_terminations": incorrect_terminations,
        "operator_route_counts": dict(sorted(route_counts.items())),
        "per_ledger": per_ledger,
    }


def source_profile(
    ledgers: Sequence[Mapping[str, Any]], *, primary_horizon: int,
    source_splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Derive a source-specific, target-free applicability profile."""

    selected = list(ledgers)
    if source_splits is not None:
        allowed = set(map(str, source_splits))
        selected = [row for row in selected if _split_name(row) in allowed]
    candidate_counts = sorted(int(row["candidate_count"]) for row in selected)
    return_gaps = sorted(float(row["return_gap"]) for row in selected)
    verified_ranks = [int(row["verified_candidate_rank"]) for row in selected]
    if not candidate_counts or not return_gaps:
        raise ValueError("cannot profile an empty source ledger set")

    def lower_quartile(values: Sequence[float]) -> float:
        return float(values[max(0, (len(values) - 1) // 4)])

    body = {
        "primary_horizon": int(primary_horizon),
        "eligible_ledgers": len(ledgers),
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "candidate_count_median": candidate_counts[len(candidate_counts) // 2],
        "return_gap_min": min(return_gaps),
        "return_gap_lower_quartile": lower_quartile(return_gaps),
        "return_gap_median": return_gaps[len(return_gaps) // 2],
        "verified_rank_distribution": {
            str(rank): verified_ranks.count(rank) for rank in sorted(set(verified_ranks))
        },
    }
    return body | {"profile_sha256": stable_hash(body)}


def build_lineage_report(
    *,
    source_game: str,
    rows_path: str | Path,
    primary_horizon: int,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    ledgers, ledger_audit = load_source_ledgers(
        rows_path, primary_horizon=primary_horizon,
    )
    authentic_examples = decision_examples_from_ledgers(
        ledgers, source_game=source_game,
    )
    shuffled_examples = decision_examples_from_ledgers(
        ledgers, source_game=source_game, shuffled_effect_binding=True,
    )
    kwargs = {
        "maximum_literals": int(thresholds["maximum_literals"]),
        "minimum_discovery_support": int(thresholds["minimum_discovery_support"]),
        "minimum_qualification_support": int(thresholds["minimum_qualification_support"]),
        "minimum_qualification_precision": float(thresholds["minimum_qualification_precision"]),
        "minimum_qualification_coverage": float(thresholds["minimum_qualification_coverage"]),
    }
    induction_examples = [
        row for row in authentic_examples
        if row.source_split in {"discovery", "calibration"}
    ]
    induction_receipts_sha256 = stable_hash([
        asdict(row) for row in induction_examples
    ])
    authentic = induce_program(
        authentic_examples,
        source_game=source_game,
        source_induction_receipts_sha256=induction_receipts_sha256,
        **kwargs,
    )
    shuffled = induce_program(
        shuffled_examples,
        source_game=source_game,
        source_induction_receipts_sha256=stable_hash([
            asdict(row) for row in shuffled_examples
            if row.source_split in {"discovery", "calibration"}
        ]),
        **kwargs,
    )
    authentic_heldout = evaluate_program_on_split(
        authentic, authentic_examples, source_split="heldout",
    )
    shuffled_heldout = evaluate_program_on_split(
        shuffled, authentic_examples, source_split="heldout",
    )
    authentic_execution = execute_program_on_ledgers(
        authentic, ledgers, source_split="heldout",
    )
    shuffled_execution = execute_program_on_ledgers(
        shuffled, ledgers, source_split="heldout",
    )
    gates = {
        "authentic_program_qualification_passed": (
            authentic["status"] == "SOURCE_INDUCED_PROGRAM_QUALIFIED"
        ),
        "heldout_authentic_coverage": (
            authentic_heldout["coverage"]
            >= float(thresholds["minimum_heldout_coverage"])
        ),
        "heldout_authentic_selective_accuracy": (
            authentic_heldout["selective_accuracy"]
            >= float(thresholds["minimum_heldout_selective_accuracy"])
        ),
        "heldout_authentic_strictly_beats_shuffled": (
            authentic_heldout["overall_accuracy"]
            > shuffled_heldout["overall_accuracy"]
        ),
        "shuffled_does_not_match_authentic": (
            shuffled_heldout["overall_accuracy"]
            <= float(thresholds["maximum_shuffled_heldout_accuracy"])
        ),
        "heldout_authentic_closed_loop_success": (
            authentic_execution["success_rate"]
            >= float(thresholds["minimum_heldout_closed_loop_success_rate"])
        ),
        "heldout_authentic_closed_loop_strictly_beats_shuffled": (
            authentic_execution["success_rate"]
            > shuffled_execution["success_rate"]
        ),
        "source_only_provenance": True,
    }
    body = {
        "schema_version": "phase3-source-lineage-induction-report-v1",
        "status": (
            "SOURCE_ONLY_INDUCTION_HELDOUT_VALIDATED"
            if all(gates.values()) else "SOURCE_ONLY_INDUCTION_HELDOUT_NOT_VALIDATED"
        ),
        "source_game": source_game,
        "rows_path": str(rows_path),
        "rows_file_sha256": ledger_audit["rows_file_sha256"],
        "primary_horizon": primary_horizon,
        "ledger_audit": ledger_audit,
        "source_profile": source_profile(
            ledgers,
            primary_horizon=primary_horizon,
            source_splits=("discovery", "calibration"),
        ),
        "authentic_program": authentic,
        "shuffled_effect_program": shuffled,
        "heldout": {
            "authentic": authentic_heldout,
            "shuffled_effect_program_on_authentic_effects": shuffled_heldout,
            "authentic_closed_loop": authentic_execution,
            "shuffled_program_closed_loop_on_authentic_effects": shuffled_execution,
        },
        "thresholds": dict(thresholds),
        "gates": gates,
        "claim_boundary": (
            "SOURCE_ONLY_ANONYMOUS_OPERATOR_INDUCTION_AND_HELDOUT_REPLAY;"
            "NO_TARGET_CAUSAL_UTILITY_CLAIM"
        ),
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "ACTIVATE_DELTA",
    "AnonymousOperator",
    "DecisionExample",
    "PredicateLiteral",
    "RELEASE_DELTA",
    "SUSPEND_DELTA",
    "TERMINATE_DELTA",
    "build_lineage_report",
    "decision_examples_from_ledgers",
    "evaluate_examples",
    "evaluate_program_on_split",
    "execute_program_on_ledgers",
    "file_sha256",
    "induce_program",
    "infer_minimal_preconditions",
    "load_source_ledgers",
    "operators_from_program",
    "route_state",
    "source_profile",
    "validate_program",
    "validate_row_hashes",
]
