"""Induce anonymous structural operators from source state transitions.

The Phase-3 temporal-effect experiments only exposed reward prefixes and state
hashes to the inducer.  Distinct coefficient vectors over those correlated
measurements did not imply distinct behavior.  This module moves the
abstraction boundary down to auditable state deltas:

``state -> action -> next_state`` is converted to an alpha-renaming-invariant
multiset of graph edits.  Source object names, action ordinals, coordinates,
and task identity are not part of the exported operator type.  A program is a
source-supported sequence of those anonymous edit types with guards learned
from generic graph cardinalities.

The graph vocabulary is deliberately structural rather than procedural.  It
contains no EXPLORE/BACKTRACK/COMMIT policy templates and does not prescribe a
target action.  A target-native grounder is responsible for predicting which
target action realizes an operator type.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from .contracts import stable_hash


STATE_FEATURES = (
    "carrier_cardinality",
    "entity_cardinality",
    "mutable_false_cardinality",
    "mutable_true_cardinality",
    "relation_cardinality",
)


def _finite_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"{label} is not a nonnegative integer")
    return int(number)


def _entity_signature(record: Mapping[str, Any]) -> tuple[str, str]:
    """Use source labels only to establish within-transition correspondence."""

    return str(record.get("type", "")), str(record.get("color", ""))


def _relations(state: Mapping[str, Any]) -> set[tuple[tuple[str, str], ...]]:
    output = set()
    for raw in state.get("relations") or ():
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("state relation is not a sequence")
        members = tuple(sorted((str(item[0]), str(item[1])) for item in raw))
        if len(members) != 2:
            raise ValueError("only binary source relations are supported")
        output.add(members)
    return output


def structural_state_features(state: Mapping[str, Any]) -> dict[str, int]:
    """Return generic graph counts used by the learned guard hypothesis class."""

    objects = state.get("objects") or ()
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("state objects are not a sequence")
    mutable_false = mutable_true = 0
    for row in objects:
        if not isinstance(row, Mapping):
            raise ValueError("state object is not a mapping")
        for key in ("is_open", "is_locked"):
            value = row.get(key)
            mutable_false += int(value is False)
            mutable_true += int(value is True)
    output = {
        "carrier_cardinality": int(state.get("carrying") is not None),
        "entity_cardinality": len(objects),
        "mutable_false_cardinality": mutable_false,
        "mutable_true_cardinality": mutable_true,
        "relation_cardinality": len(_relations(state)),
    }
    if tuple(sorted(output)) != tuple(sorted(STATE_FEATURES)):
        raise RuntimeError("structural state feature schema drifted")
    return output


def _object_attributes_by_position(
    state: Mapping[str, Any],
) -> dict[tuple[int, ...], tuple[tuple[str, str], tuple[Any, Any]]]:
    output = {}
    for row in state.get("objects") or ():
        position = row.get("position")
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)):
            continue
        key = tuple(int(round(float(value))) for value in position)
        output[key] = (
            _entity_signature(row),
            (row.get("is_open"), row.get("is_locked")),
        )
    return output


def _atom(
    operation: str, family: str, arity: int, *, value_kind: str,
) -> dict[str, Any]:
    return {
        "operation": str(operation),
        "predicate_family": str(family),
        "arity": int(arity),
        "value_kind": str(value_kind),
    }


def structural_delta_descriptor(
    before: Mapping[str, Any], after: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a source transition to a name-free graph-edit descriptor.

    Source entity labels are used only to match an entity across the two
    simulator states.  The returned descriptor contains edit shape and
    multiplicity, never the matched type/color, action, coordinate, or task.
    """

    atoms: list[dict[str, Any]] = []
    if list(before.get("agent_position") or ()) != list(
        after.get("agent_position") or ()
    ):
        atoms.append(_atom(
            "UPDATE", "CONTROL_STATE", 1, value_kind="POSITION",
        ))

    before_carrying = before.get("carrying")
    after_carrying = after.get("carrying")
    if before_carrying is None and after_carrying is not None:
        atoms.append(_atom(
            "ADD", "ENTITY_SLOT", 1, value_kind="ENTITY_REFERENCE",
        ))
    elif before_carrying is not None and after_carrying is None:
        atoms.append(_atom(
            "REMOVE", "ENTITY_SLOT", 1, value_kind="ENTITY_REFERENCE",
        ))
    elif before_carrying is not None and after_carrying is not None:
        if _entity_signature(before_carrying) != _entity_signature(after_carrying):
            atoms.append(_atom(
                "UPDATE", "ENTITY_SLOT", 1, value_kind="ENTITY_REFERENCE",
            ))

    before_attributes = _object_attributes_by_position(before)
    after_attributes = _object_attributes_by_position(after)
    for position in sorted(set(before_attributes) & set(after_attributes)):
        old_signature, old_values = before_attributes[position]
        new_signature, new_values = after_attributes[position]
        if old_signature == new_signature and old_values != new_values:
            atoms.append(_atom(
                "UPDATE", "ENTITY_ATTRIBUTE", 1, value_kind="BOOLEAN_VECTOR",
            ))

    old_relations = _relations(before)
    new_relations = _relations(after)
    atoms.extend(
        _atom("ADD", "ENTITY_RELATION", 2, value_kind="ENTITY_REFERENCE")
        for _ in sorted(new_relations - old_relations)
    )
    atoms.extend(
        _atom("REMOVE", "ENTITY_RELATION", 2, value_kind="ENTITY_REFERENCE")
        for _ in sorted(old_relations - new_relations)
    )

    counts = Counter(stable_hash(row) for row in atoms)
    unique = {stable_hash(row): row for row in atoms}
    canonical_atoms = [
        {**unique[key], "multiplicity": int(counts[key])}
        for key in sorted(unique)
    ]
    core = {
        "schema_version": "anonymous-structural-delta-v1",
        "atoms": canonical_atoms,
    }
    return core | {"delta_type_id": f"TYPE_{stable_hash(core)[:16]}"}


def validate_delta_descriptor(delta: Mapping[str, Any]) -> None:
    body = dict(delta)
    claimed = str(body.pop("delta_type_id", ""))
    if not claimed or claimed != f"TYPE_{stable_hash(body)[:16]}":
        raise ValueError("structural delta type hash mismatch")
    if body.get("schema_version") != "anonymous-structural-delta-v1":
        raise ValueError("unsupported structural delta schema")
    atoms = body.get("atoms")
    if not isinstance(atoms, Sequence) or isinstance(atoms, (str, bytes)):
        raise ValueError("structural delta atoms are invalid")
    for atom in atoms:
        if not isinstance(atom, Mapping):
            raise ValueError("structural delta atom is invalid")
        if str(atom.get("operation")) not in {"ADD", "REMOVE", "UPDATE"}:
            raise ValueError("structural delta operation is invalid")
        if _finite_integer(atom.get("arity"), label="delta arity") not in {1, 2}:
            raise ValueError("structural delta arity is unsupported")
        if _finite_integer(atom.get("multiplicity"), label="delta multiplicity") < 1:
            raise ValueError("structural delta multiplicity is invalid")


def is_empty_delta(delta: Mapping[str, Any]) -> bool:
    validate_delta_descriptor(delta)
    return not bool(delta.get("atoms"))


def is_control_only_delta(delta: Mapping[str, Any]) -> bool:
    validate_delta_descriptor(delta)
    atoms = list(delta.get("atoms") or ())
    return bool(atoms) and all(
        str(row.get("predicate_family")) == "CONTROL_STATE" for row in atoms
    )


def structural_atom_descriptors(
    delta: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Expand a compound transition into alpha-invariant atomic operators."""

    validate_delta_descriptor(delta)
    output = []
    for row in delta.get("atoms") or ():
        if str(row.get("predicate_family")) == "CONTROL_STATE":
            continue
        core = {
            "schema_version": "anonymous-structural-operator-type-v1",
            "operation": str(row["operation"]),
            "predicate_family": str(row["predicate_family"]),
            "arity": int(row["arity"]),
            "value_kind": str(row["value_kind"]),
        }
        descriptor = core | {
            "operator_type_id": f"ATYPE_{stable_hash(core)[:16]}"
        }
        output.extend(
            dict(descriptor) for _ in range(int(row["multiplicity"]))
        )
    return tuple(output)


def abstract_effect_sequence(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Strip navigation/no-op scaffolding and collapse repeated effect types."""

    output: list[str] = []
    for step in steps:
        delta = step.get("delta")
        if not isinstance(delta, Mapping):
            raise ValueError("path step omitted structural delta")
        validate_delta_descriptor(delta)
        if is_empty_delta(delta) or is_control_only_delta(delta):
            continue
        for atom in structural_atom_descriptors(delta):
            type_id = str(atom["operator_type_id"])
            if not output or output[-1] != type_id:
                output.append(type_id)
    return tuple(output)


def _lcs(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    table: list[list[tuple[str, ...]]] = [
        [tuple() for _ in range(len(right) + 1)]
        for _ in range(len(left) + 1)
    ]
    for i, a in enumerate(left, start=1):
        for j, b in enumerate(right, start=1):
            if a == b:
                table[i][j] = (*table[i - 1][j - 1], a)
                continue
            above = table[i - 1][j]
            beside = table[i][j - 1]
            table[i][j] = max(
                (above, beside), key=lambda row: (len(row), stable_hash(list(row))),
            )
    return table[-1][-1]


def common_effect_subsequence(
    paths: Iterable[Sequence[str]],
) -> tuple[str, ...]:
    selected = [tuple(map(str, path)) for path in paths]
    if not selected:
        return ()
    output = selected[0]
    for path in selected[1:]:
        output = _lcs(output, path)
        if not output:
            break
    return output


def sequence_contains(sequence: Sequence[str], program: Sequence[str]) -> bool:
    position = 0
    for value in sequence:
        if position < len(program) and str(value) == str(program[position]):
            position += 1
    return position == len(program)


def _stable_guards(
    occurrences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not occurrences:
        return []
    guards = []
    for name in STATE_FEATURES:
        values = {
            _finite_integer(row[name], label=f"guard feature {name}")
            for row in occurrences
        }
        if len(values) == 1:
            guards.append({"feature": name, "operator": "EQ", "value": values.pop()})
    return guards


@dataclass(frozen=True)
class StructuralPath:
    split: str
    success: bool
    steps: tuple[Mapping[str, Any], ...]

    @property
    def effects(self) -> tuple[str, ...]:
        return abstract_effect_sequence(self.steps)


def induce_structural_program(
    paths: Sequence[StructuralPath], *, source_receipts_sha256: str,
    discovery_split: str = "development",
    minimum_success_paths: int = 2,
    minimum_qualification_support: float = 0.75,
) -> dict[str, Any]:
    """Induce a typed operator chain and fail-closed abstention decision."""

    discovery_success = [
        row for row in paths if row.split == discovery_split and row.success
    ]
    discovery_controls = [
        row for row in paths if row.split == discovery_split and not row.success
    ]
    sequence = common_effect_subsequence(row.effects for row in discovery_success)
    qualification = [row for row in paths if row.split == "qualification"]
    qualification_success = [row for row in qualification if row.success]
    qualification_controls = [row for row in qualification if not row.success]

    def support(rows: Sequence[StructuralPath]) -> float:
        if not rows or not sequence:
            return 0.0
        return sum(sequence_contains(row.effects, sequence) for row in rows) / len(rows)

    success_support = support(qualification_success)
    control_support = support(qualification_controls)
    gates = {
        "minimum_discovery_success_paths": (
            len(discovery_success) >= minimum_success_paths
        ),
        "nonempty_induced_operator_sequence": bool(sequence),
        "qualification_success_support": (
            success_support >= minimum_qualification_support
        ),
    }
    qualified = all(gates.values())

    descriptors: dict[str, Mapping[str, Any]] = {}
    guard_occurrences: dict[str, list[Mapping[str, Any]]] = {
        type_id: [] for type_id in sequence
    }
    for path in discovery_success:
        remaining = list(sequence)
        for step in path.steps:
            delta = step.get("delta")
            if not isinstance(delta, Mapping):
                continue
            validate_delta_descriptor(delta)
            for atom in structural_atom_descriptors(delta):
                type_id = str(atom["operator_type_id"])
                descriptors[type_id] = atom
                if remaining and type_id == remaining[0]:
                    features = step.get("before_features")
                    if isinstance(features, Mapping):
                        guard_occurrences[type_id].append(features)
                    remaining.pop(0)

    operators = []
    for index, type_id in enumerate(sequence):
        descriptor = descriptors[type_id]
        core = {
            "position": index,
            "operator_type_id": type_id,
            "operator_type_descriptor": dict(descriptor),
            "learned_guards": _stable_guards(guard_occurrences[type_id]),
            "target_binding": "TARGET_NATIVE_NEURAL_DELTA_GROUNDER",
        }
        operators.append(core | {"operator_id": f"OP_{stable_hash(core)[:16]}"})
    transitions = [
        {
            "from": operators[index]["operator_id"],
            "guard": "OBSERVED_MATCHING_STRUCTURAL_DELTA",
            "to": (
                operators[index + 1]["operator_id"]
                if index + 1 < len(operators) else "TERMINAL_OUTCOME_CHECK"
            ),
        }
        for index in range(len(operators))
    ]
    body = {
        "schema_version": "source-induced-structural-program-v1",
        "status": (
            "SOURCE_STRUCTURAL_PROGRAM_QUALIFIED" if qualified
            else "SOURCE_STRUCTURAL_PROGRAM_ABSTAINING"
        ),
        "source_receipts_sha256": str(source_receipts_sha256),
        "induction_authority": (
            "SOURCE_STATE_ACTION_NEXT_STATE_DELTAS_ONLY;NO_TASK_OR_ACTION_"
            "IDENTITY_FEATURE;NO_TARGET_DATA"
        ),
        "operators": operators if qualified else [],
        "induced_sequence": list(sequence),
        "transition_graph": transitions if qualified else [],
        "qualification_metrics": {
            "success_paths": len(qualification_success),
            "control_paths": len(qualification_controls),
            "success_support": success_support,
            "control_support": control_support,
            "control_support_authority": (
                "DIAGNOSTIC_ONLY;A_NECESSARY_SUBGOAL_PROGRAM_MAY_APPEAR_IN_"
                "AN_UNFINISHED_CONTROL_PATH"
            ),
        },
        "qualification_gates": gates,
        "abstention_rule": {
            "source_not_qualified": "ABSTAIN",
            "missing_target_delta_prediction": "ABSTAIN",
            "nonunique_target_operator_binding": "ABSTAIN",
            "observed_delta_mismatch": "ABSTAIN",
        },
        "source_task_identity_used_as_feature": False,
        "source_action_identity_exported": False,
        "target_data_read": False,
        "forbidden_named_policy_tokens": [
            "EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY",
        ],
    }
    return body | {"program_sha256": stable_hash(body)}


def validate_structural_program(program: Mapping[str, Any]) -> None:
    body = dict(program)
    claimed = str(body.pop("program_sha256", ""))
    if not claimed or claimed != stable_hash(body):
        raise ValueError("structural program hash mismatch")
    if program.get("schema_version") != "source-induced-structural-program-v1":
        raise ValueError("unsupported structural program schema")
    if program.get("source_task_identity_used_as_feature") is not False:
        raise ValueError("source task identity leaked into structural program")
    if program.get("source_action_identity_exported") is not False:
        raise ValueError("source action identity leaked into structural program")
    if program.get("target_data_read") is not False:
        raise ValueError("target data leaked into structural program")
    serialized = str(program)
    for token in program.get("forbidden_named_policy_tokens") or ():
        if serialized.count(str(token)) != 1:
            raise ValueError("named policy template leaked into structural program")
    for operator in program.get("operators") or ():
        descriptor = operator.get("operator_type_descriptor")
        if not isinstance(descriptor, Mapping):
            raise ValueError("operator type descriptor is missing")
        core = dict(descriptor)
        claimed_type = str(core.pop("operator_type_id", ""))
        if claimed_type != f"ATYPE_{stable_hash(core)[:16]}":
            raise ValueError("operator type descriptor hash mismatch")
        if operator.get("operator_type_id") != claimed_type:
            raise ValueError("operator type binding mismatch")


__all__ = [
    "STATE_FEATURES", "StructuralPath", "abstract_effect_sequence",
    "common_effect_subsequence", "induce_structural_program",
    "is_control_only_delta", "is_empty_delta", "sequence_contains",
    "structural_atom_descriptors", "structural_delta_descriptor",
    "structural_state_features",
    "validate_delta_descriptor", "validate_structural_program",
]
