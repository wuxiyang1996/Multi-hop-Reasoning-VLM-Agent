"""Execute source structural subprograms with ALFWorld-native neural bindings."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .alfworld_structural_induction import (
    ADD_ID,
    REMOVE_ID,
    UPDATE_ID,
    observed_transition_operator_ids,
    repeated_source_support,
)
from .alfworld_multiplicity_grounder import candidate_effect, workflow_status
from .contracts import stable_hash


CONDITIONS = (
    "neural_only",
    "source_induced",
    "source_permuted",
    "generic_scaffold",
    "target_native_ceiling",
)


def _neural_score(row: Mapping[str, Any], repeats: int) -> float:
    return float(row["behavior_probability"]) / math.sqrt(1.0 + int(repeats))


class ALFWorldStructuralSelector:
    """A condition-specific executor over the shared anonymous operator IR.

    The source condition supplies operator order.  Target neural heads supply
    all concrete action/entity/receptacle bindings.  Low-confidence or absent
    bindings abstain to the same target-native behavioral policy used by the
    neural-only control.
    """

    def __init__(
        self, *, condition: str, target_sequence: Sequence[str],
        source_sequence: Sequence[str] = (), threshold: float = 0.5,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unsupported ALFWorld structural condition: {condition}")
        self.condition = str(condition)
        self.target_sequence = tuple(map(str, target_sequence))
        self.source_sequence = tuple(map(str, source_sequence))
        self.threshold = float(threshold)
        if self.condition == "source_induced":
            support = repeated_source_support(self.source_sequence, self.target_sequence)
            if not support["applicable"]:
                raise ValueError("source-induced program does not explain target function")
            self.controller_sequence = self.source_sequence * int(support["repeat_count"])
        elif self.condition == "source_permuted":
            self.controller_sequence = self.source_sequence
        else:
            self.controller_sequence = ()
        self.cursor = 0
        self.pending_operator: str | None = None
        self.pending_source_action = False
        self.observed_operator_sequence: list[str] = []
        self.source_admissions = 0
        self.source_abstentions = 0
        self.transition_mismatches = 0

    @staticmethod
    def _best(
        actions: Sequence[str], rows: Mapping[str, Mapping[str, Any]],
        history: Sequence[str], *, score,
    ) -> str:
        return max(
            map(str, actions),
            key=lambda action: (
                float(score(action, rows[action])) / math.sqrt(1.0 + history.count(action)),
                action,
            ),
        )

    def _expected_candidates(
        self, expected: str, rows: Mapping[str, Mapping[str, Any]],
        *, goal: str, history: Sequence[str],
    ) -> list[str]:
        output = []
        for action, row in rows.items():
            operator = float(row["operator_probabilities"].get(expected, 0.0))
            if operator < self.threshold:
                continue
            entity = float(row["entity_binding_probability"])
            destination = float(row["destination_binding_probability"])
            if expected in {ADD_ID, REMOVE_ID} and entity < self.threshold:
                continue
            if expected == REMOVE_ID and destination < self.threshold:
                continue
            if expected == ADD_ID and candidate_effect(
                goal, history, str(action),
            )["reverses_completed_binding"]:
                continue
            output.append(str(action))
        return output

    def select(
        self, *, rows: Mapping[str, Mapping[str, Any]], history: Sequence[str],
        goal: str, expert_action: str | None = None,
    ) -> dict[str, Any]:
        actions = tuple(sorted(map(str, rows)))
        if not actions:
            raise ValueError("target grounder returned no ALFWorld action")
        neural = max(actions, key=lambda action: (
            _neural_score(rows[action], history.count(action)), action,
        ))
        if self.condition == "neural_only":
            return {
                "action": neural, "neural_action": neural,
                "source_admitted": False, "reason": "TARGET_NATIVE_NEURAL_BEHAVIOR",
            }
        if self.condition == "target_native_ceiling":
            if expert_action not in rows:
                raise ValueError("target-native expert action is not admissible")
            return {
                "action": str(expert_action), "neural_action": neural,
                "source_admitted": False, "reason": "TARGET_NATIVE_EXPERT_CEILING",
            }
        if self.condition == "generic_scaffold":
            selected = max(actions, key=lambda action: (
                max(map(float, rows[action]["operator_probabilities"].values())),
                float(rows[action]["entity_binding_probability"]),
                float(rows[action]["destination_binding_probability"]),
                _neural_score(rows[action], history.count(action)),
                action,
            ))
            return {
                "action": selected, "neural_action": neural,
                "source_admitted": False,
                "reason": "SOURCE_FREE_UNORDERED_STRUCTURAL_SCAFFOLD",
            }

        if self.condition == "source_induced" and self.source_sequence == (
            ADD_ID, REMOVE_ID,
        ):
            # The target learned multiplicity ledger owns binding identity and
            # count.  Re-synchronizing from it prevents one physical object
            # from satisfying two anonymous source repetitions.
            status = workflow_status(goal, history)
            self.cursor = min(
                len(self.controller_sequence),
                2 * status.placed_count + int(status.held),
            )
        expected = (
            self.controller_sequence[self.cursor]
            if self.cursor < len(self.controller_sequence) else None
        )
        if expected is None:
            self.source_abstentions += 1
            selected = neural
            return {
                "action": selected, "neural_action": neural,
                "source_admitted": False, "reason": "SOURCE_PROGRAM_TERMINATED",
                "source_cursor": self.cursor,
            }
        candidates = self._expected_candidates(
            expected, rows, goal=goal, history=history,
        )
        if not candidates:
            self.source_abstentions += 1
            selected = neural
            self.pending_operator = None
            self.pending_source_action = False
            return {
                "action": selected, "neural_action": neural,
                "source_admitted": False, "reason": "MISSING_NEURAL_OPERATOR_BINDING",
                "expected_operator": expected, "source_cursor": self.cursor,
            }

        def source_score(action: str, row: Mapping[str, Any]) -> float:
            value = float(row["operator_probabilities"][expected])
            if expected in {ADD_ID, REMOVE_ID}:
                value *= float(row["entity_binding_probability"])
            if expected == REMOVE_ID:
                value *= float(row["destination_binding_probability"])
            return value * (0.25 + float(row["behavior_probability"]))

        selected = self._best(candidates, rows, history, score=source_score)
        self.pending_operator = expected
        self.pending_source_action = True
        self.source_admissions += 1
        return {
            "action": selected, "neural_action": neural,
            "source_admitted": True,
            "reason": "SOURCE_OPERATOR_TARGET_NEURAL_BINDING",
            "expected_operator": expected, "source_cursor": self.cursor,
            "selected_binding": dict(rows[selected]),
        }

    def observe_transition(self, *, after_observation: str) -> dict[str, Any]:
        observed = observed_transition_operator_ids({
            "after_observation": str(after_observation),
        })
        self.observed_operator_sequence.extend(observed)
        advanced = False
        if self.pending_source_action and self.pending_operator is not None:
            if self.pending_operator in observed:
                self.cursor += 1
                advanced = True
            else:
                self.transition_mismatches += 1
        body = {
            "pending_operator": self.pending_operator,
            "observed_operators": list(observed),
            "advanced": advanced,
            "source_cursor_after": self.cursor,
        }
        self.pending_operator = None
        self.pending_source_action = False
        return body | {"receipt_sha256": stable_hash(body)}


__all__ = ["ALFWorldStructuralSelector", "CONDITIONS"]
