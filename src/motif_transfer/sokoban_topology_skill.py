"""Source-qualified topology executor distilled from Sokoban interventions.

The transferable object contains no Sokoban coordinates or action tokens.  It
is the small program ``bind graph -> execute typed edges -> verify every edge ->
commit the unique sequence that reaches the goal``.  Source-native simulation
is used only to qualify that program against phase, relation, and marginal
controls.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .sokoban_commit_skill import (
    DELTAS,
    parse_state,
    shortest_solution,
    simulate,
    validate_plan,
)


ARTIFACT_VERSION = "SOKOBAN_TOPOLOGY_EXECUTOR_V1"
CONFIRMATION_VERSION = "SOKOBAN_TOPOLOGY_CONFIRMATION_V1"
CONDITIONS = (
    "canonical_topology_executor",
    "direction_permuted_executor",
    "phase_reversed_executor",
    "sequence_length_marginal",
)
_DIRECTION_CYCLE = {"up": "right", "right": "down", "down": "left", "left": "up"}


def _map_direction(action: str, mapping: Mapping[str, str]) -> str:
    direction = action.split()[-1]
    mapped = mapping[direction]
    return f"push {mapped}" if action.startswith("push ") else mapped


def _execute(state, actions: Sequence[str]) -> bool:
    current = state
    for action in actions:
        transition = simulate(current, str(action))
        if not transition.state_changed:
            return False
        current = transition.after
    return bool(current.solved)


def _candidate_sequences(state, snapshot_id: str) -> list[dict[str, Any]]:
    solution = tuple(shortest_solution(state))
    if not solution:
        return []
    cyclic = tuple(_map_direction(action, _DIRECTION_CYCLE) for action in solution)
    reversed_phase = tuple(reversed(solution))
    shuffled = tuple(
        action for _, action in sorted(
            enumerate(solution),
            key=lambda action_index: stable_hash({
                "snapshot_id": snapshot_id,
                "occurrence": action_index[0],
                "action": action_index[1],
            }),
        )
    )
    candidates = [
        ("AUTHENTIC", solution),
        ("DIRECTION_CORRUPT", cyclic),
        ("PHASE_CORRUPT", reversed_phase),
        ("ORDER_CORRUPT", shuffled),
    ]
    # Stable deduplication prevents a symmetric source state from providing a
    # fake control contrast.
    unique: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for name, actions in candidates:
        if actions not in seen:
            unique.append((name, actions))
            seen.add(actions)
    if len(unique) != 4:
        return []
    solved = [name for name, actions in unique if _execute(state, actions)]
    if solved != ["AUTHENTIC"]:
        return []
    return [
        {"candidate_id": name, "actions": list(actions)}
        for name, actions in sorted(
            unique,
            key=lambda row: stable_hash({
                "snapshot_id": snapshot_id, "candidate_id": row[0],
            }),
        )
    ]


def _select(condition: str, state, candidates: Sequence[Mapping[str, Any]]) -> str | None:
    if condition == "canonical_topology_executor":
        executable = [
            str(row["candidate_id"]) for row in candidates
            if _execute(state, tuple(map(str, row["actions"])))
        ]
    elif condition == "direction_permuted_executor":
        executable = [
            str(row["candidate_id"]) for row in candidates
            if _execute(state, tuple(
                _map_direction(str(action), _DIRECTION_CYCLE)
                for action in row["actions"]
            ))
        ]
    elif condition == "phase_reversed_executor":
        executable = [
            str(row["candidate_id"]) for row in candidates
            if _execute(state, tuple(reversed(tuple(map(str, row["actions"])))))
        ]
    elif condition == "sequence_length_marginal":
        minimum = min(len(row["actions"]) for row in candidates)
        executable = [
            str(row["candidate_id"]) for row in candidates
            if len(row["actions"]) == minimum
        ]
    else:
        raise ValueError(f"unsupported topology condition: {condition}")
    return executable[0] if len(executable) == 1 else None


def _evaluate_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for snapshot in validate_plan(plan):
        state = parse_state(str(snapshot["state"]))
        candidates = _candidate_sequences(state, str(snapshot["snapshot_id"]))
        if not candidates:
            continue
        predictions = {
            condition: _select(condition, state, candidates)
            for condition in CONDITIONS
        }
        rows.append({
            "snapshot_id": str(snapshot["snapshot_id"]),
            "episode_id": str(snapshot["episode_id"]),
            "candidate_order": [row["candidate_id"] for row in candidates],
            "gold_candidate_evaluator_only": "AUTHENTIC",
            "predictions": predictions,
        })
    if not rows:
        raise ValueError("source plan has no uniquely executable topology examples")
    metrics = {
        condition: {
            "examples": len(rows),
            "accuracy": sum(
                row["predictions"][condition] == "AUTHENTIC" for row in rows
            ) / len(rows),
            "abstentions": sum(
                row["predictions"][condition] is None for row in rows
            ),
        }
        for condition in CONDITIONS
    }
    return metrics, rows


def build_topology_artifact(plan: Mapping[str, Any]) -> dict[str, Any]:
    metrics, rows = _evaluate_plan(plan)
    authentic = metrics["canonical_topology_executor"]["accuracy"]
    if authentic < 0.99 or not all(
        authentic > metrics[name]["accuracy"] for name in CONDITIONS[1:]
    ):
        raise ValueError("source topology executor does not dominate controls")
    body = {
        "artifact_version": ARTIFACT_VERSION,
        "status": "SOURCE_DISCOVERY_FROZEN_AWAITING_FRESH_CONFIRMATION",
        "claim_boundary": (
            "TRANSFER_ANONYMOUS_GRAPH_EDGE_EXECUTION_AND_GOAL_VERIFICATION_ONLY;"
            "EXCLUDE_SOURCE_COORDINATES_ACTION_TOKENS_OBJECTS_AND_PATH_LENGTHS"
        ),
        "source_plan_sha256": str(plan["plan_sha256"]),
        "program": {
            "predicates": [
                "NODE_BOUND", "EDGE_PASSABLE", "SEQUENCE_PREFIX_VALID",
                "GOAL_REACHED", "UNIQUE_SEQUENCE_VERIFIED",
            ],
            "rules": [
                {"when": "UNBOUND_GRAPH", "select": "BIND_TOPOLOGY"},
                {"when": "VALID_PREFIX_AND_NOT_GOAL", "select": "EXECUTE_EDGE"},
                {"when": "EDGE_BLOCKED", "select": "REFUTE_SEQUENCE"},
                {"when": "UNIQUE_SEQUENCE_VERIFIED", "select": "COMMIT"},
                {"when": "AMBIGUOUS_OR_UNBOUND", "select": "ABSTAIN"},
            ],
            "target_permission": (
                "TARGET_NATIVE_NEURAL_GROUNDER_BINDS_NODES_RELATIONS_AND_GOAL;"
                "SOURCE_PROGRAM_MAY_ONLY_EXECUTE_OR_REFUTE_ANONYMOUS_EDGES_AND_COMMIT"
            ),
        },
        "source_discovery": {
            "eligible_examples": len(rows),
            "episodes": len(set(row["episode_id"] for row in rows)),
            "condition_metrics": metrics,
            "snapshot_ids": [row["snapshot_id"] for row in rows],
        },
    }
    return body | {"artifact_sha256": stable_hash(body)}


def validate_topology_artifact(artifact: Mapping[str, Any]) -> None:
    # V2 is induced from source intervention tuples and shares the anonymous
    # structural operator schema with the other Phase-3 targets.  Keep the V1
    # validator for historical receipts, but dispatch explicitly rather than
    # silently treating the two artifact contracts as interchangeable.
    if artifact.get("artifact_version") == (
        "SOURCE_INDUCED_RELATIONAL_STRUCTURAL_PROGRAM_V2"
    ):
        from .relational_structural_induction import (
            validate_relational_structural_program,
        )

        validate_relational_structural_program(artifact)
        return
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("invalid Sokoban topology artifact self hash")
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported Sokoban topology artifact")


def confirm_topology_artifact(
    artifact: Mapping[str, Any], plan: Mapping[str, Any],
    *, minimum_examples: int = 48,
) -> dict[str, Any]:
    validate_topology_artifact(artifact)
    if plan.get("status") != "FROZEN_FRESH_CONFIRMATION_BEFORE_ARTIFACT_PREDICTIONS":
        raise ValueError("confirmation plan was not frozen before predictions")
    metrics, rows = _evaluate_plan(plan)
    authentic = metrics["canonical_topology_executor"]["accuracy"]
    coverage = len(rows) >= minimum_examples and len(set(
        row["episode_id"] for row in rows
    )) >= 12
    superiority = all(
        authentic > metrics[name]["accuracy"] for name in CONDITIONS[1:]
    )
    passed = coverage and authentic >= 0.99 and superiority
    body = {
        "confirmation_version": CONFIRMATION_VERSION,
        "status": (
            "SOURCE_TOPOLOGY_EXECUTOR_CONFIRMED" if passed
            else "SOURCE_TOPOLOGY_EXECUTOR_REJECTED"
        ),
        "claim_boundary": "FRESH_SOURCE_CONFIRMATION_ONLY_NO_TARGET_EVIDENCE",
        "artifact_sha256": str(artifact["artifact_sha256"]),
        "source_plan_sha256": str(plan["plan_sha256"]),
        "eligible_examples": len(rows),
        "condition_metrics": metrics,
        "gates": {
            "coverage": coverage,
            "canonical_accuracy": authentic >= 0.99,
            "control_superiority": superiority,
        },
        "source_gate_passed": passed,
        "example_receipts": rows,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "ARTIFACT_VERSION",
    "CONDITIONS",
    "build_topology_artifact",
    "confirm_topology_artifact",
    "validate_topology_artifact",
]
