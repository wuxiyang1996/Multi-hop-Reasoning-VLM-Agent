from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .real_source_interventions import content_hash


ACTION_PATTERN = re.compile(
    r"\(\((\d+),(\d+)\),\((\d+),(\d+)\)\)"
)


def _parse_board(observation: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in observation.splitlines():
        match = re.match(r"^\d+\|\s+(.+)$", line.strip())
        if match:
            rows.append(match.group(1).split())
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
        raise ValueError("expected an 8x8 Candy board")
    return rows


def _run_length(board: Sequence[Sequence[str]], row: int, col: int, dr: int, dc: int) -> int:
    token = board[row][col]
    total = 1
    for sign in (-1, 1):
        rr, cc = row + sign * dr, col + sign * dc
        while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == token:
            total += 1
            rr += sign * dr
            cc += sign * dc
    return total


def candy_action_features(observation: str, action: str) -> tuple[float, ...]:
    board = _parse_board(observation)
    match = ACTION_PATTERN.fullmatch(action.replace(" ", ""))
    if match is None:
        raise ValueError(f"unrecognized Candy action: {action}")
    r1, c1, r2, c2 = (int(value) for value in match.groups())
    before_same = float(board[r1][c1] == board[r2][c2])
    board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]
    runs = [
        _run_length(board, row, col, dr, dc)
        for row, col in ((r1, c1), (r2, c2))
        for dr, dc in ((1, 0), (0, 1))
    ]
    matched_runs = [value for value in runs if value >= 3]
    colors = ("R", "G", "B", "P", "C", "Y", "O")
    first = board[r1][c1]
    second = board[r2][c2]
    return (
        r1 / 7.0,
        c1 / 7.0,
        r2 / 7.0,
        c2 / 7.0,
        float(r1 == r2),
        before_same,
        float(bool(matched_runs)),
        max(runs) / 8.0,
        sum(matched_runs) / 16.0,
        *[float(first == color) for color in colors],
        *[float(second == color) for color in colors],
    )


def normalized_rows(
    receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    states: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in receipts:
        if str(row.get("status")) == "VALID":
            states[str(row["snapshot_id"])].append(row)
    result: list[dict[str, Any]] = []
    for snapshot_id, state_rows in sorted(states.items()):
        rewards = [float(row["immediate_reward"]) for row in state_rows]
        low, high = min(rewards), max(rewards)
        scale = high - low
        for row, reward in zip(state_rows, rewards, strict=True):
            result.append(
                dict(row)
                | {
                    "normalized_reward": (reward - low) / scale if scale > 0 else 0.0,
                    "features": candy_action_features(
                        str(row["grounding_state"]), str(row["action"])
                    ),
                }
            )
    return result


def _shuffled_labels(rows: Sequence[Mapping[str, Any]], namespace: str) -> np.ndarray:
    labels = np.array([float(row["normalized_reward"]) for row in rows])
    by_state: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_state[str(row["snapshot_id"])].append(index)
    shuffled = labels.copy()
    for snapshot_id, indices in sorted(by_state.items()):
        ordered_targets = sorted(
            indices,
            key=lambda index: content_hash(
                [namespace, snapshot_id, str(rows[index]["action"])]
            ),
        )
        rotated = ordered_targets[1:] + ordered_targets[:1]
        for target, source in zip(ordered_targets, rotated, strict=True):
            shuffled[target] = labels[source]
    return shuffled


def _fit(rows: Sequence[Mapping[str, Any]], labels: np.ndarray):
    features = np.asarray([row["features"] for row in rows], dtype=float)
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(16,),
            activation="tanh",
            solver="lbfgs",
            alpha=0.1,
            max_iter=2000,
            random_state=1701,
        ),
    )
    model.fit(features, labels)
    return model


def _evaluate(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[float]
) -> dict[str, float | int | None]:
    truth = np.asarray([float(row["normalized_reward"]) for row in rows])
    predicted = np.asarray(predictions, dtype=float)
    by_state: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_state[str(row["snapshot_id"])].append(index)
    regrets = []
    top_ties = []
    correlations = []
    for indices in by_state.values():
        local_truth = truth[indices]
        local_predicted = predicted[indices]
        chosen = int(np.argmax(local_predicted))
        regrets.append(float(np.max(local_truth) - local_truth[chosen]))
        top_ties.append(float(local_truth[chosen] == np.max(local_truth)))
        if np.std(local_truth) > 0 and np.std(local_predicted) > 0:
            correlations.append(float(np.corrcoef(local_truth, local_predicted)[0, 1]))
    return {
        "rows": len(rows),
        "states": len(by_state),
        "mse": float(np.mean((truth - predicted) ** 2)),
        "mean_normalized_regret": float(np.mean(regrets)),
        "top_action_tie_accuracy": float(np.mean(top_ties)),
        "mean_within_state_pearson": (
            float(np.mean(correlations)) if correlations else None
        ),
    }


def run_source_grounder_gate(
    receipts: Sequence[Mapping[str, Any]], namespace: str
) -> dict[str, Any]:
    rows = normalized_rows(receipts)
    development = [row for row in rows if row["split"] == "development"]
    authentic_labels = np.asarray(
        [float(row["normalized_reward"]) for row in development]
    )
    shuffled_labels = _shuffled_labels(development, namespace)
    authentic = _fit(development, authentic_labels)
    shuffled = _fit(development, shuffled_labels)
    marginal = float(np.mean(authentic_labels))
    splits: dict[str, Any] = {}
    for split in ("qualification", "heldout"):
        evaluation = [row for row in rows if row["split"] == split]
        features = np.asarray([row["features"] for row in evaluation], dtype=float)
        splits[split] = {
            "authentic": _evaluate(evaluation, authentic.predict(features)),
            "within_state_shuffled": _evaluate(evaluation, shuffled.predict(features)),
            "source_marginal": _evaluate(
                evaluation, np.full(len(evaluation), marginal)
            ),
        }
    heldout = splits["heldout"]
    authentic_regret = heldout["authentic"]["mean_normalized_regret"]
    control_regret = min(
        heldout["within_state_shuffled"]["mean_normalized_regret"],
        heldout["source_marginal"]["mean_normalized_regret"],
    )
    passed = authentic_regret < control_regret
    return {
        "schema_version": "candy-native-neural-grounder-gate-v1",
        "status": "SOURCE_GROUNDER_GATE_PASSED" if passed else "SOURCE_GROUNDER_GATE_FAILED",
        "model": "StandardScaler + MLPRegressor(16 tanh units, LBFGS)",
        "feature_authority": "SOURCE_NATIVE_ONLY",
        "transferred_fields": [],
        "training_rows": len(development),
        "gate": "heldout authentic mean normalized regret < both controls",
        "splits": splits,
        "cross_domain_transfer_supported": False,
    }


__all__ = [
    "candy_action_features",
    "normalized_rows",
    "run_source_grounder_gate",
]
