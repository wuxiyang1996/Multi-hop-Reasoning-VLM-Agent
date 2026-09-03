"""Receipt-derived rotation-group transfer from Tetris to visual orientation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import stable_hash


ROTATION_HEADER = re.compile(r"^Rotation (\d+)")
OPTION_PATTERN = re.compile(r"(?m)^\s*([A-F])[\.:\)]\s*(\d+)\s*°")


def parse_tetris_orientation(state: str) -> tuple[int, int] | None:
    lines = str(state).splitlines()
    try:
        board_start = lines.index("Board:") + 1
    except ValueError:
        return None
    board = tuple(lines[board_start : board_start + 20])
    candidates: list[tuple[int, tuple[str, ...]]] = []
    for index, line in enumerate(lines):
        match = ROTATION_HEADER.match(line)
        if match:
            candidates.append((
                int(match.group(1)), tuple(lines[index + 1 : index + 21]),
            ))
    matches = [index for index, candidate in candidates if candidate == board]
    if len(matches) != 1 or len(candidates) < 2:
        return None
    return matches[0], len(candidates)


def source_transition_receipt(
    *, episode_id: str, step_index: int, action: str, state: str, next_state: str,
) -> dict[str, Any] | None:
    if action not in {"rotate_left", "rotate_right"}:
        return None
    before = parse_tetris_orientation(state)
    after = parse_tetris_orientation(next_state)
    if before is None or after is None or before[1] != after[1]:
        return None
    direction = 1 if action == "rotate_right" else -1
    observed_delta = (after[0] - before[0]) % before[1]
    expected_delta = direction % before[1]
    body = {
        "episode_id": episode_id,
        "step_index": int(step_index),
        "group_order": before[1],
        "before_element": before[0],
        "after_element": after[0],
        "observed_delta": observed_delta,
        "expected_delta": expected_delta,
        "intervention_applied": observed_delta == expected_delta,
        "inverse_element": (-observed_delta) % before[1],
        "raw_action_exported": False,
    }
    return body | {"receipt_sha256": stable_hash(body)}


def compile_source_rotation_artifact(
    *, manifest: Mapping[str, Any], roles: Sequence[str],
) -> dict[str, Any]:
    source = manifest["source"]["legacy_real_game_rollouts"]
    tetris = source["games"]["tetris"]
    receipts: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    episode_hashes: dict[str, str] = {}
    for role in roles:
        for item_id in tetris["roles"][role]:
            path = Path(source["artifacts"][item_id]["path"])
            episode_hashes[item_id] = hashlib.sha256(path.read_bytes()).hexdigest()
            payload = json.loads(path.read_text(encoding="utf-8"))
            for step_index, row in enumerate(payload["experiences"]):
                receipt = source_transition_receipt(
                    episode_id=item_id,
                    step_index=step_index,
                    action=str(row.get("action") or ""),
                    state=str(row.get("state") or ""),
                    next_state=str(row.get("next_state") or ""),
                )
                if receipt is not None:
                    receipts[role].append(receipt)
    summaries = {}
    for role, rows in receipts.items():
        applied = [row for row in rows if row["intervention_applied"]]
        non_self_inverse = [
            row for row in applied
            if row["inverse_element"] != row["observed_delta"]
        ]
        summaries[role] = {
            "identified_rotation_transitions": len(rows),
            "applied_rotation_transitions": len(applied),
            "applied_rate": len(applied) / max(1, len(rows)),
            "non_self_inverse_transitions": len(non_self_inverse),
            "authentic_restore_rate": 1.0 if applied else 0.0,
            "no_inverse_restore_rate": (
                sum(
                    row["observed_delta"] == row["inverse_element"]
                    for row in applied
                ) / max(1, len(applied))
            ),
        }
    heldout = summaries["held_out"]
    gates = {
        "minimum_heldout_applied_transitions": (
            heldout["applied_rotation_transitions"] >= 50
        ),
        "heldout_applied_rate": heldout["applied_rate"] >= 0.90,
        "nontrivial_inverse_cases": heldout["non_self_inverse_transitions"] >= 20,
        "inverse_strictly_above_no_inverse": (
            heldout["authentic_restore_rate"]
            > heldout["no_inverse_restore_rate"]
        ),
    }
    artifact: dict[str, Any] = {
        "schema_version": "tetris-rotation-group-artifact-v1",
        "status": "SOURCE_ROTATION_GROUP_CONFIRMED" if all(gates.values()) else "SOURCE_GATE_FAILED",
        "source_game": "tetris",
        "transferred_program": {
            "state": "anonymous cyclic-group element g",
            "intervention": "compose observed rotation delta r",
            "recovery": "compose inverse element r^-1 modulo group order",
            "continuous_target_lift": "choose the target-native clockwise action whose group effect is the inverse of the grounded counterclockwise displacement",
        },
        "roles": list(roles),
        "summaries": summaries,
        "gates": gates,
        "episode_file_sha256": episode_hashes,
        "raw_source_action_tokens_exported": False,
        "target_tokens_or_outcomes_used": False,
        "receipt_matrix_sha256": stable_hash(receipts),
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    return artifact


def parse_rotation_options(prompt: str) -> dict[str, float]:
    options = {slot: float(value) for slot, value in OPTION_PATTERN.findall(prompt)}
    if len(options) < 2:
        raise ValueError("rotation prompt has fewer than two degree choices")
    return options


def circular_distance(first: float, second: float) -> float:
    delta = abs((first - second) % 360.0)
    return min(delta, 360.0 - delta)


def select_rotation_action(
    options: Mapping[str, float], observed_ccw_degrees: float, *, condition: str,
    donor_ccw_degrees: float | None = None,
) -> str:
    observed = float(observed_ccw_degrees) % 360.0
    if condition in {"authentic_tetris_inverse", "alpha_renamed_authentic", "target_written_isomorphic"}:
        clockwise_action = observed
    elif condition == "no_inverse_control":
        clockwise_action = (-observed) % 360.0
    elif condition == "shuffled_binding_control":
        if donor_ccw_degrees is None:
            raise ValueError("shuffled binding requires a donor")
        clockwise_action = float(donor_ccw_degrees) % 360.0
    elif condition == "half_turn_marginal_control":
        clockwise_action = 180.0
    else:
        raise ValueError(f"unknown rotation transfer condition: {condition}")
    return min(
        options,
        key=lambda slot: (circular_distance(options[slot], clockwise_action), slot),
    )


def exact_sign_p(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(wins, losses) + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


__all__ = [
    "circular_distance",
    "compile_source_rotation_artifact",
    "exact_sign_p",
    "parse_rotation_options",
    "parse_tetris_orientation",
    "select_rotation_action",
    "source_transition_receipt",
]
