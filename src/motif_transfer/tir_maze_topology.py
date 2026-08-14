"""Target-native visual grounding for Sokoban topology transfer to TIR maze."""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .active_video_transfer import exact_binomial_two_sided
from .contracts import stable_hash
from .sokoban_topology_skill import validate_topology_artifact


OPTION_PATTERN = re.compile(r"(?m)^\s*([A-E])[\.:]\s*([RLUD]+)\s*$")
CANONICAL_DELTAS = {
    "R": (1, 0), "L": (-1, 0), "U": (0, -1), "D": (0, 1),
}
CONDITIONS = (
    "raw_target_only",
    "authentic_sokoban_topology_plus_target",
    "alpha_renamed_authentic",
    "direction_permuted_source_control",
    "endpoint_only_target_control",
    "path_length_marginal_control",
)


def parse_maze_options(prompt: str) -> dict[str, str]:
    options = dict(OPTION_PATTERN.findall(prompt))
    if len(options) < 2:
        raise ValueError("TIR maze prompt has fewer than two path options")
    return options


def validate_neural_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    if binding.get("role") != "TARGET_NATIVE_NEURAL_MAZE_BINDING":
        raise ValueError("maze binding has the wrong authority")
    if binding.get("answer_or_gold_seen"):
        raise ValueError("maze binding may not see answer/gold")
    moves = binding.get("move_deltas") or {}
    resolved = {
        str(token): tuple(map(int, delta)) for token, delta in moves.items()
    }
    if set(resolved) != set(CANONICAL_DELTAS):
        raise ValueError("maze binding must bind exactly R/L/U/D")
    if set(resolved.values()) != set(CANONICAL_DELTAS.values()):
        raise ValueError("maze move binding is not a cardinal bijection")
    start_rgb = tuple(map(int, binding.get("start_color_rgb") or ()))
    goal_rgb = tuple(map(int, binding.get("goal_color_rgb") or ()))
    if len(start_rgb) != 3 or len(goal_rgb) != 3:
        raise ValueError("maze binding must ground start/goal RGB colors")
    if any(not 0 <= value <= 255 for value in (*start_rgb, *goal_rgb)):
        raise ValueError("maze binding RGB value is outside [0,255]")
    start_channel = int(np.argmax(start_rgb))
    goal_channel = int(np.argmax(goal_rgb))
    if start_channel == goal_channel:
        raise ValueError("maze start/goal neural colors are not distinguishable")
    return {
        "move_deltas": resolved,
        "start_color_rgb": start_rgb,
        "goal_color_rgb": goal_rgb,
        "start_channel": start_channel,
        "goal_channel": goal_channel,
    }


def _color_centroid(array: np.ndarray, *, channel: int) -> tuple[float, float]:
    primary = array[:, :, channel].astype(float)
    other = [index for index in range(3) if index != channel]
    mask = (
        (primary > 60)
        & (primary > array[:, :, other[0]].astype(float) * 1.5)
        & (primary > array[:, :, other[1]].astype(float) * 1.5)
    )
    ys, xs = np.where(mask)
    if len(xs) < 8:
        raise ValueError("maze start/goal color grounding failed")
    return float(xs.mean()), float(ys.mean())


def _execute(
    sequence: str,
    *,
    node_count: int,
    start: tuple[float, float],
    goal: tuple[float, float],
    passable: np.ndarray,
    deltas: Mapping[str, tuple[int, int]],
    check_edges: bool,
) -> tuple[bool, dict[str, Any]]:
    x = y = 0
    pitch_x = (goal[0] - start[0]) / (node_count - 1)
    pitch_y = (goal[1] - start[1]) / (node_count - 1)
    first_invalid = None
    for step, token in enumerate(sequence, 1):
        dx, dy = deltas[token]
        x += dx
        y += dy
        if not (0 <= x < node_count and 0 <= y < node_count):
            first_invalid = {"step": step, "reason": "OUT_OF_BOUNDS", "node": [x, y]}
            break
        if check_edges:
            pixel_x = round(start[0] + x * pitch_x)
            pixel_y = round(start[1] + y * pitch_y)
            if not bool(passable[pixel_y, pixel_x]):
                first_invalid = {
                    "step": step, "reason": "BLOCKED_NODE", "node": [x, y],
                    "pixel": [pixel_x, pixel_y],
                }
                break
    reached = first_invalid is None and (x, y) == (node_count - 1, node_count - 1)
    return reached, {
        "node_count": node_count,
        "steps_executed": step if sequence else 0,
        "terminal_node": [x, y],
        "goal_node": [node_count - 1, node_count - 1],
        "first_invalid": first_invalid,
        "goal_reached": reached,
    }


def execute_maze_topology(
    image: Image.Image,
    prompt: str,
    *,
    neural_binding: Mapping[str, Any],
    source_artifact: Mapping[str, Any],
    condition: str = "authentic_sokoban_topology_plus_target",
) -> dict[str, Any]:
    """Return one answer without reading the benchmark gold label."""

    validate_topology_artifact(source_artifact)
    grounded = validate_neural_binding(neural_binding)
    deltas = grounded["move_deltas"]
    options: dict[str, Any] = parse_maze_options(prompt)
    array = np.asarray(image.convert("RGB"))
    start_channel = int(grounded["start_channel"])
    goal_channel = int(grounded["goal_channel"])
    start = _color_centroid(array, channel=start_channel)
    goal = _color_centroid(array, channel=goal_channel)
    start_other = [index for index in range(3) if index != start_channel]
    goal_other = [index for index in range(3) if index != goal_channel]
    start_mask = (
        (array[:, :, start_channel] > 60)
        & (array[:, :, start_channel] > array[:, :, start_other[0]] * 1.5)
        & (array[:, :, start_channel] > array[:, :, start_other[1]] * 1.5)
    )
    goal_mask = (
        (array[:, :, goal_channel] > 60)
        & (array[:, :, goal_channel] > array[:, :, goal_other[0]] * 1.5)
        & (array[:, :, goal_channel] > array[:, :, goal_other[1]] * 1.5)
    )
    passable = (array.mean(axis=2) > 100) | start_mask | goal_mask
    if condition == "alpha_renamed_authentic":
        alpha = {token: f"z{index}" for index, token in enumerate(sorted(deltas))}
        options = {
            slot: tuple(alpha[token] for token in sequence)
            for slot, sequence in options.items()
        }
        deltas = {alpha[token]: delta for token, delta in deltas.items()}
    elif condition == "direction_permuted_source_control":
        cycle = {"R": "D", "D": "L", "L": "U", "U": "R"}
        deltas = {token: deltas[cycle[token]] for token in deltas}
    check_edges = condition != "endpoint_only_target_control"
    if condition == "path_length_marginal_control":
        selected = min(options, key=lambda slot: (len(options[slot]), slot))
        body = {
            "condition": condition,
            "selected_answer": selected,
            "source_option": "COMMIT",
            "target_native_action": "path_length_marginal",
            "candidate_receipts": [],
            "binding_sha256": stable_hash(neural_binding),
        }
        return body | {"receipt_sha256": stable_hash(body)}
    candidate_receipts = []
    successful: list[str] = []
    # Odd square grids are a target-native grounding uncertainty.  The source
    # executor accepts an answer only if all compatible resolutions agree.
    for node_count in range(3, 102, 2):
        for slot, sequence in options.items():
            reached, receipt = _execute(
                sequence,
                node_count=node_count,
                start=start,
                goal=goal,
                passable=passable,
                deltas=deltas,
                check_edges=check_edges,
            )
            if reached:
                successful.append(slot)
                candidate_receipts.append({
                    "answer_slot": slot,
                    "sequence_sha256": stable_hash(sequence),
                    **receipt,
                })
    unique = sorted(set(successful))
    selected = unique[0] if len(unique) == 1 else None
    body = {
        "condition": condition,
        "selected_answer": selected,
        "source_option": "COMMIT" if selected else "ABSTAIN",
        "target_native_action": "execute_bound_path_edges",
        "candidate_receipts": candidate_receipts,
        "grounded_start_pixel": list(start),
        "grounded_goal_pixel": list(goal),
        "binding_sha256": stable_hash(neural_binding),
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _validate_confirmation(
    artifact: Mapping[str, Any], confirmation: Mapping[str, Any],
) -> None:
    validate_topology_artifact(artifact)
    body = dict(confirmation)
    claimed = str(body.pop("report_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("invalid Sokoban topology confirmation self hash")
    if confirmation.get("artifact_sha256") != artifact.get("artifact_sha256"):
        raise ValueError("Sokoban topology artifact/confirmation mismatch")
    if not confirmation.get("source_gate_passed"):
        raise ValueError("Sokoban topology source gate did not pass")


def evaluate_tir_maze_transfer(
    receipts: list[Mapping[str, Any]],
    *,
    source_artifact: Mapping[str, Any],
    source_confirmation: Mapping[str, Any],
    expected_ids: list[str],
    evidence_tier: str,
    claim_boundary: str,
) -> dict[str, Any]:
    """Evaluate already-emitted condition answers against evaluator-only gold."""

    _validate_confirmation(source_artifact, source_confirmation)
    observed = [str(row["sample_id"]) for row in receipts]
    if observed != list(map(str, expected_ids)) or len(observed) != len(set(observed)):
        raise ValueError("TIR-maze receipt coverage/order differs from frozen split")
    traces = []
    for receipt in receipts:
        baseline = str(receipt["baseline_answer"])
        gold = str(receipt["gold_answer_evaluator_only"])
        conditions = receipt["conditions"]
        if set(conditions) != set(CONDITIONS) - {"raw_target_only"}:
            raise ValueError("TIR-maze condition receipt matrix is incomplete")
        for condition in CONDITIONS:
            native = None if condition == "raw_target_only" else conditions[condition]
            selected = baseline if native is None else native.get("selected_answer")
            committed = str(selected or baseline)
            body = {
                "sample_id": str(receipt["sample_id"]),
                "condition": condition,
                "baseline_answer": baseline,
                "native_selected_answer": selected,
                "committed_answer": committed,
                "gold_answer_evaluator_only": gold,
                "correct_evaluator_only": committed == gold,
                "source_option": (
                    "TARGET_ONLY" if native is None else native["source_option"]
                ),
                "native_receipt_sha256": (
                    None if native is None else native["receipt_sha256"]
                ),
            }
            traces.append(body | {"trace_sha256": stable_hash(body)})
    by_condition = {
        condition: [row for row in traces if row["condition"] == condition]
        for condition in CONDITIONS
    }
    summaries = {
        condition: {
            "tasks": len(rows),
            "successes": sum(row["correct_evaluator_only"] for row in rows),
            "success_rate": sum(row["correct_evaluator_only"] for row in rows) / len(rows),
            "action_changes_vs_raw": sum(
                row["committed_answer"] != row["baseline_answer"] for row in rows
            ),
            "source_abstentions": sum(row["source_option"] == "ABSTAIN" for row in rows),
        }
        for condition, rows in by_condition.items()
    }
    authentic_name = "authentic_sokoban_topology_plus_target"
    authentic = {row["sample_id"]: row for row in by_condition[authentic_name]}
    paired = {}
    for comparator in CONDITIONS:
        if comparator == authentic_name:
            continue
        other = {row["sample_id"]: row for row in by_condition[comparator]}
        wins = losses = 0
        for sample_id in observed:
            a = bool(authentic[sample_id]["correct_evaluator_only"])
            b = bool(other[sample_id]["correct_evaluator_only"])
            wins += a and not b
            losses += b and not a
        paired[comparator] = {
            "wins": wins,
            "losses": losses,
            "ties": len(observed) - wins - losses,
            "net_wins": wins - losses,
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        }
    authentic_summary = summaries[authentic_name]
    controls = (
        "raw_target_only", "direction_permuted_source_control",
        "endpoint_only_target_control", "path_length_marginal_control",
    )
    gates = {
        "fresh_source_topology_gate_passed": True,
        "receipt_matrix_complete": len(receipts) == len(expected_ids),
        "all_target_neural_bindings_valid": all(
            bool(row.get("neural_binding_valid")) for row in receipts
        ),
        "authentic_nontrivial_action_contrast": (
            authentic_summary["action_changes_vs_raw"] >= 2
        ),
        "authentic_zero_negative_transfer_vs_raw": (
            paired["raw_target_only"]["losses"] == 0
        ),
        "authentic_strictly_beats_all_controls": all(
            authentic_summary["successes"] > summaries[name]["successes"]
            for name in controls
        ),
        "alpha_rename_invariance": all(
            authentic[sample_id]["committed_answer"]
            == next(
                row["committed_answer"] for row in by_condition["alpha_renamed_authentic"]
                if row["sample_id"] == sample_id
            )
            for sample_id in observed
        ),
    }
    passed = all(gates.values())
    status = (
        "CONSUMED_DEVELOPMENT_GATE_PASSED"
        if passed and evidence_tier == "CONSUMED_DEVELOPMENT"
        else "FRESH_QUALIFICATION_GATE_PASSED"
        if passed and evidence_tier == "FRESH_QUALIFICATION"
        else "FRESH_FORMAL_TRANSFER_VALIDATED"
        if passed and evidence_tier == "FRESH_FORMAL_CONFIRMATION"
        else "TRANSFER_GATE_FAILED"
    )
    body = {
        "schema_version": "tir-maze-sokoban-topology-transfer-v1",
        "status": status,
        "evidence_tier": evidence_tier,
        "claim_boundary": claim_boundary,
        "source_artifact_sha256": str(source_artifact["artifact_sha256"]),
        "source_confirmation_sha256": str(source_confirmation["report_sha256"]),
        "tasks": observed,
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "traces": traces,
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "CONDITIONS",
    "execute_maze_topology",
    "evaluate_tir_maze_transfer",
    "parse_maze_options",
    "validate_neural_binding",
]
