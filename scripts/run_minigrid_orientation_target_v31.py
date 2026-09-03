#!/usr/bin/env python3
"""Run frozen Tetris-V28 to MiniGrid orientation-recovery transfer."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import gymnasium as gym
import minigrid  # noqa: F401  # register environments
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.minigrid_neural_grounder import (  # noqa: E402
    predict_neural_binding,
    train_grounder_artifact,
    validate_grounder_artifact,
)
from motif_transfer.minigrid_orientation_recovery import (  # noqa: E402
    CONDITIONS,
    DIRECTION_NAMES,
    TOKENS,
    rotated_donor_bindings,
    select_recovery,
    task_spec,
)
from motif_transfer.tetris_rotation_transfer import exact_sign_p  # noqa: E402


DEFAULT_CONFIG = REPO / "configs/minigrid_orientation_target_v31.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _env(config: Mapping[str, Any], seed: int):
    env = gym.make(
        str(config["target"]["environment_id"]), render_mode="rgb_array",
    )
    env.reset(seed=int(seed))
    return env


def _turn(env: Any, effect: int) -> int:
    actions = env.unwrapped.actions
    value = int(effect) % 4
    if value == 3:
        env.step(actions.left)
        return 1
    for _ in range(value):
        env.step(actions.right)
    return value


def _render_panels(
    config: Mapping[str, Any], seed: int,
) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    spec = task_spec(seed, str(config["target"]["namespace"]))
    current = _env(config, seed)
    try:
        panels = {"I": Image.fromarray(current.render()).convert("RGB")}
        for effect in spec.probe_effects:
            _turn(current, effect)
        panels["P"] = Image.fromarray(current.render()).convert("RGB")
    finally:
        current.close()
    calibration_seed = int(seed) + int(config["target"]["calibration_seed_offset"])
    calibration = _env(config, calibration_seed)
    try:
        panels["C0"] = Image.fromarray(calibration.render()).convert("RGB")
    finally:
        calibration.close()
    for token, effect in spec.token_to_effect.items():
        candidate = _env(config, calibration_seed)
        try:
            _turn(candidate, effect)
            panels[token] = Image.fromarray(candidate.render()).convert("RGB")
        finally:
            candidate.close()
    return panels, {
        "seed": int(seed),
        "probe_steps": len(spec.probe_effects),
        "probe_sequence_sha256": stable_hash(list(spec.probe_effects)),
        "candidate_token_count": len(spec.token_effects),
        "calibration_seed_sha256": stable_hash(calibration_seed),
    }


def _evaluator_labels(
    config: Mapping[str, Any], seed: int,
) -> tuple[dict[str, str], dict[str, int], str, int]:
    spec = task_spec(seed, str(config["target"]["namespace"]))
    current = _env(config, seed)
    try:
        initial = int(current.unwrapped.agent_dir)
        for effect in spec.probe_effects:
            _turn(current, effect)
        post = int(current.unwrapped.agent_dir)
    finally:
        current.close()
    calibration_seed = int(seed) + int(config["target"]["calibration_seed_offset"])
    calibration = _env(config, calibration_seed)
    try:
        calibration_direction = int(calibration.unwrapped.agent_dir)
    finally:
        calibration.close()
    values = {"I": initial, "P": post, "C0": calibration_direction}
    for token, effect in spec.token_to_effect.items():
        values[token] = (calibration_direction + effect) % 4
    direct = next(
        token for token, effect in spec.token_to_effect.items()
        if (spec.probe_effect + effect) % 4 == 0
    )
    return (
        {label: DIRECTION_NAMES[value] for label, value in values.items()},
        spec.token_to_effect, direct, spec.probe_effect,
    )


def _development_rows(
    config: Mapping[str, Any], seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        panels, _ = _render_panels(config, int(seed))
        directions, _, direct, _ = _evaluator_labels(config, int(seed))
        rows.append({
            "seed": int(seed), "panels": panels,
            "directions": directions, "direct_recovery": direct,
        })
    return rows


def _native_suffix(config: Mapping[str, Any], seed: int) -> list[int]:
    env = _env(config, seed)
    try:
        base = env.unwrapped
        start = (
            int(base.agent_pos[0]), int(base.agent_pos[1]), int(base.agent_dir),
        )
        goal = None
        for x in range(base.width):
            for y in range(base.height):
                cell = base.grid.get(x, y)
                if cell is not None and getattr(cell, "type", None) == "goal":
                    goal = (x, y)
        if goal is None:
            raise RuntimeError("MiniGrid target omitted its goal")
        left = int(base.actions.left)
        right = int(base.actions.right)
        forward = int(base.actions.forward)
        vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            state, path = queue.popleft()
            x, y, direction = state
            if (x, y) == goal:
                return path
            candidates = [
                ((x, y, (direction - 1) % 4), left),
                ((x, y, (direction + 1) % 4), right),
            ]
            dx, dy = vectors[direction]
            nx, ny = x + dx, y + dy
            cell = base.grid.get(nx, ny)
            if cell is None or bool(cell.can_overlap()):
                candidates.append(((nx, ny, direction), forward))
            for next_state, action in candidates:
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append((next_state, [*path, action]))
        raise RuntimeError("MiniGrid target goal is unreachable")
    finally:
        env.close()


def _execute(
    config: Mapping[str, Any], seed: int, selected_token: str,
) -> dict[str, Any]:
    if selected_token not in TOKENS:
        return {
            "selected_token": "ABSTAIN", "executed": False,
            "native_success": False, "terminated": False, "truncated": False,
            "primitive_transitions": 0,
        }
    spec = task_spec(seed, str(config["target"]["namespace"]))
    suffix = _native_suffix(config, seed)
    env = _env(config, seed)
    transitions = 0
    terminated = truncated = False
    try:
        for effect in spec.probe_effects:
            transitions += _turn(env, effect)
        transitions += _turn(env, spec.token_to_effect[selected_token])
        for action in suffix:
            _, _, terminated, truncated, _ = env.step(action)
            transitions += 1
            if terminated or truncated:
                break
        cell = env.unwrapped.grid.get(
            int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1]),
        )
        on_goal = cell is not None and getattr(cell, "type", None) == "goal"
        return {
            "selected_token": selected_token, "executed": True,
            "native_success": bool(terminated and on_goal and not truncated),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "primitive_transitions": transitions, "suffix_length": len(suffix),
            "suffix_sha256": stable_hash(suffix),
        }
    finally:
        env.close()


def _collect_one(
    config: Mapping[str, Any], artifact: Mapping[str, Any], *, seed: int,
    contract_sha256: str,
) -> dict[str, Any]:
    panels, metadata = _render_panels(config, seed)
    binding = predict_neural_binding(
        artifact, panels,
        orientation_minimum_confidence=float(
            config["grounder"]["orientation_minimum_confidence"]
        ),
        direct_minimum_confidence=float(
            config["grounder"]["direct_minimum_confidence"]
        ),
    )
    directions, effects, direct, probe = _evaluator_labels(config, seed)
    panel_correct = {
        label: binding["directions"].get(label) == expected
        for label, expected in directions.items()
    }
    body = {
        "schema_version": "minigrid-orientation-target-receipt-v31",
        "collection_contract_sha256": contract_sha256,
        **metadata,
        "binding": binding,
        "evaluator_only": {
            "directions": directions, "token_effects": effects,
            "direct_recovery": direct, "probe_effect": probe,
            "panel_correct": panel_correct,
        },
        "source_program_or_identity_exposed_to_neural_inference": False,
        "goal_or_native_success_exposed_to_neural_inference": False,
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _paired(rows: Sequence[Mapping[str, Any]], right: str) -> dict[str, Any]:
    wins = sum(
        row["outcomes"]["source_induced"]["native_success"]
        and not row["outcomes"][right]["native_success"] for row in rows
    )
    losses = sum(
        not row["outcomes"]["source_induced"]["native_success"]
        and row["outcomes"][right]["native_success"] for row in rows
    )
    return {
        "wins": wins, "losses": losses, "ties": len(rows) - wins - losses,
        "net_wins": wins - losses, "exact_two_sided_p": exact_sign_p(wins, losses),
    }


def _evaluate(
    config: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]],
    program: Mapping[str, Any], artifact: Mapping[str, Any], *, split: str,
) -> dict[str, Any]:
    donor = rotated_donor_bindings(
        receipts, namespace=str(config["controls"]["shuffle_namespace"]),
    )
    rows = []
    for receipt in sorted(receipts, key=lambda row: int(row["seed"])):
        seed = int(receipt["seed"])
        outcomes = {}
        for condition in CONDITIONS:
            selected = select_recovery(
                program, receipt["binding"], condition=condition,
                shuffled_token_effects=(
                    donor.get(seed) if condition == "shuffled_binding_control"
                    else None
                ),
            )
            outcomes[condition] = _execute(config, seed, selected)
        rows.append({
            "seed": seed, "binding": receipt["binding"],
            "evaluator_only": receipt["evaluator_only"],
            "source_program_or_identity_exposed_to_neural_inference": receipt[
                "source_program_or_identity_exposed_to_neural_inference"
            ],
            "goal_or_native_success_exposed_to_neural_inference": receipt[
                "goal_or_native_success_exposed_to_neural_inference"
            ],
            "outcomes": outcomes,
        })
    panel_total = sum(len(row["evaluator_only"]["panel_correct"]) for row in rows)
    panel_correct = sum(
        sum(row["evaluator_only"]["panel_correct"].values()) for row in rows
    )
    grounding = {
        "qualified_tasks": sum(row["binding"]["qualified"] for row in rows),
        "task_coverage": sum(row["binding"]["qualified"] for row in rows) / len(rows),
        "panel_correct": panel_correct, "panel_total": panel_total,
        "panel_accuracy": panel_correct / panel_total,
        "effect_binding_exact_tasks": sum(
            row["binding"].get("probe_effect") == row["evaluator_only"]["probe_effect"]
            and row["binding"].get("token_effects")
            == row["evaluator_only"]["token_effects"] for row in rows
        ),
    }
    metrics = {
        condition: {
            "tasks": len(rows),
            "executed": sum(row["outcomes"][condition]["executed"] for row in rows),
            "native_success": sum(
                row["outcomes"][condition]["native_success"] for row in rows
            ),
            "success_rate": sum(
                row["outcomes"][condition]["native_success"] for row in rows
            ) / len(rows),
        }
        for condition in CONDITIONS
    }
    paired = {
        condition: _paired(rows, condition)
        for condition in CONDITIONS if condition != "source_induced"
    }
    thresholds = config["gates"][split]
    gates = {
        "exact_frozen_task_count": len(rows) == len(config["splits"][split]),
        "source_program_is_v28_source_only": (
            program.get("status") == "SOURCE_CYCLIC_IDENTITY_PROGRAM_INDUCED"
            and program.get("target_data_read") is False
            and program.get("raw_source_action_tokens_exported") is False
        ),
        "target_grounder_used_only_development_labels": (
            artifact["training"]["target_native_success_or_reward_read"] == 0
            and artifact["training"]["complete_target_trajectories_read"] == 0
            and artifact["training"]["source_program_or_identity_read"] is False
        ),
        "zero_source_or_goal_leak_to_neural_inference": all(
            row["source_program_or_identity_exposed_to_neural_inference"] is False
            and row["goal_or_native_success_exposed_to_neural_inference"] is False
            for row in rows
        ),
        "grounder_task_coverage": grounding["task_coverage"]
        >= float(thresholds["minimum_grounder_task_coverage"]),
        "grounder_panel_accuracy": grounding["panel_accuracy"]
        >= float(thresholds["minimum_grounder_panel_accuracy"]),
        "source_minimum_success": metrics["source_induced"]["success_rate"]
        >= float(thresholds["minimum_source_success_rate"]),
        "alpha_rename_invariance": all(
            row["outcomes"]["source_induced"]["selected_token"]
            == row["outcomes"]["alpha_renamed_source"]["selected_token"]
            for row in rows
        ),
        "target_written_isomorphic_equivalence": all(
            row["outcomes"]["source_induced"]["selected_token"]
            == row["outcomes"]["target_written_isomorphic"]["selected_token"]
            for row in rows
        ),
        "source_not_below_neural_only": (
            paired["neural_only_direct"]["losses"]
            <= paired["neural_only_direct"]["wins"]
        ),
        "source_above_neural_only_when_required": (
            not bool(thresholds["require_source_above_neural_only"])
            or metrics["source_induced"]["native_success"]
            > metrics["neural_only_direct"]["native_success"]
        ),
        "paired_neural_only_significance_when_required": (
            not bool(thresholds["require_source_above_neural_only"])
            or (
                paired["neural_only_direct"]["wins"]
                > paired["neural_only_direct"]["losses"]
                and paired["neural_only_direct"]["exact_two_sided_p"]
                <= float(thresholds["maximum_neural_only_p_value"])
            )
        ),
        "source_strictly_above_each_destructive_control": all(
            metrics["source_induced"]["native_success"]
            > metrics[name]["native_success"]
            for name in (
                "copy_effect_control", "fixed_token_control",
                "shuffled_binding_control",
            )
        ),
        "wrong_program_negative_transfer_bounded": all(
            metrics[name]["success_rate"]
            <= float(thresholds["maximum_destructive_control_success_rate"])
            for name in (
                "copy_effect_control", "fixed_token_control",
                "shuffled_binding_control",
            )
        ),
    }
    passed = all(gates.values())
    stage = {
        "development": "CONSUMED_DEVELOPMENT",
        "qualification": "FRESH_QUALIFICATION",
        "formal_reserve": "UNTOUCHED_FORMAL_RESERVE",
    }[split]
    body = {
        "schema_version": "minigrid-orientation-target-report-v31",
        "status": f"{stage}_MINIGRID_CYCLIC_TRANSFER_{'PASSED' if passed else 'FAILED'}",
        "split": split, "grounding": grounding, "metrics": metrics,
        "paired_source": paired, "gates": gates,
        "resource_accounting": {
            "provider_calls": 0, "reported_cost_usd": 0.0,
            "target_rendered_grounding_panels_per_task": 7,
            "target_development_tasks_for_grounder": artifact["training"][
                "development_tasks"
            ],
            "target_development_recovery_labels_for_neural_only_baseline": artifact[
                "training"
            ]["target_native_recovery_labels_read"],
            "complete_target_trajectories_used_to_acquire_source_program": 0,
        },
        "program_sha256": str(program["program_sha256"]),
        "grounder_artifact_sha256": str(artifact["artifact_sha256"]),
        "claim_boundary": str(config["claim_boundary"]), "rows": rows,
    }
    return body | {"report_sha256": stable_hash(body)}


def _authority(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    artifact_path = REPO / str(config["outputs"]["grounder_artifact"])
    if split == "development":
        if artifact_path.exists():
            artifact = _read(artifact_path)
            validate_grounder_artifact(artifact)
            return artifact
        rows = _development_rows(
            config, [int(value) for value in config["splits"]["development"]],
        )
        artifact = train_grounder_artifact(
            rows, namespace=str(config["grounder"]["artifact_namespace"]),
            feature_side=int(config["grounder"]["feature_side"]),
            crop_radius=int(config["grounder"]["crop_radius"]),
            orientation_hidden=tuple(config["grounder"]["orientation_hidden"]),
            direct_hidden=tuple(config["grounder"]["direct_hidden"]),
            random_state=int(config["grounder"]["random_state"]),
        )
        _write(artifact_path, artifact)
        return artifact
    development = _read(REPO / str(config["authority"]["development_report"]))
    _self_hash(development, "report_sha256")
    if development["status"] != "CONSUMED_DEVELOPMENT_MINIGRID_CYCLIC_TRANSFER_PASSED":
        raise RuntimeError("development did not authorize qualification")
    artifact = _read(artifact_path)
    validate_grounder_artifact(artifact)
    if artifact["artifact_sha256"] != development["grounder_artifact_sha256"]:
        raise RuntimeError("grounder artifact changed after development")
    if split == "formal_reserve":
        qualification = _read(REPO / str(config["authority"]["qualification_report"]))
        _self_hash(qualification, "report_sha256")
        if qualification["status"] != "FRESH_QUALIFICATION_MINIGRID_CYCLIC_TRANSFER_PASSED":
            raise RuntimeError("qualification did not authorize formal reserve")
        if qualification["grounder_artifact_sha256"] != artifact["artifact_sha256"]:
            raise RuntimeError("qualification used another grounder artifact")
    return artifact


def run(config_path: Path, *, split: str) -> dict[str, Any]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    if config.get("status") != "FROZEN_BEFORE_ANY_V31_TARGET_PROTOCOL_SEED":
        raise ValueError("MiniGrid target protocol is not frozen")
    for hash_field, path_field in config["dependency_fields"].items():
        path = Path(str(config[path_field]))
        if not path.is_absolute():
            path = REPO / path
        if _sha(path) != config[hash_field]:
            raise ValueError(f"frozen dependency changed: {path}")
    source_report = _read(REPO / str(config["source_report"]))
    _self_hash(source_report, "report_sha256")
    program = source_report["development"]["first_qualified"]["program"]
    _self_hash(program, "program_sha256")
    artifact = _authority(config, split)
    contract_sha256 = stable_hash({
        "config_sha256": config["config_sha256"], "split": split,
        "program_sha256": program["program_sha256"],
        "grounder_artifact_sha256": artifact["artifact_sha256"],
    })
    run_dir = REPO / str(config["outputs"]["run_dir"])
    receipts_dir = run_dir / f"{split}_receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    for raw_seed in config["splits"][split]:
        seed = int(raw_seed)
        path = receipts_dir / f"{seed}.json"
        if path.exists():
            receipt = _read(path)
            _self_hash(receipt, "receipt_sha256")
            if receipt["collection_contract_sha256"] != contract_sha256:
                raise ValueError(f"receipt contract drift: {seed}")
        else:
            receipt = _collect_one(
                config, artifact, seed=seed, contract_sha256=contract_sha256,
            )
            _write(path, receipt)
        receipts.append(receipt)
    report = _evaluate(config, receipts, program, artifact, split=split)
    output = REPO / str(config["outputs"][f"{split}_report"])
    _write(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--split", choices=("development", "qualification", "formal_reserve"),
        required=True,
    )
    args = parser.parse_args()
    report = run(args.config.resolve(), split=args.split)
    print(json.dumps({
        "status": report["status"], "grounding": report["grounding"],
        "metrics": report["metrics"], "paired_source": report["paired_source"],
        "gates": report["gates"],
        "resource_accounting": report["resource_accounting"],
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
