#!/usr/bin/env python3
"""Freeze a risk-first target-native admission threshold for ALFWorld."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_hierarchical_grounder import mlp_probability  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_alfworld_typed_grounder import (  # noqa: E402
    validate_artifact,
)
from motif_transfer.phase3_source_portfolio import (  # noqa: E402
    select_source_program_portfolio,
)
from motif_transfer.phase3_typed_effect_induction import (  # noqa: E402
    target_trial_order,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_state_builder():
    path = REPO / "scripts/train_phase3_alfworld_typed_grounder.py"
    spec = importlib.util.spec_from_file_location("phase3_alfworld_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load typed-grounder state builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._episode_states


def _decisions(
    states: Sequence[Mapping[str, Any]], *, artifact: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    effects = tuple(artifact["effect_types"])
    result = []
    for state in states:
        actions = [str(row["action"]) for row in state["rows"]]
        policy = [
            mlp_probability(row["features"], artifact["target_policy_head"])
            for row in state["rows"]
        ]
        grounded = []
        for index, row in enumerate(state["rows"]):
            grounded.append({
                effect: mlp_probability(
                    row["features"], artifact["typed_effect_heads"][effect],
                ) * policy[index] ** float(artifact["policy_support_exponent"])
                for effect in effects
            })
        ids = [stable_hash({"target_native_action": action}) for action in actions]
        receipt = select_source_program_portfolio(
            source_artifacts,
            candidate_ids=ids,
            candidate_effects=grounded,
            target_grounding_sha256=stable_hash(grounded),
        )
        selected = next(
            row for row in source_artifacts
            if row["artifact_sha256"] == receipt["selected_artifact_sha256"]
        )
        order, reason = target_trial_order(
            selected["typed_effect_program"], grounded,
        )
        if reason is not None:
            raise RuntimeError(f"qualified portfolio failed to bind: {reason}")
        source_index = order[0]
        neural_index = max(range(len(actions)), key=lambda index: (
            policy[index], actions[index],
        ))
        result.append({
            "source_correct": actions[source_index] == state["expert_action"],
            "neural_correct": actions[neural_index] == state["expert_action"],
            "changed_action": source_index != neural_index,
            "policy_support_ratio": policy[source_index] / max(
                policy[neural_index], 1e-300,
            ),
        })
    return result


def _metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    admitted = [
        row["policy_support_ratio"] >= threshold and row["changed_action"]
        for row in rows
    ]
    correct = [
        row["source_correct"] if is_admitted else row["neural_correct"]
        for row, is_admitted in zip(rows, admitted)
    ]
    return {
        "states": len(rows),
        "threshold": threshold,
        "expert_action_top1": sum(correct) / len(rows),
        "neural_only_expert_action_top1": sum(
            row["neural_correct"] for row in rows
        ) / len(rows),
        "changed_action_admission_rate": sum(admitted) / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite calibrated artifact: {args.output}")
    config = _read(args.config.resolve())
    artifact = _read(args.input.resolve())
    validate_artifact(artifact)
    receipts = _read(Path(artifact["target_adaptation_receipts"]["path"]))
    builder = _load_state_builder()
    source_artifacts = [
        _read(Path(row["path"])) for row in artifact["source_programs"]
    ]
    partitions = {}
    for partition in ("adaptation_train", "adaptation_validation"):
        episodes = [
            row for row in receipts["episodes"] if row["partition"] == partition
        ]
        states = builder(
            episodes,
            feature_bins=int(artifact["feature_bins"]),
            label_granularity=str(
                artifact.get("label_granularity", "exact_action")
            ),
        )
        partitions[partition] = _decisions(
            states, artifact=artifact, source_artifacts=source_artifacts,
        )
    grid = tuple(map(float, config["threshold_grid"]))
    train_grid = [_metrics(partitions["adaptation_train"], value) for value in grid]
    train_gate = config["train_selection"]
    eligible = [
        row for row in train_grid
        if row["changed_action_admission_rate"] >= float(
            train_gate["minimum_changed_action_admission_rate"]
        )
        and row["changed_action_admission_rate"] <= float(
            train_gate["maximum_changed_action_admission_rate"]
        )
        and row["expert_action_top1"] >= (
            row["neural_only_expert_action_top1"]
            - float(train_gate["maximum_top1_drop_vs_neural"])
        )
    ]
    if not eligible:
        print(json.dumps({
            "status": "NO_RISK_FIRST_THRESHOLD_QUALIFIED",
            "train_grid": train_grid,
            "train_selection": train_gate,
        }, indent=2, sort_keys=True))
        raise SystemExit("no risk-first admission threshold qualified on train")
    # Risk-first: maximize the target support threshold, then predictive score.
    selected = max(eligible, key=lambda row: (
        row["threshold"], row["expert_action_top1"],
    ))
    validation = _metrics(
        partitions["adaptation_validation"], float(selected["threshold"]),
    )
    gates_config = config["validation_gates"]
    gates = {
        "changed_action_admission_nontrivial": (
            validation["changed_action_admission_rate"] >= float(
                gates_config["minimum_changed_action_admission_rate"]
            )
        ),
        "expert_action_top1_noninferior": (
            validation["expert_action_top1"] >= (
                validation["neural_only_expert_action_top1"]
                - float(gates_config["maximum_top1_drop_vs_neural"])
            )
        ),
    }
    body = dict(artifact)
    body.pop("artifact_sha256")
    body.update({
        "status": (
            "ALFWORLD_TYPED_GROUNDING_AND_ABSTENTION_QUALIFIED"
            if all(gates.values()) else
            "ALFWORLD_TYPED_GROUNDING_AND_ABSTENTION_BLOCKED"
        ),
        "minimum_source_policy_support_ratio": float(selected["threshold"]),
        "abstention_calibration": {
            "schema_version": "phase3-alfworld-risk-first-abstention-v1",
            "blocked_online_predecessor": (
                "PHASE3_ALFWORLD_REPLICATION_V1_SOURCE_1_OF_11_VS_NEURAL_5_OF_11;"
                "CEILING_INFRASTRUCTURE_FAILED_BEFORE_COMPLETION"
            ),
            "formal_resets_before_calibration": 0,
            "selection_partition": "adaptation_train",
            "selection_rule": (
                "MAXIMIZE_TARGET_POLICY_SUPPORT_THRESHOLD_SUBJECT_TO_"
                "NONTRIVIAL_ADMISSION_AND_TOP1_NONINFERIORITY"
            ),
            "train_grid": train_grid,
            "selected_train": selected,
            "adaptation_validation": validation,
            "frozen_validation_gates": gates_config,
            "gates": gates,
        },
    })
    calibrated = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibrated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": calibrated["status"],
        "artifact_sha256": calibrated["artifact_sha256"],
        "selected_threshold": selected["threshold"],
        "train": selected,
        "validation": validation,
        "gates": gates,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
