#!/usr/bin/env python3
"""Calibrate a loss-free target-policy support gate for intervention effects."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_alfworld_typed_grounder import validate_artifact  # noqa: E402
from motif_transfer.phase3_source_portfolio import (  # noqa: E402
    permute_selected_effect_binding,
    select_source_program_portfolio,
)
from motif_transfer.phase3_typed_effect_induction import target_trial_order  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_trainer():
    path = REPO / "scripts/train_phase3_alfworld_intervention_grounder.py"
    spec = importlib.util.spec_from_file_location("phase3_intervention_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load intervention grounder utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decisions(
    states: Sequence[Mapping[str, Any]], *, artifact: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]], trainer: Any,
) -> list[dict[str, Any]]:
    result = []
    for state in states:
        effects, policy = trainer._predictions(
            state,
            heads=artifact["typed_effect_heads"],
            exponent=float(artifact["policy_support_exponent"]),
        )
        rows = list(state["rows"])
        ids = [str(row["candidate_id"]) for row in rows]
        neural = max(range(len(rows)), key=lambda index: (
            policy[index], rows[index]["option"],
        ))
        source = neural
        permuted = neural
        selected_effect = None
        receipt = select_source_program_portfolio(
            source_artifacts,
            candidate_ids=ids,
            candidate_effects=effects,
            target_grounding_sha256=stable_hash(effects),
        )
        selected_sha = receipt["selected_artifact_sha256"]
        if selected_sha is not None:
            source_artifact = next(
                row for row in source_artifacts
                if row["artifact_sha256"] == selected_sha
            )
            program = source_artifact["typed_effect_program"]
            order, reason = target_trial_order(program, effects)
            if reason is not None:
                raise RuntimeError(f"source binding failed during calibration: {reason}")
            source = order[0]
            shuffled, _ = permute_selected_effect_binding(
                program, candidate_ids=ids, candidate_effects=effects,
            )
            shuffled_order, reason = target_trial_order(program, shuffled)
            if reason is not None:
                raise RuntimeError(f"control binding failed during calibration: {reason}")
            permuted = shuffled_order[0]
            selected_effect = str(program["selected_effect_type"])
        result.append({
            "neural_h8": float(
                rows[neural]["actual_effects"]["EFFECT_BY_TRANSITION_8"]
            ),
            "source_h8": float(
                rows[source]["actual_effects"]["EFFECT_BY_TRANSITION_8"]
            ),
            "neural_index": neural,
            "source_index": source,
            "permuted_index": permuted,
            "source_support_ratio": policy[source] / max(policy[neural], 1e-300),
            "permuted_support_ratio": policy[permuted] / max(
                policy[neural], 1e-300,
            ),
            "selected_effect_type": selected_effect,
        })
    return result


def _metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    source_utility = 0.0
    neural_utility = 0.0
    changes = 0
    contrasts = 0
    wins = 0
    losses = 0
    effects = Counter()
    for row in rows:
        source_admitted = row["source_support_ratio"] >= threshold
        permuted_admitted = row["permuted_support_ratio"] >= threshold
        source_index = (
            row["source_index"] if source_admitted else row["neural_index"]
        )
        permuted_index = (
            row["permuted_index"] if permuted_admitted else row["neural_index"]
        )
        source_value = (
            row["source_h8"] if source_admitted else row["neural_h8"]
        )
        neural_value = row["neural_h8"]
        neural_utility += neural_value
        source_utility += source_value
        changes += int(source_index != row["neural_index"])
        contrasts += int(source_index != permuted_index)
        wins += int(source_value > neural_value)
        losses += int(source_value < neural_value)
        if source_admitted and row["selected_effect_type"] is not None:
            effects[str(row["selected_effect_type"])] += 1
    total = len(rows)
    return {
        "states": total,
        "threshold": threshold,
        "neural_mean_h8_utility": neural_utility / max(1, total),
        "source_mean_h8_utility": source_utility / max(1, total),
        "source_neural_change_rate": changes / max(1, total),
        "source_permuted_contrast_rate": contrasts / max(1, total),
        "source_h8_wins": wins,
        "source_h8_losses": losses,
        "admitted_selected_effect_counts": dict(effects),
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
    if artifact.get("grounding_supervision") != (
        "DEVELOPMENT_ONLY_TARGET_NATIVE_OPTION_INTERVENTION_FORKS_"
        "WITH_H1_H4_H8_TRANSITION_EFFECTS"
    ):
        raise SystemExit("input did not use intervention-grounded supervision")
    trainer = _load_trainer()
    receipts = _read(Path(artifact["target_adaptation_receipts"]["path"]))
    source_artifacts = [
        _read(Path(row["path"])) for row in artifact["source_programs"]
    ]
    partitions = {
        partition: trainer._states(
            receipts,
            partition=partition,
            feature_bins=int(artifact["feature_bins"]),
            policy_head=artifact["target_policy_head"],
        )
        for partition in ("adaptation_train", "adaptation_validation")
    }
    decision_rows = {
        partition: _decisions(
            states, artifact=artifact, source_artifacts=source_artifacts,
            trainer=trainer,
        )
        for partition, states in partitions.items()
    }
    train_grid = [
        _metrics(decision_rows["adaptation_train"], float(threshold))
        for threshold in config["threshold_grid"]
    ]
    selection = config["train_selection"]
    eligible = [row for row in train_grid if (
        row["source_h8_losses"]
        <= int(selection["maximum_source_h8_loss_states"])
        and row["source_mean_h8_utility"] >= row["neural_mean_h8_utility"]
        and row["source_neural_change_rate"]
        >= float(selection["minimum_source_neural_change_rate"])
        and row["source_permuted_contrast_rate"]
        >= float(selection["minimum_source_permuted_contrast_rate"])
    )]
    if not eligible:
        raise SystemExit("no loss-free nontrivial support threshold qualified on train")
    # The smallest safe threshold retains the most source intervention
    # opportunity; validation and online qualification remain unseen here.
    selected = min(eligible, key=lambda row: row["threshold"])
    validation = _metrics(
        decision_rows["adaptation_validation"], float(selected["threshold"]),
    )
    thresholds = config["frozen_validation_thresholds"]
    gates = {
        "zero_h8_loss_states": validation["source_h8_losses"]
        <= int(thresholds["maximum_source_h8_loss_states"]),
        "h8_utility_noninferior": validation["source_mean_h8_utility"]
        >= validation["neural_mean_h8_utility"],
        "source_change_nontrivial": validation["source_neural_change_rate"]
        >= float(thresholds["minimum_source_neural_change_rate"]),
        "permuted_contrast": validation["source_permuted_contrast_rate"]
        >= float(thresholds["minimum_source_permuted_contrast_rate"]),
        "multiple_source_effects_admitted": len(
            validation["admitted_selected_effect_counts"]
        ) >= int(thresholds["minimum_admitted_selected_effect_types"]),
    }
    body = dict(artifact)
    body.pop("artifact_sha256")
    body.update({
        "status": (
            "ALFWORLD_INTERVENTION_GROUNDER_AND_ABSTENTION_QUALIFIED"
            if all(gates.values()) else
            "ALFWORLD_INTERVENTION_GROUNDER_AND_ABSTENTION_BLOCKED"
        ),
        "minimum_source_policy_support_ratio": float(selected["threshold"]),
        "intervention_abstention_calibration": {
            "schema_version": "phase3-alfworld-intervention-abstention-v10",
            "blocked_predecessor_artifact_sha256": artifact["artifact_sha256"],
            "formal_resets_before_calibration": 0,
            "selection_partition": "adaptation_train",
            "selection_rule": (
                "MINIMUM_TARGET_POLICY_SUPPORT_THRESHOLD_WITH_ZERO_H8_LOSS_"
                "AND_NONTRIVIAL_SOURCE_CHANGE"
            ),
            "train_grid": train_grid,
            "selected_train": selected,
            "adaptation_validation": validation,
            "frozen_validation_thresholds": thresholds,
            "gates": gates,
        },
    })
    output = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": output["status"],
        "artifact_sha256": output["artifact_sha256"],
        "selected_threshold": selected["threshold"],
        "selected_train": selected,
        "validation": validation,
        "gates": gates,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
