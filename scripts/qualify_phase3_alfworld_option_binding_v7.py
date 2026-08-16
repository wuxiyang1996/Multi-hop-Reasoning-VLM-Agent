#!/usr/bin/env python3
"""Qualify target-native option binding for the unchanged Phase-3 IR."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_hierarchical_grounder import (  # noqa: E402
    action_option,
    mlp_probability,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_alfworld_typed_grounder import (  # noqa: E402
    validate_artifact,
)
from motif_transfer.phase3_source_portfolio import (  # noqa: E402
    permute_selected_effect_binding,
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


def _metrics(
    states: Sequence[Mapping[str, Any]], *, artifact: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    effects = tuple(artifact["effect_types"])
    counts = Counter()
    selected_effects = Counter()
    for state in states:
        groups: dict[str, list[tuple[str, float, dict[str, float]]]] = defaultdict(list)
        for row in state["rows"]:
            policy = mlp_probability(
                row["features"], artifact["target_policy_head"],
            )
            values = {
                effect: mlp_probability(
                    row["features"], artifact["typed_effect_heads"][effect],
                ) * policy ** float(artifact["policy_support_exponent"])
                for effect in effects
            }
            groups[action_option(str(row["action"]))].append((
                str(row["action"]), policy, values,
            ))
        options = sorted(groups)
        policy = [max(row[1] for row in groups[option]) for option in options]
        neural_index = max(range(len(options)), key=lambda index: (
            policy[index], options[index],
        ))
        source_index = generic_index = neural_index
        permuted_index = None
        if len(options) >= 2:
            counts["multi_option_states"] += 1
            grounded = [{
                effect: max(row[2][effect] for row in groups[option])
                for effect in effects
            } for option in options]
            ids = [
                stable_hash({"target_native_option": option}) for option in options
            ]
            receipt = select_source_program_portfolio(
                source_artifacts,
                candidate_ids=ids,
                candidate_effects=grounded,
                target_grounding_sha256=stable_hash(grounded),
            )
            selected_sha = receipt["selected_artifact_sha256"]
            if selected_sha is not None:
                selected = next(
                    row for row in source_artifacts
                    if row["artifact_sha256"] == selected_sha
                )
                program = selected["typed_effect_program"]
                order, reason = target_trial_order(program, grounded)
                if reason is not None:
                    raise RuntimeError(f"option binding failed: {reason}")
                source_index = order[0]
                permuted, _ = permute_selected_effect_binding(
                    program, candidate_ids=ids, candidate_effects=grounded,
                )
                permuted_order, reason = target_trial_order(program, permuted)
                if reason is not None:
                    raise RuntimeError(f"option control failed: {reason}")
                permuted_index = permuted_order[0]
                selected_effects[str(program["selected_effect_type"])] += 1
                counts["applicable"] += 1
            generic_index = max(range(len(options)), key=lambda index: (
                sum(grounded[index].values()), policy[index], options[index],
            ))
        expert_option = action_option(str(state["expert_action"]))
        counts["states"] += 1
        counts["neural_hits"] += options[neural_index] == expert_option
        counts["source_hits"] += options[source_index] == expert_option
        counts["generic_hits"] += options[generic_index] == expert_option
        counts["source_neural_changes"] += source_index != neural_index
        counts["source_permuted_contrasts"] += (
            permuted_index is not None and source_index != permuted_index
        )
    total = counts["states"]
    return {
        "states": total,
        "multi_option_states": counts["multi_option_states"],
        "applicable_multi_option_states": counts["applicable"],
        "conditional_applicability_rate": (
            counts["applicable"] / max(1, counts["multi_option_states"])
        ),
        "neural_only_expert_option_accuracy": counts["neural_hits"] / total,
        "source_induced_expert_option_accuracy": counts["source_hits"] / total,
        "generic_scaffold_expert_option_accuracy": counts["generic_hits"] / total,
        "source_neural_option_change_rate": counts["source_neural_changes"] / total,
        "source_permuted_option_contrast_rate": (
            counts["source_permuted_contrasts"] / total
        ),
        "selected_effect_counts": dict(selected_effects),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite qualified artifact: {args.output}")
    config = _read(args.config.resolve())
    artifact = _read(args.input.resolve())
    validate_artifact(artifact)
    if artifact.get("label_granularity") != "target_native_option":
        raise SystemExit("input heads were not trained on target-native options")
    receipts = _read(Path(artifact["target_adaptation_receipts"]["path"]))
    builder = _load_state_builder()
    source_artifacts = [
        _read(Path(row["path"])) for row in artifact["source_programs"]
    ]
    results = {}
    for partition in ("adaptation_train", "adaptation_validation"):
        episodes = [
            row for row in receipts["episodes"] if row["partition"] == partition
        ]
        states = builder(
            episodes,
            feature_bins=int(artifact["feature_bins"]),
            label_granularity="target_native_option",
        )
        results[partition] = _metrics(
            states, artifact=artifact, source_artifacts=source_artifacts,
        )
    validation = results["adaptation_validation"]
    thresholds = config["frozen_validation_thresholds"]
    gates = {
        "all_neural_effect_heads_auc": all(
            row["validation_auc"] >= float(thresholds["minimum_each_head_auc"])
            for row in artifact["head_validation"].values()
        ),
        "portfolio_applicability": validation["conditional_applicability_rate"] >= float(
            thresholds["minimum_applicability_rate"]
        ),
        "source_option_noninferior": (
            validation["source_induced_expert_option_accuracy"] >= (
                validation["neural_only_expert_option_accuracy"]
                - float(thresholds["maximum_source_option_accuracy_drop"])
            )
        ),
        "source_option_change_nontrivial": (
            validation["source_neural_option_change_rate"] >= float(
                thresholds["minimum_source_neural_option_change_rate"]
            )
        ),
        "source_permuted_contrast": (
            validation["source_permuted_option_contrast_rate"] >= float(
                thresholds["minimum_source_permuted_option_contrast_rate"]
            )
        ),
        "multiple_source_effect_types_selected": (
            len(validation["selected_effect_counts"])
            >= int(thresholds["minimum_selected_effect_types"])
        ),
    }
    body = dict(artifact)
    body.pop("artifact_sha256")
    body.update({
        "status": (
            "ALFWORLD_TYPED_OPTION_BINDING_QUALIFIED"
            if all(gates.values()) else "ALFWORLD_TYPED_OPTION_BINDING_BLOCKED"
        ),
        "binding_level": "target_native_option",
        "minimum_source_policy_support_ratio": 0.0,
        "option_binding_qualification": {
            "schema_version": "phase3-alfworld-option-binding-v1",
            "formal_resets_before_qualification": 0,
            "target_native_action_realization": (
                "MAX_FROZEN_TARGET_POLICY_WITHIN_SOURCE_SELECTED_OPTION"
            ),
            "train": results["adaptation_train"],
            "validation": validation,
            "frozen_validation_thresholds": thresholds,
            "gates": gates,
        },
        "formal_success_read_for_training_or_qualification": False,
    })
    output = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": output["status"],
        "artifact_sha256": output["artifact_sha256"],
        "train": results["adaptation_train"],
        "validation": validation,
        "gates": gates,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
