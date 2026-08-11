#!/usr/bin/env python3
"""Freeze a development-only typed-IR/ALFWorld compatibility artifact."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from motif_transfer.alfworld_hierarchical_grounder import action_option
from motif_transfer.alfworld_masked_effect_grounder import (
    score_actions,
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.typed_alfworld_harness import (
    CONCRETE_ACTION_RANKINGS,
    choose_typed_action,
    target_effect,
    validate_typed_effect_ir,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_states(
    episodes: list[Mapping[str, Any]], target: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for episode in episodes:
        history: list[str] = []
        options: list[str] = []
        for transition in episode["transitions"]:
            grounded = score_actions(
                goal=str(transition["goal"]),
                observation=str(transition["before_observation"]),
                native_actions=tuple(map(str, transition["native_actions"])),
                step=int(transition["step"]),
                action_history=history,
                artifact=target,
            )
            expert = str(transition["expert_action"])
            result.append({
                "grounded": grounded,
                "history": tuple(history),
                "grounded_history_options": tuple(options),
                "expert_action": expert,
                "expert_effect": target_effect(action_option(expert)),
            })
            history.append(expert)
            options.append(action_option(expert))
    return result


def _metrics(
    states: list[Mapping[str, Any]],
    *,
    source_ir: Mapping[str, Any],
    threshold: float,
    concrete_action_ranking: str = "realization_first",
    minimum_target_policy_ratio: float = 0.0,
) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {
        "target_only": Counter(),
        "authentic_typed_ir": Counter(),
    }
    per_effect: dict[str, Counter[str]] = {}
    for state in states:
        expert_effect = str(state["expert_effect"])
        per_effect.setdefault(expert_effect, Counter())
        for condition in counts:
            decision = choose_typed_action(
                condition=condition,
                grounded=state["grounded"],
                history=state["history"],
                grounded_history_options=state["grounded_history_options"],
                source_ir=source_ir,
                minimum_realization_score=threshold,
                concrete_action_ranking=concrete_action_ranking,
                minimum_target_policy_ratio=minimum_target_policy_ratio,
            )
            row = counts[condition]
            row["states"] += 1
            row["effect_hits"] += int(
                decision["target_realized_effect"] == expert_effect
            )
            row["action_hits"] += int(
                decision["action"] == state["expert_action"]
            )
            row["source_admissions"] += int(decision["source_admitted"])
            row["changed_effects"] += int(decision["changed_effect"])
            if expert_effect != "POSITION":
                row["nonposition_states"] += 1
                row["nonposition_effect_hits"] += int(
                    decision["target_realized_effect"] == expert_effect
                )
            effect_row = per_effect[expert_effect]
            effect_row[f"{condition}_states"] += 1
            effect_row[f"{condition}_hits"] += int(
                decision["target_realized_effect"] == expert_effect
            )
    summaries = {}
    for condition, row in counts.items():
        summaries[condition] = {
            "states": row["states"],
            "effect_accuracy": row["effect_hits"] / row["states"],
            "nonposition_effect_recall": (
                row["nonposition_effect_hits"] / row["nonposition_states"]
            ),
            "expert_action_top1": row["action_hits"] / row["states"],
            "source_admission_rate": row["source_admissions"] / row["states"],
            "changed_effect_rate": row["changed_effects"] / row["states"],
        }
    result = {
        "threshold": threshold,
        "conditions": summaries,
        "per_expert_effect": {
            effect: {
                condition: (
                    row[f"{condition}_hits"] / row[f"{condition}_states"]
                )
                for condition in counts
            }
            for effect, row in sorted(per_effect.items())
        },
    }
    if concrete_action_ranking != "realization_first" or minimum_target_policy_ratio:
        result["concrete_action_ranking"] = concrete_action_ranking
        result["minimum_target_policy_ratio"] = minimum_target_policy_ratio
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--target-grounder", type=Path, required=True)
    parser.add_argument("--adaptation-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold-grid", default="0,0.05,0.1,0.2,0.3,0.4,0.5"
    )
    parser.add_argument(
        "--concrete-action-ranking",
        choices=CONCRETE_ACTION_RANKINGS,
        default="realization_first",
    )
    parser.add_argument("--policy-ratio-grid", default="0")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen Harness: {args.output}")
    source = _read(args.source_report)
    target = _read(args.target_grounder)
    receipts = _read(args.adaptation_receipts)
    if source.get("overall_status") != "SOURCE_TYPED_GATE_PASSED":
        raise SystemExit("typed source V4 gates did not pass")
    if source.get("edge_replication_gate", {}).get("status") != "EDGE_REPLICATION_GATE_PASSED":
        raise SystemExit("typed source edge replication gate did not pass")
    if source.get("effect_value_gate", {}).get("status") != "EFFECT_VALUE_GATE_PASSED":
        raise SystemExit("typed source effect value gate did not pass")
    source_ir = source["effect_ir"]
    validate_typed_effect_ir(source_ir)
    validate_target_artifact(target)
    if target.get("status") != "ADAPTATION_GATE_PASSED":
        raise SystemExit("target-native neural grounder did not pass adaptation gate")
    if receipts.get("qualification_or_heldout_read"):
        raise SystemExit("adaptation receipts crossed target evaluation boundary")
    train = _cache_states(
        [row for row in receipts["episodes"] if row["partition"] == "adaptation_train"],
        target,
    )
    validation = _cache_states(
        [row for row in receipts["episodes"] if row["partition"] == "adaptation_validation"],
        target,
    )
    thresholds = tuple(float(value) for value in args.threshold_grid.split(","))
    policy_ratios = tuple(
        float(value) for value in args.policy_ratio_grid.split(",")
    )
    v6_admission = args.concrete_action_ranking == "target_policy_within_effect"
    training_grid = [
        _metrics(
            train,
            source_ir=source_ir,
            threshold=threshold,
            concrete_action_ranking=args.concrete_action_ranking,
            minimum_target_policy_ratio=policy_ratio,
        )
        for threshold in thresholds
        for policy_ratio in policy_ratios
    ]
    if v6_admission:
        eligible_training = [
            row for row in training_grid
            if row["conditions"]["authentic_typed_ir"]["changed_effect_rate"] >= 0.02
            and 0.05 <= row["conditions"]["authentic_typed_ir"]["source_admission_rate"] <= 0.30
        ]
        if not eligible_training:
            raise SystemExit(
                "no adaptation-training candidate retained nontrivial transfer"
            )
        selected_training = max(
            eligible_training,
            key=lambda row: (
                row["conditions"]["authentic_typed_ir"]["effect_accuracy"],
                row["conditions"]["authentic_typed_ir"]["expert_action_top1"],
                row["conditions"]["authentic_typed_ir"]["nonposition_effect_recall"],
                row["minimum_target_policy_ratio"],
                row["threshold"],
            ),
        )
    else:
        selected_training = max(
            training_grid,
            key=lambda row: (
                row["conditions"]["authentic_typed_ir"]["effect_accuracy"],
                row["conditions"]["authentic_typed_ir"]["nonposition_effect_recall"],
                -row["threshold"],
            ),
        )
    threshold = float(selected_training["threshold"])
    policy_ratio = float(
        selected_training.get("minimum_target_policy_ratio", 0.0)
    )
    validation_metrics = _metrics(
        validation,
        source_ir=source_ir,
        threshold=threshold,
        concrete_action_ranking=args.concrete_action_ranking,
        minimum_target_policy_ratio=policy_ratio,
    )
    authentic = validation_metrics["conditions"]["authentic_typed_ir"]
    target_only = validation_metrics["conditions"]["target_only"]
    gates = {
        "source_and_target_prerequisites": True,
        "validation_nonposition_effect_recall": (
            authentic["nonposition_effect_recall"] >= 0.90
        ),
        "validation_effect_accuracy_noninferior": (
            authentic["effect_accuracy"] >= target_only["effect_accuracy"] - 0.01
        ),
        "validation_source_admission_nonconstant": (
            0.05 <= authentic["source_admission_rate"] <= 0.30
        ),
    }
    if v6_admission:
        gates["validation_changed_effect_nontrivial"] = (
            authentic["changed_effect_rate"] >= 0.02
        )
    passed = all(gates.values())
    body = {
        "schema_version": (
            "typed-multisource-alfworld-harness-v6"
            if v6_admission else "typed-multisource-alfworld-harness-v5"
        ),
        "status": (
            "DEVELOPMENT_DIAGNOSTIC_AUTHORIZED"
            if passed else "BLOCKED_BEFORE_ONLINE_DIAGNOSTIC"
        ),
        "claim_boundary": (
            "TARGET_ADAPTATION_ONLY_COMPATIBILITY; ONLINE_USE_RESTRICTED_TO_"
            "ALREADY_CONSUMED_QUALIFICATION; TARGET_HELDOUT_FORBIDDEN"
        ),
        "source_report": {
            "path": str(args.source_report.resolve()),
            "file_sha256": _sha256(args.source_report),
            "effect_ir_sha256": source_ir["ir_sha256"],
        },
        "source_effect_ir": source_ir,
        "target_grounder": {
            "path": str(args.target_grounder.resolve()),
            "file_sha256": _sha256(args.target_grounder),
            "artifact_sha256": target["artifact_sha256"],
        },
        "adaptation_receipts": {
            "path": str(args.adaptation_receipts.resolve()),
            "file_sha256": _sha256(args.adaptation_receipts),
        },
        "minimum_realization_score": {
            "selection_partition": "adaptation_train",
            "candidate_grid": list(thresholds),
            "selected": threshold,
            "training_grid": training_grid,
        },
        "adaptation_validation": validation_metrics,
        "gates": gates,
        "permissions": {
            "source_ir": ["FILTER_TARGET_EFFECT_OPTIONS", "ORDER_EFFECT_CYCLES"],
            "target_grounder": [
                "SCORE_TARGET_NATIVE_ACTIONS",
                "REALIZE_ACTION_WITHIN_SOURCE_SELECTED_EFFECT",
                "ABSTAIN_TO_TARGET_POLICY",
            ],
            "forbidden": [
                "SOURCE_ACTION_OR_COORDINATE_AT_RUNTIME",
                "SOURCE_SELECTS_CONCRETE_ALFWORLD_ACTION",
                "AUTHENTIC_PATH_READS_REQUIRED_OPTION_DIAGNOSTIC",
                "TARGET_HELDOUT_RESET",
            ],
        },
    }
    if v6_admission:
        body["concrete_action_ranking"] = args.concrete_action_ranking
        body["minimum_target_policy_ratio"] = {
            "selection_partition": "adaptation_train",
            "candidate_grid": list(policy_ratios),
            "selected": policy_ratio,
        }
        body["permissions"]["target_grounder"].append(
            "REQUIRE_RELATIVE_TARGET_POLICY_SUPPORT"
        )
    payload = body | {"harness_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_summary = {
        "output": str(args.output.resolve()),
        "status": payload["status"],
        "harness_sha256": payload["harness_sha256"],
        "selected_threshold": threshold,
        "adaptation_validation": validation_metrics,
        "gates": gates,
    }
    if v6_admission:
        output_summary["selected_policy_ratio"] = policy_ratio
    print(json.dumps(output_summary, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
