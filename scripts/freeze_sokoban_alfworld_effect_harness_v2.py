#!/usr/bin/env python3
"""Freeze V2 source-effect/target-grounder compatibility on adaptation data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from motif_transfer.alfworld_hierarchical_grounder import action_option
from motif_transfer.alfworld_masked_effect_grounder import (
    score_actions,
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash
from motif_transfer.sokoban_alfworld_effect_harness import (
    choose_action,
    ground_effect_predicates,
)
from motif_transfer.sokoban_effect_program import validate_effect_program


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _states(episodes: list[dict], target: dict) -> list[dict]:
    result = []
    for episode in episodes:
        history: list[str] = []
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
                "task_id": str(episode["task_id"]),
                "step": int(transition["step"]),
                "goal": str(transition["goal"]),
                "history": tuple(history),
                "grounded": grounded,
                "expert_action": expert,
                "expert_group": (
                    "POSITION" if action_option(expert) == "SEARCH" else "COMMIT"
                ),
            })
            history.append(expert)
    return result


def _metrics(states: list[dict], threshold: float) -> dict:
    confusion: Counter[tuple[str, str]] = Counter()
    for row in states:
        predicates = ground_effect_predicates(
            row["grounded"], effect_threshold=threshold,
        )
        selected = (
            "COMMIT" if predicates["direct_progress_available"] else "POSITION"
        )
        confusion[(row["expert_group"], selected)] += 1
    commit_total = sum(
        count for (expert, _selected), count in confusion.items()
        if expert == "COMMIT"
    )
    position_total = sum(
        count for (expert, _selected), count in confusion.items()
        if expert == "POSITION"
    )
    commit_recall = confusion[("COMMIT", "COMMIT")] / commit_total
    position_recall = confusion[("POSITION", "POSITION")] / position_total
    return {
        "states": len(states),
        "threshold": threshold,
        "option_accuracy": sum(
            count for (expert, selected), count in confusion.items()
            if expert == selected
        ) / len(states),
        "balanced_option_accuracy": 0.5 * (commit_recall + position_recall),
        "commit_recall": commit_recall,
        "position_recall": position_recall,
        "confusion": {
            f"{expert}->{selected}": count
            for (expert, selected), count in sorted(confusion.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--fresh-source-confirmation", type=Path, required=True)
    parser.add_argument("--target-artifact", type=Path, required=True)
    parser.add_argument("--adaptation-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold-grid", default="0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen Harness: {args.output}")
    source = _read(args.source_artifact)
    source_confirmation = _read(args.fresh_source_confirmation)
    target = _read(args.target_artifact)
    receipts = _read(args.adaptation_receipts)
    validate_effect_program(source)
    validate_target_artifact(target)
    if not source_confirmation.get("source_gate_passed"):
        raise SystemExit("fresh source effect confirmation did not pass")
    if source_confirmation.get("artifact_sha256") != source.get("artifact_sha256"):
        raise SystemExit("fresh source confirmation/artifact mismatch")
    if target.get("status") != "ADAPTATION_GATE_PASSED":
        raise SystemExit("masked target grounder did not pass")
    if receipts.get("qualification_or_heldout_read"):
        raise SystemExit("adaptation receipts crossed target evaluation boundary")
    training_states = _states([
        row for row in receipts["episodes"] if row["partition"] == "adaptation_train"
    ], target)
    validation_states = _states([
        row for row in receipts["episodes"]
        if row["partition"] == "adaptation_validation"
    ], target)
    thresholds = tuple(float(value) for value in args.threshold_grid.split(","))
    training_grid = [_metrics(training_states, value) for value in thresholds]
    selected = max(
        training_grid,
        key=lambda row: (
            row["balanced_option_accuracy"], row["option_accuracy"],
            -row["threshold"],
        ),
    )
    threshold = float(selected["threshold"])
    validation = _metrics(validation_states, threshold)
    concrete_hits = 0
    for row in validation_states:
        decision = choose_action(
            condition="authentic_source_effect_harness",
            grounded=row["grounded"],
            history=row["history"],
            source_artifact=source,
            effect_threshold=threshold,
        )
        concrete_hits += int(decision["action"] == row["expert_action"])
    validation["concrete_action_top1"] = concrete_hits / len(validation_states)
    gates = {
        "balanced_option_accuracy": validation["balanced_option_accuracy"] >= 0.75,
        "commit_recall": validation["commit_recall"] >= 0.75,
        "position_recall": validation["position_recall"] >= 0.75,
    }
    passed = all(gates.values())
    body = {
        "schema_version": "sokoban-alfworld-effect-harness-v2",
        "status": "QUALIFICATION_AUTHORIZED" if passed else "BLOCKED_BEFORE_QUALIFICATION",
        "claim_boundary": (
            "SOURCE_TRANSFERS_EFFECT_TRIGGERED_CONTROL_FLOW_ONLY; TARGET_NATIVE_"
            "NEURAL_HEADS_GROUND_EFFECT_AND_REALIZE_ACTION; AUTHENTIC_PATH_DOES_"
            "NOT_READ_REQUIRED_OPTION; NO_TARGET_EVALUATION_TASK_RESET"
        ),
        "source_artifact": {
            "path": str(args.source_artifact.resolve()),
            "file_sha256": _sha256(args.source_artifact),
            "artifact_sha256": source["artifact_sha256"],
        },
        "fresh_source_confirmation": {
            "path": str(args.fresh_source_confirmation.resolve()),
            "file_sha256": _sha256(args.fresh_source_confirmation),
            "report_sha256": source_confirmation["report_sha256"],
        },
        "masked_target_grounder": {
            "path": str(args.target_artifact.resolve()),
            "file_sha256": _sha256(args.target_artifact),
            "artifact_sha256": target["artifact_sha256"],
        },
        "adaptation_receipts": {
            "path": str(args.adaptation_receipts.resolve()),
            "file_sha256": _sha256(args.adaptation_receipts),
        },
        "effect_threshold": {
            "selection_partition": "adaptation_train",
            "candidate_grid": list(thresholds),
            "selected": threshold,
            "training_grid": training_grid,
        },
        "adaptation_validation_compatibility": validation,
        "gates": gates,
        "permissions": {
            "source": ["SELECT_POSITION_OR_COMMIT", "VERIFY", "REPLAN"],
            "target_harness": [
                "GROUND_EFFECT_PREDICATES_WITH_MASKED_NEURAL_HEADS",
                "REALIZE_ONE_NATIVE_ACTION_INSIDE_SELECTED_OPTION",
                "REPORT_NEXT_OBSERVATION_FOR_RECOMPUTATION",
            ],
            "forbidden": [
                "AUTHENTIC_PATH_READS_TARGET_REQUIRED_OPTION",
                "SOURCE_RANKS_CONCRETE_TARGET_ACTIONS",
                "TARGET_EVALUATION_UPDATES_SOURCE_PROGRAM_OR_THRESHOLD",
            ],
        },
    }
    payload = body | {"harness_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "harness_sha256": payload["harness_sha256"],
        "status": payload["status"],
        "selected_effect_threshold": threshold,
        "adaptation_validation": validation,
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
