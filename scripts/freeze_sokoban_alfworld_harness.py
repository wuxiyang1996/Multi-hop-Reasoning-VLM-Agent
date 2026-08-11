#!/usr/bin/env python3
"""Bind a frozen Sokoban skill to an adaptation-only ALFWorld Harness."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from motif_transfer.alfworld_hierarchical_grounder import action_option, score_actions
from motif_transfer.contracts import stable_hash
from motif_transfer.sokoban_alfworld_harness import choose_action
from motif_transfer.sokoban_commit_skill import validate_artifact


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--source-qualification", type=Path, required=True)
    parser.add_argument("--fresh-confirmation", type=Path, required=True)
    parser.add_argument("--target-artifact", type=Path, required=True)
    parser.add_argument("--adaptation-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-option-accuracy", type=float, default=0.75)
    parser.add_argument("--minimum-commit-recall", type=float, default=0.90)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen Harness: {args.output}")
    source = _read(args.source_artifact)
    source_qualification = _read(args.source_qualification)
    fresh = _read(args.fresh_confirmation)
    target = _read(args.target_artifact)
    adaptation = _read(args.adaptation_receipts)
    validate_artifact(source)
    if not source_qualification.get("source_gate_passed"):
        raise SystemExit("source qualification did not pass")
    if not fresh.get("source_gate_passed") or not fresh.get("fresh_confirmation"):
        raise SystemExit("fresh source confirmation did not pass")
    if target.get("status") != "QUALIFICATION_AUTHORIZED":
        raise SystemExit("target-native grounder was not authorized")
    if not target.get("target_grounder_gate", {}).get("passed"):
        raise SystemExit("target-native grounder gate did not pass")
    if adaptation.get("qualification_or_heldout_read"):
        raise SystemExit("target adaptation receipts crossed the evaluation boundary")

    counts: Counter[tuple[str, str]] = Counter()
    concrete_correct = 0
    states = 0
    for episode in adaptation["episodes"]:
        if episode.get("partition") != "adaptation_validation":
            continue
        history: list[str] = []
        for transition in episode["transitions"]:
            grounded = score_actions(
                goal=str(transition["goal"]),
                observation=str(transition["before_observation"]),
                native_actions=tuple(map(str, transition["native_actions"])),
                step=int(transition["step"]),
                action_history=history,
                artifact=target["target_grounder"],
            )
            identity = f"{episode['task_id']}:{transition['step']}"
            decision = choose_action(
                condition="authentic_source_plus_harness",
                grounded=grounded,
                goal=str(transition["goal"]),
                history=history,
                source_artifact=source,
                identity=identity,
            )
            expert_action = str(transition["expert_action"])
            expert_option = (
                "POSITION" if action_option(expert_action) == "SEARCH" else "COMMIT"
            )
            source_option = str(decision["source_selected_option"])
            counts[(expert_option, source_option)] += 1
            concrete_correct += int(decision["action"] == expert_action)
            states += 1
            history.append(expert_action)
    correct = sum(count for (expert, selected), count in counts.items()
                  if expert == selected)
    commit_total = sum(count for (expert, _selected), count in counts.items()
                       if expert == "COMMIT")
    commit_hits = counts[("COMMIT", "COMMIT")]
    option_accuracy = correct / states
    commit_recall = commit_hits / commit_total if commit_total else 0.0
    compatibility_passed = (
        option_accuracy >= args.minimum_option_accuracy
        and commit_recall >= args.minimum_commit_recall
    )
    compatibility = {
        "authority": "TARGET_ADAPTATION_VALIDATION_ONLY",
        "states": states,
        "option_confusion": {
            f"{expert}->{selected}": count
            for (expert, selected), count in sorted(counts.items())
        },
        "option_accuracy": option_accuracy,
        "commit_recall": commit_recall,
        "concrete_action_top1": concrete_correct / states,
        "minimum_option_accuracy": args.minimum_option_accuracy,
        "minimum_commit_recall": args.minimum_commit_recall,
        "passed": compatibility_passed,
    }
    body = {
        "schema_version": "sokoban-alfworld-harness-v1",
        "status": (
            "QUALIFICATION_AUTHORIZED"
            if compatibility_passed else "BLOCKED_BEFORE_QUALIFICATION"
        ),
        "claim_boundary": (
            "SOURCE_SELECTS_POSITION_OR_COMMIT_ONLY; TARGET_HARNESS_GROUNDS_"
            "PREDICATES_AND_REALIZES_ONE_NATIVE_ACTION; NO_SELECTED_VALID_UNSEEN_"
            "TASK_WAS_RESET_DURING_FREEZE"
        ),
        "source_artifact": {
            "path": str(args.source_artifact.resolve()),
            "file_sha256": _sha256(args.source_artifact),
            "artifact_sha256": source["artifact_sha256"],
        },
        "source_qualification": {
            "path": str(args.source_qualification.resolve()),
            "file_sha256": _sha256(args.source_qualification),
            "report_sha256": source_qualification["report_sha256"],
        },
        "fresh_source_confirmation": {
            "path": str(args.fresh_confirmation.resolve()),
            "file_sha256": _sha256(args.fresh_confirmation),
            "report_sha256": fresh["report_sha256"],
        },
        "target_grounder_artifact": {
            "path": str(args.target_artifact.resolve()),
            "file_sha256": _sha256(args.target_artifact),
            "target_grounder_gate": target["target_grounder_gate"],
        },
        "adaptation_receipts": {
            "path": str(args.adaptation_receipts.resolve()),
            "file_sha256": _sha256(args.adaptation_receipts),
        },
        "compatibility_gate": compatibility,
        "permissions": {
            "source": ["SELECT_POSITION_OR_COMMIT", "ABSTAIN"],
            "target_harness": [
                "GROUND_CANONICAL_PREDICATES",
                "REFUTE_UNSATISFIED_COMMIT_PRECONDITION",
                "REALIZE_ONE_NATIVE_ACTION_INSIDE_SELECTED_OPTION",
                "REPORT_OBSERVED_EFFECT",
            ],
            "forbidden": [
                "SOURCE_RANKS_CONCRETE_TARGET_ACTIONS",
                "HARNESS_CHOOSES_SOURCE_OPTION",
                "TARGET_OUTCOME_UPDATES_FROZEN_SOURCE_SKILL",
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
        "status": payload["status"],
        "harness_sha256": payload["harness_sha256"],
        "compatibility_gate": compatibility,
    }, indent=2, sort_keys=True))
    return 0 if compatibility_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
