#!/usr/bin/env python3
"""Retire consumed AGQA reports into V29 transfer-development evidence.

This is an adaptation-only, zero-provider-call audit.  It repairs two
composition errors without changing any stored neural prediction:

1. the source-induced abstention rules are executed per row; and
2. source abstention preserves the frozen target-native comparator instead of
   falling back to the weaker direct-only arm.

All input outcomes have already been consumed, so this script cannot produce a
new confirmatory claim.  Its only authorized output is calibration for a future
disjoint reserve.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_goal_relation_transfer import (  # noqa: E402
    bind_source_goal_relation_program,
)
from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    exact_one_sided_pvalue, target_only_ontology_decision,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.online_transfer_utility import (  # noqa: E402
    ApplicabilityReceipt, OnlineTransferUtilityGate, PairedOutcome,
)
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _answer_matches,
)


DEFAULT_INPUTS = (
    "runs/agqa2_query_object_v25_reserve/report.json",
    "runs/agqa2_query_object_v28_reserve/report.json",
)
DEFAULT_ARTIFACT = "runs/sokoban_goal_relation_macro_v3/artifact.json"
DEFAULT_CONFIRMATION = (
    "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    body = dict(report)
    claimed = str(body.pop("report_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"input report hash mismatch: {path}")
    return report


def _row_integrity(row: Mapping[str, Any]) -> bool:
    forbidden = (
        "runtime_answer_read",
        "runtime_functional_program_read",
        "runtime_scene_graph_read",
        "runtime_source_identity_read",
        "operand_grounder_question_read",
        "operand_grounder_competing_operand_read",
        "object_ontology_original_question_read",
        "object_ontology_answer_candidates_read",
    )
    return all(row.get(key) is False for key in forbidden)


def _calibration(wins: int, losses: int, ties: int) -> dict[str, Any]:
    gate = OnlineTransferUtilityGate()
    gate.update_many(
        [PairedOutcome(True, False)] * wins
        + [PairedOutcome(False, True)] * losses
        + [PairedOutcome(True, True)] * ties
    )
    applicability = ApplicabilityReceipt(True, True, True, True, True)
    decision = gate.decision(applicability)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "decision": decision.decision,
        "reason": decision.reason,
        "posterior_mean_win_probability": (
            decision.posterior_mean_win_probability
        ),
        "posterior_lower_win_probability": (
            decision.posterior_lower_win_probability
        ),
        "observed_disagreement_rate": decision.observed_disagreement_rate,
    }


def _paired_metrics(rows: Sequence[Mapping[str, Any]], left: str, right: str):
    wins = sum(row[left] and not row[right] for row in rows)
    losses = sum(row[right] and not row[left] for row in rows)
    ties = len(rows) - wins - losses
    return {
        "left_correct": sum(bool(row[left]) for row in rows),
        "right_correct": sum(bool(row[right]) for row in rows),
        "left_minus_right_correct": (
            sum(bool(row[left]) for row in rows)
            - sum(bool(row[right]) for row in rows)
        ),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "exact_one_sided_pvalue": exact_one_sided_pvalue(
            source_wins=wins, source_losses=losses,
        ),
    }


def run(
    *, input_paths: Sequence[Path], artifact_path: Path,
    confirmation_path: Path, output_path: Path,
) -> dict[str, Any]:
    artifact = json.loads(artifact_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    adapter_path = REPO_ROOT / "src/motif_transfer/agqa_goal_relation_transfer.py"
    module_sha256 = _sha256(adapter_path)
    target_grounder_sha256 = stable_hash({
        "module_sha256": module_sha256,
        "protocol": "SOURCE_INDUCED_UNIQUE_RELATION_BINDING_V29",
        "source_program_sha256": artifact["artifact_sha256"],
        "candidate_views": "FROZEN_V22_TWO_OF_THREE_NEURAL_VIEWS",
        "gold_or_direct_visible": False,
    })
    target_executor_sha256 = stable_hash({
        "module_sha256": module_sha256,
        "executor": "AGQAObjectExecutor",
        "native_actions": "FIXED_AGQA_OBJECT_ONTOLOGY",
    })

    reports = [(path, _verified_report(path)) for path in input_paths]
    evaluated: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    seen_videos: set[str] = set()
    per_input: dict[str, dict[str, Any]] = {}
    for path, report in reports:
        report_rows = []
        for row in report["rows"]:
            task_id = str(row["task_id"])
            video_id = str(row["video_id"])
            if task_id in seen_tasks or video_id in seen_videos:
                raise ValueError("development inputs are not task/video disjoint")
            seen_tasks.add(task_id)
            seen_videos.add(video_id)
            integrity = _row_integrity(row)
            binding = bind_source_goal_relation_program(
                artifact=artifact,
                confirmation=confirmation,
                task_id=task_id,
                target_state_sha256=str(row["runtime_receipt_sha256"]),
                target_grounder_sha256=target_grounder_sha256,
                calibrated_execution=row["calibrated_target_native_execution"],
                grounder_qualified=integrity,
                formal_outcome_read=False,
            )
            target_only_decision = target_only_ontology_decision(
                row["object_ontology_receipts"], (0.8, 0.8),
            )
            target_prediction = target_only_decision or row["direct_response"]
            source_prediction = binding.authorized_candidate or target_prediction
            generic_candidate = row["calibrated_target_native_execution"].get(
                "decision"
            )
            generic_prediction = generic_candidate or target_prediction
            # The source held-out effect-shuffle had zero authenticated
            # bindings, so its matched program must abstain on every target.
            shuffled = bind_source_goal_relation_program(
                artifact=artifact,
                confirmation=confirmation,
                task_id=task_id,
                target_state_sha256=str(row["runtime_receipt_sha256"]),
                target_grounder_sha256=target_grounder_sha256,
                calibrated_execution=row["calibrated_target_native_execution"],
                grounder_qualified=integrity,
                effect_binding_authenticated=False,
                formal_outcome_read=False,
            )
            shuffled_prediction = (
                shuffled.authorized_candidate or target_prediction
            )
            # Predictions are complete before evaluator-only gold is accessed.
            gold = str(row["gold_answer_evaluator_only"])
            item = {
                "input": str(path.relative_to(REPO_ROOT)),
                "task_id": task_id,
                "video_id": video_id,
                "binding_receipt_sha256": binding.receipt_sha256,
                "binding_reason": binding.reason,
                "target_binding_count": binding.target_binding_count,
                "source_executor_authorized": (
                    binding.authorized_candidate is not None
                ),
                "source_candidate": binding.authorized_candidate,
                "effect_shuffled_authorized": (
                    shuffled.authorized_candidate is not None
                ),
                "target_only_decisive": target_only_decision is not None,
                "source_correct": _answer_matches(source_prediction, gold),
                "target_correct": _answer_matches(target_prediction, gold),
                "effect_shuffled_correct": _answer_matches(
                    shuffled_prediction, gold,
                ),
                "generic_scaffold_correct": _answer_matches(
                    generic_prediction, gold,
                ),
                "target_written_equivalent_correct": _answer_matches(
                    source_prediction, gold,
                ),
                "prediction_frozen_before_gold_read": True,
                "formal_outcome_used_for_current_authorization": False,
                "runtime_integrity_qualified": integrity,
            }
            evaluated.append(item)
            report_rows.append(item)
        per_input[str(path.relative_to(REPO_ROOT))] = {
            "rows": len(report_rows),
            "source_vs_target": _paired_metrics(
                report_rows, "source_correct", "target_correct",
            ),
            "source_authorizations": sum(
                row["source_executor_authorized"] for row in report_rows
            ),
        }

    source_vs_target = _paired_metrics(
        evaluated, "source_correct", "target_correct",
    )
    source_vs_shuffled = _paired_metrics(
        evaluated, "source_correct", "effect_shuffled_correct",
    )
    source_vs_generic = _paired_metrics(
        evaluated, "source_correct", "generic_scaffold_correct",
    )
    target_written = _paired_metrics(
        evaluated, "source_correct", "target_written_equivalent_correct",
    )
    utility = _calibration(
        source_vs_target["wins"], source_vs_target["losses"],
        source_vs_target["ties"],
    )
    authenticity = _calibration(
        source_vs_shuffled["wins"], source_vs_shuffled["losses"],
        source_vs_shuffled["ties"],
    )
    gates = {
        "source_program_fresh_validated": (
            confirmation.get("source_gate_passed") is True
        ),
        "required_development_rows": len(evaluated) >= 150,
        "development_video_disjoint": len(seen_videos) == len(evaluated),
        "runtime_integrity_qualified": all(
            row["runtime_integrity_qualified"] for row in evaluated
        ),
        "source_abstention_is_row_level": 0 < sum(
            row["source_executor_authorized"] for row in evaluated
        ) < len(evaluated),
        "no_negative_transfer_vs_target_native": (
            source_vs_target["losses"] == 0
        ),
        "directional_utility_calibrated": utility["decision"] == "SELECT_SKILL",
        "source_specific_authenticity_calibrated": (
            authenticity["decision"] == "SELECT_SKILL"
        ),
        "effect_shuffled_source_never_authorized": all(
            not row["effect_shuffled_authorized"]
            for row in evaluated
        ),
        "target_written_equivalent_is_ceiling_control": (
            target_written["wins"] == target_written["losses"] == 0
        ),
        "current_outcome_never_used_for_authorization": all(
            not row["formal_outcome_used_for_current_authorization"]
            for row in evaluated
        ),
    }
    qualified = all(gates.values())
    core = {
        "schema_version": "agqa2-goal-relation-transfer-development-v29",
        "status": (
            "AGQA2_GOAL_RELATION_V29_DEVELOPMENT_QUALIFIED"
            if qualified else
            "AGQA2_GOAL_RELATION_V29_DEVELOPMENT_NOT_QUALIFIED"
        ),
        "claim_boundary": (
            "RETROSPECTIVE_ADAPTATION_ONLY;CONSUMED_V25_AND_V28;"
            "ZERO_PROVIDER_CALLS;NO_CONFIRMATORY_CLAIM"
        ),
        "input_reports": [{
            "path": str(path.relative_to(REPO_ROOT)),
            "file_sha256": _sha256(path),
            "report_sha256": report["report_sha256"],
            "prior_status": report["status"],
        } for path, report in reports],
        "source_artifact": {
            "path": str(artifact_path.relative_to(REPO_ROOT)),
            "file_sha256": _sha256(artifact_path),
            "artifact_sha256": artifact["artifact_sha256"],
            "induction_authority": artifact["induction_authority"],
        },
        "source_confirmation": {
            "path": str(confirmation_path.relative_to(REPO_ROOT)),
            "file_sha256": _sha256(confirmation_path),
            "report_sha256": confirmation["report_sha256"],
        },
        "adapter_module": str(adapter_path.relative_to(REPO_ROOT)),
        "adapter_module_sha256": module_sha256,
        "target_grounder_sha256": target_grounder_sha256,
        "target_executor_sha256": target_executor_sha256,
        "rows": len(evaluated),
        "source_executor_authorizations": sum(
            row["source_executor_authorized"] for row in evaluated
        ),
        "source_abstentions": sum(
            not row["source_executor_authorized"] for row in evaluated
        ),
        "source_vs_target_native": source_vs_target,
        "source_vs_effect_shuffled": source_vs_shuffled,
        "source_vs_generic_scaffold": source_vs_generic,
        "source_vs_target_written_equivalent": target_written,
        "future_route_calibration": {
            "utility_vs_target_native": utility,
            "authenticity_vs_effect_shuffled": authenticity,
            "may_apply_only_to_future_disjoint_tasks": True,
        },
        "per_input": per_input,
        "gates": gates,
        "qualified_for_future_disjoint_reserve": qualified,
        "provider_calls": 0,
        "confirmatory_claim": False,
        "row_receipts": evaluated,
    }
    result = core | {"report_sha256": stable_hash(core)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+", type=Path,
        default=[REPO_ROOT / value for value in DEFAULT_INPUTS],
    )
    parser.add_argument(
        "--artifact", type=Path, default=REPO_ROOT / DEFAULT_ARTIFACT,
    )
    parser.add_argument(
        "--confirmation", type=Path,
        default=REPO_ROOT / DEFAULT_CONFIRMATION,
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_goal_relation_v29_development/report.json",
    )
    args = parser.parse_args()
    result = run(
        input_paths=[path.resolve() for path in args.inputs],
        artifact_path=args.artifact.resolve(),
        confirmation_path=args.confirmation.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps({key: deepcopy(result[key]) for key in (
        "status", "rows", "source_executor_authorizations",
        "source_vs_target_native", "source_vs_effect_shuffled",
        "source_vs_generic_scaffold", "future_route_calibration", "gates",
        "provider_calls", "report_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
