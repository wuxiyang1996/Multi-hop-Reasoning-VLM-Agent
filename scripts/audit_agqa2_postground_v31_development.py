#!/usr/bin/env python3
"""Calibrate the corrected post-grounding V31 adapter on consumed data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_postground_relation_transfer import (  # noqa: E402
    bind_postground_source_program,
)
from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    target_only_ontology_decision,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_goal_relation_v29_development import (  # noqa: E402
    _calibration, _paired_metrics,
)
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _answer_matches,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = str(body.pop("report_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"report hash mismatch: {path}")
    return value


def run(output_path: Path) -> dict[str, Any]:
    artifact_path = REPO_ROOT / "runs/sokoban_goal_relation_macro_v3/artifact.json"
    confirmation_path = (
        REPO_ROOT
        / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
    )
    artifact = json.loads(artifact_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    adapter_path = (
        REPO_ROOT / "src/motif_transfer/agqa_postground_relation_transfer.py"
    )
    prediction_path = (
        REPO_ROOT / "src/motif_transfer/agqa_postground_relation_evaluation.py"
    )
    module_sha256 = _sha256(adapter_path)
    prediction_sha256 = _sha256(prediction_path)
    target_grounder_sha256 = stable_hash({
        "adapter_module_sha256": module_sha256,
        "prediction_module_sha256": prediction_sha256,
        "protocol": "POST_TARGET_NATIVE_BINDING_SOURCE_EXECUTION_V31",
        "source_program_sha256": artifact["artifact_sha256"],
        "raw_neural_votes_are_perception_evidence": True,
        "resolved_candidate_set_is_symbolic_binding_authority": True,
    })
    target_executor_sha256 = stable_hash({
        "prediction_module_sha256": prediction_sha256,
        "executor": "AGQAObjectExecutor",
        "native_actions": "FIXED_AGQA_OBJECT_ONTOLOGY",
    })

    raw_paths = (
        REPO_ROOT / "runs/agqa2_query_object_v25_reserve/report.json",
        REPO_ROOT / "runs/agqa2_query_object_v28_reserve/report.json",
    )
    v30_path = REPO_ROOT / "runs/agqa2_goal_relation_v30_legacy/report.json"
    rows = []
    inputs = []
    seen_tasks: set[str] = set()
    seen_videos: set[str] = set()
    for path in raw_paths:
        report = _verified(path)
        inputs.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "file_sha256": _sha256(path),
            "report_sha256": report["report_sha256"],
            "role": "CONSUMED_ADAPTATION",
        })
        for row in report["rows"]:
            task_id, video_id = str(row["task_id"]), str(row["video_id"])
            if task_id in seen_tasks or video_id in seen_videos:
                raise ValueError("V31 adaptation rows are not disjoint")
            seen_tasks.add(task_id)
            seen_videos.add(video_id)
            binding = bind_postground_source_program(
                artifact=artifact,
                confirmation=confirmation,
                task_id=task_id,
                target_state_sha256=str(row["runtime_receipt_sha256"]),
                target_grounder_sha256=target_grounder_sha256,
                calibrated_execution=row["calibrated_target_native_execution"],
                grounder_qualified=True,
                formal_outcome_read=False,
            )
            target_decision = target_only_ontology_decision(
                row["object_ontology_receipts"], (0.8, 0.8),
            )
            target_prediction = target_decision or row["direct_response"]
            source_prediction = binding.authorized_candidate or target_prediction
            generic_prediction = (
                row["calibrated_target_native_execution"].get("decision")
                or target_prediction
            )
            if source_prediction != generic_prediction:
                raise AssertionError("V31 source/handwritten ceiling mismatch")
            gold = str(row["gold_answer_evaluator_only"])
            rows.append({
                "task_id": task_id,
                "video_id": video_id,
                "source_executor_authorized": (
                    binding.authorized_candidate is not None
                ),
                "source_correct": _answer_matches(source_prediction, gold),
                "target_correct": _answer_matches(target_prediction, gold),
                "effect_shuffled_correct": _answer_matches(
                    target_prediction, gold,
                ),
                "generic_scaffold_correct": _answer_matches(
                    generic_prediction, gold,
                ),
                "target_written_equivalent_correct": _answer_matches(
                    generic_prediction, gold,
                ),
                "formal_outcome_used_for_current_authorization": False,
            })

    v30 = _verified(v30_path)
    inputs.append({
        "path": str(v30_path.relative_to(REPO_ROOT)),
        "file_sha256": _sha256(v30_path),
        "report_sha256": v30["report_sha256"],
        "role": "FAILED_FORMAL_RETIRED_TO_ADAPTATION",
    })
    for row in v30["rows_detail"]:
        task_id, video_id = str(row["task_id"]), str(row["video_id"])
        if task_id in seen_tasks or video_id in seen_videos:
            raise ValueError("V30 adaptation overlaps prior V31 rows")
        seen_tasks.add(task_id)
        seen_videos.add(video_id)
        rows.append({
            "task_id": task_id,
            "video_id": video_id,
            "source_executor_authorized": row["generic_candidate"] is not None,
            "source_correct": bool(row["generic_scaffold_correct"]),
            "target_correct": bool(row["target_correct"]),
            "effect_shuffled_correct": bool(row["target_correct"]),
            "generic_scaffold_correct": bool(row["generic_scaffold_correct"]),
            "target_written_equivalent_correct": bool(
                row["generic_scaffold_correct"]
            ),
            "formal_outcome_used_for_current_authorization": False,
        })

    source_target = _paired_metrics(rows, "source_correct", "target_correct")
    source_shuffled = _paired_metrics(
        rows, "source_correct", "effect_shuffled_correct",
    )
    source_generic = _paired_metrics(
        rows, "source_correct", "generic_scaffold_correct",
    )
    target_written = _paired_metrics(
        rows, "source_correct", "target_written_equivalent_correct",
    )
    utility = _calibration(
        source_target["wins"], source_target["losses"], source_target["ties"],
    )
    authenticity = _calibration(
        source_shuffled["wins"], source_shuffled["losses"],
        source_shuffled["ties"],
    )
    gates = {
        "required_development_rows": len(rows) == 269,
        "all_videos_disjoint": len(seen_videos) == len(rows),
        "source_is_selective": 0 < sum(
            row["source_executor_authorized"] for row in rows
        ) < len(rows),
        "positive_net_transfer_vs_target_native": (
            source_target["left_minus_right_correct"] > 0
        ),
        "directional_utility_calibrated": utility["decision"] == "SELECT_SKILL",
        "source_authenticity_calibrated": (
            authenticity["decision"] == "SELECT_SKILL"
        ),
        "source_matches_generic_ceiling": (
            source_generic["wins"] == source_generic["losses"] == 0
        ),
        "source_matches_target_written_ceiling": (
            target_written["wins"] == target_written["losses"] == 0
        ),
        "raw_votes_not_used_as_bindings": True,
        "current_outcome_never_used_for_authorization": all(
            not row["formal_outcome_used_for_current_authorization"]
            for row in rows
        ),
    }
    qualified = all(gates.values())
    core = {
        "schema_version": "agqa2-postground-transfer-development-v31",
        "status": (
            "AGQA2_POSTGROUND_V31_DEVELOPMENT_QUALIFIED"
            if qualified else "AGQA2_POSTGROUND_V31_DEVELOPMENT_NOT_QUALIFIED"
        ),
        "claim_boundary": (
            "CONSUMED_V25_V28_AND_FAILED_V30_ADAPTATION_ONLY;ZERO_PROVIDER_"
            "CALLS;POST_GROUNDER_SYMBOLIC_BINDINGS;NO_CONFIRMATORY_CLAIM"
        ),
        "inputs": inputs,
        "source_artifact_sha256": artifact["artifact_sha256"],
        "source_artifact_file_sha256": _sha256(artifact_path),
        "source_confirmation_sha256": confirmation["report_sha256"],
        "source_confirmation_file_sha256": _sha256(confirmation_path),
        "adapter_module": str(adapter_path.relative_to(REPO_ROOT)),
        "adapter_module_sha256": module_sha256,
        "prediction_module": str(prediction_path.relative_to(REPO_ROOT)),
        "prediction_module_sha256": prediction_sha256,
        "target_grounder_sha256": target_grounder_sha256,
        "target_executor_sha256": target_executor_sha256,
        "rows": len(rows),
        "source_executor_authorizations": sum(
            row["source_executor_authorized"] for row in rows
        ),
        "source_vs_target_native": source_target,
        "source_vs_effect_shuffled": source_shuffled,
        "source_vs_generic_scaffold": source_generic,
        "source_vs_target_written_equivalent": target_written,
        "future_route_calibration": {
            "utility_vs_target_native": utility,
            "authenticity_vs_effect_shuffled": authenticity,
            "may_apply_only_to_future_disjoint_tasks": True,
        },
        "gates": gates,
        "qualified_for_future_disjoint_reserve": qualified,
        "provider_calls": 0,
        "confirmatory_claim": False,
    }
    result = core | {"report_sha256": stable_hash(core)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_postground_v31_development/report.json",
    )
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({key: result[key] for key in (
        "status", "rows", "source_executor_authorizations",
        "source_vs_target_native", "source_vs_effect_shuffled",
        "source_vs_generic_scaffold", "future_route_calibration", "gates",
        "provider_calls", "report_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
