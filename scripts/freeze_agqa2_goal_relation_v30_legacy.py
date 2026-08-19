#!/usr/bin/env python3
"""Freeze V30 before reading outcomes of 119 completed V27 receipts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.evaluate_agqa2_goal_relation_v30_legacy import (  # noqa: E402
    evaluation_protocol_core,
)


MISSING_TASK = "QMIKJ-29239"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, hash_field: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = str(body.pop(hash_field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return value


def main() -> None:
    output = REPO_ROOT / "runs/agqa2_goal_relation_v30_legacy/report.json"
    if output.is_file():
        raise RuntimeError("V30 formal outcomes are already consumed")
    development_path = (
        REPO_ROOT / "runs/agqa2_goal_relation_v29_development/report.json"
    )
    development = _verified(development_path, "report_sha256")
    if not development.get("qualified_for_future_disjoint_reserve"):
        raise ValueError("V29 development did not qualify the future route")
    calibration = development["future_route_calibration"]
    if not all(
        calibration[label]["decision"] == "SELECT_SKILL"
        for label in (
            "utility_vs_target_native", "authenticity_vs_effect_shuffled",
        )
    ):
        raise ValueError("V29 development calibration does not authorize transfer")

    abort_path = REPO_ROOT / "docs/results/agqa2_query_object_v27_runtime_abort.json"
    abort = _verified(abort_path, "abort_sha256")
    if (
        abort["status"] != "V27_RUNTIME_INCOMPLETE_BEFORE_FORMAL_EVALUATION"
        or abort["completed_runtime_receipts"] != 119
        or abort["required_runtime_receipts"] != 120
        or abort["formal_gold_evaluation_started"] is not False
        or abort["formal_report_created"] is not False
        or abort["official_answers_inspected_for_repair"] is not False
        or abort["terminal_worker_errors"] != {
            MISSING_TASK: "ValueError: provider response omitted a JSON object"
        }
    ):
        raise ValueError("V27 is not an outcome-unread transport-only abort")

    parent_manifest_path = (
        REPO_ROOT / "configs/agqa2_query_object_v27_replay_manifest.json"
    )
    parent = _verified(parent_manifest_path, "manifest_sha256")
    runtime_root = REPO_ROOT / "runs/agqa2_query_object_v27_replay/runtime_receipts"
    runtime_paths = {path.stem: path for path in runtime_root.glob("*.json")}
    if set(runtime_paths) != {
        str(row["task_id"]) for row in parent["samples"]
        if str(row["task_id"]) != MISSING_TASK
    }:
        raise ValueError("V27 completed runtime receipt set drifted")

    development_videos = set()
    for report_path in (
        REPO_ROOT / "runs/agqa2_query_object_v25_reserve/report.json",
        REPO_ROOT / "runs/agqa2_query_object_v28_reserve/report.json",
    ):
        report = _verified(report_path, "report_sha256")
        development_videos.update(str(row["video_id"]) for row in report["rows"])
    samples = []
    for row in parent["samples"]:
        task_id = str(row["task_id"])
        if task_id == MISSING_TASK:
            continue
        if str(row["video_id"]) in development_videos:
            raise ValueError("V30 legacy formal overlaps V29 development video")
        receipt_path = runtime_paths[task_id]
        receipt = json.loads(receipt_path.read_text())
        body = dict(receipt)
        claimed = str(body.pop("runtime_receipt_sha256", ""))
        if not claimed or stable_hash(body) != claimed:
            raise ValueError(f"runtime receipt hash mismatch: {task_id}")
        if (
            receipt["task_id"] != task_id
            or receipt["video_id"] != row["video_id"]
            or receipt["question_sha256"] != row["question_sha256"]
            or receipt["video_sha256"] != row["video_sha256"]
        ):
            raise ValueError(f"runtime receipt/manifest mismatch: {task_id}")
        samples.append(deepcopy(row) | {
            "runtime_receipt_path": str(receipt_path.relative_to(REPO_ROOT)),
            "runtime_receipt_file_sha256": _sha256(receipt_path),
            "runtime_receipt_sha256": claimed,
        })
    if len(samples) != 119 or len({row["video_id"] for row in samples}) != 119:
        raise ValueError("V30 requires 119 unique completed videos")
    manifest_core = {
        "schema_version": "agqa2-goal-relation-legacy-manifest-v30",
        "status": "FROZEN_V30_OUTCOME_UNREAD_RUNTIME_RECEIPTS",
        "split": "formal",
        "claim_boundary": (
            "119_OF_120_V27_RUNTIME_RECEIPTS_FROZEN_BEFORE_ANY_FORMAL_GOLD_"
            "READ;ONE_TRANSPORT_MISSING_ROW_EXCLUDED;ZERO_NEW_PROVIDER_CALLS;"
            "VIDEO_DISJOINT_FROM_V29_DEVELOPMENT"
        ),
        "archive_path": parent["archive_path"],
        "archive_sha256": parent["archive_sha256"],
        "entry": parent["entry"],
        "video_root": parent["video_root"],
        "samples": sorted(samples, key=lambda row: row["task_id"]),
        "sample_count": 119,
        "unique_video_count": 119,
        "parent_v27_manifest": str(parent_manifest_path.relative_to(REPO_ROOT)),
        "parent_v27_manifest_file_sha256": _sha256(parent_manifest_path),
        "parent_v27_manifest_sha256": parent["manifest_sha256"],
        "parent_v27_abort": str(abort_path.relative_to(REPO_ROOT)),
        "parent_v27_abort_file_sha256": _sha256(abort_path),
        "parent_v27_abort_sha256": abort["abort_sha256"],
        "missing_task_id": MISSING_TASK,
        "missingness_cause": "PROVIDER_JSON_TRANSPORT_BEFORE_GOLD_READ",
        "answer_read_during_freeze": False,
        "scene_graph_read_during_freeze": False,
        "current_outcome_read_during_freeze": False,
        "development_video_overlap": 0,
    }
    manifest = manifest_core | {"manifest_sha256": stable_hash(manifest_core)}
    manifest_path = (
        REPO_ROOT / "configs/agqa2_goal_relation_v30_legacy_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    artifact_path = REPO_ROOT / "runs/sokoban_goal_relation_macro_v3/artifact.json"
    confirmation_path = (
        REPO_ROOT
        / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
    )
    inducer_path = REPO_ROOT / "src/motif_transfer/source_goal_relation_induction.py"
    adapter_path = REPO_ROOT / "src/motif_transfer/agqa_goal_relation_transfer.py"
    prediction_path = (
        REPO_ROOT / "src/motif_transfer/agqa_goal_relation_evaluation.py"
    )
    evaluator_path = (
        REPO_ROOT / "scripts/evaluate_agqa2_goal_relation_v30_legacy.py"
    )
    config = {
        "schema_version": "agqa2-goal-relation-legacy-config-v30",
        "status": "FROZEN_V30_LEGACY_FORMAL_EVALUATION",
        "split": "formal",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "source": {
            "artifact": str(artifact_path.relative_to(REPO_ROOT)),
            "artifact_file_sha256": _sha256(artifact_path),
            "confirmation": str(confirmation_path.relative_to(REPO_ROOT)),
            "confirmation_file_sha256": _sha256(confirmation_path),
            "inducer": str(inducer_path.relative_to(REPO_ROOT)),
            "inducer_file_sha256": _sha256(inducer_path),
            "inducer_artifact_sha256": _sha256(inducer_path),
        },
        "development": {
            "report": str(development_path.relative_to(REPO_ROOT)),
            "report_file_sha256": _sha256(development_path),
            "report_sha256": development["report_sha256"],
            "confirmatory": False,
        },
        "adapter": {
            "module": str(adapter_path.relative_to(REPO_ROOT)),
            "module_file_sha256": _sha256(adapter_path),
            "prediction_module": str(prediction_path.relative_to(REPO_ROOT)),
            "prediction_module_file_sha256": _sha256(prediction_path),
            "evaluator": str(evaluator_path.relative_to(REPO_ROOT)),
            "evaluator_file_sha256": _sha256(evaluator_path),
            "target_grounder_sha256": development["target_grounder_sha256"],
            "target_executor_sha256": development["target_executor_sha256"],
            "minimum_ontology_confidences": [0.8, 0.8],
        },
        "calibration": {
            label: {
                key: calibration[label][key] for key in (
                    "wins", "losses", "ties",
                )
            } for label in (
                "utility_vs_target_native", "authenticity_vs_effect_shuffled",
            )
        },
        "controls": {
            "neural_only": "TWO_ONTOLOGY_AGREEMENT_ELSE_MATCHED_DIRECT",
            "source_induced": (
                "UNIQUE_BINDING_SOURCE_PROGRAM_ELSE_NEURAL_ONLY"
            ),
            "source_effect_shuffled": (
                "SOURCE_HELDOUT_SHUFFLED_EFFECT_FAIL_CLOSED_ELSE_NEURAL_ONLY"
            ),
            "generic_scaffold": (
                "TWO_OF_THREE_NEURAL_MAJORITY_ELSE_NEURAL_ONLY"
            ),
            "target_written_equivalent": (
                "EXTENSIONALLY_IDENTICAL_CEILING_EXPECTED_TO_MATCH_SOURCE"
            ),
        },
        "qualification_gates": {
            "required_valid_rows": 119,
            "minimum_source_authorizations": 30,
            "minimum_source_vs_target_wins": 5,
            "maximum_source_vs_target_losses": 1,
            "minimum_source_minus_target_correct": 5,
            "maximum_exact_one_sided_pvalue": 0.05,
        },
        "failure_policy": (
            "EVALUATE_EXACT_119_ROWS_ONCE;REPORT_ALL_CONTROLS;NO_THRESHOLD_"
            "CHANGES;NO_CLAIM_IF_ANY_GATE_FAILS"
        ),
        "preregistration": (
            "configs/agqa2_goal_relation_v30_legacy_preregistration.json"
        ),
    }
    protocol_sha256 = stable_hash(evaluation_protocol_core(config))
    prereg_core = {
        "schema_version": "agqa2-goal-relation-preregistration-v30",
        "status": "FROZEN_BEFORE_V30_FORMAL_GOLD_READ",
        "claim_boundary": manifest["claim_boundary"],
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_protocol_sha256": protocol_sha256,
        "development_report_sha256": development["report_sha256"],
        "calibration": deepcopy(config["calibration"]),
        "controls": deepcopy(config["controls"]),
        "qualification_gates": deepcopy(config["qualification_gates"]),
        "failure_policy": config["failure_policy"],
        "formal_gold_read_before_freeze": False,
        "provider_calls_authorized": 0,
        "confirmatory_endpoint": (
            "SOURCE_INDUCED_VS_TARGET_NATIVE_PAIRED_CORRECTNESS"
        ),
        "multiplicity_policy": "ONE_PRIMARY_ENDPOINT;CONTROLS_ALWAYS_REPORTED",
    }
    prereg = prereg_core | {
        "preregistration_sha256": stable_hash(prereg_core)
    }
    prereg_path = REPO_ROOT / config["preregistration"]
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_evaluation_protocol_sha256": protocol_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_goal_relation_v30_legacy.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "sample_count": manifest["sample_count"],
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_protocol_sha256": protocol_sha256,
        "development_report_sha256": development["report_sha256"],
        "formal_gold_read_before_freeze": False,
        "provider_calls_authorized": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
