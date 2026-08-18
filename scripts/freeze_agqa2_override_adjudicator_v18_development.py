#!/usr/bin/env python3
"""Freeze independent focused adjudication of all V17 source overrides."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402


def _allowed(row: dict) -> list[str]:
    comparison = row["query_plan"]["comparison"]
    if comparison in {"EXISTS", "VERIFY_A_LONGER", "VERIFY_A_SHORTER"}:
        return ["yes", "no"]
    if comparison == "BEFORE_AFTER":
        return ["before", "after"]
    if comparison in {"SELECT_LONGER", "SELECT_SHORTER", "CHOOSE_OBJECT"}:
        return [row["query_plan"]["operand_a"], row["query_plan"]["operand_b"]]
    raise ValueError(f"V18 does not adjudicate {comparison}")


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_override_adjudicator_v18_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V18 adjudicator development is already consumed")
    report_path = REPO_ROOT / "runs/agqa2_active_grounding_v17_powered_reserve/report.json"
    report = json.loads(report_path.read_text())
    if report.get("grounder_qualified"):
        raise ValueError("V17 unexpectedly qualified")
    parent_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_active_grounding_v17_powered_reserve_manifest.json",
        "manifest_sha256",
    )
    frozen_by_task = {
        str(row["task_id"]): row for row in parent_manifest["samples"]
    }
    samples = []
    for row in report["rows"]:
        if row["calibrated_authorization_class"] != "SOURCE_TYPED_OVERRIDE":
            continue
        task_id = str(row["task_id"])
        frozen = frozen_by_task[task_id]
        samples.append({
            "task_id": task_id,
            "video_id": row["video_id"],
            "video_path": frozen["video_path"],
            "video_sha256": row["video_sha256"],
            "question_sha256": row["question_sha256"],
            "query_plan_sha256": row["query_plan"]["plan_sha256"],
            "grounding_receipt_sha256": row["grounding_receipt"]["receipt_sha256"],
            "typed_decision_sha256": stable_hash(
                row["calibrated_target_native_execution"]["decision"]
            ),
            "direct_response_sha256": stable_hash(row["direct_response"]),
            "allowed_decisions": _allowed(row),
            "selection_reason": "ALL_V17_SOURCE_TYPED_OVERRIDES",
        })
    samples.sort(key=lambda row: stable_hash(
        f"agqa-v18-focused-adjudication:{row['task_id']}"
    ))
    if len(samples) != report["metrics"]["source_typed_overrides"]:
        raise AssertionError("V18 did not select every V17 source override")
    core = {
        "schema_version": "agqa2-override-adjudicator-manifest-v18-development",
        "status": "FROZEN_BEFORE_ANY_V18_ADJUDICATOR_CALL",
        "split": "consumed_v17_development",
        "claim_boundary": (
            "ALL_AND_ONLY_V17_SOURCE_TYPED_OVERRIDES;INDEPENDENT_CLAUDE_"
            "FULL_TIMELINE_ADJUDICATION;NO_TYPED_OR_DIRECT_CANDIDATE_VISIBLE_"
            "TO_MODEL;DEVELOPMENT_ONLY"
        ),
        "parent_v17_report": str(report_path.relative_to(REPO_ROOT)),
        "parent_v17_report_file_sha256": _sha256(report_path),
        "parent_v17_report_sha256": report["report_sha256"],
        "parent_v17_manifest_sha256": parent_manifest["manifest_sha256"],
        "archive_path": parent_manifest["archive_path"],
        "archive_sha256": parent_manifest["archive_sha256"],
        "entry": parent_manifest["entry"],
        "samples": samples,
        "sample_count": len(samples),
        "selection_reads_gold_or_correctness": False,
        "selection_reads_only_frozen_authorization_class": True,
        "adjudicator_visible_fields": [
            "public_question", "chronological_frames", "frame_timestamps",
        ],
        "adjudicator_forbidden_fields": [
            "typed_decision", "direct_response", "gold_answer",
            "functional_program", "scene_graph", "source_identity",
        ],
    }
    manifest = core | {"manifest_sha256": stable_hash(core)}
    manifest_path = REPO_ROOT / "configs/agqa2_override_adjudicator_v18_development_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    config = {
        "schema_version": "agqa2-override-adjudicator-config-v18-development",
        "status": "FROZEN_V18_FOCUSED_ADJUDICATOR_DEVELOPMENT",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "parent_v17_report": str(report_path.relative_to(REPO_ROOT)),
        "parent_v17_report_file_sha256": _sha256(report_path),
        "model": {
            "provider": "openrouter",
            "api_key_name": "OPENROUTER_API_KEY",
            "id": "anthropic/claude-sonnet-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "timeout_seconds": 300,
            "max_retries": 2,
            "schema_retries": 3,
            "max_tokens": 700,
            "temperature": 0,
        },
        "media": {
            "frame_count": 48,
            "max_side": 512,
            "frames_per_panel": 6,
            "panel_frame_width": 192,
            "jpeg_quality": 80,
        },
        "authorization": {
            "minimum_confidence": 0.8,
            "requires_exact_typed_decision_match": True,
            "requires_cited_visual_evidence": True,
            "unknown_abstains": True,
        },
        "qualification_gates": {
            "required_rows": len(samples),
            "minimum_adjudicator_correct": 5,
            "minimum_retained_typed_vs_direct_wins": 3,
            "maximum_authorized_typed_vs_direct_losses": 0,
            "minimum_final_vs_direct_delta": 3,
            "maximum_reported_provider_cost_usd": 0.20,
        },
        "failure_policy": {
            "development": "DO_NOT_INTEGRATE_OR_FREEZE_FRESH_IF_ANY_GATE_FAILS",
            "fresh": "REQUIRES_NEW_VIDEO_DISJOINT_RESERVE_AFTER_INTEGRATION",
        },
        "module": "src/motif_transfer/agqa_override_adjudicator.py",
        "module_sha256": _sha256(
            REPO_ROOT / "src/motif_transfer/agqa_override_adjudicator.py"
        ),
        "collector": "scripts/collect_agqa2_override_adjudicator_v18.py",
        "collector_sha256": _sha256(
            REPO_ROOT / "scripts/collect_agqa2_override_adjudicator_v18.py"
        ),
    }
    config_path = REPO_ROOT / "configs/agqa2_override_adjudicator_v18_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": len(samples),
        "task_ids": [row["task_id"] for row in samples],
        "config_file_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
