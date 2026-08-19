#!/usr/bin/env python3
"""Freeze one operator-unfiltered official-test AGQA V61 formal run."""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402
from scripts.collect_agqa2_temporal_localized_query_v59 import _sha256  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v16_reserve import (  # noqa: E402
    _configured_video_ids,
)


MAX_ROWS_PER_VIDEO = 3
NONCE = "agqa2-v61-full-distribution-test-up-to-three-rows-per-fresh-video"
ARCHIVE = Path(
    "/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip"
)
ENTRY = "AGQA_balanced/test_balanced.txt"
VIDEO_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades"
)
SELECTION = REPO_ROOT / "configs/agqa2_full_distribution_v61_formal_selection.json"
MANIFEST = REPO_ROOT / "configs/agqa2_full_distribution_v61_formal_manifest.json"
PREREG = REPO_ROOT / (
    "configs/agqa2_full_distribution_v61_formal_preregistration.json"
)
CONFIG = REPO_ROOT / "configs/agqa2_full_distribution_v61_formal.json"
DOWNLOAD = REPO_ROOT / "runs/agqa2_full_distribution_v61_download/receipt.json"
QUALIFICATION_CONFIG = REPO_ROOT / (
    "configs/agqa2_temporal_localized_query_v60_qualification.json"
)
QUALIFICATION_REPORT = REPO_ROOT / (
    "runs/agqa2_temporal_localized_query_v60_qualification/report.json"
)
EVALUATOR = REPO_ROOT / "scripts/evaluate_agqa2_full_distribution_v61.py"


def _verified(path: Path, field: str) -> dict:
    payload = json.loads(path.read_text())
    body = dict(payload)
    claimed = body.pop(field)
    if stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return payload


def _new_selection() -> dict:
    excluded = _configured_video_ids() | {
        path.stem for path in VIDEO_ROOT.glob("*.mp4")
    } | {"YSKX3"}
    by_video = {}
    with zipfile.ZipFile(ARCHIVE) as bundle:
        with bundle.open(ENTRY, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if not video_id or video_id in excluded:
                    continue
                question = str(row.get("question", ""))
                program = str(row.get("program", ""))
                candidate = {
                    "priority": stable_hash(NONCE + "|" + task_id),
                    "task_id": task_id,
                    "video_id": video_id,
                    "video_path": str(VIDEO_ROOT / f"{video_id}.mp4"),
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                }
                rows = by_video.setdefault(video_id, [])
                rows.append(candidate)
    samples = []
    for video_id, rows in sorted(by_video.items()):
        # AGQA contains one official-test video with only two questions.  Keep
        # every fresh video in scope and freeze up to three rows rather than
        # silently dropping that video from the claimed distribution.
        chosen = sorted(rows, key=lambda row: row["priority"])[:MAX_ROWS_PER_VIDEO]
        if not chosen:
            raise ValueError(f"test video {video_id} has no questions")
        samples.extend(
            {key: value for key, value in row.items() if key != "priority"}
            for row in chosen
        )
    video_count = len(by_video)
    if video_count < 500:
        raise ValueError("V61 full-distribution pool has fewer than 500 fresh videos")
    body = {
        "schema_version": "agqa2-full-distribution-selection-v61",
        "status": "FROZEN_V61_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V61_CALLS",
        "split": "official_test_fresh_video_full_distribution",
        "selection_nonce": NONCE,
        "selection": (
            "ALL_FRESH_VIDEO_IDS;UP_TO_THREE_LOWEST_HASH_QUESTIONS_PER_VIDEO;"
            "NO_OPERATOR_QUESTION_FAMILY_PROGRAM_OR_TAXONOMY_FILTER;"
            "NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "maximum_rows_per_video": MAX_ROWS_PER_VIDEO,
        "sample_count": len(samples),
        "unique_video_count": video_count,
        "explicitly_excluded_accidentally_viewed_video_ids": ["YSKX3"],
        "samples": samples,
        "raw_video_archive": {
            "url": (
                "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/"
                "charades/Charades_v1_480.zip"
            ),
            "archive_prefix": "Charades_v1_480/",
            "content_length": 16339546533,
            "etag": "d37c91565b08ce1f432a46e11351751e-1948",
        },
        "answer_or_scene_graph_read_during_freeze": False,
        "operator_filter_used": False,
        "confirmatory_claim": True,
    }
    return body | {"manifest_sha256": stable_hash(body)}


def main() -> None:
    qualification = _verified(QUALIFICATION_REPORT, "report_sha256")
    if not qualification.get("grounder_qualified") or not all(
        qualification.get("qualification_gates", {}).values()
    ):
        raise ValueError("V61 is sealed until V60 passes every frozen gate")
    if SELECTION.is_file():
        selection = _verified(SELECTION, "manifest_sha256")
    else:
        selection = _new_selection()
        SELECTION.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    missing = sorted({
        row["video_id"] for row in selection["samples"]
        if not Path(row["video_path"]).is_file()
    })
    if missing:
        print(json.dumps({
            "status": selection["status"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "sample_count": selection["sample_count"],
            "unique_video_count": selection["unique_video_count"],
            "missing_video_count": len(missing),
            "next": (
                "python scripts/download_agqa2_active_grounding_v4_reserve.py "
                f"--selection {SELECTION.relative_to(REPO_ROOT)} --receipt "
                f"{DOWNLOAD.relative_to(REPO_ROOT)}"
            ),
        }, indent=2, sort_keys=True))
        return
    receipt = json.loads(DOWNLOAD.read_text())
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256")
        != selection["manifest_sha256"]
        or len(receipt.get("videos") or ())
        != selection["unique_video_count"]
    ):
        raise ValueError("V61 download receipt is incomplete")
    receipt_by_video = {
        str(row["video_id"]): row for row in receipt["videos"]
    }
    samples = [
        dict(row) | {"video_sha256": receipt_by_video[row["video_id"]]["sha256"]}
        for row in selection["samples"]
    ]
    manifest_body = {
        "schema_version": "agqa2-full-distribution-manifest-v61",
        "status": "FROZEN_V61_FULL_DISTRIBUTION_FORMAL_BEFORE_PROVIDER_CALLS",
        "split": "official_test_fresh_video_full_distribution",
        "selection_manifest_sha256": selection["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(DOWNLOAD),
        "sample_count": selection["sample_count"],
        "unique_video_count": selection["unique_video_count"],
        "maximum_rows_per_video": MAX_ROWS_PER_VIDEO,
        "samples": samples,
        "answer_or_scene_graph_read_during_freeze": False,
        "operator_filter_used": False,
        "confirmatory_claim": True,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    base = json.loads(QUALIFICATION_CONFIG.read_text())
    config = deepcopy(base)
    row_count = selection["sample_count"]
    video_count = selection["unique_video_count"]
    config.update({
        "schema_version": "agqa2-full-distribution-config-v61",
        "status": "FROZEN_V61_FULL_DISTRIBUTION_FORMAL",
        "split": "official_test_fresh_video_full_distribution",
        "claim_boundary": (
            f"{video_count}_NEW_OFFICIAL_TEST_VIDEOS_X_UP_TO_{MAX_ROWS_PER_VIDEO}_HASHED_"
            "QUESTIONS_WITHOUT_OPERATOR_FILTER;SELECTIVE_SOURCE_TRANSFER_WITH_"
            "IDENTICAL_DIRECT_FALLBACK;VIDEO_CLUSTERED_PRIMARY_INFERENCE"
        ),
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(MANIFEST),
        "expected_grounder_sha256": qualification["grounder_sha256"],
        "dataset": dict(base["dataset"]) | {"entry": ENTRY},
        # These gates permit base-report construction. Confirmatory gates are
        # frozen separately below and evaluated by the V61 evaluator.
        "qualification_gates": {
            "required_valid_rows": row_count,
            "required_unique_videos": video_count,
            "minimum_candidate_predictions": 1,
            "minimum_wins": 0,
            "maximum_losses": row_count,
            "minimum_net_gain": -row_count,
            "maximum_exact_one_sided_pvalue": 1.0,
        },
    })
    row_count_histogram = {}
    for sample_count in sorted({
        sum(row["video_id"] == video_id for row in selection["samples"])
        for video_id in {row["video_id"] for row in selection["samples"]}
    }):
        row_count_histogram[str(sample_count)] = sum(
            1 for video_id in {row["video_id"] for row in selection["samples"]}
            if sum(row["video_id"] == video_id for row in selection["samples"])
            == sample_count
        )
    formal_gates = {
        "required_rows": row_count,
        "required_unique_videos": video_count,
        "required_row_count_histogram": row_count_histogram,
        "minimum_applicable_rows": 250,
        "minimum_source_authorizations": 100,
        "minimum_row_wins": 10,
        "maximum_row_losses": 2,
        "minimum_row_net_gain": 8,
        "maximum_row_exact_pvalue": 0.05,
        "minimum_positive_video_clusters": 8,
        "maximum_negative_video_clusters": 2,
        "maximum_cluster_exact_pvalue": 0.05,
        "maximum_reported_provider_cost_usd": 8.0,
    }
    prereg = {
        "schema_version": "agqa2-full-distribution-prereg-v61",
        "status": "FROZEN_BEFORE_ANY_V61_PROVIDER_OR_OUTCOME_CALL",
        "claim_boundary": config["claim_boundary"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(DOWNLOAD),
        "qualification_report_sha256": qualification["report_sha256"],
        "qualification_grounder_sha256": qualification["grounder_sha256"],
        "evaluator_file_sha256": _sha256(EVALUATOR),
        "formal_gates": formal_gates,
        "selection_without_operator_filter": True,
        "primary_statistical_unit": "VIDEO_CLUSTER",
        "failure_policy": {
            "formal": "RUN_ONCE;NO_POST_OUTCOME_RULE_OR_THRESHOLD_CHANGE",
            "failed_gate": "REPORT_FULL_DISTRIBUTION_TRANSFER_NOT_VALIDATED",
            "passed": "REPORT_SELECTIVE_TRANSFER_ON_FULL_AGQA_DISTRIBUTION",
        },
        "answer_or_scene_graph_read_during_freeze": False,
        "confirmatory_claim": True,
    }
    PREREG.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": str(PREREG.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(PREREG),
        "full_distribution_evaluation": {
            "evaluator": str(EVALUATOR.relative_to(REPO_ROOT)),
            "evaluator_file_sha256": _sha256(EVALUATOR),
            "qualification_report": str(
                QUALIFICATION_REPORT.relative_to(REPO_ROOT)
            ),
            "qualification_report_file_sha256": _sha256(QUALIFICATION_REPORT),
            "formal_gates": formal_gates,
        },
    })
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"],
        "sample_count": row_count,
        "unique_video_count": video_count,
        "maximum_rows_per_video": MAX_ROWS_PER_VIDEO,
        "row_count_histogram": row_count_histogram,
        "manifest_sha256": manifest["manifest_sha256"],
        "grounder_sha256": qualification["grounder_sha256"],
        "formal_gates": formal_gates,
        "config_file_sha256": _sha256(CONFIG),
        "preregistration_file_sha256": _sha256(PREREG),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
