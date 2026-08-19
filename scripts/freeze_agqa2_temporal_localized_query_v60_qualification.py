#!/usr/bin/env python3
"""Freeze a fresh-train qualification for the V59 composite grounder."""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_program_transfer import profile_program  # noqa: E402
from motif_transfer.agqa_temporal_localized_query import (  # noqa: E402
    parse_temporal_localized_object_question,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402
from scripts.collect_agqa2_temporal_localized_query_v59 import _sha256  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v16_reserve import (  # noqa: E402
    _configured_video_ids,
)


N = 240
NONCE = "agqa2-v60-temporal-localized-fresh-train-qualification-240"
ARCHIVE = Path(
    "/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip"
)
ENTRY = "AGQA_balanced/train_balanced.txt"
VIDEO_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades"
)
SIGNATURE = "Query>OnlyItem>Iterate>Localize>Filter"
SELECTION = REPO_ROOT / (
    "configs/agqa2_temporal_localized_query_v60_qualification_selection.json"
)
MANIFEST = REPO_ROOT / (
    "configs/agqa2_temporal_localized_query_v60_qualification_manifest.json"
)
PREREG = REPO_ROOT / (
    "configs/agqa2_temporal_localized_query_v60_qualification_preregistration.json"
)
CONFIG = REPO_ROOT / (
    "configs/agqa2_temporal_localized_query_v60_qualification.json"
)
DOWNLOAD = REPO_ROOT / (
    "runs/agqa2_temporal_localized_query_v60_download/receipt.json"
)
DEVELOPMENT_CONFIG = REPO_ROOT / (
    "configs/agqa2_temporal_localized_query_v59_development.json"
)
DEVELOPMENT_REPORT = REPO_ROOT / (
    "runs/agqa2_temporal_localized_query_v59_development/report.json"
)


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
    }
    best_by_video = {}
    with zipfile.ZipFile(ARCHIVE) as bundle:
        with bundle.open(ENTRY, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if not video_id or video_id in excluded:
                    continue
                program = str(row.get("program", ""))
                profile = profile_program(task_id=task_id, program=program)
                if (
                    ">".join(profile.functions) != SIGNATURE
                    or "[relations," not in program
                ):
                    continue
                question = str(row.get("question", ""))
                plan = parse_temporal_localized_object_question(question)
                if plan is None:
                    continue
                candidate = {
                    "priority": stable_hash(NONCE + "|" + task_id),
                    "task_id": task_id,
                    "video_id": video_id,
                    "video_path": str(VIDEO_ROOT / f"{video_id}.mp4"),
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                    "temporal_operator": plan.temporal_operator,
                    "relation": plan.relation,
                }
                prior = best_by_video.get(video_id)
                if prior is None or candidate["priority"] < prior["priority"]:
                    best_by_video[video_id] = candidate
    selected = sorted(best_by_video.values(), key=lambda row: row["priority"])[:N]
    if len(selected) != N:
        raise ValueError("not enough fresh train videos for V60")
    samples = [
        {key: value for key, value in row.items() if key != "priority"}
        for row in selected
    ]
    body = {
        "schema_version": "agqa2-temporal-localized-query-selection-v60",
        "status": "FROZEN_V60_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V60_CALLS",
        "split": "official_train_fresh_qualification",
        "selection_nonce": NONCE,
        "selection": (
            "HASH_ORDERED_EXACT_TEMPORAL_LOCALIZED_RELATION_ROWS;"
            "ONE_ROW_PER_PREVIOUSLY_UNCONFIGURED_AND_NONLOCAL_VIDEO;"
            "NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "sample_count": len(samples),
        "unique_video_count": len({row["video_id"] for row in samples}),
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
        "confirmatory_claim": False,
    }
    return body | {"manifest_sha256": stable_hash(body)}


def main() -> None:
    if SELECTION.is_file():
        selection = _verified(SELECTION, "manifest_sha256")
    else:
        selection = _new_selection()
        SELECTION.parent.mkdir(parents=True, exist_ok=True)
        SELECTION.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    missing = [
        row["video_id"] for row in selection["samples"]
        if not Path(row["video_path"]).is_file()
    ]
    if missing:
        print(json.dumps({
            "status": selection["status"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "sample_count": len(selection["samples"]),
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
        or len(receipt.get("videos") or ()) != N
    ):
        raise ValueError("V60 download receipt is incomplete")
    receipt_by_video = {
        str(row["video_id"]): row for row in receipt["videos"]
    }
    samples = [
        dict(row) | {"video_sha256": receipt_by_video[row["video_id"]]["sha256"]}
        for row in selection["samples"]
    ]
    manifest_body = {
        "schema_version": "agqa2-temporal-localized-query-manifest-v60",
        "status": "FROZEN_V60_FRESH_TRAIN_QUALIFICATION_BEFORE_PROVIDER_CALLS",
        "split": "official_train_fresh_qualification",
        "selection_manifest_sha256": selection["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(DOWNLOAD),
        "sample_count": N,
        "unique_video_count": N,
        "samples": samples,
        "answer_or_scene_graph_read_during_freeze": False,
        "confirmatory_claim": False,
    }
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    development = _verified(DEVELOPMENT_REPORT, "report_sha256")
    if development["source_vs_direct"]["net_gain"] < 1:
        raise ValueError("V60 requires positive consumed-development net gain")
    base = json.loads(DEVELOPMENT_CONFIG.read_text())
    config = deepcopy(base)
    config.update({
        "schema_version": "agqa2-temporal-localized-query-config-v60",
        "status": "FROZEN_V60_FRESH_TRAIN_QUALIFICATION",
        "split": "official_train_fresh_qualification",
        "claim_boundary": (
            "240_NEW_CROSS_EXPERIMENT_VIDEO_DISJOINT_EXACT_TEMPORAL_"
            "LOCALIZED_RELATION_ROWS_FROM_OFFICIAL_TRAIN;QUALIFICATION_ONLY;"
            "NO_FULL_AGQA_OR_TEST_CLAIM"
        ),
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(MANIFEST),
        "expected_grounder_sha256": development["grounder_sha256"],
        "qualification_gates": {
            "required_valid_rows": N,
            "required_unique_videos": N,
            "minimum_candidate_predictions": 40,
            "minimum_wins": 5,
            "maximum_losses": 0,
            "minimum_net_gain": 5,
            "maximum_exact_one_sided_pvalue": 0.05,
        },
    })
    protocol = {
        "schema_version": "agqa2-temporal-localized-query-prereg-v60",
        "status": "FROZEN_BEFORE_ANY_V60_PROVIDER_OR_OUTCOME_CALL",
        "selection_manifest_sha256": selection["manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(DOWNLOAD),
        "development_report_sha256": development["report_sha256"],
        "grounder_sha256": development["grounder_sha256"],
        "qualification_gates": config["qualification_gates"],
        "cost_cap_usd": 4.0,
        "failure_policy": {
            "failed_gate": "STOP_BEFORE_FULL_DISTRIBUTION_TEST_FORMAL",
            "passed": "FREEZE_ONE_FULL_DISTRIBUTION_TEST_FORMAL",
            "post_outcome_rule_change": "FORBIDDEN",
        },
        "answer_or_scene_graph_read_during_freeze": False,
        "confirmatory_claim": False,
    }
    PREREG.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    config["preregistration"] = str(PREREG.relative_to(REPO_ROOT))
    config["preregistration_file_sha256"] = _sha256(PREREG)
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"],
        "sample_count": N,
        "manifest_sha256": manifest["manifest_sha256"],
        "grounder_sha256": development["grounder_sha256"],
        "config_file_sha256": _sha256(CONFIG),
        "preregistration_file_sha256": _sha256(PREREG),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
