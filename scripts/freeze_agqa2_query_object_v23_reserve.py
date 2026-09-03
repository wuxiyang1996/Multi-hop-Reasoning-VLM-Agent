#!/usr/bin/env python3
"""Freeze one fresh 30-video confirmation of the qualified V22 grounder."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.agqa_program_transfer import RELATION_ROUTE, profile_program  # noqa: E402
from motif_transfer.agqa_query_object_grounder import atomic_query_object_plan  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object, _load_sources,
)
from scripts.collect_agqa2_query_object_v20 import _evaluation_core, _semantic_core  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v16_reserve import (  # noqa: E402
    _configured_video_ids,
)
from scripts.freeze_agqa2_query_object_v20_development import _relation_group  # noqa: E402


NONCE = "agqa2-query-object-v23-final-30-video-disjoint-confirmation"
PER_GROUP = 10


def _query_object_program_answer_space_matches(program: str) -> bool:
    """AGQA object-valued public queries have a Query functional root."""

    return program.strip().split("(", 1)[0] == "Query"


def _development_summary() -> tuple[Path, dict]:
    report_path = REPO_ROOT / "runs/agqa2_query_object_v22_development/report.json"
    report = json.loads(report_path.read_text())
    body = dict(report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed:
        raise ValueError("V22 development report hash mismatch")
    if not report.get("grounder_qualified"):
        raise ValueError("V22 QUERY_OBJECT development did not qualify")
    fields = (
        "status", "grounder_qualified", "grounder_sha256",
        "evaluation_protocol_sha256", "metrics", "controls",
        "qualification_gates", "reported_provider_cost_usd", "report_sha256",
    )
    core = {key: deepcopy(report[key]) for key in fields}
    core.update({
        "schema_version": "agqa2-query-object-v22-development-summary",
        "development_report_file_sha256": _sha256(report_path),
        "claim_scope": "ATOMIC_QUERY_OBJECT_ONLY",
        "confirmatory": False,
    })
    summary = core | {"summary_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_query_object_v22_development_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path, summary


def _select(development_manifest: dict, excluded: set[str]) -> dict:
    candidates: dict[str, list[dict]] = defaultdict(list)
    archive_path = Path(development_manifest["archive_path"])
    video_root = Path(development_manifest["video_root"])
    with zipfile.ZipFile(archive_path) as bundle, bundle.open(
        "AGQA_balanced/test_balanced.txt", "r"
    ) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if (
                    not video_id or video_id in excluded
                    or (video_root / f"{video_id}.mp4").is_file()
                ):
                    continue
                question = str(row.get("question", ""))
                plan = parse_public_question_plan(question)
                if plan is None or not atomic_query_object_plan(plan):
                    continue
                program = str(row.get("program", ""))
                route = profile_program(task_id=task_id, program=program).route_kind
                if (
                    route != RELATION_ROUTE
                    or not _query_object_program_answer_space_matches(program)
                ):
                    continue
                group = _relation_group(plan.operand_a)
                candidates[group].append({
                    "task_id": task_id,
                    "video_id": video_id,
                    "video_path": str(video_root / f"{video_id}.mp4"),
                    "video_present_at_selection": False,
                    "oracle_route": RELATION_ROUTE,
                    "comparison": "QUERY_OBJECT",
                    "relation_group": group,
                    "public_relation": plan.operand_a,
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                    "public_parser_plan_sha256": plan.plan_sha256,
                    "applicability_rule": (
                        "ATOMIC_QUERY_OBJECT_PUBLIC_GRAMMAR_PLUS_PROGRAM_"
                        "ANSWER_SPACE_COMPATIBILITY"
                    ),
                    "rank_sha256": stable_hash(f"{NONCE}:{group}:{task_id}"),
                })
    selected, used = [], set()
    groups = ("MANIPULATION_CONTACT", "SPATIAL_SUPPORT", "PERCEPTION")
    for group in groups:
        for row in sorted(candidates[group], key=lambda item: item["rank_sha256"]):
            if row["video_id"] in used:
                continue
            selected.append(row)
            used.add(row["video_id"])
            if sum(item["relation_group"] == group for item in selected) == PER_GROUP:
                break
    counts = {
        group: sum(row["relation_group"] == group for row in selected)
        for group in groups
    }
    if any(value != PER_GROUP for value in counts.values()):
        raise RuntimeError(f"insufficient fresh QUERY_OBJECT candidates: {counts}")
    core = {
        "schema_version": "agqa2-query-object-reserve-selection-v23",
        "status": "FROZEN_V23_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V23_CALLS",
        "split": "reserve",
        "claim_boundary": (
            "UNCHANGED_V22_QUERY_OBJECT_GROUNDER;30_NEW_CROSS_EXPERIMENT_"
            "VIDEO_DISJOINT_TEST_ROWS;ONE_OUTCOME_BLIND_CONFIRMATION;"
            "NOT_UNTOUCHED_METADATA"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_VIDEO_IDS_IN_PRIOR_CONFIGS_AND_ALL_MP4S_PRESENT_IN_"
            "SHARED_CHARADES_ROOT;ATOMIC_QUERY_OBJECT_ONLY;PROGRAM_ANSWER_SPACE_"
            "COMPATIBILITY;HASH_RANK_WITHIN_THREE_RELATION_GROUPS;ONE_TASK_PER_"
            "VIDEO;NO_ANSWER_SCENE_GRAPH_OR_DIRECT_RESPONSE_READ"
        ),
        "archive_path": str(archive_path),
        "archive_sha256": development_manifest["archive_sha256"],
        "entry": "AGQA_balanced/test_balanced.txt",
        "video_root": str(video_root),
        "samples": sorted(selected, key=lambda row: row["task_id"]),
        "sample_count": len(selected),
        "unique_video_count": len(used),
        "relation_group_counts": counts,
        "excluded_prior_config_or_present_video_count": len(excluded),
        "excluded_video_ids_sha256": stable_hash(sorted(excluded)),
        "answer_read_during_freeze": False,
        "scene_graph_read_during_freeze": False,
        "direct_response_read_during_freeze": False,
        "functional_program_root_read_for_answer_space_only": True,
        "prior_v23_neural_grounder_exposure": False,
        "raw_video_archive": {
            "url": (
                "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/"
                "charades/Charades_v1_480.zip"
            ),
            "archive_prefix": "Charades_v1_480/",
        },
    }
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: dict) -> dict:
    samples = []
    for row in selection["samples"]:
        path = Path(row["video_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        samples.append(dict(row) | {
            "video_sha256": _sha256(path),
            "video_bytes": path.stat().st_size,
        })
    core = {
        key: deepcopy(value) for key, value in selection.items()
        if key not in {"manifest_sha256", "raw_video_archive"}
    }
    core.update({
        "schema_version": "agqa2-query-object-reserve-manifest-v23",
        "status": "FROZEN_V23_RAW_VIDEO_UNSEEN_BY_NEURAL_GROUNDER_BEFORE_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": len(samples),
        "local_integrity_decode_probe_completed": True,
        "prior_neural_grounder_or_model_video_exposure": False,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v23_reserve"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V23 QUERY_OBJECT reserve is already consumed")
    summary_path, development = _development_summary()
    development_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_query_object_v22_development_manifest.json",
        "manifest_sha256",
    )
    excluded = _configured_video_ids()
    video_root = Path(development_manifest["video_root"])
    excluded.update(path.stem for path in video_root.glob("*.mp4"))
    selection_path = REPO_ROOT / "configs/agqa2_query_object_v23_reserve_selection.json"
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path.is_file() else _select(development_manifest, excluded)
    )
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    missing = [
        row["video_id"] for row in selection["samples"]
        if not Path(row["video_path"]).is_file()
    ]
    if missing:
        print(json.dumps({
            "status": selection["status"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "sample_count": selection["sample_count"],
            "missing_video_ids": missing,
            "next": "download exact frozen videos and rerun",
        }, indent=2))
        return

    receipt_path = REPO_ROOT / "runs/agqa2_query_object_v23_download/receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("V23 download receipt is missing")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256") != selection["manifest_sha256"]
    ):
        raise ValueError("V23 download receipt is incomplete or belongs to another pool")
    manifest = _seal(selection)
    manifest_path = REPO_ROOT / "configs/agqa2_query_object_v23_reserve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v22_development.json"
    ).read_text())
    config.update({
        "schema_version": "agqa2-query-object-reserve-config-v23",
        "status": "FROZEN_V23_QUERY_OBJECT_CONFIRMATION",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V23_RESERVE_CALL",
        "development_qualification_report": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V22_QUERY_OBJECT",
    })
    config["qualification_gates"] = {
        "required_valid_runtime_rows": 30,
        "minimum_route_correct": 30,
        "minimum_decisive_executions": 15,
        "minimum_decisive_accuracy": 0.75,
        "maximum_typed_vs_direct_losses": 0,
        "minimum_typed_vs_direct_wins": 2,
        "required_source_permuted_abstentions": 30,
        "required_target_written_equivalent_matches": 30,
        "maximum_reported_provider_cost_usd": 0.35,
    }
    config["preregistration"] = "configs/agqa2_query_object_v23_reserve_preregistration.json"
    for key in (
        "preregistration_file_sha256", "expected_grounder_sha256",
        "expected_evaluation_protocol_sha256",
    ):
        config.pop(key, None)
    sources, _ = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    evaluation_sha256 = stable_hash(_evaluation_core(config))
    if grounder_sha256 != development["grounder_sha256"]:
        raise AssertionError("V23 reserve changed the qualified V22 grounder")
    prereg = {
        "schema_version": "agqa2-query-object-reserve-preregistration-v23",
        "status": "FROZEN_BEFORE_ANY_V23_RESERVE_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "qualified_grounder_sha256": grounder_sha256,
        "reserve_evaluation_protocol_sha256": evaluation_sha256,
        "development_qualification_summary": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(summary_path),
        "selection_manifest_sha256": selection["manifest_sha256"],
        "sealed_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "reserve_gates": deepcopy(config["qualification_gates"]),
        "grounder_changed_after_development": False,
        "failure_policy": (
            "RUN_ONCE;FAIL_CLOSED_TO_MATCHED_DIRECT;NO_POST_RESERVE_TUNING_OR_"
            "ADDITIONAL_FRESH_SEED"
        ),
    }
    prereg_path = REPO_ROOT / config["preregistration"]
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": grounder_sha256,
        "expected_evaluation_protocol_sha256": evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_query_object_v23_reserve.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "sample_count": manifest["sample_count"],
        "gates": config["qualification_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
