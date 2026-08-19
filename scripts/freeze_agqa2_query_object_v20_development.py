#!/usr/bin/env python3
"""Freeze an isolated AGQA train-development QUERY_OBJECT qualification."""

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
from scripts.collect_agqa2_query_object_v20 import (  # noqa: E402
    _evaluation_core, _semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v4 import _sha256  # noqa: E402


NONCE = "agqa2-query-object-v20-train-development-18-video-disjoint"
ARCHIVE = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip")
ENTRY = "AGQA_balanced/train_balanced.txt"
VIDEO_ROOT = Path("/fs/gamma-projects/vlm-robot/datasets/STAR-official/videos/charades")
PER_GROUP = 6


def _relation_group(relation: str) -> str:
    text = relation.casefold()
    if text.startswith("watching"):
        return "PERCEPTION"
    if any(marker in text for marker in (
        "sitting on", "standing on", "lying on", "leaning on", "above",
        "beneath", "behind", "in front of", "on the side of", "covered by",
    )) or text == "in":
        return "SPATIAL_SUPPORT"
    return "MANIPULATION_CONTACT"


def _select() -> dict:
    candidates: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(ARCHIVE) as bundle, bundle.open(ENTRY, "r") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                video_path = VIDEO_ROOT / f"{video_id}.mp4"
                if not video_id or not video_path.is_file():
                    continue
                question = str(row.get("question", ""))
                plan = parse_public_question_plan(question)
                if plan is None or not atomic_query_object_plan(plan):
                    continue
                program = str(row.get("program", ""))
                if profile_program(task_id=task_id, program=program).route_kind != RELATION_ROUTE:
                    continue
                group = _relation_group(plan.operand_a)
                candidates[group].append({
                    "task_id": task_id,
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "oracle_route": RELATION_ROUTE,
                    "comparison": "QUERY_OBJECT",
                    "relation_group": group,
                    "public_relation": plan.operand_a,
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                    "public_parser_plan_sha256": plan.plan_sha256,
                    "rank_sha256": stable_hash(f"{NONCE}:{group}:{task_id}"),
                })
    selected = []
    used_videos: set[str] = set()
    for group in ("MANIPULATION_CONTACT", "SPATIAL_SUPPORT", "PERCEPTION"):
        ranked = sorted(candidates[group], key=lambda row: row["rank_sha256"])
        for row in ranked:
            if row["video_id"] in used_videos:
                continue
            selected.append(row)
            used_videos.add(row["video_id"])
            if sum(item["relation_group"] == group for item in selected) == PER_GROUP:
                break
    counts = {
        group: sum(row["relation_group"] == group for row in selected)
        for group in ("MANIPULATION_CONTACT", "SPATIAL_SUPPORT", "PERCEPTION")
    }
    if any(value != PER_GROUP for value in counts.values()):
        raise RuntimeError(f"insufficient local train QUERY_OBJECT candidates: {counts}")
    samples = []
    for row in sorted(selected, key=lambda item: item["task_id"]):
        path = Path(row["video_path"])
        samples.append(row | {
            "video_sha256": _sha256(path),
            "video_bytes": path.stat().st_size,
        })
    core = {
        "schema_version": "agqa2-query-object-development-manifest-v20",
        "status": "FROZEN_V20_QUERY_OBJECT_TRAIN_DEVELOPMENT_BEFORE_CALLS",
        "split": "development",
        "claim_boundary": (
            "AGQA_TRAIN_ONLY_DEVELOPMENT;ATOMIC_QUERY_OBJECT_WITHOUT_TEMPORAL_"
            "SUBQUERY;18_VIDEO_DISJOINT_ROWS;NOT_CONFIRMATORY"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "LOCAL_AGQA_TRAIN_VIDEOS_ONLY;PUBLIC_QUESTION_ATOMIC_QUERY_OBJECT_"
            "GRAMMAR;HASH_RANK_WITHIN_THREE_RELATION_GROUPS;ONE_TASK_PER_VIDEO;"
            "NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "archive_path": str(ARCHIVE),
        "archive_sha256": _sha256(ARCHIVE),
        "entry": ENTRY,
        "video_root": str(VIDEO_ROOT),
        "samples": samples,
        "sample_count": len(samples),
        "unique_video_count": len(used_videos),
        "relation_group_counts": counts,
        "answer_read_during_freeze": False,
        "scene_graph_read_during_freeze": False,
        "functional_program_used_only_for_route_compatibility": True,
        "source_identity_visible_to_grounder": False,
        "per_question_answer_candidates_visible": False,
    }
    return core | {"manifest_sha256": stable_hash(core)}


def _config(manifest_path: Path, manifest: dict) -> dict:
    base = json.loads((
        REPO_ROOT / "configs/agqa2_temporal_selective_v19_development.json"
    ).read_text())
    config = deepcopy(base)
    config.update({
        "schema_version": "agqa2-query-object-development-config-v20",
        "status": "FROZEN_V20_QUERY_OBJECT_DEVELOPMENT",
        "split": "development",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V20_QUERY_OBJECT_CALL",
        "query_parser_mode": "DETERMINISTIC_EXPLICIT_OPERAND_GRAMMAR_V1",
        "applicability_mode": "ATOMIC_QUERY_OBJECT_ONLY_V1",
        "execution_calibration": None,
        "runtime_selection": None,
        "report_version": "V20_QUERY_OBJECT",
    })
    for key in (
        "expected_grounder_sha256", "expected_evaluation_protocol_sha256",
        "development_qualification_report", "development_qualification_file_sha256",
    ):
        config.pop(key, None)
    collector = REPO_ROOT / "scripts/collect_agqa2_query_object_v20.py"
    base_collector = REPO_ROOT / "scripts/collect_agqa2_active_grounding_v3.py"
    module = REPO_ROOT / "src/motif_transfer/agqa_query_object_grounder.py"
    config["grounder"].update({
        "collector": str(collector.relative_to(REPO_ROOT)),
        "collector_sha256": _sha256(collector),
        "protocol": "SOURCE_RECURRENT_DUAL_TARGET_NATIVE_QUERY_OBJECT_V20",
    })
    ontology_model = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_name": "OPENROUTER_API_KEY",
        "id": "google/gemini-2.5-flash-lite",
        "temperature": 0,
        "timeout_seconds": 240,
        "max_retries": 2,
        "schema_retries": 2,
        "max_ontology_tokens": 300,
    }
    config["query_object_grounder"] = {
        "module": str(module.relative_to(REPO_ROOT)),
        "module_sha256": _sha256(module),
        "base_collector": str(base_collector.relative_to(REPO_ROOT)),
        "base_collector_sha256": _sha256(base_collector),
        "model": ontology_model,
        "fixed_dataset_level_ontology": True,
        "per_question_answer_candidates_read": False,
        "original_question_read": False,
        "visible_fields": [
            "one_requested_relation", "global_object_ontology",
            "chronological_proxy_frames", "frame_timestamps",
        ],
        "forbidden_fields": [
            "original_question", "answer", "functional_program", "scene_graph",
            "per_question_answer_candidates", "direct_response", "source_identity",
        ],
    }
    config["query_object_calibration"] = {
        "mode": "TWO_TARGET_NATIVE_NEURAL_VIEWS_V1",
        "minimum_ontology_confidence": 0.8,
        "override_requires_base_and_ontology_exact_canonical_agreement": True,
        "direct_response_visible_to_calibrator_only_after_both_views_freeze": True,
    }
    config["qualification_gates"] = {
        "required_valid_runtime_rows": 18,
        "minimum_route_correct": 18,
        "minimum_decisive_executions": 10,
        "minimum_decisive_accuracy": 0.75,
        "maximum_typed_vs_direct_losses": 0,
        "minimum_typed_vs_direct_wins": 2,
        "required_source_permuted_abstentions": 18,
        "required_target_written_equivalent_matches": 18,
        "maximum_reported_provider_cost_usd": 0.30,
    }
    config["preregistration"] = "configs/agqa2_query_object_v20_development_preregistration.json"
    return config


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_query_object_v20_development"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V20 QUERY_OBJECT development is already consumed")
    manifest_path = REPO_ROOT / "configs/agqa2_query_object_v20_development_manifest.json"
    manifest = _select()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    config = _config(manifest_path, manifest)
    sources, _ = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    evaluation_sha256 = stable_hash(_evaluation_core(config))
    prereg = {
        "schema_version": "agqa2-query-object-development-preregistration-v20",
        "status": "FROZEN_BEFORE_ANY_V20_QUERY_OBJECT_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "sample_count": manifest["sample_count"],
        "unique_video_count": manifest["unique_video_count"],
        "qualification_gates": config["qualification_gates"],
        "development_adaptation_allowed": True,
        "confirmatory_claim_allowed": False,
        "atomic_route_artifacts_modified": False,
        "reserve_policy": (
            "ONLY_FREEZE_A_NEW_TEST_VIDEO_DISJOINT_RESERVE_IF_ALL_DEVELOPMENT_"
            "GATES_PASS;DO_NOT_REUSE_V14_V19_FORMAL_ROWS"
        ),
    }
    prereg_path = REPO_ROOT / config["preregistration"]
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": grounder_sha256,
        "expected_evaluation_protocol_sha256": evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_query_object_v20_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "manifest_sha256": manifest["manifest_sha256"],
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_sha256,
        "selected": [
            {key: row[key] for key in (
                "task_id", "video_id", "public_relation", "relation_group",
            )}
            for row in manifest["samples"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
