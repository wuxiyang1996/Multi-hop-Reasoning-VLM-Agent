#!/usr/bin/env python3
"""Freeze one fresh AGQA robust temporal-pair formal confirmation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import (  # noqa: E402
    parse_public_question_plan,
)
from motif_transfer.agqa_program_transfer import (  # noqa: E402
    TEMPORAL_PAIR_ROUTE,
    profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object,
    _load_sources,
)
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core,
    _grounder_semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v12_reserve import (  # noqa: E402
    _eligible,
)
from scripts.freeze_agqa2_active_grounding_v14_replication import (  # noqa: E402
    _program_answer_space_matches,
)
from scripts.freeze_agqa2_active_grounding_v16_reserve import (  # noqa: E402
    _configured_video_ids,
)


NONCE = "agqa2-v34-candy-robust-temporal-pair-formal-100"
SAMPLE_COUNT = 100
DEVELOPMENT_REPORT = (
    "runs/agqa2_robust_temporal_v33_development/report.json"
)
DEVELOPMENT_SUMMARY = (
    "docs/results/agqa2_robust_temporal_v33_development_summary.json"
)
PARENT_CONFIG = "configs/agqa2_temporal_selective_v19_reserve.json"
DEVELOPMENT_MANIFEST = (
    "configs/agqa2_temporal_selective_v19_development_manifest.json"
)
SELECTION = "configs/agqa2_robust_temporal_v34_formal_selection.json"
MANIFEST = "configs/agqa2_robust_temporal_v34_formal_manifest.json"
PREREGISTRATION = (
    "configs/agqa2_robust_temporal_v34_formal_preregistration.json"
)
CONFIG = "configs/agqa2_robust_temporal_v34_formal.json"
DOWNLOAD_RECEIPT = "runs/agqa2_robust_temporal_v34_download/receipt.json"
ADAPTER_MODULE = "src/motif_transfer/agqa_robust_temporal_transfer.py"
EVALUATOR_MODULE = "scripts/collect_agqa2_robust_temporal_v34_formal.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"report hash mismatch: {path}")
    return value


def _verified_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop("manifest_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"manifest hash mismatch: {path}")
    return value


def _write_development_summary(report: Mapping[str, Any]) -> Path:
    path = REPO_ROOT / DEVELOPMENT_SUMMARY
    core = {
        "schema_version": "agqa2-robust-temporal-v33-development-summary-v1",
        "status": report["status"],
        "grounder_qualified": (
            report["status"]
            == "AGQA2_ROBUST_TEMPORAL_V33_DEVELOPMENT_QUALIFIED"
        ),
        # The base collector validates this parent acquisition identity.  The
        # post-ground identity is separately frozen below and in the final
        # evaluator.
        "grounder_sha256": report["parent_target_grounder_sha256"],
        "postground_target_grounder_sha256": report[
            "target_grounder_sha256"
        ],
        "source_program_sha256": report["source_program_sha256"],
        "rows": report["rows"],
        "source_authorizations": report["source_authorizations"],
        "source_vs_target_native": report["source_vs_target_native"],
        "future_route_calibration": report["future_route_calibration"],
        "qualification_gates": report["qualification_gates"],
        "development_report_sha256": report["report_sha256"],
        "development_report_file_sha256": _sha256(
            REPO_ROOT / DEVELOPMENT_REPORT
        ),
        "confirmatory_claim": False,
        "provider_calls": 0,
        "reported_provider_cost_usd": 0.0,
    }
    summary = core | {"summary_sha256": stable_hash(core)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def _selection(development: Mapping[str, Any], excluded: set[str]) -> dict:
    candidates = []
    root = Path(development["video_root"])
    with zipfile.ZipFile(development["archive_path"]) as bundle, bundle.open(
        development["entry"], "r",
    ) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if (
                    not video_id
                    or video_id in excluded
                    or (root / f"{video_id}.mp4").is_file()
                ):
                    continue
                question = str(row.get("question", ""))
                plan = parse_public_question_plan(question)
                if (
                    plan is None
                    or not _eligible(plan)
                    or plan.obligation_kind != TEMPORAL_PAIR_ROUTE
                ):
                    continue
                program = str(row.get("program", ""))
                if (
                    profile_program(
                        task_id=task_id, program=program,
                    ).route_kind != TEMPORAL_PAIR_ROUTE
                    or not _program_answer_space_matches(plan, program)
                ):
                    continue
                candidates.append({
                    "task_id": str(task_id),
                    "video_id": video_id,
                    "oracle_route": TEMPORAL_PAIR_ROUTE,
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                    "public_parser_plan_sha256": plan.plan_sha256,
                    "applicability_rule": (
                        "ATOMIC_BEFORE_AFTER_PUBLIC_GRAMMAR_PLUS_PROGRAM_"
                        "ANSWER_SPACE_COMPATIBILITY"
                    ),
                    "rank_sha256": stable_hash(f"{NONCE}:{task_id}"),
                })
    candidates.sort(key=lambda row: row["rank_sha256"])
    selected = []
    used_videos = set()
    for row in candidates:
        if row["video_id"] in used_videos:
            continue
        selected.append(row | {
            "video_path": str(root / f"{row['video_id']}.mp4"),
            "video_present_at_selection": False,
        })
        used_videos.add(row["video_id"])
        if len(selected) == SAMPLE_COUNT:
            break
    if len(selected) != SAMPLE_COUNT:
        raise RuntimeError(
            f"insufficient fresh temporal-pair videos: {len(selected)}"
        )
    core = {
        "schema_version": "agqa2-robust-temporal-selection-v34-formal",
        "status": "FROZEN_V34_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V34_CALLS",
        "split": "reserve",
        "claim_boundary": (
            "ONE_HUNDRED_NEW_CROSS_EXPERIMENT_VIDEO_DISJOINT_ATOMIC_"
            "BEFORE_AFTER_ROWS;NOT_UNTOUCHED_METADATA"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_VIDEO_IDS_REFERENCED_BY_PRIOR_CONFIGS_AND_ALL_"
            "MP4S_PRESENT_IN_SHARED_CHARADES_ROOT;REQUIRE_ATOMIC_"
            "TEMPORAL_PAIR_TYPED_ARITY_AND_PROGRAM_ANSWER_SPACE_"
            "COMPATIBILITY;FIXED_HASH_RANK;ONE_TASK_PER_VIDEO;NO_ANSWER_OR_"
            "SCENE_GRAPH_READ"
        ),
        "archive_path": development["archive_path"],
        "archive_sha256": development["archive_sha256"],
        "entry": development["entry"],
        "video_root": development["video_root"],
        "route_counts": {TEMPORAL_PAIR_ROUTE: SAMPLE_COUNT},
        "samples": selected,
        "sample_count": len(selected),
        "unique_video_count": len(used_videos),
        "excluded_prior_config_or_present_video_count": len(excluded),
        "excluded_video_ids_sha256": stable_hash(sorted(excluded)),
        "answer_read_during_freeze": False,
        "scene_graph_grounding_read_during_freeze": False,
        "direct_response_read_during_freeze": False,
        "functional_program_root_read_for_answer_space_only": True,
        "prior_v34_neural_grounder_exposure": False,
        "raw_video_archive": {
            "url": (
                "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/"
                "charades/Charades_v1_480.zip"
            ),
            "archive_prefix": "Charades_v1_480/",
        },
    }
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: Mapping[str, Any]) -> dict:
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
        "schema_version": "agqa2-robust-temporal-manifest-v34-formal",
        "status": "FROZEN_V34_RAW_VIDEO_UNSEEN_BEFORE_FORMAL_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": len(samples),
        "local_integrity_decode_probe_completed": True,
        "prior_neural_grounder_or_model_video_exposure": False,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    report = _verified_report(REPO_ROOT / DEVELOPMENT_REPORT)
    if report["status"] != (
        "AGQA2_ROBUST_TEMPORAL_V33_DEVELOPMENT_QUALIFIED"
    ):
        raise ValueError("V33 robust temporal development did not qualify")
    if not all(report["qualification_gates"].values()):
        raise ValueError("V33 development has a failed gate")
    summary_path = _write_development_summary(report)
    development = _verified_manifest(REPO_ROOT / DEVELOPMENT_MANIFEST)

    selection_path = REPO_ROOT / SELECTION
    if selection_path.is_file():
        selection = _verified_manifest(selection_path)
    else:
        excluded = _configured_video_ids()
        excluded.update(
            path.stem
            for path in Path(development["video_root"]).glob("*.mp4")
        )
        selection = _selection(development, excluded)
        selection_path.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n"
        )
    missing = [
        row["video_id"] for row in selection["samples"]
        if not Path(row["video_path"]).is_file()
    ]
    if missing:
        print(json.dumps({
            "status": selection["status"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "candidate_count": selection["sample_count"],
            "missing_video_count": len(missing),
            "missing_video_ids": missing,
            "next": (
                "download exact frozen videos with the generic AGQA range "
                "extractor, then rerun this freezer"
            ),
        }, indent=2))
        return

    receipt_path = REPO_ROOT / DOWNLOAD_RECEIPT
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256")
        != selection["manifest_sha256"]
        or len(receipt.get("videos") or ()) != SAMPLE_COUNT
    ):
        raise ValueError("V34 download receipt is incomplete or mismatched")
    manifest = _seal(selection)
    manifest_path = REPO_ROOT / MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    config = deepcopy(parent)
    config.pop("runtime_selection", None)
    config["qualification_gates"] = {
        # This base report is a receipt transport/integrity diagnostic.  The
        # impossible decisive-count gate prevents its legacy global gate from
        # authorizing source execution; V34 post-ground gates are separate.
        "required_valid_runtime_rows": SAMPLE_COUNT,
        "minimum_route_correct": SAMPLE_COUNT,
        "minimum_decisive_executions": SAMPLE_COUNT + 1,
        "minimum_decisive_accuracy": 0.0,
        "minimum_typed_vs_direct_wins": 0,
        "maximum_typed_vs_direct_losses": SAMPLE_COUNT,
        "required_source_permuted_abstentions": SAMPLE_COUNT,
        "required_target_written_equivalent_matches": SAMPLE_COUNT,
        "maximum_reported_provider_cost_usd": 0.9,
    }
    config.update({
        "schema_version": "agqa2-robust-temporal-v34-base-config-v1",
        "status": "FROZEN_V34_ROBUST_TEMPORAL_FORMAL_BASE",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": MANIFEST,
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL"
        ),
        "development_qualification_report": DEVELOPMENT_SUMMARY,
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V34_BASE",
    })
    sources, _ = _load_sources(config)
    parent_grounder_sha256 = stable_hash(
        _grounder_semantic_core(config, sources)
    )
    if parent_grounder_sha256 != report["parent_target_grounder_sha256"]:
        raise AssertionError("V34 changed the qualified neural acquisition")
    evaluation_protocol_sha256 = stable_hash(
        _evaluation_protocol_core(config)
    )
    adapter_sha256 = _sha256(REPO_ROOT / ADAPTER_MODULE)
    if (
        report["target_grounder_core"][
            "postground_adapter_module_sha256"
        ] != adapter_sha256
    ):
        raise AssertionError("V34 post-ground adapter changed after development")
    source_program_sha256 = report["source_program_sha256"]
    final_gates = {
        "required_valid_rows": SAMPLE_COUNT,
        "required_unique_videos": SAMPLE_COUNT,
        "minimum_source_authorizations": 20,
        "minimum_source_wins": 7,
        "maximum_source_losses": 1,
        "minimum_source_minus_target_correct": 5,
        "maximum_exact_one_sided_pvalue": 0.05,
        "required_effect_shuffled_abstentions": SAMPLE_COUNT,
        "required_wrong_source_abstentions": SAMPLE_COUNT,
        "required_generic_scaffold_matches": SAMPLE_COUNT,
        "required_target_written_equivalent_matches": SAMPLE_COUNT,
        "maximum_reported_provider_cost_usd": 0.9,
    }
    final_protocol_core = {
        "schema_version": "agqa2-robust-temporal-v34-evaluation-protocol-v1",
        "sample_count": SAMPLE_COUNT,
        "primary_endpoint": "SOURCE_INDUCED_VS_TARGET_NATIVE_PAIRED_ACCURACY",
        "source_program_sha256": source_program_sha256,
        "target_executor_sha256": report["target_executor_sha256"],
        "postground_target_grounder_sha256": report[
            "target_grounder_sha256"
        ],
        "adapter_module_sha256": adapter_sha256,
        "evaluator_module_sha256": _sha256(REPO_ROOT / EVALUATOR_MODULE),
        "development_calibration": report["future_route_calibration"],
        "fallback": "PRESERVE_MATCHED_TARGET_NATIVE_DIRECT_ON_ABSTENTION",
        "controls": [
            "SOURCE_EFFECT_SHUFFLED",
            "WRONG_SOURCE_TEMPORAL_SINGLE",
            "HANDWRITTEN_GENERIC_EQUIVALENT",
            "TARGET_WRITTEN_EQUIVALENT",
        ],
        "formal_gates": final_gates,
        "current_outcome_authorization": False,
    }
    final_protocol_sha256 = stable_hash(final_protocol_core)
    preregistration = {
        "schema_version": "agqa2-robust-temporal-v34-preregistration-v1",
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "qualified_v33_development_report_sha256": report["report_sha256"],
        "qualified_parent_grounder_sha256": parent_grounder_sha256,
        "qualified_postground_target_grounder_sha256": report[
            "target_grounder_sha256"
        ],
        "source_program_sha256": source_program_sha256,
        "target_executor_sha256": report["target_executor_sha256"],
        "source_program_induced_from_interventions": True,
        "source_program_target_data_read": False,
        "development_calibration": report["future_route_calibration"],
        "base_evaluation_protocol_sha256": evaluation_protocol_sha256,
        "postground_evaluation_protocol": final_protocol_core,
        "postground_evaluation_protocol_sha256": final_protocol_sha256,
        "formal_gates": final_gates,
        "cost_projection": {
            "v19_temporal_pair_mean_cost_per_row_usd": 0.0069045976,
            "projected_100_row_cost_usd": 0.69045976,
            "frozen_cap_usd": 0.9,
        },
        "failure_policy": {
            "formal": "RUN_ONCE_ON_FROZEN_POOL;NO_POST_OUTCOME_ADAPTATION",
            "failed_gate": "REPORT_NOT_QUALIFIED;DO_NOT_RESAMPLE",
        },
    }
    prereg_path = REPO_ROOT / PREREGISTRATION
    prereg_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n"
    )
    config.update({
        "preregistration": PREREGISTRATION,
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": parent_grounder_sha256,
        "expected_evaluation_protocol_sha256": evaluation_protocol_sha256,
        "postground": {
            "adapter_module": ADAPTER_MODULE,
            "adapter_module_sha256": adapter_sha256,
            "evaluator_module": EVALUATOR_MODULE,
            "evaluator_module_sha256": _sha256(REPO_ROOT / EVALUATOR_MODULE),
            "target_grounder_sha256": report["target_grounder_sha256"],
            "target_executor_sha256": report["target_executor_sha256"],
            "source_program_sha256": source_program_sha256,
            "development_calibration": report[
                "future_route_calibration"
            ],
            "evaluation_protocol_sha256": final_protocol_sha256,
            "formal_gates": final_gates,
        },
    })
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": preregistration["status"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": SAMPLE_COUNT,
        "parent_grounder_sha256": parent_grounder_sha256,
        "postground_target_grounder_sha256": report[
            "target_grounder_sha256"
        ],
        "postground_evaluation_protocol_sha256": final_protocol_sha256,
        "provider_cost_cap_usd": 0.9,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
