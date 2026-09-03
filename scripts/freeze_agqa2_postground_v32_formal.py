#!/usr/bin/env python3
"""Freeze a fresh 120-video V32 post-grounding confirmation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_postground_v32_formal import (  # noqa: E402
    evaluation_protocol_core as _postground_protocol_core,
)
from scripts.collect_agqa2_query_object_v20 import (  # noqa: E402
    _evaluation_core as _base_evaluation_core, _semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v16_reserve import (  # noqa: E402
    _configured_video_ids,
)
from scripts.freeze_agqa2_active_grounding_v4 import _verified_json  # noqa: E402
import scripts.freeze_agqa2_query_object_v23_reserve as v23  # noqa: E402


NONCE = "agqa2-postground-v32-fresh-120-formal-confirmation"
PER_GROUP = 40
TOTAL_ROWS = 120


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_report(path: Path) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = str(body.pop("report_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"report hash mismatch: {path}")
    return value


def _selection(development_manifest: dict, excluded: set[str]) -> dict:
    v23.NONCE = NONCE
    v23.PER_GROUP = PER_GROUP
    inherited = v23._select(development_manifest, excluded)
    core = deepcopy(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-postground-selection-v32",
        "status": "FROZEN_V32_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V32_CALLS",
        "claim_boundary": (
            "FRESH_120_VIDEO_DISJOINT_POSTGROUND_CONFIRMATION;SOURCE_PROGRAM_"
            "AND_TARGET_GROUNDER_FROZEN_AFTER_V31_DEVELOPMENT;ONE_PRIMARY_"
            "PAIRED_ENDPOINT;NO_OUTCOME_ADAPTATION"
        ),
        "selection_nonce": NONCE,
        "prior_v32_neural_grounder_exposure": False,
        "sample_size_rationale": {
            "fixed_total_rows": TOTAL_ROWS,
            "rows_per_relation_group": PER_GROUP,
            "development_rows": 269,
            "development_source_wins": 15,
            "development_source_losses": 1,
            "development_disagreement_rate": 16 / 269,
            "formal_sample_size_frozen_before_video_download_or_calls": True,
        },
    })
    core.pop("prior_v23_neural_grounder_exposure", None)
    return core | {"manifest_sha256": stable_hash(core)}


def _manifest(selection: dict) -> dict:
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
        "schema_version": "agqa2-postground-formal-manifest-v32",
        "status": "FROZEN_V32_RAW_VIDEO_UNSEEN_BEFORE_PROVIDER_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": len(samples),
        "prior_neural_grounder_or_model_video_exposure": False,
        "answer_read_during_freeze": False,
        "scene_graph_read_during_freeze": False,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    output = REPO_ROOT / "runs/agqa2_postground_v32_formal/report.json"
    if output.is_file():
        raise RuntimeError("V32 formal outcomes are already consumed")
    development_path = (
        REPO_ROOT / "runs/agqa2_postground_v31_development/report.json"
    )
    development = _verified_report(development_path)
    if not development.get("qualified_for_future_disjoint_reserve"):
        raise ValueError("V31 postground development did not qualify")
    base_development_path = (
        REPO_ROOT / "docs/results/agqa2_query_object_v28_development_summary.json"
    )
    base_development = _verified_json(base_development_path, "summary_sha256")
    if not base_development.get("grounder_qualified"):
        raise ValueError("V32 base neural grounder dependency did not qualify")
    development_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_query_object_v24_development_manifest.json",
        "manifest_sha256",
    )
    excluded = _configured_video_ids()
    video_root = Path(development_manifest["video_root"])
    excluded.update(path.stem for path in video_root.glob("*.mp4"))
    selection_path = (
        REPO_ROOT / "configs/agqa2_postground_v32_formal_selection.json"
    )
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path.is_file()
        else _selection(development_manifest, excluded)
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
            "relation_group_counts": selection["relation_group_counts"],
            "missing_video_ids": missing,
            "next": (
                "run download_agqa2_active_grounding_v4_reserve.py on the "
                "frozen selection, then rerun this freezer"
            ),
        }, indent=2))
        return

    receipt_path = REPO_ROOT / "runs/agqa2_postground_v32_download/receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("V32 frozen-video download receipt is missing")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256")
        != selection["manifest_sha256"]
        or len(receipt.get("videos", [])) != TOTAL_ROWS
    ):
        raise ValueError("V32 download receipt is incomplete or mismatched")
    manifest = _manifest(selection)
    manifest_path = (
        REPO_ROOT / "configs/agqa2_postground_v32_formal_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = json.loads((
        REPO_ROOT / "configs/agqa2_query_object_v28_development.json"
    ).read_text())
    config.update({
        "schema_version": "agqa2-postground-base-config-v32",
        "status": "FROZEN_V32_POSTGROUND_FORMAL",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V32_PROVIDER_OR_GOLD_CALL"
        ),
        "development_qualification_report": str(
            base_development_path.relative_to(REPO_ROOT)
        ),
        "development_qualification_file_sha256": _sha256(
            base_development_path
        ),
        "report_version": "V32_BASE_QUERY_OBJECT",
        "preregistration": (
            "configs/agqa2_postground_v32_formal_preregistration.json"
        ),
    })
    config.pop("source_specific_evaluation", None)
    for key in (
        "preregistration_file_sha256", "expected_grounder_sha256",
        "expected_evaluation_protocol_sha256",
    ):
        config.pop(key, None)
    config["qualification_gates"] = {
        "required_valid_runtime_rows": TOTAL_ROWS,
        "minimum_route_correct": TOTAL_ROWS,
        "minimum_decisive_executions": 60,
        "minimum_decisive_accuracy": 0.75,
        "maximum_typed_vs_direct_losses": 2,
        "minimum_typed_vs_direct_wins": 10,
        "required_source_permuted_abstentions": TOTAL_ROWS,
        "required_target_written_equivalent_matches": TOTAL_ROWS,
        "maximum_reported_provider_cost_usd": 1.30,
    }

    artifact_path = REPO_ROOT / "runs/sokoban_goal_relation_macro_v3/artifact.json"
    confirmation_path = (
        REPO_ROOT
        / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json"
    )
    inducer_path = REPO_ROOT / "src/motif_transfer/source_goal_relation_induction.py"
    adapter_path = (
        REPO_ROOT / "src/motif_transfer/agqa_postground_relation_transfer.py"
    )
    prediction_path = (
        REPO_ROOT / "src/motif_transfer/agqa_postground_relation_evaluation.py"
    )
    collector_path = (
        REPO_ROOT / "scripts/collect_agqa2_postground_v32_formal.py"
    )
    calibration = development["future_route_calibration"]
    postground = {
        "schema_version": "agqa2-postground-formal-protocol-v32",
        "claim_boundary": manifest["claim_boundary"],
        "manifest_sha256": manifest["manifest_sha256"],
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
        },
        "adapter": {
            "module": str(adapter_path.relative_to(REPO_ROOT)),
            "module_file_sha256": _sha256(adapter_path),
            "prediction_module": str(prediction_path.relative_to(REPO_ROOT)),
            "prediction_module_file_sha256": _sha256(prediction_path),
            "collector": str(collector_path.relative_to(REPO_ROOT)),
            "collector_file_sha256": _sha256(collector_path),
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
                "POSTGROUND_UNIQUE_BINDING_PLUS_SOURCE_RECURRENT_ACQUISITION"
            ),
            "source_effect_shuffled": (
                "SOURCE_HELDOUT_SHUFFLED_EFFECT_FAIL_CLOSED"
            ),
            "generic_scaffold": (
                "MATCHED_HANDWRITTEN_THREE_VIEW_CONSENSUS_CEILING"
            ),
            "target_written_equivalent": (
                "EXTENSIONALLY_IDENTICAL_CEILING_EXPECTED_TO_MATCH_SOURCE"
            ),
        },
        "qualification_gates": {
            "required_valid_rows": TOTAL_ROWS,
            "minimum_source_authorizations": 50,
            "minimum_source_vs_target_wins": 5,
            "maximum_source_vs_target_losses": 2,
            "minimum_source_minus_target_correct": 4,
            "maximum_exact_one_sided_pvalue": 0.05,
            "maximum_reported_provider_cost_usd": 1.30,
        },
        "failure_policy": (
            "RUN_EXACT_120_ROWS_ONCE;REPORT_ALL_CONTROLS;NO_RULE_THRESHOLD_"
            "OR_SAMPLE_CHANGES_AFTER_OUTCOMES;NO_CLAIM_IF_ANY_GATE_FAILS"
        ),
        "preregistration": config["preregistration"],
    }
    config["postground_formal"] = postground
    sources, _ = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    if grounder_sha256 != base_development["grounder_sha256"]:
        raise AssertionError("V32 changed the V28 bounded neural grounder")
    base_protocol_sha256 = stable_hash(_base_evaluation_core(config))
    config["expected_grounder_sha256"] = grounder_sha256
    config["expected_evaluation_protocol_sha256"] = base_protocol_sha256
    postground_protocol_sha256 = stable_hash(_postground_protocol_core(config))
    config["postground_formal"]["expected_evaluation_protocol_sha256"] = (
        postground_protocol_sha256
    )
    prereg_core = {
        "schema_version": "agqa2-postground-preregistration-v32",
        "status": "FROZEN_BEFORE_ANY_V32_PROVIDER_OR_GOLD_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "sealed_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "development_report_sha256": development["report_sha256"],
        "source_program_sha256": development["source_artifact_sha256"],
        "target_grounder_sha256": development["target_grounder_sha256"],
        "target_executor_sha256": development["target_executor_sha256"],
        "base_neural_grounder_sha256": grounder_sha256,
        "base_evaluation_protocol_sha256": base_protocol_sha256,
        "postground_evaluation_protocol_sha256": postground_protocol_sha256,
        "calibration": deepcopy(postground["calibration"]),
        "controls": deepcopy(postground["controls"]),
        "qualification_gates": deepcopy(postground["qualification_gates"]),
        "sample_size_rationale": deepcopy(selection["sample_size_rationale"]),
        "failure_policy": postground["failure_policy"],
        "provider_cost_cap_usd": 1.30,
        "provider_or_gold_calls_before_freeze": False,
        "primary_endpoint": (
            "SOURCE_INDUCED_VS_TARGET_NATIVE_PAIRED_CORRECTNESS"
        ),
        "multiplicity_policy": "ONE_PRIMARY_ENDPOINT;ALL_CONTROLS_REPORTED",
    }
    prereg = prereg_core | {
        "preregistration_sha256": stable_hash(prereg_core)
    }
    prereg_path = REPO_ROOT / config["preregistration"]
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config["preregistration_file_sha256"] = _sha256(prereg_path)
    config["postground_formal"]["preregistration_file_sha256"] = _sha256(
        prereg_path
    )
    config_path = REPO_ROOT / "configs/agqa2_postground_v32_formal.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["status"],
        "sample_count": manifest["sample_count"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "sealed_manifest_sha256": manifest["manifest_sha256"],
        "development_report_sha256": development["report_sha256"],
        "postground_evaluation_protocol_sha256": postground_protocol_sha256,
        "base_neural_grounder_unchanged": True,
        "provider_or_gold_calls_before_freeze": False,
        "provider_cost_cap_usd": 1.30,
    }, indent=2))


if __name__ == "__main__":
    main()
