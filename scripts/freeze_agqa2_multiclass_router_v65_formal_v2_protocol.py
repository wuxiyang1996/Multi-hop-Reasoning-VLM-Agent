#!/usr/bin/env python3
"""Prospectively freeze the untouched AGQA multi-route V2 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from motif_transfer.contracts import stable_hash  # noqa: E402

SELECTION = REPO_ROOT / "configs/agqa2_multiclass_router_formal_v2_selection.json"
PRIOR_SELECTION = REPO_ROOT / "configs/agqa2_router_heldout_formal_v1_selection.json"
ROUTER_REPORT = REPO_ROOT / "runs/agqa2_multiclass_program_router_v2/qualification_report.json"
ROUTER_MODEL = REPO_ROOT / "runs/agqa2_multiclass_program_router_v2/router.joblib"
EVALUATOR = REPO_ROOT / "scripts/evaluate_agqa2_multiclass_router_v65_formal_v2.py"
OUTPUT = REPO_ROOT / "configs/agqa2_multiclass_router_v65_formal_v2_protocol.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified(path: Path, key: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop(key)
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid {key}: {path}")
    return value


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("V2 protocol is immutable once frozen")
    selection = _verified(SELECTION, "manifest_sha256")
    prior = _verified(PRIOR_SELECTION, "manifest_sha256")
    router = _verified(ROUTER_REPORT, "report_sha256")
    if selection["status"] != "FROZEN_V78_SELECTION_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS":
        raise ValueError("V2 selection is not in its prospective state")
    if any(Path(row["video_path"]).is_file() for row in selection["samples"]):
        raise ValueError("a V2 video exists locally before protocol freeze")
    if selection["answer_read_during_selection"] or selection["program_read_during_selection"]:
        raise ValueError("selection accessed prohibited formal labels")
    if selection["v1_formal_outcomes_used_for_selection"]:
        raise ValueError("V1 outcomes contaminated V2 selection")
    if router["status"] != "MULTICLASS_PROGRAM_ROUTER_V2_QUALIFIED":
        raise ValueError("multi-class router is not qualified")
    if router["v1_formal_outcomes_used_for_training_or_thresholds"]:
        raise ValueError("V1 outcomes contaminated the multi-class router")
    if _sha256(ROUTER_MODEL) != selection["router_model_file_sha256"]:
        raise ValueError("router model lineage mismatch")
    if _sha256(ROUTER_REPORT) != selection["router_qualification_file_sha256"]:
        raise ValueError("router report lineage mismatch")
    selected_videos = {str(row["video_id"]) for row in selection["samples"]}
    prior_videos = {str(row["video_id"]) for row in prior["samples"]}
    if selected_videos & prior_videos:
        raise ValueError("V2 overlaps V1")

    body = {
        "schema_version": "agqa2-multiclass-router-v65-formal-protocol-v2",
        "status": "FROZEN_BEFORE_VIDEO_DOWNLOAD_PROVIDER_OR_FORMAL_LABEL_ACCESS",
        "claim_boundary": (
            "91_VIDEO_DISJOINT_AGQA_TRAIN_FORMAL_QUESTIONS;"
            "TRAIN_DEV_ONLY_MULTICLASS_PROGRAM_ROUTER;"
            "UNCHANGED_V65_QWEN235_VISUAL_GROUNDER;"
            "RELATION_TEMPORAL_PAIR_TEMPORAL_SINGLE;FOUR_ARM_CONFIRMATORY_TRANSFER"
        ),
        "prospective_context": {
            "v1_result_status": "KNOWN_FAILED_BEFORE_V2_PROTOCOL_FREEZE",
            "v1_outcomes_used_for_router_training_thresholds_or_selection": False,
            "v2_is_new_multiroute_replication_not_v1_gate_revision": True,
        },
        "cohort": {
            "sample_count": len(selection["samples"]),
            "one_question_per_video": True,
            "selection_status": selection["status"],
            "selection_file_sha256": _sha256(SELECTION),
            "selection_manifest_sha256": selection["manifest_sha256"],
            "prior_v1_selection_file_sha256": _sha256(PRIOR_SELECTION),
            "prior_v1_selection_manifest_sha256": prior["manifest_sha256"],
            "route_counts": selection["route_counts"],
            "video_disjoint_from_v1_and_all_prior_runtime": True,
        },
        "controls": {
            "neural_only": "QWEN235_MATCHED_DIRECT_RESPONSE",
            "source_induced": "V65_TYPED_GROUNDING_PLUS_SOURCE_INDUCED_IR_WITH_DIRECT_FAIL_CLOSED_FALLBACK",
            "source_permuted": "DETERMINISTIC_WRONG_PROGRAM_TYPE_MUST_ABSTAIN_TO_MATCHED_DIRECT",
            "target_written_equivalent": "SOURCE_BLIND_EXTENSIONALLY_IDENTICAL_TARGET_CONTROLLER",
        },
        "gates": {
            "required_valid_runtime_rows": len(selection["samples"]),
            "required_route_correct": len(selection["samples"]),
            "required_source_permuted_abstentions": len(selection["samples"]),
            "required_target_written_equivalent_matches": len(selection["samples"]),
            "minimum_source_authorizations": 70,
            "maximum_losses": 9,
            "minimum_net_gain": 6,
            "minimum_wins": 10,
            "maximum_one_sided_exact_pvalue": 0.05,
            "maximum_reported_provider_cost_usd": 0.75,
        },
        "evaluator": {
            "file": str(EVALUATOR.relative_to(REPO_ROOT)),
            "file_sha256": _sha256(EVALUATOR),
            "gold_access": "ONLY_AFTER_ALL_PROVIDER_RUNTIME_RECEIPTS_FREEZE",
            "paired_test": "ONE_SIDED_EXACT_BINOMIAL_ON_SOURCE_VS_NEURAL_DISCORDANT_PAIRS",
        },
        "lineage": {
            "expected_grounder_sha256": "f8e3e500c273858b5cb70a2ae3e0551e51be8dff4d76e33c6144a578209cbed1",
            "frozen_runtime_git_commit": "ded7448839183851aa10c3cd3e12d253f04e1ceb",
            "v65_collector_sha256": "c845a0446fe5edc60f29dedbbb8eca3527a1f0c087f130924529a64cb8cdd5f1",
            "v65_grounder_module_sha256": "87a41b64a77aae9cd8899f714061276fd3fcee05e8950a050fffb8849b81761c",
            "dependency_overlay_sha256": {
                "src/motif_transfer/phase3_source_function_induction.py": "5bd04fa4b0d9b3a90b61d9108e19b8366080b167a63e5ac2556d351356fdcd6d"
            },
            "program_router_model_sha256": selection["router_model_file_sha256"],
            "program_router_qualification_file_sha256": selection["router_qualification_file_sha256"],
            "program_router_qualification_report_sha256": selection["router_qualification_report_sha256"],
            "program_router_training_scope": "AGQA_OFFICIAL_TRAIN_DEVELOPMENT_PARTITIONS_ONLY",
        },
        "prohibited_adaptation": [
            "NO_V65_GROUNDER_PARAMETER_PROMPT_MODEL_OR_THRESHOLD_CHANGE",
            "NO_FORMAL_VIDEO_PROGRAM_ANSWER_OR_SCENE_GRAPH_USED_FOR_ROUTER_TRAINING",
            "NO_V1_OR_V2_FORMAL_RESULT_DRIVEN_THRESHOLD_OR_GATE_CHANGE",
            "NO_RETRY_ON_THE_SAME_V2_COHORT_AFTER_GATE_FAILURE",
        ],
    }
    output = body | {"protocol_sha256": stable_hash(body)}
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": output["status"], "sample_count": body["cohort"]["sample_count"],
        "route_counts": body["cohort"]["route_counts"],
        "evaluator_file_sha256": body["evaluator"]["file_sha256"],
        "protocol_sha256": output["protocol_sha256"],
        "protocol_file_sha256": _sha256(OUTPUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
