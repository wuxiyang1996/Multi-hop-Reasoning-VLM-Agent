#!/usr/bin/env python3
"""Freeze the one-shot AGQA Qwen32 formal protocol after development passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _component(relative: str) -> str:
    return _sha256(REPO_ROOT / relative)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-evaluation", type=Path, required=True)
    parser.add_argument("--development-preoutcome", type=Path, required=True)
    parser.add_argument("--development-semantic-runtime", type=Path, required=True)
    parser.add_argument("--development-grounding-view", type=Path, action="append", required=True)
    parser.add_argument("--development-routed-grounding", type=Path, required=True)
    parser.add_argument("--development-fallback", type=Path, required=True)
    parser.add_argument("--exclusion-ledger", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--anonymous-controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V17 formal protocol is immutable")
    if len(args.development_grounding_view) != 2:
        raise ValueError("exactly two development grounding views are required")

    evaluation = json.loads(args.development_evaluation.read_text())
    pre = json.loads(args.development_preoutcome.read_text())
    runtime = json.loads(args.development_semantic_runtime.read_text())
    views = [json.loads(path.read_text()) for path in args.development_grounding_view]
    routed = json.loads(args.development_routed_grounding.read_text())
    fallback = json.loads(args.development_fallback.read_text())
    ledger = json.loads(args.exclusion_ledger.read_text())
    source = json.loads(args.source_capabilities.read_text())
    controller = json.loads(args.anonymous_controller.read_text())

    if evaluation.get("status") != "V18_COMPOSITIONAL_DEVELOPMENT_TRANSFER_SIGNAL_PASSED":
        raise ValueError("development transfer qualification did not pass")
    if not all(evaluation.get("gates", {}).values()):
        raise ValueError("development transfer gates are incomplete")
    if evaluation.get("preoutcome_file_sha256") != _sha256(args.development_preoutcome):
        raise ValueError("development evaluation/preoutcome mismatch")
    if pre.get("status") != "V18_COMPOSITIONAL_DEVELOPMENT_DECISIONS_FROZEN":
        raise ValueError("development preoutcome did not pass")
    if pre.get("executor_version") != "v3" or evaluation.get("executor_version") != "v3":
        raise ValueError("development did not qualify the V3 typed executor adapter")
    if not all(pre.get("gates", {}).values()):
        raise ValueError("development preoutcome gates are incomplete")
    if runtime.get("status") != "SEMANTIC_RUNTIME_FROZEN_BEFORE_VIDEO_OR_OUTCOME":
        raise ValueError("development semantic runtime is invalid")
    if any(view.get("status") != "RAW_VIDEO_GROUNDING_FROZEN_BEFORE_OUTCOMES" for view in views):
        raise ValueError("development grounding views are invalid")
    if [int(view.get("frame_budget", -1)) for view in views] != [48, 96]:
        raise ValueError("development frame views are not the preregistered 48/96 pair")
    if any(view.get("model") != "qwen/qwen3-vl-32b-instruct" for view in views):
        raise ValueError("development grounder is not Qwen3-VL-32B")
    if pre.get("grounding_view_file_sha256s") != [
        _sha256(path) for path in args.development_grounding_view
    ]:
        raise ValueError("development preoutcome does not bind grounding views")
    if routed.get("candidate_report_sha256s") != [view["report_sha256"] for view in views]:
        raise ValueError("development router does not bind ordered grounding views")
    if pre.get("routed_grounding_file_sha256") != _sha256(args.development_routed_grounding):
        raise ValueError("development preoutcome does not bind routed grounding")
    if fallback.get("status") != "SHARED_FALLBACK_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("development fallback is invalid")
    if fallback.get("model") != "Qwen/Qwen3.5-9B":
        raise ValueError("development fallback model differs from protocol")
    if pre.get("fallback_file_sha256") != _sha256(args.development_fallback):
        raise ValueError("development preoutcome does not bind fallback")
    if ledger.get("status") != "ALL_EXISTING_AGQA_COHORT_VIDEOS_FROZEN_AS_EXCLUDED":
        raise ValueError("prior-video exclusion ledger is invalid")
    if source.get("status") != "SOURCE_CAPABILITIES_INDUCED" or source.get("target_data_read"):
        raise ValueError("source capabilities crossed the target boundary")
    if controller.get("status") != "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED":
        raise ValueError("anonymous source controller is invalid")

    components = {
        "semantic_parser_sha256": _component("scripts/run_agqa_layer_b_semantic_parser.py"),
        "qwen32_collector_sha256": _component("scripts/collect_agqa_layer_b_qwen235_grounding.py"),
        "grounding_merger_sha256": _component("scripts/merge_agqa_layer_b_grounding_shards.py"),
        "grounding_router_sha256": _component("scripts/route_agqa_layer_b_shared_grounding.py"),
        "fallback_sha256": _component("scripts/collect_agqa_layer_b_shared_fallback.py"),
        "preoutcome_sha256": _component(
            "scripts/freeze_agqa_offtheshelf_qwen32_multiview_development_preoutcome_v16.py"
        ),
        "evaluator_sha256": _component(
            "scripts/evaluate_agqa_offtheshelf_compositional_development_v15.py"
        ),
        "executor_adapter_sha256": _component(
            "src/motif_transfer/agqa_layer_b_executor_v3.py"
        ),
        "cohort_freezer_sha256": _component(
            "scripts/freeze_agqa_offtheshelf_qwen32_formal_cohort_v17.py"
        ),
        "downloader_sha256": _component(
            "scripts/download_agqa2_active_grounding_v4_reserve.py"
        ),
    }
    body = {
        "schema_version": "agqa-offtheshelf-qwen32-compositional-formal-protocol-v17",
        "status": "QWEN32_COMPOSITIONAL_FORMAL_PROTOCOL_FROZEN_AFTER_DEVELOPMENT",
        "claim_scope": "ONE_SHOT_FRESH_VIDEO_AND_TASK_DISJOINT_BALANCED_TRAIN_COMPOSITIONAL_TRANSFER",
        "development": {
            "evaluation_file_sha256": _sha256(args.development_evaluation),
            "evaluation_report_sha256": evaluation["report_sha256"],
            "preoutcome_file_sha256": _sha256(args.development_preoutcome),
            "preoutcome_receipt_sha256": pre["receipt_sha256"],
            "cohort_sha256": pre["cohort_sha256"],
            "all_transfer_gates_passed": True,
            "outcomes_previously_consumed": True,
        },
        "exclusion_ledger_file_sha256": _sha256(args.exclusion_ledger),
        "exclusion_ledger_sha256": ledger["ledger_sha256"],
        "excluded_prior_videos": ledger["excluded_videos"],
        "source_harness": {
            "source_capability_file_sha256": _sha256(args.source_capabilities),
            "source_capability_sha256": source["artifact_sha256"],
            "source_induction_authority": source["induction_authority"],
            "anonymous_controller_file_sha256": _sha256(args.anonymous_controller),
            "anonymous_controller_sha256": controller["artifact_sha256"],
        },
        "runtime": {
            "semantic_parser_sha256": runtime["parser_sha256"],
            "semantic_parser_qualification_sha256": runtime["qualification_sha256"],
            "grounder_model": "qwen/qwen3-vl-32b-instruct",
            "qwen32_grounder_backend_sha256s": [view["grounder_backend_sha256"] for view in views],
            "frame_views": [48, 96],
            "sampling": "uniform_full_video",
            "grounder_response_mode": "json_schema",
            "grounder_max_tokens": 1800,
            "grounder_temperature": None,
            "router": "FIRST_GENERIC_COMMIT_ELSE_FIRST_CANDIDATE_V1",
            "fallback_model": "Qwen/Qwen3.5-9B",
            "fallback_temperature": 0,
            "fallback_thinking": False,
            "executor_version": "v3",
            "same_frames_grounding_parser_executor_and_fallback_for_all_five_arms": True,
        },
        "preoutcome_gates": {
            "minimum_source_commit_fraction": 0.20,
            "maximum_permuted_commit_fraction": 0.05,
            "minimum_disagreement_fraction": 0.05,
            "minimum_two_event_fraction": 0.50,
            "target_written_isomorphic_equivalence": 1.0,
        },
        "formal_gates": {
            "source_strictly_beats_neural_only": True,
            "source_vs_neural_exact_mcnemar_p_below": 0.05,
            "source_strictly_beats_matched_permuted": True,
            "source_vs_matched_permuted_exact_mcnemar_p_below": 0.05,
            "negative_transfer_fraction_at_most": 0.05,
            "target_written_isomorphic_equivalence": 1.0,
            "generic_scaffold_is_reported_ceiling_not_a_pass_gate": True,
            "source_accuracy_strictly_above_55_percent_is_secondary_not_a_pass_gate": True,
        },
        "formal_cohort": {
            "tasks": 256,
            "videos": 256,
            "per_semantic_root": 128,
            "semantic_roots": ["duration_choice", "duration_extremum"],
            "must_exclude_every_video_in_ledger": True,
            "outcome_open_count": 1,
        },
        "components": components,
        "authority": {
            "grounder_may_read": ["raw_video_frames", "question", "operator_free_semantic_slots", "public_AGQA_object_action_ontology"],
            "runtime_may_not_read": ["gold_answer", "official_STSG", "functional_program", "source_controller", "target_outcome"],
            "formal_outcomes_unread": True,
            "provider_calls_at_protocol_freeze": 0,
        },
    }
    body["protocol_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "development_report_sha256": evaluation["report_sha256"],
        "excluded_prior_videos": ledger["excluded_videos"],
        "protocol_sha256": body["protocol_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
