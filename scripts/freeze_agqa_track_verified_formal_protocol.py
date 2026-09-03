#!/usr/bin/env python3
"""Freeze the AGQA track-verified five-arm formal protocol after qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMPLEMENTATION_PATHS = (
    "scripts/freeze_agqa_query_grounder_v2_strict_boundary_formal.py",
    "scripts/run_agqa_layer_b_semantic_parser.py",
    "scripts/pilot_agqa_action_genome_sgdet.py",
    "scripts/probe_agqa_layer_b_charades_action_model.py",
    "scripts/merge_agqa_layer_b_action_probe_shards.py",
    "scripts/build_agqa_action_genome_sgdet_query_plans.py",
    "scripts/compile_agqa_slowfast_action_bindings.py",
    "scripts/compile_agqa_action_genome_query_grounder_v2.py",
    "scripts/compile_agqa_action_genome_query_grounder_v2_strict_temporal.py",
    "scripts/verify_agqa_query_candidates_with_stable_tracks.py",
    "scripts/adapt_agqa_query_grounder_v2_to_layer_b.py",
    "scripts/collect_agqa_layer_b_shared_fallback.py",
    "scripts/freeze_agqa_query_grounder_v2_strict_boundary_preoutcome.py",
    "scripts/evaluate_agqa_query_grounder_v2_strict_boundary_formal.py",
    "src/motif_transfer/agqa_layer_b_harness.py",
    "src/motif_transfer/agqa_layer_b_executor_v2.py",
    "src/motif_transfer/agqa_query_grounder_v2.py",
    "src/motif_transfer/agqa_action_genome_grounder.py",
    "src/motif_transfer/agqa_strict_temporal_projection.py",
    "src/motif_transfer/agqa_track_verified_candidate.py",
    "src/motif_transfer/anonymous_video_harness.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_protocol(
    qualification: dict,
    qualification_protocol: dict,
    *,
    qualification_file: Path,
    qualification_protocol_file: Path,
    videos: int,
    tasks_per_video: int,
    selection_salt: str,
) -> dict:
    if qualification.get("status") != "QUERY_GROUNDER_V2_STRICT_BOUNDARY_QUALIFIED":
        raise ValueError("formal protocol requires a passed grounder qualification")
    if not qualification.get("gates") or not all(qualification["gates"].values()):
        raise ValueError("grounder qualification gates did not all pass")
    if qualification.get("protocol_file_sha256") != _sha256(qualification_protocol_file):
        raise ValueError("qualification is not bound to its protocol")
    frozen = qualification_protocol["frozen_grounder"]
    if "candidate_verifier" not in frozen:
        raise ValueError("qualification did not freeze the stable-track verifier")
    source = qualification_protocol["source_harness"]
    tasks = int(videos) * int(tasks_per_video)
    typed_evidence = (
        frozen["candidate_verifier"].get("grounding_schema_version")
        == "agqa-query-grounder-v2-typed-evidence-verified-v1"
    )
    return {
        "schema_version": (
            "agqa-query-grounder-v2-typed-evidence-formal-protocol-v1"
            if typed_evidence
            else "agqa-query-grounder-v2-track-verified-formal-protocol-v1"
        ),
        "status": "FROZEN_BEFORE_FORMAL_SELECTION_VIDEO_ACQUISITION_GROUNDING_FALLBACK_OR_OUTCOMES",
        "claim": (
            "A source-only game-induced symbolic Harness selectively improves final AGQA "
            "query-object QA over neural-only and a capacity-matched source-permuted control "
            "when all arms share the same answer-blind target-native raw-video runtime."
        ),
        "claim_boundary": {
            "split": "official_balanced_train",
            "official_test_claim": False,
            "sota_claim": False,
            "target_native_grounder_training_overlap": (
                "TEMPURA SGDET and SlowFast are off-the-shelf components trained on "
                "Action Genome/Charades train data; the experiment measures matched-arm "
                "mechanism transfer, not held-out perception generalization."
            ),
            "qualification_cohort_is_transfer_evidence": False,
            "formal_cohort_must_be_video_and_task_disjoint_from_all_prior_raw_runtime": True,
            "candidate_threshold_was_selected_on_consumed_development_only": True,
        },
        "formal_cohort": {
            "source_split": "AGQA balanced train / Action Genome train",
            "structural_type": "query",
            "semantic_type": "object",
            "videos": int(videos),
            "tasks_per_video": int(tasks_per_video),
            "query_object_tasks": tasks,
            "selection_salt": str(selection_salt),
            "outcomes_unavailable_until_all_five_arm_predictions_are_immutable": True,
        },
        "qualified_grounder": {
            **frozen,
            "qualification_file": str(qualification_file),
            "qualification_file_sha256": _sha256(qualification_file),
            "qualification_report_sha256": qualification["report_sha256"],
            "qualification_protocol_file": str(qualification_protocol_file),
            "qualification_protocol_file_sha256": _sha256(qualification_protocol_file),
        },
        "semantic_parser": {
            "model": "target-native Flan-T5-small question-to-operator-free-semantics",
            "model_weights_sha256": "80bfb6921705ad61f852db4452669c6c55d4de4dc99e7b7db5103da1c8527964",
            "qualification_file_sha256": "0d04d4bafbae118172cd57cc4cb150be136fbc123d70fea3016648a063d662d0",
            "qualification_report_sha256": "dbb95ee168af674190d818ceefb24ca5a69a367576cf43867911c9a8f90a0e31",
            "formal_answers_or_functional_programs_visible": False,
        },
        "fallback_actor": {
            "model": "Qwen/Qwen3.5-9B", "temperature": 0, "thinking": False,
            "input": "shared question, operator-free semantic slots, and shared pixel-grounded events",
            "provider_calls": 0,
        },
        "source_harness": {
            **source,
            "target_data_read_during_induction": False,
            "unchanged_from_prior_clevrer_and_agqa_formals": True,
        },
        "arms": [
            "neural_only", "generic_scaffold", "source_permuted",
            "source_induced", "target_written_isomorphic",
        ],
        "shared_arm_contract": {
            "raw_video_frames": True, "grounding_receipts": True,
            "question_parser": True, "typed_executor": True,
            "fallback_actor": True, "only_symbolic_harness_varies": True,
            "source_permuted_uses_same_operator_inventory_and_permuted_source_composition_edges": True,
            "generic_scaffold_is_reported_as_a_ceiling_not_a_pass_gate": True,
        },
        "formal_gates": {
            "minimum_source_symbolic_commit_fraction": 0.2,
            "maximum_source_permuted_commit_fraction": 0.05,
            "source_accuracy_strictly_above_neural": True,
            "source_accuracy_strictly_above_matched_permuted": True,
            "source_vs_neural_exact_two_sided_p_maximum": 0.05,
            "source_vs_permuted_exact_two_sided_p_maximum": 0.05,
            "negative_transfer_loss_fraction_maximum": 0.05,
            "target_written_isomorphic_prediction_equivalence": 1.0,
            "shared_runtime_receipt_fraction": 1.0,
        },
        "secondary_target": {
            "overall_source_accuracy_strictly_above": 0.55,
            "is_formal_pass_gate": False,
        },
        "forbidden_runtime_inputs": [
            "task_answer", "official_stsg", "functional_program",
            "source_controller_in_grounder", "target_outcome",
        ],
        "failure_policy": (
            "Open outcomes once only after the pre-outcome receipt passes. Preserve a "
            "failed formal result; do not tune or retry on the same cohort."
        ),
        "implementation_file_sha256s": {
            path: _sha256(Path(path)) for path in dict.fromkeys((
                *IMPLEMENTATION_PATHS,
                *frozen["candidate_verifier"].get("component_paths", {}).values(),
            ))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--qualification-protocol", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=512)
    parser.add_argument("--tasks-per-video", type=int, default=2)
    parser.add_argument("--selection-salt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("formal protocol is immutable")
    qualification = json.loads(args.qualification.read_text())
    qualification_protocol = json.loads(args.qualification_protocol.read_text())
    protocol = build_protocol(
        qualification, qualification_protocol,
        qualification_file=args.qualification,
        qualification_protocol_file=args.qualification_protocol,
        videos=args.videos,
        tasks_per_video=args.tasks_per_video,
        selection_salt=args.selection_salt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": protocol["status"],
        "videos": protocol["formal_cohort"]["videos"],
        "tasks": protocol["formal_cohort"]["query_object_tasks"],
        "qualification_report_sha256": protocol["qualified_grounder"]["qualification_report_sha256"],
        "output_file_sha256": _sha256(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
