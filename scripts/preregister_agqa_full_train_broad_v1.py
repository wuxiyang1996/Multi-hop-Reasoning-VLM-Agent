#!/usr/bin/env python3
"""Freeze the AGQA full-train broad Layer-B protocol before runtime inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parser-qualification", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--anonymous-controller", type=Path, required=True)
    parser.add_argument("--tasks-per-video-per-stratum", type=int, default=3)
    parser.add_argument("--raw-grounder-model", default="qwen/qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--raw-grounder-frame-budget", type=int, default=24)
    parser.add_argument("--raw-grounder-qualification-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("AGQA broad preregistration is immutable")
    cohort, manifest = _load(args.cohort), _load(args.manifest)
    parser_q, source, controller = map(
        _load, (args.parser_qualification, args.source_capabilities, args.anonymous_controller),
    )
    if manifest.get("status") != "AGQA_FRESH_BROAD_RESERVE_FROZEN":
        raise ValueError("fresh cohort did not pass")
    if cohort.get("cohort_sha256") != manifest.get("cohort_sha256"):
        raise ValueError("cohort/manifest mismatch")
    if parser_q.get("status") != "SEMANTIC_PARSER_QUALIFIED":
        raise ValueError("question-only parser not qualified")
    if controller.get("status") != "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED":
        raise ValueError("anonymous source controller not qualified")
    scripts = [
        Path("scripts/run_agqa_layer_b_semantic_parser.py"),
        Path("scripts/collect_agqa_layer_b_qwen235_grounding.py"),
        Path("scripts/collect_agqa_layer_b_atomic_claims.py"),
        Path("scripts/collect_agqa_layer_b_shared_fallback.py"),
        Path("scripts/freeze_agqa_layer_b_epistemic_preoutcome.py"),
        Path("scripts/evaluate_agqa_layer_b_epistemic_five_arm.py"),
        Path("src/motif_transfer/anonymous_video_harness.py"),
    ]
    body = {
        "schema_version": "agqa-full-train-broad-layer-b-preregistration-v1",
        "status": "FROZEN_BEFORE_PARSER_GROUNDER_CLAIMS_FALLBACK_OR_OUTCOME",
        "claim": "source-only anonymous game controller selectively improves final AGQA QA on a task- and prior-raw-runtime-video-disjoint broad reserve",
        "cohort": {
            "cohort_sha256": cohort["cohort_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "tasks": len(cohort["rows"]),
            "videos": len(cohort["video_receipts"]),
            "structural_strata": ["choose", "compare", "logic", "query", "verify"],
            "tasks_per_video_per_stratum": args.tasks_per_video_per_stratum,
            "raw_video_runtime_disjoint": True,
            "semantic_parser_exact_task_supervision_overlap": 0,
            "official_split": "balanced_train",
        },
        "frozen_components": {
            "semantic_parser": {
                "model": "target-native Flan-T5-small question-to-operator-free-semantics",
                "qualification_report_sha256": parser_q["report_sha256"],
                "heldout_semantic_exact_rate": parser_q["metrics"]["semantic_exact_rate"],
                "formal_task_ids_seen_in_supervision": 0,
            },
            "raw_event_grounder": {
                "model": args.raw_grounder_model,
                "provider": "OpenRouter",
                "frame_budget": args.raw_grounder_frame_budget,
                "sampling": "uniform_full_video",
                "maximum_events": 24,
                "source_controller_visible": False,
            },
            "atomic_claim_grounder": {
                "model": "qwen/qwen3-vl-32b-instruct",
                "provider": "OpenRouter",
                "frame_budget": 96,
                "one_operator_free_proposition_per_call": True,
                "source_controller_visible": False,
            },
            "fallback_actor": {
                "model": "Qwen/Qwen3.5-9B",
                "thinking": False,
                "input": "shared question, semantic slots, and grounded events",
            },
            "source_vm_capabilities_sha256": source["artifact_sha256"],
            "anonymous_source_controller_sha256": controller["artifact_sha256"],
        },
        "arms": [
            "neural_only", "generic_scaffold", "source_permuted",
            "source_induced", "target_written_isomorphic",
        ],
        "arm_contract": {
            "all_arms_share_raw_videos_frames_grounding_parser_executor_fallback": True,
            "neural_only": "shared Qwen9B fallback only",
            "generic_scaffold": "eager target-native VM ceiling",
            "source_permuted": "capacity-matched failed source control; fail closed to shared fallback",
            "source_induced": "anonymous source attempt/commit-or-release controller over strict VM candidate",
            "target_written_isomorphic": "independently instantiated extensionally identical ceiling",
        },
        "formal_gates": {
            "source_accuracy_strictly_above_neural": True,
            "source_vs_neural_exact_two_sided_p_maximum": 0.05,
            "source_vs_permuted_exact_two_sided_p_maximum": 0.05,
            "negative_transfer_loss_fraction_maximum": 0.05,
            "target_written_isomorphic_prediction_equivalence": 1.0,
            "source_permuted_prediction_equivalence_to_neural": 1.0,
            "shared_grounder_receipt_fraction": 1.0,
            "minimum_source_symbolic_commit_fraction": 0.20,
            "generic_symbolic_is_reported_ceiling_not_a_pass_gate": True,
        },
        "failure_policy": "One immutable evaluator opening. Do not tune on this reserve or retry a failed deterministic result; any future run must use new task and raw-video-runtime-disjoint evidence.",
        "authority": {
            "source_or_threshold_selected_from_formal_outcome": False,
            "formal_answer_available_before_evaluator": False,
            "official_functional_program_available_at_runtime": False,
            "official_scene_graph_available_at_runtime": False,
            "target_structural_metadata_used_only_for_outcome-blind cohort balancing": True,
        },
        "input_file_sha256s": {
            "cohort": _sha(args.cohort),
            "manifest": _sha(args.manifest),
            "parser_qualification": _sha(args.parser_qualification),
            "source_capabilities": _sha(args.source_capabilities),
            "anonymous_controller": _sha(args.anonymous_controller),
            "raw_grounder_qualification_evidence": (
                _sha(args.raw_grounder_qualification_evidence)
                if args.raw_grounder_qualification_evidence else None
            ),
        },
        "implementation_file_sha256s": {str(path): _sha(path) for path in scripts},
    }
    body["preregistration_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": body["status"], "tasks": body["cohort"]["tasks"],
        "videos": body["cohort"]["videos"],
        "preregistration_sha256": body["preregistration_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
