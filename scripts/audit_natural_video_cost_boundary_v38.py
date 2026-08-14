#!/usr/bin/env python3
"""Compact the completed video evidence into a cost-aware stop/go audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_costs(value: Any) -> list[float]:
    if isinstance(value, dict):
        costs = [float(value["cost"])] if isinstance(value.get("cost"), (int, float)) else []
        for key, child in value.items():
            if key != "cost":
                costs.extend(_all_costs(child))
        return costs
    if isinstance(value, list):
        costs = []
        for child in value:
            costs.extend(_all_costs(child))
        return costs
    return []


def _metric(report: dict, name: str) -> dict:
    row = report["condition_metrics_vs_matched_direct"][name]
    return {key: row[key] for key in (
        "n", "correct", "accuracy", "wins", "losses", "ties",
        "net_wins", "accuracy_delta", "exact_two_sided_p",
    )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-report",
        type=Path,
        default=REPO / "runs/natural_video_matched_cate_v37_formal/formal_report.json",
    )
    parser.add_argument(
        "--formal-receipts",
        type=Path,
        default=REPO / "runs/natural_video_matched_cate_v37_formal/formal_receipts.json",
    )
    parser.add_argument(
        "--clevrer-report",
        type=Path,
        default=REPO / "runs/sokoban_clevrer_proof_v14_formal/formal_report.json",
    )
    parser.add_argument(
        "--star-v27-summary",
        type=Path,
        default=REPO / "docs/results/star_interaction_grounding_factorial_v27_summary.json",
    )
    parser.add_argument(
        "--typed-v33-summary",
        type=Path,
        default=REPO / "docs/results/three_video_typed_grounding_v30_v33_summary.json",
    )
    parser.add_argument(
        "--video-holmes-summary",
        type=Path,
        default=REPO / "docs/results/active_video_tir_neurosymbolic_preflight_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs/results/natural_video_cost_boundary_v38_audit.json",
    )
    args = parser.parse_args()

    formal = json.loads(args.formal_report.read_text(encoding="utf-8"))
    receipts = json.loads(args.formal_receipts.read_text(encoding="utf-8"))
    clevrer = json.loads(args.clevrer_report.read_text(encoding="utf-8"))
    star = json.loads(args.star_v27_summary.read_text(encoding="utf-8"))
    typed = json.loads(args.typed_v33_summary.read_text(encoding="utf-8"))
    holmes = json.loads(args.video_holmes_summary.read_text(encoding="utf-8"))
    costs = _all_costs(receipts)

    artifact = {
        "schema_version": 1,
        "status": "NATURAL_VIDEO_TRANSFER_FORMALLY_FAILED_STOP_NEW_API_SPEND",
        "formal_natural_video": {
            "status": formal["status"],
            "questions": formal["rows"],
            "video_clusters": formal["video_clusters"],
            "same_model_and_frames_direct_vs_proof": formal[
                "same_model_and_frames_direct_vs_proof"
            ],
            "zero_sample_overlap_with_adaptation": formal[
                "zero_sample_overlap_with_v36_adaptation"
            ],
            "zero_video_overlap_with_adaptation": formal[
                "zero_video_overlap_with_v36_adaptation"
            ],
            "matched_direct": _metric(formal, "matched_direct"),
            "raw_typed_proof": _metric(formal, "raw_typed_proof"),
            "authentic_source_cate": _metric(formal, "source_proof_cate"),
            "inverted_source_control": _metric(formal, "inverted_source_contract"),
            "same_rate_marginal_control": _metric(formal, "same_rate_marginal"),
            "all_formal_gates_passed": formal["all_formal_gates_passed"],
            "failed_gate_count": sum(not value for value in formal["formal_gates"].values()),
            "estimated_provider_cost_usd": sum(costs),
            "provider_call_records": len(costs),
        },
        "mechanism_boundary": {
            "clevrer_structured_video": {
                "status": clevrer["status"],
                "samples": clevrer["samples"],
                "target_correct": clevrer["conditions"]["target_explicit_no_recovery"]["correct"],
                "authentic_correct": clevrer["conditions"][
                    "authentic_sokoban_proof_cate_recover"
                ]["correct"],
                "paired": clevrer["paired_authentic"]["target_explicit_no_recovery"],
            },
            "star_active_frame_navigation": {
                "status": star["status"],
                "direct_correct": star["conditions"]["active_direct"]["correct"],
                "source_correct": star["conditions"]["active_source"]["correct"],
                "paired": star["active_source_vs_active_direct"],
                "estimated_provider_cost_usd": star["mechanism_diagnostics"][
                    "estimated_total_cost_usd"
                ],
            },
            "natural_candidate_grounding": typed["natural_candidate_grounding"],
            "video_holmes_latest": {
                name: holmes["experiments"][name]
                for name in (
                    "active_video_wrapper_neurosymbolic_v7",
                    "active_video_wrapper_neurosymbolic_v9",
                    "active_video_wrapper_neurosymbolic_v10",
                    "active_video_wrapper_neurosymbolic_v11",
                )
            },
        },
        "what_is_not_missing": [
            "More question samples: the matched formal set already has 201 questions over 28 independent videos.",
            "More frames: direct and proof used the same 24 frames, and active STAR navigation was significantly worse.",
            "Generic candidate grounding: the typed natural-video grounder reached 33/35 correct covered candidates.",
            "A stronger target model: the matched Gemini direct baseline was already 150/201; NExT-QA was 57/65.",
            "More source gating of the current proof: authentic source lost one answer while its inversion gained two.",
        ],
        "what_is_missing": [
            "A target evidence operation with material intrinsic answer headroom; raw typed proof was only +1/201 (8W/7L, p=1).",
            "Source-specific applicability: authentic source must beat isomorphic, marginal, binding, and inverted controls.",
            "For natural video, a source causal object aligned to event dependency/counterfactual intervention rather than generic COMMIT->VERIFY.",
        ],
        "decision": {
            "new_video_provider_calls": "STOP",
            "increase_frame_count": "DO_NOT_DO",
            "video_holmes": "DEPRIORITIZE",
            "retain_clevrer": "YES_AS_VALIDATED_STRUCTURED_VIDEO_MECHANISM",
            "cheap_next_action": (
                "Use existing V37 receipts only for a leave-video-out error audit. Reopen natural-video "
                "collection only if a preregistered target evidence operator first shows stable intrinsic "
                "headroom and the authentic source predicts its wins over destructive controls."
            ),
        },
        "input_hashes": {
            "formal_report": file_sha256(args.formal_report),
            "formal_receipts": file_sha256(args.formal_receipts),
            "clevrer_report": file_sha256(args.clevrer_report),
            "star_v27_summary": file_sha256(args.star_v27_summary),
            "typed_v33_summary": file_sha256(args.typed_v33_summary),
            "video_holmes_summary": file_sha256(args.video_holmes_summary),
        },
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": artifact["status"],
        "formal_natural_video": artifact["formal_natural_video"],
        "decision": artifact["decision"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
