#!/usr/bin/env python3
"""Summarize strict V7/V8 video adaptation without opening confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "status": report["status"],
        "samples": report["samples"],
        "baseline_correct": report["baseline"]["correct"],
        "oracle_correct": report["oracle"]["correct"],
        "conditions_correct": {
            name: value["correct"] for name, value in report["conditions"].items()
        },
        "distinct_wrong_control_candidate_fraction": report.get(
            "distinct_wrong_control_candidate_fraction"
        ),
        "distinct_shuffled_control_candidate_fraction": report.get(
            "distinct_shuffled_control_candidate_fraction"
        ),
        "gates": report["gates"],
    }


def build_summary(paths: Mapping[str, Path], manifest: Path) -> dict[str, Any]:
    experiments = {name: _compact(path) for name, path in paths.items()}
    all_controls_identifiable = all(
        bool(row["gates"]["wrong_control_executes_distinct_observation"])
        and bool(row["gates"]["shuffled_control_executes_distinct_observation"])
        for row in experiments.values()
    )
    all_transfer_failed = all(
        str(row["status"]).endswith("_FAIL") for row in experiments.values()
    )
    return {
        "schema_version": "video-neurosymbolic-v7-v8-summary-v1",
        "status": (
            "VIDEO_HARNESS_REPAIRED_TRANSFER_NOT_VALIDATED"
            if all_controls_identifiable and all_transfer_failed
            else "VIDEO_V7_V8_AUDIT_REQUIRED"
        ),
        "scope": "VIDEO_ONLY; validated TIR/WebShop/ALFWorld untouched",
        "repairs": {
            "question_only_temporal_localization": True,
            "dense_resampling_inside_localized_window": True,
            "same_window_and_pixel_budget_across_forks": True,
            "independent_decoy_bind_and_identity_audit": True,
            "wrong_control_requires_distinct_observation": True,
            "failed_bind_transition": "NOOP_TO_BASELINE",
            "confirmation_baseline_collector_implemented_with_exact_adaptation_prompt": True,
        },
        "experiments": experiments,
        "strict_findings": {
            "nextqa_bind_relate": (
                "baseline=8/12, authentic=8/12, target-unbound=8/12"
            ),
            "star_bind_relate": (
                "baseline=6/16, authentic=6/16, target-unbound=11/16"
            ),
            "star_bind_mutate": (
                "baseline=6/16, authentic=7/16, target-unbound=12/16"
            ),
            "clevrer_bind_relate": (
                "baseline=0/12, authentic=4/12, target-unbound=4/12"
            ),
            "clevrer_bind_mutate": (
                "baseline=0/12, authentic=0/12, target-unbound=1/12"
            ),
        },
        "interpretation": (
            "The repaired target-native video programs have real perception and "
            "reasoning headroom, especially STAR target-unbound and CLEVRER RELATE, "
            "but the source-transferred BIND routing is not strictly better than "
            "the strong target-native control. Cross-domain success-rate transfer "
            "is therefore not validated by these adaptation receipts."
        ),
        "confirmation": {
            "opened": False,
            "authorized": False,
            "reason": "No raw or selective adaptation policy passed strict gates.",
            "frozen_manifest": str(manifest.resolve()),
            "frozen_manifest_sha256": _sha256(manifest),
            "reserve_remains_sealed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/video_v7_v8_adaptation_summary.json",
    )
    args = parser.parse_args()
    paths = {
        "nextqa_bind_relate_v7": REPO / "runs/typed_nextqa_v4b_gpt54/candidate_claim_v7_1_temporal_control_merged_adaptation_report.json",
        "star_bind_relate_v7": REPO / "runs/typed_star_v4b_gpt54/candidate_claim_v7_1_temporal_control_merged_adaptation_report.json",
        "clevrer_bind_relate_v7": REPO / "runs/typed_clevrer_v4b_gpt54/candidate_claim_v7_1_temporal_control_merged_adaptation_report.json",
        "star_bind_mutate_v8": REPO / "runs/typed_star_v4b_gpt54/candidate_mutation_v8_temporal_decoy_merged_adaptation_report.json",
        "clevrer_bind_mutate_v8": REPO / "runs/typed_clevrer_v4b_gpt54/candidate_mutation_v8_temporal_decoy_merged_adaptation_report.json",
    }
    manifest = REPO / "configs/three_video_v7_confirmation_splits.json"
    summary = build_summary(paths, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"], "output": str(args.output.resolve()),
    }))


if __name__ == "__main__":
    main()
