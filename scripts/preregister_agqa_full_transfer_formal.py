#!/usr/bin/env python3
"""Authorize the one-shot AGQA formal join from frozen, outcome-blind artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--controls-manifest", type=Path, required=True)
    parser.add_argument("--source-capabilities", type=Path, required=True)
    parser.add_argument("--compiler-qualification", type=Path, required=True)
    parser.add_argument("--executor-development", type=Path, required=True)
    parser.add_argument("--compiler-runtime", type=Path, required=True)
    parser.add_argument("--neural-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preregistration is immutable")

    cohort = load(args.cohort_manifest)
    controls = load(args.controls_manifest)
    source = load(args.source_capabilities)
    compiler = load(args.compiler_qualification)
    executor = load(args.executor_development)
    compiler_runtime = load(args.compiler_runtime)
    neural_runtime = load(args.neural_runtime)

    compiler_metrics = compiler["metrics"]
    executor_metrics = executor["metrics"]
    runtime_rows = compiler_runtime["rows"]
    admitted = sum(row["program_admission"]["status"] == "COMPILED" for row in runtime_rows)
    runtime_admission_rate = admitted / len(runtime_rows)
    checks = {
        "cohort_frozen_before_outcome_or_program": cohort["status"] == "FROZEN_BEFORE_OUTCOME_OR_PROGRAM_ACCESS",
        "controls_frozen_before_formal_outcome": controls["status"] == "SIX_ARMS_FROZEN" and not controls["formal_outcomes_read"],
        "source_capabilities_induced": source["status"] == "SOURCE_CAPABILITIES_INDUCED",
        "compiler_full_heldout_exact_at_least_0_98": compiler_metrics["program_exact_rate"] >= .98,
        "compiler_full_heldout_admission_at_least_0_995": compiler_metrics["source_admission_rate"] >= .995,
        "executor_development_coverage_at_least_0_90": executor_metrics["coverage"] >= .90,
        "executor_development_conditional_accuracy_at_least_0_995": executor_metrics["conditional_accuracy"] >= .995,
        "fresh_compiler_runtime_admission_at_least_0_995": runtime_admission_rate >= .995,
        "fresh_runtime_cohort_alignment": (
            compiler_runtime["cohort_sha256"] == cohort["cohort_sha256"]
            and neural_runtime["cohort_sha256"] == cohort["cohort_sha256"]
            and len(compiler_runtime["rows"]) == cohort["questions"]
            and len(neural_runtime["rows"]) == cohort["questions"]
        ),
        "fresh_runtimes_outcome_blind": (
            not compiler_runtime["answer_read"] and not compiler_runtime["oracle_program_read"]
            and not neural_runtime["answer_read"] and not neural_runtime["oracle_program_read"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal authorization failed: {checks}")

    body = {
        "schema_version": "agqa-full-transfer-preregistration-v1",
        "status": "FORMAL_AUTHORIZED",
        "claim_boundary": "FRESH_QUESTIONS_AND_OUTCOMES;VIDEOS_AND_OFFICIAL_STSG_REUSED",
        "primary_hypotheses": {
            "source_beats_neural": "McNemar exact p<0.05 and net wins>0",
            "source_beats_source_permuted": "McNemar exact p<0.05 and net wins>0",
            "source_beats_generic_scaffold": "McNemar exact p<0.05 and net wins>0",
            "source_matches_target_written_isomorphic": "absolute accuracy gap<=1 percentage point",
            "negative_transfer": "source losses versus neural<=max(5,1% of reserve)",
            "family_coverage": "all structural families represented",
        },
        "qualification_checks": checks,
        "qualification_metrics": {
            "compiler_program_exact_rate": compiler_metrics["program_exact_rate"],
            "compiler_source_admission_rate": compiler_metrics["source_admission_rate"],
            "executor_coverage": executor_metrics["coverage"],
            "executor_conditional_accuracy": executor_metrics["conditional_accuracy"],
            "fresh_compiler_admission_rate": runtime_admission_rate,
        },
        "artifact_hashes": {
            "cohort_manifest_sha256": cohort["manifest_sha256"],
            "controls_manifest_sha256": controls["manifest_sha256"],
            "source_capabilities_sha256": source["artifact_sha256"],
            "compiler_qualification_sha256": compiler["report_sha256"],
            "executor_development_sha256": executor["report_sha256"],
            "compiler_runtime_sha256": compiler_runtime["runtime_sha256"],
            "neural_runtime_sha256": neural_runtime["runtime_sha256"],
            "compiler_runtime_file_sha256": file_sha256(args.compiler_runtime),
            "neural_runtime_file_sha256": file_sha256(args.neural_runtime),
        },
        "formal_outcomes_read": False,
    }
    body["preregistration_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
