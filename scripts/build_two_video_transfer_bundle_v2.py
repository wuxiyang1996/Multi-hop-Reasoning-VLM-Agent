#!/usr/bin/env python3
"""Build the paper-facing fresh CLEVRER + AGQA Layer-B evidence bundle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_stable(value: dict, key: str) -> None:
    claimed = value[key]
    body = {name: item for name, item in value.items() if name != key}
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid embedded {key}")


def aggregate(rows: list[dict], public: dict, field: str) -> dict:
    metadata = {str(row["task_id"]): row for row in public["rows"]}
    buckets: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        key = str(metadata[str(row["task_id"])][field])
        buckets[key].update(
            tasks=1,
            neural_correct=int(row["correct"]["neural_only"]),
            source_correct=int(row["correct"]["source_induced"]),
            generic_correct=int(row["correct"]["generic_scaffold"]),
            source_commits=int(row["source_open_world_commit"]),
            wins=int(row["correct"]["source_induced"] and not row["correct"]["neural_only"]),
            losses=int(row["correct"]["neural_only"] and not row["correct"]["source_induced"]),
        )
    return {key: dict(value) for key, value in sorted(buckets.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clevrer-formal", type=Path, required=True)
    parser.add_argument("--clevrer-substitution", type=Path, required=True)
    parser.add_argument("--clevrer-taxonomy", type=Path, required=True)
    parser.add_argument("--agqa-formal", type=Path, required=True)
    parser.add_argument("--agqa-cohort", type=Path, required=True)
    parser.add_argument("--agqa-manifest", type=Path, required=True)
    parser.add_argument("--agqa-grounding", type=Path, required=True)
    parser.add_argument("--agqa-claims", type=Path, required=True)
    parser.add_argument("--agqa-fallback", type=Path, required=True)
    parser.add_argument("--agqa-preoutcome", type=Path, required=True)
    parser.add_argument("--agqa-runtime-freeze", type=Path, required=True)
    parser.add_argument("--anonymous-controller", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.verify_existing:
        raise FileExistsError("two-video V2 bundle is immutable")
    clevrer = load(args.clevrer_formal); csub = load(args.clevrer_substitution)
    ctax = load(args.clevrer_taxonomy); agqa = load(args.agqa_formal)
    cohort = load(args.agqa_cohort); manifest = load(args.agqa_manifest)
    grounding = load(args.agqa_grounding); claims = load(args.agqa_claims)
    fallback = load(args.agqa_fallback); preoutcome = load(args.agqa_preoutcome)
    freeze = load(args.agqa_runtime_freeze); controller = load(args.anonymous_controller)
    for value, key in ((clevrer, "report_sha256"), (csub, "report_sha256"),
                       (ctax, "report_sha256"), (agqa, "report_sha256"),
                       (grounding, "report_sha256"), (claims, "report_sha256"),
                       (fallback, "report_sha256"), (preoutcome, "receipt_sha256"),
                       (controller, "artifact_sha256")):
        verify_stable(value, key)
    if clevrer["status"] != "CLEVRER_FULL_LAYER_B_TRANSFER_VALIDATED":
        raise ValueError("CLEVRER formal did not pass")
    if csub["status"] != "CLEVRER_ANONYMOUS_HARNESS_SUBSTITUTION_VERIFIED":
        raise ValueError("CLEVRER anonymous substitution did not pass")
    if agqa["status"] != "LAYER_B_GATES_PASSED" or not all(agqa["gates"].values()):
        raise ValueError("AGQA powered formal did not pass")
    if preoutcome["status"] != "ALL_RUNTIME_ARTIFACTS_FROZEN_BEFORE_OUTCOMES":
        raise ValueError("AGQA pre-outcome freeze did not pass")
    if len({cohort["cohort_sha256"], manifest["cohort_sha256"],
            grounding["cohort_sha256"], claims["cohort_sha256"],
            fallback["cohort_sha256"], agqa["cohort_sha256"]}) != 1:
        raise ValueError("AGQA artifacts do not share a cohort")
    if agqa["anonymous_controller_sha256"] != controller["artifact_sha256"]:
        raise ValueError("AGQA did not execute the frozen anonymous controller")
    if csub["controller_artifact_sha256"] != controller["artifact_sha256"]:
        raise ValueError("CLEVRER did not execute the same anonymous controller")

    failure = Counter()
    for row in agqa["rows"]:
        neural = row["correct"]["neural_only"]; source = row["correct"]["source_induced"]
        generic = row["correct"]["generic_scaffold"]; commit = row["source_open_world_commit"]
        if source and not neural: failure["symbolic_recovery"] += 1
        if neural and not source: failure["negative_transfer"] += 1
        if commit and source and neural: failure["committed_both_correct"] += 1
        if commit and not source and not neural: failure["committed_shared_failure"] += 1
        if not commit and neural: failure["fallback_correct"] += 1
        if not commit and not neural and generic: failure["fallback_generic_headroom"] += 1
        if not commit and not neural and not generic: failure["fallback_shared_failure"] += 1

    n = int(agqa["summaries"]["source_induced"]["total"])
    neural = agqa["summaries"]["neural_only"]; source = agqa["summaries"]["source_induced"]
    generic = agqa["summaries"]["generic_scaffold"]
    comparison = agqa["comparisons"]["neural_only"]
    provider_errors = sum(row.get("provider_error") is not None for row in grounding["rows"])
    nonempty = sum(bool(row["grounding_receipt"]["events"]) for row in grounding["rows"])
    event_count = sum(len(row["grounding_receipt"]["events"]) for row in grounding["rows"])
    body = {
        "schema_version": "two-video-fresh-layer-b-transfer-bundle-v2",
        "status": "BOTH_VIDEO_BENCHMARKS_FRESH_LAYER_B_VALIDATED",
        "anonymous_controller_artifact_sha256": controller["artifact_sha256"],
        "claim": "One source-only anonymous game controller significantly improves final QA under benchmark-shared raw-video grounding on fresh CLEVRER and fresh AGQA broad reserves.",
        "clevrer": {
            "status": clevrer["status"], "tasks": clevrer["task_count"],
            "neural_correct": clevrer["metrics"]["neural_only"]["correct"],
            "source_correct": clevrer["metrics"]["source_induced"]["correct"],
            "generic_correct": clevrer["metrics"]["generic_symbolic"]["correct"],
            "source_vs_neural": clevrer["paired"]["source_vs_neural"],
            "negative_transfer_loss_fraction": clevrer["negative_transfer_loss_fraction"],
            "anonymous_substitution_exact_tasks": csub["tasks"],
            "formal_report_sha256": clevrer["report_sha256"],
        },
        "agqa2": {
            "status": agqa["status"], "split": "official_balanced_train",
            "freshness": "task-disjoint, parser-supervision-task-disjoint, and prior-raw-runtime-video-disjoint",
            "videos": len(cohort["video_receipts"]), "tasks": n,
            "neural_correct": neural["correct"], "neural_accuracy": neural["accuracy"],
            "source_correct": source["correct"], "source_accuracy": source["accuracy"],
            "gain_correct": source["correct"] - neural["correct"],
            "gain_percentage_points": 100 * (source["accuracy"] - neural["accuracy"]),
            "generic_ceiling_correct": generic["correct"],
            "generic_ceiling_accuracy": generic["accuracy"],
            "source_vs_neural": comparison,
            "negative_transfer_loss_fraction": comparison["losses"] / n,
            "source_symbolic_commits": source["symbolic_commits"],
            "source_symbolic_commit_fraction": source["symbolic_commits"] / n,
            "structural_breakdown": aggregate(agqa["rows"], cohort, "structural"),
            "semantic_breakdown": aggregate(agqa["rows"], cohort, "semantic"),
            "observable_failure_taxonomy": dict(failure),
            "grounding": {"nonempty_event_graphs": nonempty, "events": event_count,
                          "provider_errors_fail_closed": provider_errors,
                          "provider_cost_usd": grounding["reported_receipt_provider_cost_usd"]},
            "atomic_claim_provider_cost_usd": claims["reported_receipt_provider_cost_usd"],
            "total_receipt_provider_cost_usd": (
                grounding["reported_receipt_provider_cost_usd"]
                + claims["reported_receipt_provider_cost_usd"]
            ),
            "gates": agqa["gates"], "formal_report_sha256": agqa["report_sha256"],
            "preoutcome_receipt_sha256": preoutcome["receipt_sha256"],
        },
        "shared_invariants": {
            "same_anonymous_source_controller_across_benchmarks": True,
            "within_benchmark_all_arms_share_frames_grounder_parser_executor_fallback": True,
            "source_permuted_equals_neural": True,
            "target_written_isomorphic_equals_source": True,
            "generic_is_reported_ceiling_not_pass_gate": True,
            "target_outcomes_unavailable_before_final_evaluators": True,
        },
        "paper_boundaries": {
            "raw_video_QA_SOTA_claimed": False,
            "target_native_grounding_is_designer_selected": True,
            "universal_typed_VM_is_designer_specified": True,
            "source_provenance_identifiable_against_isomorphic_target_controller": False,
            "clevrer_predictive_counterfactual_source_commit": False,
            "agqa_official_test_claimed": False,
        },
        "artifact_file_sha256s": {
            str(path): file_sha(path) for path in (
                args.clevrer_formal, args.clevrer_substitution, args.clevrer_taxonomy,
                args.agqa_formal, args.agqa_cohort, args.agqa_manifest,
                args.agqa_grounding, args.agqa_claims, args.agqa_fallback,
                args.agqa_preoutcome, args.agqa_runtime_freeze, args.anonymous_controller,
            )
        },
    }
    body["bundle_sha256"] = stable_hash(body)
    if args.verify_existing:
        if not args.output.exists() or load(args.output) != body:
            raise ValueError("existing two-video V2 bundle does not reproduce exactly")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "clevrer_tasks": body["clevrer"]["tasks"],
        "agqa_tasks": n, "agqa_gain_correct": body["agqa2"]["gain_correct"],
        "agqa_p": comparison["exact_two_sided_p"], "bundle_sha256": body["bundle_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
