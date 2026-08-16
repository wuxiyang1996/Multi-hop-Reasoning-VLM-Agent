#!/usr/bin/env python3
"""Independently reconstruct and audit the validated Phase-2 WebShop V4 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase2_webshop_utility_v1 import (  # noqa: E402
    PASSED_STATUS,
    build_report,
    file_sha256,
    validate_manifest,
    validate_self_hash,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase2_webshop_utility_v4/manifest.json",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=REPO / "runs/phase2_webshop_utility_v4",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/phase2_webshop_utility_v4_audit.json",
    )
    args = parser.parse_args()

    manifest = _read(args.manifest)
    validate_manifest(manifest, repo=REPO)
    saved = _read(args.run_dir / "report.json")
    validate_self_hash(saved, "report_sha256")
    receipt_paths = sorted((args.run_dir / "receipts").glob("*.json"))
    receipts = [_read(path) for path in receipt_paths]
    rebuilt = build_report(
        manifest, receipts, cache_usage=saved.get("cache_usage"),
    )

    cohort_manifests = {
        version: _read(REPO / f"configs/phase2_webshop_utility_{version}/manifest.json")
        for version in ("v1", "v2", "v3", "v4")
    }
    for value in cohort_manifests.values():
        validate_manifest(value, repo=REPO)
    asin_sets = {
        version: {str(row["asin"]) for row in value["tasks"]}
        for version, value in cohort_manifests.items()
    }
    goal_sets = {
        version: {str(row["goal_sha256"]) for row in value["tasks"]}
        for version, value in cohort_manifests.items()
    }
    pairwise_disjoint = all(
        not asin_sets[left] & asin_sets[right]
        and not goal_sets[left] & goal_sets[right]
        for i, left in enumerate(asin_sets)
        for right in list(asin_sets)[i + 1:]
    )

    v2_report = _read(REPO / "runs/phase2_webshop_utility_v2/report.json")
    validate_self_hash(v2_report, "report_sha256")
    v2_failed_gates = sorted(
        key for key, value in v2_report["gates"].items() if not value
    )
    v3_failed_preflight = _read(
        REPO / "docs/results/phase2_webshop_utility_v3_preflight.json"
    )
    validate_self_hash(v3_failed_preflight, "preflight_sha256")
    v3_singlethread = _read(
        REPO / "docs/results/phase2_webshop_utility_v3_singlethread_diagnostic.json"
    )
    validate_self_hash(v3_singlethread, "preflight_sha256")
    v4_preflight = _read(
        REPO / "docs/results/phase2_webshop_utility_v4_preflight.json"
    )
    validate_self_hash(v4_preflight, "preflight_sha256")

    fallback_attempts = [
        attempt
        for receipt in receipts
        for step in receipt.get("steps") or ()
        for attempt in step.get("decision_attempts") or ()
        if attempt.get("deterministic_fallback") is not None
    ]
    gates = {
        "frozen_manifest_valid": True,
        "exactly_160_receipts": len(receipts) == 160,
        "independent_rebuild_byte_equivalent": rebuilt == saved,
        "saved_report_hash_reproduced": (
            rebuilt.get("report_sha256") == saved.get("report_sha256")
        ),
        "validated_status_reproduced": rebuilt.get("status") == PASSED_STATUS,
        "all_17_preregistered_gates_pass": (
            len(rebuilt.get("gates") or {}) == 17
            and all(rebuilt["gates"].values())
        ),
        "all_four_phase2_cohorts_pairwise_disjoint": pairwise_disjoint,
        "v2_failure_preserved_not_rewritten": v2_failed_gates == ["all_receipts_complete"],
        "v3_threaded_preflight_failure_preserved": (
            v3_failed_preflight.get("status") == "PHASE2_WEBSHOP_LIVE_PREFLIGHT_FAILED"
        ),
        "singlethread_diagnosis_reproduced_all_goals": (
            v3_singlethread.get("status")
            == "PHASE2_WEBSHOP_SINGLETHREAD_PREFLIGHT_PASSED"
        ),
        "v4_singlethread_preflight_passed": (
            v4_preflight.get("status")
            == "PHASE2_WEBSHOP_SINGLETHREAD_PREFLIGHT_PASSED"
        ),
        "singlethread_formal_server_frozen": (
            manifest.get("server_concurrency_policy", {}).get("threaded") is False
        ),
        "failclosed_policy_is_task_source_condition_blind": all(
            manifest["candidate_generation_failure_policy"].get(key) is False
            for key in (
                "task_or_goal_information_used",
                "source_information_used",
                "condition_information_used",
                "transport_errors_caught",
            )
        ),
        "all_formal_fallbacks_audited_safe_noop": all(
            row.get("deterministic_fallback", {}).get("provider_call") is False
            for row in fallback_attempts
        ),
        "six_distinct_source_artifact_hashes": (
            len({row["artifact_sha256"] for row in manifest["sources"].values()}) == 6
        ),
        "zero_historical_target_outcome_reuse": all(
            receipt.get("historical_target_outcome_reused") is False
            for receipt in receipts
        ),
        "exactly_one_target_reset_per_receipt": all(
            receipt.get("target_reset_or_sample_open_count") == 1
            for receipt in receipts
        ),
    }
    files = {
        "manifest": args.manifest,
        "report": args.run_dir / "report.json",
        "decision_cache": args.run_dir / "decision_cache.json",
        "formal_log": args.run_dir / "formal.log",
        "server_log": args.run_dir / "server.log",
        "v4_preflight": REPO / "docs/results/phase2_webshop_utility_v4_preflight.json",
    }
    body = {
        "schema_version": "phase2-webshop-utility-v4-independent-audit-v1",
        "status": (
            "PHASE2_WEBSHOP_V4_INDEPENDENT_AUDIT_PASSED"
            if all(gates.values())
            else "PHASE2_WEBSHOP_V4_INDEPENDENT_AUDIT_FAILED"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "report_sha256": saved["report_sha256"],
        "receipt_count": len(receipts),
        "receipt_sha256_aggregate": stable_hash([
            row["receipt_sha256"] for row in receipts
        ]),
        "formal_deterministic_fallback_attempts": len(fallback_attempts),
        "v2_failed_gates": v2_failed_gates,
        "v4_result": {
            "authentic_strict_successes": saved["summaries"][
                "authentic_search_automaton_plus_target"
            ]["strict_successes"],
            "raw_strict_successes": saved["summaries"]["raw_target_only"][
                "strict_successes"
            ],
            "authentic_vs_raw": saved["paired"]["raw_target_only"],
            "authentic_mean_reward": saved["summaries"][
                "authentic_search_automaton_plus_target"
            ]["mean_reward"],
            "raw_mean_reward": saved["summaries"]["raw_target_only"]["mean_reward"],
        },
        "gates": gates,
        "file_sha256": {
            name: file_sha256(path) for name, path in files.items()
        },
    }
    audit = body | {"audit_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite audit: {args.output}")
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
