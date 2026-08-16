#!/usr/bin/env python3
"""Independently rebuild and audit ALFWorld Phase-2 V3 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase2_alfworld_utility_v3 import (  # noqa: E402
    PASSED_STATUS, UNSUPPORTED_FAMILY, build_report, validate_manifest,
)
from motif_transfer.phase2_webshop_utility_v1 import file_sha256, validate_self_hash  # noqa: E402
from motif_transfer.webshop_search_automaton_v16 import AUTHENTIC, CONDITIONS, RAW  # noqa: E402


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "configs/phase2_alfworld_utility_v3/manifest.json")
    parser.add_argument("--run-dir", type=Path, default=REPO / "runs/phase2_alfworld_utility_v3")
    parser.add_argument("--output", type=Path, default=REPO / "docs/results/phase2_alfworld_utility_v3_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite audit: {args.output}")
    manifest = _read(args.manifest)
    validate_manifest(manifest, repo=REPO)
    saved = _read(args.run_dir / "report.json")
    validate_self_hash(saved, "report_sha256")
    receipt_paths = sorted((args.run_dir / "receipts").glob("*.json"))
    receipts = [_read(path) for path in receipt_paths]
    rebuilt = build_report(manifest, receipts)
    starts = [_read(path) for path in sorted((args.run_dir / "started").glob("*.json"))]
    v1 = _read(REPO / "configs/phase2_alfworld_utility_v1/manifest.json")
    v2 = _read(REPO / "configs/phase2_alfworld_utility_v2/manifest.json")
    v1_preflight = _read(REPO / "docs/results/phase2_alfworld_utility_v1_preflight.json")
    validate_self_hash(v1_preflight, "preflight_sha256")
    v2_report = _read(REPO / "runs/phase2_alfworld_utility_v2/report.json")
    validate_self_hash(v2_report, "report_sha256")
    selected = {row["target_identity"] for row in manifest["tasks"]}
    consumed = set(manifest["excluded_prior_task_ids"])
    indexed = {(row["target_identity"], row["condition"]): row for row in receipts}
    out_scope = [row for row in manifest["tasks"] if row["task_family"] == UNSUPPORTED_FAMILY]
    selector_exact = all(
        indexed[task["target_identity"], condition]["effective_condition"] == RAW
        and indexed[task["target_identity"], condition]["strict_success"]
            == indexed[task["target_identity"], RAW]["strict_success"]
        and [step["selected_action"] for step in indexed[task["target_identity"], condition]["steps"]]
            == [step["selected_action"] for step in indexed[task["target_identity"], RAW]["steps"]]
        for task in out_scope for condition in CONDITIONS
    )
    gates = {
        "manifest_valid": True,
        "exactly_375_receipts": len(receipts) == 375,
        "exactly_375_pre_reset_markers": len(starts) == 375,
        "independent_rebuild_byte_equivalent": rebuilt == saved,
        "saved_report_hash_reproduced": rebuilt.get("report_sha256") == saved.get("report_sha256"),
        "validated_status_reproduced": rebuilt.get("status") == PASSED_STATUS,
        "all_13_preregistered_gates_pass": len(rebuilt.get("gates") or {}) == 13 and all(rebuilt["gates"].values()),
        "v1_order_failure_preserved": v1_preflight.get("status") == "PHASE2_ALFWORLD_PREFLIGHT_FAILED",
        "v2_negative_result_preserved": v2_report.get("status") == "PHASE2_ALFWORLD_CAUSAL_UTILITY_NOT_VALIDATED",
        "v1_v2_v3_target_sets_disjoint": not selected.intersection({row["target_identity"] for row in v1["tasks"]} | {row["target_identity"] for row in v2["tasks"]}),
        "complete_140_task_partition": len(selected) == 75 and len(consumed) == 65 and not selected.intersection(consumed),
        "exactly_14_arity_abstentions": len(out_scope) == 14,
        "out_of_scope_conditions_exactly_match_raw": selector_exact,
        "zero_historical_outcome_reuse": all(row.get("historical_target_outcome_reused") is False for row in receipts),
        "one_reset_per_receipt": all(row.get("target_reset_or_sample_open_count") == 1 for row in receipts),
    }
    files = {
        "manifest": args.manifest,
        "report": args.run_dir / "report.json",
        "preflight": REPO / "docs/results/phase2_alfworld_utility_v3_preflight.json",
    }
    body = {
        "schema_version": "phase2-alfworld-selective-independent-audit-v3",
        "status": "PHASE2_ALFWORLD_V3_INDEPENDENT_AUDIT_PASSED" if all(gates.values()) else "PHASE2_ALFWORLD_V3_INDEPENDENT_AUDIT_FAILED",
        "manifest_sha256": manifest["manifest_sha256"],
        "report_sha256": saved["report_sha256"],
        "receipt_count": len(receipts),
        "receipt_sha256_aggregate": stable_hash([row["receipt_sha256"] for row in receipts]),
        "result": {
            "raw_successes": saved["summaries"][RAW]["strict_successes"],
            "authentic_successes": saved["summaries"][AUTHENTIC]["strict_successes"],
            "authentic_vs_raw": saved["paired"][RAW],
            "negative_transfer_rate": saved["discordant_negative_transfer_rate"],
        },
        "gates": gates,
        "file_sha256": {name: file_sha256(path) for name, path in files.items()},
    }
    audit = body | {"audit_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
