#!/usr/bin/env python3
"""Audit an outcome-unopened stable-track verifier transport replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--prior-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("shadow audit is immutable")
    protocol = json.loads(args.protocol.read_text())
    grounding = json.loads(args.grounding.read_text())
    coverage = json.loads(args.coverage.read_text())
    prior = json.loads(args.prior_qualification.read_text())
    spec = protocol["shadow_input"]
    method = protocol["frozen_method"]
    required = protocol["outcome_blind_shadow_gates"]
    rows = coverage.get("rows", ())
    every_commit_supported = all(
        not (row.get("source_commit") or row.get("permuted_commit"))
        or bool(row.get("candidate_supported"))
        for row in rows
    )
    checks = {
        "protocol_was_frozen": protocol.get("status") == "FROZEN_BEFORE_TRACK_VERIFIED_SHADOW_REPLAY",
        "parent_grounding_bound": (
            grounding.get("parent_grounding_file_sha256") == spec["grounding_file_sha256"]
            and grounding.get("parent_grounding_report_sha256") == spec["grounding_report_sha256"]
        ),
        "frozen_formula": grounding.get("candidate_verification", {}).get("formula") == method["formula"],
        "no_fitted_weights": grounding.get("candidate_verification", {}).get("fitted_weights") is False,
        "source_controller_not_read_by_verifier": (
            grounding.get("candidate_verification", {}).get("source_controller_read") is False
            and grounding.get("source_controller_read") is False
        ),
        "prior_outcomes_remain_unopened": (
            prior.get("qualification_outcomes_opened") is False
            and prior.get("target_outcome_read") is False
        ),
        "full_shadow_cohort": int(coverage.get("tasks", -1)) == int(spec["tasks"]) == len(rows),
        "candidate_supported_fraction": (
            float(coverage.get("candidate_supported_fraction", -1))
            >= float(required["minimum_candidate_supported_fraction"])
        ),
        "source_symbolic_commit_fraction": (
            float(coverage.get("source_commit_fraction", -1))
            >= float(required["minimum_source_symbolic_commit_fraction"])
        ),
        "source_permuted_commit_fraction": (
            float(coverage.get("permuted_commit_fraction", 2))
            <= float(required["maximum_source_permuted_commit_fraction"])
        ),
        "candidate_support_for_every_commit": (
            every_commit_supported
            and required["candidate_support_required_for_every_symbolic_commit"] is True
        ),
        "coverage_binds_grounding": (
            coverage.get("query_grounding_report_sha256") == grounding.get("report_sha256")
        ),
        "authority_safe": not any(grounding.get(key) for key in (
            "answer_read", "official_scene_graph_read", "functional_program_read",
            "source_controller_read", "target_outcome_read",
        )),
    }
    body = {
        "schema_version": "agqa-track-verified-shadow-audit-v1",
        "status": "TRACK_VERIFIED_SHADOW_PASS" if all(checks.values()) else "TRACK_VERIFIED_SHADOW_FAIL",
        "protocol_file_sha256": _sha256(args.protocol),
        "grounding_file_sha256": _sha256(args.grounding),
        "grounding_report_sha256": grounding.get("report_sha256"),
        "coverage_file_sha256": _sha256(args.coverage),
        "coverage_report_sha256": coverage.get("report_sha256"),
        "prior_qualification_file_sha256": _sha256(args.prior_qualification),
        "tasks": coverage.get("tasks"),
        "candidate_supported": coverage.get("candidate_supported"),
        "candidate_supported_fraction": coverage.get("candidate_supported_fraction"),
        "source_commits": coverage.get("source_commits"),
        "source_commit_fraction": coverage.get("source_commit_fraction"),
        "permuted_commits": coverage.get("permuted_commits"),
        "permuted_commit_fraction": coverage.get("permuted_commit_fraction"),
        "checks": checks,
        "qualification_evidence": False,
        "transfer_evidence": False,
        "answers_read": False,
        "target_outcome_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
