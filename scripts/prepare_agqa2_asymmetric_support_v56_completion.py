#!/usr/bin/env python3
"""Prepare an administrative-only V56 dependency-alias completion config.

The frozen formal acquisition completed all 300 runtime receipts before the
legacy base reporter raised on a missing ``report_sha256`` alias in the V55
qualification summary.  This script preserves the original config and summary,
adds only that legacy alias in copies, and proves both semantic identities are
unchanged before the cached receipts are assembled into the formal report.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core,
    _grounder_semantic_core,
)


ORIGINAL_CONFIG = "configs/agqa2_asymmetric_support_v56_formal.json"
COMPLETION_CONFIG = "configs/agqa2_asymmetric_support_v56_formal_completion.json"
ORIGINAL_SUMMARY = (
    "docs/results/agqa2_asymmetric_support_v55_qualification_summary.json"
)
COMPLETION_SUMMARY = (
    "docs/results/agqa2_asymmetric_support_v55_qualification_summary_v56_compat.json"
)
QUALIFICATION_REPORT = (
    "runs/agqa2_asymmetric_support_v55_qualification/report.json"
)
AUDIT_RECEIPT = (
    "runs/agqa2_asymmetric_support_v56_formal/dependency_alias_completion.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = REPO_ROOT / "runs/agqa2_asymmetric_support_v56_formal/report.json"
    base_output = REPO_ROOT / "runs/agqa2_asymmetric_support_v56_formal/base_report.json"
    if output.exists() or base_output.exists():
        raise RuntimeError("V56 report already exists; completion is no longer legal")

    original_config_path = REPO_ROOT / ORIGINAL_CONFIG
    original_summary_path = REPO_ROOT / ORIGINAL_SUMMARY
    qualification_path = REPO_ROOT / QUALIFICATION_REPORT
    original = json.loads(original_config_path.read_text())
    summary = json.loads(original_summary_path.read_text())
    qualification = json.loads(qualification_path.read_text())
    if summary.get("report_sha256") is not None:
        raise ValueError("original summary unexpectedly already has the legacy alias")
    if summary["qualification_report_sha256"] != qualification["report_sha256"]:
        raise ValueError("V55 qualification report identity mismatch")
    if not summary["grounder_qualified"]:
        raise ValueError("V55 qualification summary is not qualified")

    completion_summary = deepcopy(summary)
    completion_summary.pop("summary_sha256")
    completion_summary.update({
        "schema_version": (
            "agqa2-asymmetric-support-v55-qualification-summary-v56-compat-v1"
        ),
        "report_sha256": qualification["report_sha256"],
        "compatibility_alias_only": True,
        "compatibility_alias": (
            "report_sha256=qualification_report_sha256"
        ),
        "created_after_v56_runtime_receipts_before_persisted_formal_report": True,
        "changes_prediction_grounding_execution_or_gates": False,
    })
    completion_summary["summary_sha256"] = stable_hash(completion_summary)
    completion_summary_path = REPO_ROOT / COMPLETION_SUMMARY
    completion_summary_path.write_text(
        json.dumps(completion_summary, indent=2, sort_keys=True) + "\n"
    )

    completion = deepcopy(original)
    completion["development_qualification_report"] = COMPLETION_SUMMARY
    completion["development_qualification_file_sha256"] = _sha256(
        completion_summary_path
    )
    completion_path = REPO_ROOT / COMPLETION_CONFIG
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")

    original_sources, _ = _load_sources(original)
    completion_sources, _ = _load_sources(completion)
    original_grounder = stable_hash(
        _grounder_semantic_core(original, original_sources)
    )
    completion_grounder = stable_hash(
        _grounder_semantic_core(completion, completion_sources)
    )
    original_protocol = stable_hash(_evaluation_protocol_core(original))
    completion_protocol = stable_hash(_evaluation_protocol_core(completion))
    if original_grounder != completion_grounder:
        raise AssertionError("dependency alias changed neural grounder identity")
    if original_protocol != completion_protocol:
        raise AssertionError("dependency alias changed evaluation protocol identity")
    allowed_changes = {
        "development_qualification_report",
        "development_qualification_file_sha256",
    }
    actual_changes = {
        key for key in set(original) | set(completion)
        if original.get(key) != completion.get(key)
    }
    if actual_changes != allowed_changes:
        raise AssertionError(f"unexpected completion config changes: {actual_changes}")

    runtime_dir = REPO_ROOT / "runs/agqa2_asymmetric_support_v56_formal/runtime_receipts"
    runtime_receipts = sorted(runtime_dir.glob("*.json"))
    if len(runtime_receipts) != 300:
        raise ValueError(f"expected 300 frozen runtime receipts, got {len(runtime_receipts)}")
    audit_core = {
        "schema_version": "agqa2-asymmetric-support-v56-dependency-alias-completion-v1",
        "status": "ADMINISTRATIVE_ALIAS_ONLY_BEFORE_PERSISTED_FORMAL_REPORT",
        "failure": "LEGACY_BASE_REPORTER_REQUIRED_REPORT_SHA256_ALIAS",
        "original_config": ORIGINAL_CONFIG,
        "original_config_file_sha256": _sha256(original_config_path),
        "completion_config": COMPLETION_CONFIG,
        "completion_config_file_sha256": _sha256(completion_path),
        "original_summary": ORIGINAL_SUMMARY,
        "original_summary_file_sha256": _sha256(original_summary_path),
        "completion_summary": COMPLETION_SUMMARY,
        "completion_summary_file_sha256": _sha256(completion_summary_path),
        "changed_config_fields": sorted(actual_changes),
        "runtime_receipt_count": len(runtime_receipts),
        "runtime_receipt_set_sha256": stable_hash(
            [_sha256(path) for path in runtime_receipts]
        ),
        "original_grounder_sha256": original_grounder,
        "completion_grounder_sha256": completion_grounder,
        "original_evaluation_protocol_sha256": original_protocol,
        "completion_evaluation_protocol_sha256": completion_protocol,
        "samples_prompts_models_predictions_gates_changed": False,
        "outcome_loop_started_before_reporter_failure": True,
        "persisted_formal_result_exists_before_completion": False,
    }
    audit = audit_core | {"audit_sha256": stable_hash(audit_core)}
    audit_path = REPO_ROOT / AUDIT_RECEIPT
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
