#!/usr/bin/env python3
"""Freeze the compatibility-only V47 completion of V46 qualification."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402


PARENT_CONFIG = "configs/agqa2_interval_reliability_v46_qualification.json"
PARENT_PREREG = (
    "configs/agqa2_interval_reliability_v46_qualification_preregistration.json"
)
PARENT_ABORT = "docs/results/agqa2_interval_reliability_v46_runtime_abort.json"
PREREG = "configs/agqa2_interval_reliability_v47_qualification_preregistration.json"
CONFIG = "configs/agqa2_interval_reliability_v47_qualification.json"
EVALUATOR = "scripts/evaluate_agqa2_interval_reliability_v47_qualification.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, field: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop(field)
    if stable_hash(body) != claimed:
        raise ValueError(f"hash mismatch: {path}")
    return value


def main() -> None:
    abort_path = REPO_ROOT / PARENT_ABORT
    abort = _verified(abort_path, "result_sha256")
    if (
        abort["status"]
        != "AGQA2_INTERVAL_RELIABILITY_V46_QUALIFICATION_RUNTIME_INCOMPLETE"
        or abort["calibrated_prediction_loop_entered"]
        or abort["calibrated_metrics_externalized"]
    ):
        raise ValueError("V46 is not eligible for compatibility-only completion")
    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    parent_prereg = json.loads((REPO_ROOT / PARENT_PREREG).read_text())
    artifact_hash = parent["interval_reliability_calibration"]["artifact_sha256"]
    evaluator_path = REPO_ROOT / EVALUATOR
    protocol = deepcopy(parent_prereg["postground_evaluation_protocol"])
    protocol.update({
        "schema_version": "agqa2-interval-reliability-v47-qualification-protocol-v1",
        "evaluator_module_sha256": _sha256(evaluator_path),
        "v46_base_report_sha256": abort["base_report_sha256"],
        "compatibility_change": (
            "ADD_LEGACY_EVIDENCE_ALIAS_EQUAL_TO_FROZEN_V45_ARTIFACT_HASH_ONLY"
        ),
    })
    protocol_sha = stable_hash(protocol)
    prereg = deepcopy(parent_prereg)
    prereg.update({
        "schema_version": "agqa2-interval-reliability-v47-preregistration-v1",
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v47_status": "FROZEN_BEFORE_V47_QUALIFICATION_PREDICTION_EVALUATION",
        "claim_boundary": (
            "EXACT_HASHED_V46_BASE_RECEIPTS;ONE_LEGACY_EVIDENCE_ALIAS_"
            "ADDITION_ONLY;DEVELOPMENT_QUALIFICATION"
        ),
        "qualified_v33_development_report_sha256": artifact_hash,
        "v46_runtime_abort_summary": PARENT_ABORT,
        "v46_runtime_abort_summary_file_sha256": _sha256(abort_path),
        "v46_base_report_sha256": abort["base_report_sha256"],
        "postground_evaluation_protocol": protocol,
        "postground_evaluation_protocol_sha256": protocol_sha,
        "new_provider_calls_allowed": False,
        "confirmatory_claim_allowed": False,
    })
    prereg.pop("v46_status", None)
    prereg_path = REPO_ROOT / PREREG
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config = deepcopy(parent)
    config.update({
        "schema_version": "agqa2-interval-reliability-v47-qualification-config-v1",
        "status": "V47_COMPATIBILITY_ONLY_QUALIFICATION_COMPLETION",
        "claim_boundary": prereg["claim_boundary"],
        "preregistration": PREREG,
        "preregistration_file_sha256": _sha256(prereg_path),
        "report_version": "V47_QUALIFICATION_BASE_REUSE",
    })
    config["postground"].update({
        "evaluator_module": EVALUATOR,
        "evaluator_module_sha256": _sha256(evaluator_path),
        "evaluation_protocol_sha256": protocol_sha,
    })
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["v47_status"],
        "base_report_sha256": abort["base_report_sha256"],
        "target_grounder_sha256": config["postground"]["target_grounder_sha256"],
        "evaluation_protocol_sha256": protocol_sha,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
