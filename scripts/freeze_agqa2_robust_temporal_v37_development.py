#!/usr/bin/env python3
"""Freeze V37 bookkeeping-compatible evaluation of V36 base receipts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402


PARENT_CONFIG = "configs/agqa2_robust_temporal_v36_development.json"
PARENT_ABORT = "docs/results/agqa2_robust_temporal_v36_runtime_abort.json"
V33_SUMMARY = "docs/results/agqa2_robust_temporal_v33_development_summary.json"
MANIFEST = "configs/agqa2_robust_temporal_v37_development_manifest.json"
PREREG = "configs/agqa2_robust_temporal_v37_development_preregistration.json"
CONFIG = "configs/agqa2_robust_temporal_v37_development.json"
EVALUATOR = "scripts/evaluate_agqa2_robust_temporal_v37_development.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    abort_path = REPO_ROOT / PARENT_ABORT
    abort = json.loads(abort_path.read_text())
    abort_body = dict(abort)
    claimed_abort = abort_body.pop("result_sha256")
    if stable_hash(abort_body) != claimed_abort:
        raise ValueError("V36 abort hash mismatch")
    if (
        abort["status"]
        != "AGQA2_ROBUST_TEMPORAL_V36_DEVELOPMENT_RUNTIME_INCOMPLETE"
        or abort["prediction_freeze_loop_entered"]
        or abort["official_answer_field_accessed"]
        or abort["completed_runtime_receipts"] != 100
    ):
        raise ValueError("V36 is not eligible for bookkeeping-only evaluation")

    summary_path = REPO_ROOT / V33_SUMMARY
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "AGQA2_ROBUST_TEMPORAL_V33_DEVELOPMENT_QUALIFIED":
        raise ValueError("V33 evidence lineage is not qualified")

    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    old_manifest = json.loads((REPO_ROOT / parent["manifest"]).read_text())
    old_manifest_body = dict(old_manifest)
    old_manifest_sha = old_manifest_body.pop("manifest_sha256")
    if stable_hash(old_manifest_body) != old_manifest_sha:
        raise ValueError("V36 manifest hash mismatch")
    manifest_body = deepcopy(old_manifest_body)
    manifest_body.update({
        "schema_version": "agqa2-robust-temporal-v37-development-manifest-v1",
        "status": "FROZEN_V37_BOOKKEEPING_COMPATIBILITY_DEVELOPMENT",
        "claim_boundary": (
            "EXACT_HASHED_V36_BASE_RECEIPTS;NO_NEW_GROUNDING;ONLY_RESTORE_"
            "THE_PREEXISTING_V33_EVIDENCE_LINEAGE_KEY;DEVELOPMENT_ONLY"
        ),
        "parent_v36_manifest_sha256": old_manifest_sha,
        "parent_v36_base_report_sha256": abort["base_report_sha256"],
        "parent_v36_runtime_abort_result_sha256": abort["result_sha256"],
    })
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    manifest_path = REPO_ROOT / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    evaluator_path = REPO_ROOT / EVALUATOR
    config = deepcopy(parent)
    config.update({
        "schema_version": "agqa2-robust-temporal-v37-development-config-v1",
        "status": "FROZEN_V37_ROBUST_TEMPORAL_DEVELOPMENT",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": MANIFEST,
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "report_version": "V37_EVALUATION_ONLY",
    })
    protocol = deepcopy(parent["postground"])
    protocol.pop("evaluation_protocol_sha256", None)
    protocol.update({
        "schema_version": "agqa2-robust-temporal-v37-development-protocol-v1",
        "evaluator_module": EVALUATOR,
        "evaluator_module_sha256": _sha256(evaluator_path),
        "base_report_sha256": abort["base_report_sha256"],
        "new_grounding_or_provider_calls": False,
        "compatibility_key_only_change": True,
        "confirmatory_claim": False,
    })
    protocol_sha = stable_hash(protocol)
    prereg = {
        "schema_version": "agqa2-robust-temporal-v37-preregistration-v1",
        # Compatibility status consumed by the unchanged V34 core evaluator.
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v37_status": "FROZEN_BEFORE_V37_DEVELOPMENT_OUTCOME_EVALUATION",
        "claim_boundary": manifest["claim_boundary"],
        "development_manifest_sha256": manifest["manifest_sha256"],
        "parent_v36_abort_summary": PARENT_ABORT,
        "parent_v36_abort_summary_file_sha256": _sha256(abort_path),
        "qualified_v33_development_report_sha256": summary[
            "development_report_sha256"
        ],
        "qualified_v33_summary_file_sha256": _sha256(summary_path),
        "postground_evaluation_protocol_sha256": protocol_sha,
        "base_report_sha256": abort["base_report_sha256"],
        "new_provider_calls_allowed": False,
        "confirmatory_claim_allowed": False,
        "future_policy": (
            "ONLY_IF_V37_QUALIFIES_FREEZE_ONE_NEW_VIDEO_DISJOINT_FORMAL_RUN"
        ),
    }
    prereg_path = REPO_ROOT / PREREG
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": PREREG,
        "preregistration_file_sha256": _sha256(prereg_path),
    })
    config["postground"].update({
        "evaluator_module": EVALUATOR,
        "evaluator_module_sha256": _sha256(evaluator_path),
        "evaluation_protocol_sha256": protocol_sha,
    })
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["v37_status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "base_report_sha256": abort["base_report_sha256"],
        "postground_evaluation_protocol_sha256": protocol_sha,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
