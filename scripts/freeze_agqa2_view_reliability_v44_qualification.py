#!/usr/bin/env python3
"""Freeze the serialization-only V44 completion of V43 qualification."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_view_reliability_calibrator import (  # noqa: E402
    calibrated_target_grounder_sha256,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


PARENT_CONFIG = "configs/agqa2_view_reliability_v43_qualification.json"
PARENT_PREREG = "configs/agqa2_view_reliability_v43_qualification_preregistration.json"
PARENT_ABORT = "docs/results/agqa2_view_reliability_v43_runtime_abort.json"
PREREG = "configs/agqa2_view_reliability_v44_qualification_preregistration.json"
CONFIG = "configs/agqa2_view_reliability_v44_qualification.json"
AGGREGATE_ADAPTER = "src/motif_transfer/agqa_aggregate_temporal_transfer.py"
CALIBRATOR = "src/motif_transfer/agqa_view_reliability_calibrator.py"
EVALUATOR = "scripts/evaluate_agqa2_view_reliability_v44_qualification.py"


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
        != "AGQA2_VIEW_RELIABILITY_V43_QUALIFICATION_RUNTIME_INCOMPLETE"
        or abort["calibrated_prediction_loop_entered"]
        or abort["calibrated_metrics_externalized"]
    ):
        raise ValueError("V43 is not eligible for serialization-only completion")
    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    parent_prereg = json.loads((REPO_ROOT / PARENT_PREREG).read_text())
    artifact_path = REPO_ROOT / parent["view_reliability_calibration"]["artifact"]
    artifact = _verified(artifact_path, "artifact_sha256")
    aggregate_path = REPO_ROOT / AGGREGATE_ADAPTER
    calibrator_path = REPO_ROOT / CALIBRATOR
    evaluator_path = REPO_ROOT / EVALUATOR
    target_grounder = calibrated_target_grounder_sha256(
        parent_grounder_sha256=parent["expected_grounder_sha256"],
        aggregate_adapter_sha256=_sha256(aggregate_path),
        normalization_module_sha256=parent[
            "syntax_transport_normalization"
        ]["normalization_module_sha256"],
        acquisition_collector_sha256=parent["grounder"]["collector_sha256"],
        calibrator_module_sha256=_sha256(calibrator_path),
        calibration_artifact_sha256=artifact["artifact_sha256"],
    )
    protocol = deepcopy(parent_prereg["postground_evaluation_protocol"])
    protocol.update({
        "schema_version": "agqa2-view-reliability-v44-qualification-protocol-v1",
        "target_grounder_sha256": target_grounder,
        "calibrator_module_sha256": _sha256(calibrator_path),
        "evaluator_module_sha256": _sha256(evaluator_path),
        "v43_base_report_sha256": abort["base_report_sha256"],
        "serialization_change": "JSON_LIST_TO_IDENTICAL_TUPLE_ONLY",
    })
    protocol_sha = stable_hash(protocol)
    prereg = deepcopy(parent_prereg)
    prereg.update({
        "schema_version": "agqa2-view-reliability-v44-preregistration-v1",
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v44_status": "FROZEN_BEFORE_V44_QUALIFICATION_PREDICTION_EVALUATION",
        "claim_boundary": (
            "EXACT_HASHED_V43_BASE_RECEIPTS;JSON_LIST_TO_IDENTICAL_TUPLE_"
            "CANONICALIZATION_ONLY;DEVELOPMENT_QUALIFICATION"
        ),
        "v43_runtime_abort_summary": PARENT_ABORT,
        "v43_runtime_abort_summary_file_sha256": _sha256(abort_path),
        "v43_base_report_sha256": abort["base_report_sha256"],
        "postground_evaluation_protocol": protocol,
        "postground_evaluation_protocol_sha256": protocol_sha,
        "new_provider_calls_allowed": False,
        "confirmatory_claim_allowed": False,
    })
    prereg.pop("v43_status", None)
    prereg_path = REPO_ROOT / PREREG
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config = deepcopy(parent)
    config.update({
        "schema_version": "agqa2-view-reliability-v44-qualification-config-v1",
        "status": "V44_SERIALIZATION_ONLY_QUALIFICATION_COMPLETION",
        "claim_boundary": prereg["claim_boundary"],
        "preregistration": PREREG,
        "preregistration_file_sha256": _sha256(prereg_path),
        "report_version": "V44_QUALIFICATION_BASE_REUSE",
    })
    config["view_reliability_calibration"].update({
        "module_sha256": _sha256(calibrator_path),
    })
    config["postground"].update({
        "adapter_module": AGGREGATE_ADAPTER,
        "adapter_module_sha256": _sha256(aggregate_path),
        "evaluator_module": EVALUATOR,
        "evaluator_module_sha256": _sha256(evaluator_path),
        "target_grounder_sha256": target_grounder,
        "evaluation_protocol_sha256": protocol_sha,
    })
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["v44_status"],
        "base_report_sha256": abort["base_report_sha256"],
        "target_grounder_sha256": target_grounder,
        "evaluation_protocol_sha256": protocol_sha,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
