#!/usr/bin/env python3
"""Add the V34 evaluator's legacy evidence-hash alias to V56 copies only."""

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
DEPENDENCY_CONFIG = "configs/agqa2_asymmetric_support_v56_formal_completion.json"
COMPLETION_CONFIG = "configs/agqa2_asymmetric_support_v56_formal_completion_v2.json"
ORIGINAL_PREREG = "configs/agqa2_asymmetric_support_v56_formal_preregistration.json"
COMPLETION_PREREG = (
    "configs/agqa2_asymmetric_support_v56_formal_preregistration_v2_compat.json"
)
V55_PREREG = "configs/agqa2_asymmetric_support_v55_qualification_preregistration.json"
AUDIT_RECEIPT = (
    "runs/agqa2_asymmetric_support_v56_formal/evaluator_alias_completion.json"
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
    if output.exists():
        raise RuntimeError("V56 formal report already exists")
    if not base_output.exists():
        raise RuntimeError("V56 base report must exist before evaluator completion")

    original_path = REPO_ROOT / ORIGINAL_CONFIG
    dependency_path = REPO_ROOT / DEPENDENCY_CONFIG
    prereg_path = REPO_ROOT / ORIGINAL_PREREG
    v55_prereg_path = REPO_ROOT / V55_PREREG
    original = json.loads(original_path.read_text())
    dependency = json.loads(dependency_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    v55_prereg = json.loads(v55_prereg_path.read_text())
    legacy_key = "qualified_v33_development_report_sha256"
    if legacy_key in prereg:
        raise ValueError("original V56 prereg unexpectedly has the legacy alias")
    legacy_value = v55_prereg[legacy_key]
    if legacy_value != prereg["v54_training_artifact_sha256"]:
        raise ValueError("legacy evidence alias differs from frozen V54 artifact")

    completion_prereg = deepcopy(prereg)
    completion_prereg[legacy_key] = legacy_value
    completion_prereg.update({
        "v56_compatibility_alias_only": True,
        "v56_compatibility_alias": (
            "qualified_v33_development_report_sha256="
            "v54_training_artifact_sha256"
        ),
        "changes_prediction_grounding_execution_or_gates": False,
    })
    completion_prereg_path = REPO_ROOT / COMPLETION_PREREG
    completion_prereg_path.write_text(
        json.dumps(completion_prereg, indent=2, sort_keys=True) + "\n"
    )

    completion = deepcopy(dependency)
    completion["preregistration"] = COMPLETION_PREREG
    completion["preregistration_file_sha256"] = _sha256(completion_prereg_path)
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
        raise AssertionError("evaluator alias changed neural grounder identity")
    if original_protocol != completion_protocol:
        raise AssertionError("evaluator alias changed base evaluation identity")
    if prereg["postground_evaluation_protocol_sha256"] != completion_prereg[
        "postground_evaluation_protocol_sha256"
    ]:
        raise AssertionError("evaluator alias changed post-ground protocol identity")
    if prereg["formal_gates"] != completion_prereg["formal_gates"]:
        raise AssertionError("evaluator alias changed formal gates")

    changed_from_dependency = {
        key for key in set(dependency) | set(completion)
        if dependency.get(key) != completion.get(key)
    }
    if changed_from_dependency != {"preregistration", "preregistration_file_sha256"}:
        raise AssertionError(f"unexpected completion changes: {changed_from_dependency}")
    audit_core = {
        "schema_version": "agqa2-asymmetric-support-v56-evaluator-alias-completion-v1",
        "status": "ADMINISTRATIVE_EVALUATOR_ALIAS_ONLY_BEFORE_FORMAL_REPORT",
        "failure": "V34_EVALUATOR_REQUIRED_QUALIFIED_V33_REPORT_SHA256_ALIAS",
        "original_config_file_sha256": _sha256(original_path),
        "dependency_completion_config_file_sha256": _sha256(dependency_path),
        "completion_config": COMPLETION_CONFIG,
        "completion_config_file_sha256": _sha256(completion_path),
        "original_preregistration_file_sha256": _sha256(prereg_path),
        "completion_preregistration": COMPLETION_PREREG,
        "completion_preregistration_file_sha256": _sha256(completion_prereg_path),
        "legacy_alias_value": legacy_value,
        "base_report_file_sha256": _sha256(base_output),
        "runtime_receipt_count": len(list((base_output.parent / "runtime_receipts").glob("*.json"))),
        "original_grounder_sha256": original_grounder,
        "completion_grounder_sha256": completion_grounder,
        "original_base_evaluation_protocol_sha256": original_protocol,
        "completion_base_evaluation_protocol_sha256": completion_protocol,
        "postground_evaluation_protocol_sha256": prereg[
            "postground_evaluation_protocol_sha256"
        ],
        "samples_prompts_models_predictions_gates_changed": False,
        "persisted_formal_result_exists_before_completion": False,
    }
    audit = audit_core | {"audit_sha256": stable_hash(audit_core)}
    audit_path = REPO_ROOT / AUDIT_RECEIPT
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
