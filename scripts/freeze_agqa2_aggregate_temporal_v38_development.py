#!/usr/bin/env python3
"""Freeze the outcome-informed V38 method-selection audit."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_aggregate_temporal_transfer import (  # noqa: E402
    aggregate_target_grounder_sha256,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


PARENT_CONFIG = "configs/agqa2_robust_temporal_v37_development.json"
PARENT_REPORT = "runs/agqa2_robust_temporal_v37_development/report.json"
V33_SUMMARY = "docs/results/agqa2_robust_temporal_v33_development_summary.json"
ADAPTER = "src/motif_transfer/agqa_aggregate_temporal_transfer.py"
EVALUATOR = "scripts/evaluate_agqa2_aggregate_temporal_v38.py"
PREREG = "configs/agqa2_aggregate_temporal_v38_development_preregistration.json"
CONFIG = "configs/agqa2_aggregate_temporal_v38_development.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parent_report_path = REPO_ROOT / PARENT_REPORT
    parent_report = json.loads(parent_report_path.read_text())
    body = dict(parent_report)
    claimed = body.pop("report_sha256")
    if stable_hash(body) != claimed:
        raise ValueError("V37 report hash mismatch")
    if (
        parent_report["status"]
        != "AGQA2_ROBUST_TEMPORAL_V37_DEVELOPMENT_NOT_QUALIFIED"
        or parent_report["source_vs_target_native"]["wins"] != 6
        or parent_report["source_vs_target_native"]["losses"] != 0
    ):
        raise ValueError("unexpected V37 method-selection evidence")
    v33_path = REPO_ROOT / V33_SUMMARY
    v33 = json.loads(v33_path.read_text())
    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    adapter_path = REPO_ROOT / ADAPTER
    evaluator_path = REPO_ROOT / EVALUATOR
    normalization = parent["syntax_transport_normalization"]
    parent_grounder = parent["expected_grounder_sha256"]
    target_grounder = aggregate_target_grounder_sha256(
        parent_grounder_sha256=parent_grounder,
        adapter_module_sha256=_sha256(adapter_path),
        normalization_module_sha256=normalization[
            "normalization_module_sha256"
        ],
        acquisition_collector_sha256=parent["grounder"]["collector_sha256"],
    )
    gates = deepcopy(parent["postground"]["formal_gates"])
    protocol = {
        "schema_version": "agqa2-aggregate-temporal-v38-development-protocol-v1",
        "sample_count": 100,
        "primary_endpoint": "SOURCE_INDUCED_VS_TARGET_NATIVE_PAIRED_ACCURACY",
        "source_program_sha256": parent["postground"]["source_program_sha256"],
        "target_grounder_sha256": target_grounder,
        "target_executor_sha256": parent["postground"]["target_executor_sha256"],
        "adapter_module_sha256": _sha256(adapter_path),
        "evaluator_module_sha256": _sha256(evaluator_path),
        "binding_rule": (
            "BOTH_TYPED_ARGUMENTS_GROUNDED;AT_LEAST_THREE_TOTAL_VIEW_"
            "HYPOTHESES;ALL_CROSS_VIEW_PAIRS_STRICT_AND_CONSISTENT"
        ),
        "rule_derivation": (
            "SOURCE_RECURRENT_FLAG_APPLIES_TO_THE_ARITY_TWO_OPERATOR_NOT_"
            "INDEPENDENTLY_TO_EACH_ARGUMENT"
        ),
        "fallback": "PRESERVE_MATCHED_TARGET_NATIVE_DIRECT_ON_ABSTENTION",
        "formal_gates": gates,
        "method_selection_outcome_informed": True,
        "confirmatory_claim": False,
    }
    protocol_sha = stable_hash(protocol)
    prereg = {
        "schema_version": "agqa2-aggregate-temporal-v38-preregistration-v1",
        # Compatibility status consumed by the unchanged outcome-blind core.
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v38_status": "FROZEN_AFTER_V37_DEVELOPMENT_FOR_METHOD_SELECTION_ONLY",
        "claim_boundary": (
            "V38_REUSES_CONSUMED_V36_BASE_ROWS_AFTER_V37_OUTCOME_ACCESS;"
            "METHOD_SELECTION_ONLY;CANNOT_SUPPORT_A_CONFIRMATORY_CLAIM"
        ),
        "parent_v37_report_sha256": parent_report["report_sha256"],
        "parent_v37_report_file_sha256": _sha256(parent_report_path),
        "qualified_v33_development_report_sha256": v33[
            "development_report_sha256"
        ],
        "source_program_sha256": parent["postground"]["source_program_sha256"],
        "postground_evaluation_protocol": protocol,
        "postground_evaluation_protocol_sha256": protocol_sha,
        "formal_gates": gates,
        "method_selection_outcome_informed": True,
        "confirmatory_claim_allowed": False,
        "future_policy": (
            "IF_ALL_V38_METHOD_GATES_PASS_FREEZE_EXACT_RULE_ON_ONE_NEW_"
            "VIDEO_DISJOINT_V39_FORMAL_POOL"
        ),
    }
    prereg_path = REPO_ROOT / PREREG
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config = deepcopy(parent)
    config.update({
        "schema_version": "agqa2-aggregate-temporal-v38-development-config-v1",
        "status": "V38_OUTCOME_INFORMED_METHOD_SELECTION_ONLY",
        "claim_boundary": prereg["claim_boundary"],
        "preregistration": PREREG,
        "preregistration_file_sha256": _sha256(prereg_path),
        "report_version": "V38_METHOD_SELECTION",
    })
    config["postground"].update({
        "adapter_module": ADAPTER,
        "adapter_module_sha256": _sha256(adapter_path),
        "evaluator_module": EVALUATOR,
        "evaluator_module_sha256": _sha256(evaluator_path),
        "target_grounder_sha256": target_grounder,
        "evaluation_protocol_sha256": protocol_sha,
        "formal_gates": gates,
    })
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["v38_status"],
        "target_grounder_sha256": target_grounder,
        "evaluation_protocol_sha256": protocol_sha,
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
