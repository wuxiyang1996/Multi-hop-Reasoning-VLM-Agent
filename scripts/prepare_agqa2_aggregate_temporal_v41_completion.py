#!/usr/bin/env python3
"""Prepare a schema-alias-only completion of the V40 formal runtime."""

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


PARENT_CONFIG = "configs/agqa2_aggregate_temporal_v40_formal.json"
PARENT_SUMMARY = (
    "docs/results/agqa2_aggregate_temporal_v38_development_summary.json"
)
COMPAT_SUMMARY = (
    "docs/results/agqa2_aggregate_temporal_v38_development_dependency_v2.json"
)
V40_ABORT = "docs/results/agqa2_aggregate_temporal_v40_runtime_abort.json"
CONFIG = "configs/agqa2_aggregate_temporal_v41_completion.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, hash_field: str) -> dict:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = body.pop(hash_field)
    if stable_hash(body) != claimed:
        raise ValueError(f"hash mismatch: {path}")
    return value


def main() -> None:
    abort_path = REPO_ROOT / V40_ABORT
    abort = _verified(abort_path, "result_sha256")
    if (
        abort["status"]
        != "AGQA2_AGGREGATE_TEMPORAL_V40_RUNTIME_ASSEMBLY_INCOMPLETE"
        or abort["completed_runtime_receipts"] != 100
        or abort["source_prediction_freeze_loop_entered"]
        or abort["formal_metrics_externalized"]
    ):
        raise ValueError("V40 is not eligible for deterministic completion")
    parent_summary_path = REPO_ROOT / PARENT_SUMMARY
    parent_summary = _verified(parent_summary_path, "summary_sha256")
    summary_body = deepcopy(parent_summary)
    parent_summary_sha = summary_body.pop("summary_sha256")
    summary_body.update({
        "schema_version": (
            "agqa2-aggregate-temporal-v38-development-dependency-v2"
        ),
        # Legacy reserve collector alias; same value, no new evidence.
        "report_sha256": parent_summary["development_report_sha256"],
        "parent_summary_sha256": parent_summary_sha,
        "schema_compatibility_alias_only": True,
        "outcome_metric_or_decision_change": False,
    })
    summary = summary_body | {"summary_sha256": stable_hash(summary_body)}
    summary_path = REPO_ROOT / COMPAT_SUMMARY
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    parent = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    config = deepcopy(parent)
    config.update({
        "schema_version": "agqa2-aggregate-temporal-v41-completion-config-v1",
        "status": "V41_SCHEMA_ALIAS_ONLY_COMPLETION_OF_V40_FORMAL",
        "development_qualification_report": COMPAT_SUMMARY,
        "development_qualification_file_sha256": _sha256(summary_path),
        "runtime_completion": {
            "parent_v40_abort_summary": V40_ABORT,
            "parent_v40_abort_summary_file_sha256": _sha256(abort_path),
            "reuses_exact_v40_manifest": True,
            "reuses_exact_v40_preregistration": True,
            "reuses_exact_v40_evaluator": True,
            "reuses_exact_v40_method_gates_and_predictions": True,
            "only_change": (
                "ADD_REPORT_SHA256_ALIAS_EQUAL_TO_EXISTING_DEVELOPMENT_"
                "REPORT_SHA256_IN_DEPENDENCY"
            ),
            "formal_metrics_externalized_before_change": False,
        },
    })
    sources, _ = _load_sources(config)
    grounder = stable_hash(_grounder_semantic_core(config, sources))
    evaluation = stable_hash(_evaluation_protocol_core(config))
    if grounder != parent["expected_grounder_sha256"]:
        raise AssertionError("schema alias changed the V40 grounder")
    if evaluation != parent["expected_evaluation_protocol_sha256"]:
        raise AssertionError("schema alias changed the V40 evaluation protocol")
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": config["status"],
        "grounder_sha256": grounder,
        "evaluation_protocol_sha256": evaluation,
        "postground_evaluation_protocol_sha256": config["postground"][
            "evaluation_protocol_sha256"
        ],
        "compatibility_summary_sha256": summary["summary_sha256"],
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
