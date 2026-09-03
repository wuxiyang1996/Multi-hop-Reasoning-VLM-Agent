#!/usr/bin/env python3
"""Refreeze the unexposed V39 pool with its V38 dependency for V40."""

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


PARENT_CONFIG = "configs/agqa2_aggregate_temporal_v39_formal.json"
PARENT_MANIFEST = "configs/agqa2_aggregate_temporal_v39_formal_manifest.json"
PARENT_PREREG = "configs/agqa2_aggregate_temporal_v39_formal_preregistration.json"
PARENT_ABORT = "docs/results/agqa2_aggregate_temporal_v39_preflight_abort.json"
DEVELOPMENT_SUMMARY = (
    "docs/results/agqa2_aggregate_temporal_v38_development_summary.json"
)
MANIFEST = "configs/agqa2_aggregate_temporal_v40_formal_manifest.json"
PREREG = "configs/agqa2_aggregate_temporal_v40_formal_preregistration.json"
CONFIG = "configs/agqa2_aggregate_temporal_v40_formal.json"
EVALUATOR = "scripts/collect_agqa2_aggregate_temporal_v40_formal.py"


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
    run_root = REPO_ROOT / "runs/agqa2_aggregate_temporal_v40_formal"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V40 formal run already has runtime artifacts")
    abort_path = REPO_ROOT / PARENT_ABORT
    abort = _verified(abort_path, "result_sha256")
    if (
        abort["status"]
        != "AGQA2_AGGREGATE_TEMPORAL_V39_PREFLIGHT_ABORTED"
        or abort["provider_calls"] != 0
        or abort["formal_video_model_exposure"]
        or not abort["exact_video_pool_reusable"]
    ):
        raise ValueError("V39 pool is not eligible for exact V40 reuse")
    summary_path = REPO_ROOT / DEVELOPMENT_SUMMARY
    summary = _verified(summary_path, "summary_sha256")
    if not summary["grounder_qualified"]:
        raise ValueError("V38 compact development dependency is not qualified")

    parent_manifest_path = REPO_ROOT / PARENT_MANIFEST
    parent_manifest = _verified(parent_manifest_path, "manifest_sha256")
    manifest_body = deepcopy(parent_manifest)
    parent_manifest_sha = manifest_body.pop("manifest_sha256")
    manifest_body.update({
        "schema_version": "agqa2-aggregate-temporal-manifest-v40-formal",
        "status": "FROZEN_V40_RAW_VIDEO_UNEXPOSED_BEFORE_FORMAL_CALLS",
        "claim_boundary": (
            "EXACT_V39_PREFLIGHT_ABORTED_UNEXPOSED_ONE_HUNDRED_VIDEO_POOL;"
            "V38_OPERATOR_RECURRENCE_RULE;V40_FORMAL"
        ),
        "parent_v39_manifest_sha256": parent_manifest_sha,
        "parent_v39_preflight_abort_result_sha256": abort["result_sha256"],
        "prior_v39_provider_calls": 0,
        "prior_v39_neural_video_exposure": False,
    })
    manifest = manifest_body | {"manifest_sha256": stable_hash(manifest_body)}
    manifest_path = REPO_ROOT / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    parent_config = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    parent_prereg = json.loads((REPO_ROOT / PARENT_PREREG).read_text())
    evaluator_path = REPO_ROOT / EVALUATOR
    config = deepcopy(parent_config)
    config.update({
        "schema_version": "agqa2-aggregate-temporal-v40-formal-config-v1",
        "status": "FROZEN_V40_AGGREGATE_TEMPORAL_FORMAL",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": MANIFEST,
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "development_qualification_report": DEVELOPMENT_SUMMARY,
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V40_BASE",
    })
    sources, _ = _load_sources(config)
    parent_grounder = stable_hash(_grounder_semantic_core(config, sources))
    if parent_grounder != summary["grounder_sha256"]:
        raise AssertionError("V40 reserve differs from V38 qualified acquisition")
    base_evaluation = stable_hash(_evaluation_protocol_core(config))

    protocol = deepcopy(parent_prereg["postground_evaluation_protocol"])
    protocol.update({
        "schema_version": "agqa2-aggregate-temporal-v40-formal-protocol-v1",
        "evaluator_module_sha256": _sha256(evaluator_path),
        "v39_preflight_abort_result_sha256": abort["result_sha256"],
        "exact_unexposed_pool_reused": True,
    })
    protocol_sha = stable_hash(protocol)
    prereg = deepcopy(parent_prereg)
    prereg.update({
        "schema_version": "agqa2-aggregate-temporal-v40-preregistration-v1",
        # Compatibility status consumed by the unchanged outcome-blind core.
        "status": "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "v40_status": "FROZEN_BEFORE_ANY_V40_FORMAL_PROVIDER_OR_OUTCOME_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "v39_preflight_abort_summary": PARENT_ABORT,
        "v39_preflight_abort_summary_file_sha256": _sha256(abort_path),
        "v38_development_qualification_summary": DEVELOPMENT_SUMMARY,
        "v38_development_qualification_summary_file_sha256": _sha256(
            summary_path
        ),
        "base_evaluation_protocol_sha256": base_evaluation,
        "postground_evaluation_protocol": protocol,
        "postground_evaluation_protocol_sha256": protocol_sha,
        "exact_v39_video_pool_reused": True,
        "prior_v39_provider_calls": 0,
    })
    prereg.pop("v39_status", None)
    prereg_path = REPO_ROOT / PREREG
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": PREREG,
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": parent_grounder,
        "expected_evaluation_protocol_sha256": base_evaluation,
    })
    config["postground"].update({
        "evaluator_module": EVALUATOR,
        "evaluator_module_sha256": _sha256(evaluator_path),
        "evaluation_protocol_sha256": protocol_sha,
    })
    config_path = REPO_ROOT / CONFIG
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": prereg["v40_status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": len(manifest["samples"]),
        "parent_grounder_sha256": parent_grounder,
        "target_grounder_sha256": config["postground"][
            "target_grounder_sha256"
        ],
        "evaluation_protocol_sha256": protocol_sha,
        "provider_cost_cap_usd": prereg["cost_projection"][
            "frozen_cap_usd"
        ],
        "config_file_sha256": _sha256(config_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
