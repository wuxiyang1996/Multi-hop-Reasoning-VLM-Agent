#!/usr/bin/env python3
"""Freeze V15 replication on the exact uncalled V14 36-video pool."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core,
    _grounder_semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402


def _provider_cache_flags(value: Any) -> Iterator[bool]:
    if isinstance(value, Mapping):
        usage = value.get("usage")
        if (
            "cache_reused" in value
            and isinstance(usage, Mapping)
            and not usage.get("local_non_provider_call", False)
        ):
            yield bool(value["cache_reused"])
        for child in value.values():
            yield from _provider_cache_flags(child)
    elif isinstance(value, list):
        for child in value:
            yield from _provider_cache_flags(child)


def _development_summary() -> tuple[Path, dict[str, Any]]:
    report_path = REPO_ROOT / "runs/agqa2_active_grounding_v15_development/report.json"
    report = json.loads(report_path.read_text())
    if not report.get("grounder_qualified"):
        raise ValueError("V15 development did not qualify")
    receipt_paths = sorted((report_path.parent / "runtime_receipts").glob("*.json"))
    if len(receipt_paths) != report["acquisition_candidate_count"]:
        raise ValueError("V15 development runtime receipt count is incomplete")
    provider_flags: list[bool] = []
    for path in receipt_paths:
        receipt = json.loads(path.read_text())
        if not receipt.get("direct_cache_reused"):
            raise ValueError(f"V15 development direct call was not replayed: {path}")
        provider_flags.extend(_provider_cache_flags(receipt))
    if not provider_flags or not all(provider_flags):
        raise ValueError("V15 development made a non-replayed provider call")

    fields = (
        "status", "grounder_qualified", "grounder_sha256",
        "evaluation_protocol_sha256", "metrics", "controls",
        "qualification_gates", "reported_provider_cost_usd", "report_sha256",
    )
    core = {key: deepcopy(report[key]) for key in fields}
    core.update({
        "schema_version": "agqa2-active-grounding-v15-development-summary",
        "development_report_file_sha256": _sha256(report_path),
        "accepted_provider_receipts_replayed_from_v13": len(provider_flags),
        "new_provider_calls_during_v15_requalification": 0,
        "runtime_receipt_count": len(receipt_paths),
    })
    summary = core | {"summary_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_active_grounding_v15_development_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path, summary


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v15_replication"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V15 replication is already consumed")
    summary_path, development = _development_summary()

    abort_path = REPO_ROOT / "docs/results/agqa2_active_grounding_v14_preflight_abort.json"
    abort = json.loads(abort_path.read_text())
    if (
        abort["status"] != "V14_ABORTED_BEFORE_ANY_RUNTIME_OR_PROVIDER_CALL"
        or abort["provider_calls_started"]
        or abort["runtime_receipts_created"] != 0
        or abort["raw_video_decode_or_grounder_inspection_started"]
    ):
        raise ValueError("V14 pool is not eligible for zero-exposure reuse")

    parent_manifest_path = (
        REPO_ROOT / "configs/agqa2_active_grounding_v14_replication_manifest.json"
    )
    parent = _verified_json(parent_manifest_path, "manifest_sha256")
    manifest_core = {
        key: deepcopy(value) for key, value in parent.items()
        if key != "manifest_sha256"
    }
    manifest_core.update({
        "schema_version": "agqa2-active-grounding-manifest-v15-replication",
        "status": "FROZEN_V15_REUSED_UNCALLED_V14_POOL_BEFORE_ANY_NEURAL_CALL",
        "claim_boundary": (
            "V15_HASH_BOUNDARY_REQUALIFIED;EXACT_UNCALLED_V14_36_VIDEO_"
            "DISJOINT_POOL;30_ROW_OUTCOME_BLIND_REPLICATION;NOT_UNTOUCHED_METADATA"
        ),
        "selection_rule": "REUSE_EXACT_FROZEN_UNCALLED_V14_SELECTION",
        "parent_v14_manifest_sha256": parent["manifest_sha256"],
        "v14_zero_call_abort_file_sha256": _sha256(abort_path),
        "new_video_downloads": 0,
        "inherited_content_sealed_v14_video_count": parent["sample_count"],
        "prior_v15_raw_video_exposure": False,
    })
    manifest = manifest_core | {"manifest_sha256": stable_hash(manifest_core)}
    manifest_path = REPO_ROOT / "configs/agqa2_active_grounding_v15_replication_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    dev_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v15_development.json"
    ).read_text())
    v14_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v14_replication.json"
    ).read_text())
    config = deepcopy(dev_config)
    config.update({
        "schema_version": "agqa2-active-grounding-replication-config-v15",
        "status": "FROZEN_V15_30_ROW_REPLICATION",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": (
            "FROZEN_BEFORE_ANY_V15_REPLICATION_NEURAL_CALL"
        ),
        "development_qualification_report": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(summary_path),
        "runtime_selection": deepcopy(v14_config["runtime_selection"]),
        "qualification_gates": deepcopy(v14_config["qualification_gates"]),
        "report_version": "V15",
    })
    sources, _ = _load_sources(config)
    expected_grounder_sha256 = stable_hash(_grounder_semantic_core(config, sources))
    expected_evaluation_sha256 = stable_hash(_evaluation_protocol_core(config))
    if expected_grounder_sha256 != development["grounder_sha256"]:
        raise AssertionError("V15 replication grounder differs from qualified development")
    if expected_evaluation_sha256 == development["evaluation_protocol_sha256"]:
        raise AssertionError("scaled replication should have a distinct evaluation protocol")

    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v15-replication",
        "status": "FROZEN_BEFORE_ANY_V15_REPLICATION_NEURAL_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "qualified_grounder_sha256": expected_grounder_sha256,
        "replication_evaluation_protocol_sha256": expected_evaluation_sha256,
        "development_qualification_summary": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(summary_path),
        "v14_zero_call_abort": str(abort_path.relative_to(REPO_ROOT)),
        "v14_zero_call_abort_file_sha256": _sha256(abort_path),
        "parent_v14_manifest": str(parent_manifest_path.relative_to(REPO_ROOT)),
        "parent_v14_manifest_sha256": parent["manifest_sha256"],
        "replication_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "replication_manifest_sha256": manifest["manifest_sha256"],
        "runtime_selection": deepcopy(config["runtime_selection"]),
        "execution_calibration": deepcopy(config["execution_calibration"]),
        "acquisition": deepcopy(config["acquisition"]),
        "replication_gates": deepcopy(config["qualification_gates"]),
        "failure_policy": {
            "replication": "RUN_ONCE_ON_EXACT_UNCALLED_V14_POOL;NO_POST_RESULT_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v15_replication_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": expected_grounder_sha256,
        "expected_evaluation_protocol_sha256": expected_evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v15_replication.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_count": manifest["sample_count"],
        "evaluated_count": (
            len(config["qualification_gates"])
            and config["qualification_gates"]["required_valid_runtime_rows"]
        ),
        "grounder_sha256": expected_grounder_sha256,
        "evaluation_protocol_sha256": expected_evaluation_sha256,
        "config_file_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
