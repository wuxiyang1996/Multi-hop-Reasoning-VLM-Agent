#!/usr/bin/env python3
"""Record that V39 stopped before key loading or provider calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run = REPO_ROOT / "runs/agqa2_aggregate_temporal_v39_formal"
    artifacts = list(run.rglob("*.json")) if run.exists() else []
    if artifacts:
        raise RuntimeError(f"V39 has runtime JSON artifacts: {artifacts}")
    config_path = REPO_ROOT / "configs/agqa2_aggregate_temporal_v39_formal.json"
    prereg_path = REPO_ROOT / (
        "configs/agqa2_aggregate_temporal_v39_formal_preregistration.json"
    )
    manifest_path = REPO_ROOT / (
        "configs/agqa2_aggregate_temporal_v39_formal_manifest.json"
    )
    core = {
        "schema_version": "agqa2-aggregate-temporal-v39-preflight-abort-v1",
        "status": "AGQA2_AGGREGATE_TEMPORAL_V39_PREFLIGHT_ABORTED",
        "failure_class": "MISSING_DEVELOPMENT_QUALIFICATION_DEPENDENCY_FIELD",
        "failure_detail": "development_qualification_report",
        "provider_key_loaded": False,
        "provider_calls": 0,
        "base_report_created": False,
        "formal_outcome_accessed": False,
        "formal_video_model_exposure": False,
        "exact_video_pool_reusable": True,
        "config_file_sha256": _sha256(config_path),
        "preregistration_file_sha256": _sha256(prereg_path),
        "formal_manifest_file_sha256": _sha256(manifest_path),
        "next_authorized_use": (
            "REFREEZE_THE_EXACT_UNEXPOSED_POOL_WITH_THE_V38_COMPACT_"
            "DEVELOPMENT_QUALIFICATION_DEPENDENCY"
        ),
        "confirmatory_claim_allowed": False,
    }
    result = core | {"result_sha256": stable_hash(core)}
    output = REPO_ROOT / (
        "docs/results/agqa2_aggregate_temporal_v39_preflight_abort.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
