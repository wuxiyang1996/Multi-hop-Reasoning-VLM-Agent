#!/usr/bin/env python3
"""Audit the portable compact evidence for the formal CLEVRER V14 route."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "configs/video_event_graph_v14_release.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(value: bytes) -> dict[str, Any]:
    payload = json.loads(value.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def audit(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundled: dict[str, dict[str, Any]] = {}
    for row in manifest["bundled_artifacts"]:
        path = REPO / row["path"]
        compressed = path.read_bytes()
        if _sha256_bytes(compressed) != row["gzip_sha256"]:
            raise ValueError(f"gzip hash mismatch: {path}")
        raw = gzip.decompress(compressed)
        if _sha256_bytes(raw) != row["uncompressed_sha256"]:
            raise ValueError(f"uncompressed hash mismatch: {path}")
        bundled[row["role"]] = _read_json(raw)
    for relative, expected in manifest["frozen_runtime_sha256"].items():
        path = REPO / relative
        if _sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"frozen runtime hash mismatch: {relative}")

    report = bundled["formal_report"]
    training = bundled["training_report"]
    grounder = bundled["frozen_proof_grounder"]
    expected = manifest["formal_metrics"]
    authentic = report["conditions"]["authentic_sokoban_proof_cate_recover"]
    target = report["conditions"]["target_explicit_no_recovery"]
    paired = report["paired_authentic"]["target_explicit_no_recovery"]
    observed = {
        "samples": report["samples"],
        "target_only_correct": target["correct"],
        "authentic_correct": authentic["correct"],
        "authentic_recoveries": authentic["recoveries"],
        "paired_wins": paired["wins"],
        "paired_losses": paired["losses"],
        "exact_two_sided_p": paired["exact_two_sided_p"],
    }
    if observed != expected:
        raise ValueError(f"formal metric mismatch: {observed}")
    if report["status"] != (
        "SOKOBAN_TO_CLEVRER_PROOF_NEUROSYMBOLIC_TRANSFER_FORMAL_VALIDATED"
    ):
        raise ValueError("formal report is not validated")
    if not all(report["gates"].values()):
        raise ValueError("at least one formal gate failed")
    if training["status"] != "V14_PROOF_GROUNDER_DEVELOPMENT_GATE_PASSED":
        raise ValueError("training gate failed")
    if training["artifact_sha256"] != grounder["artifact_sha256"]:
        raise ValueError("grounder/training content lineage mismatch")
    if not manifest["route"]["target_actions_remain_target_native"]:
        raise ValueError("target-native action authority is not asserted")
    return {
        "status": "PORTABLE_CLEVRER_V14_FORMAL_EVIDENCE_VALIDATED",
        "samples": observed["samples"],
        "success_delta": observed["authentic_correct"] - observed["target_only_correct"],
        "paired_wins": observed["paired_wins"],
        "paired_losses": observed["paired_losses"],
        "exact_two_sided_p": observed["exact_two_sided_p"],
        "all_formal_gates_passed": True,
        "target_native_action_authority": True,
        "claim_boundary": manifest["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest), indent=2))


if __name__ == "__main__":
    main()
