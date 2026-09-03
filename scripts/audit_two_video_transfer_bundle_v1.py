#!/usr/bin/env python3
"""Content/hash/status audit for the CLEVRER + AGQA paper evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from motif_transfer.contracts import stable_hash


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedded_hash_valid(value: dict[str, Any], field: str) -> bool:
    body = dict(value)
    claimed = body.pop(field, None)
    return isinstance(claimed, str) and stable_hash(body) == claimed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("docs/results/two_video_transfer_bundle_v1.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    paths = {
        "anonymous_controller": root / "runs/anonymous_video_harness_v1/controller.json",
        "clevrer_predictions": root / "runs/clevrer_full_raw_video_v2/five_arm_predictions.json",
        "clevrer_formal": root / "runs/clevrer_full_raw_video_v2/formal_report.json",
        "clevrer_substitution": root / "runs/clevrer_full_raw_video_v2/anonymous_harness_substitution_v1.json",
        "clevrer_taxonomy": root / "runs/clevrer_full_raw_video_v2/failure_taxonomy_v1.json",
        "agqa_broad": root / "runs/agqa2_layer_b_raw_video_v1/qualification_v4/five_arm_epistemic_qualification_full512_v4.json",
        "agqa_temporal": root / "runs/agqa2_layer_b_raw_video_v1/typed_temporal_replication_v1/five_arm_typed_temporal_full256_v1.json",
        "agqa_substitution": root / "runs/agqa2_layer_b_raw_video_v1/anonymous_harness_substitution_v1.json",
        "agqa_inventory": root / "docs/results/agqa2_untouched_inventory_v16_20260902.json",
    }
    values = {key: _load(path) for key, path in paths.items()}
    controller_sha = values["anonymous_controller"]["artifact_sha256"]
    gates = {
        "anonymous_controller_hash_valid": _embedded_hash_valid(values["anonymous_controller"], "artifact_sha256"),
        "clevrer_prediction_hash_valid": _embedded_hash_valid(values["clevrer_predictions"], "predictions_sha256"),
        "clevrer_formal_hash_valid": _embedded_hash_valid(values["clevrer_formal"], "report_sha256"),
        "clevrer_substitution_hash_valid": _embedded_hash_valid(values["clevrer_substitution"], "report_sha256"),
        "clevrer_taxonomy_hash_valid": _embedded_hash_valid(values["clevrer_taxonomy"], "report_sha256"),
        "agqa_substitution_hash_valid": _embedded_hash_valid(values["agqa_substitution"], "report_sha256"),
        "clevrer_fresh_formal_validated": values["clevrer_formal"].get("status") == "CLEVRER_FULL_LAYER_B_TRANSFER_VALIDATED",
        "clevrer_anonymous_substitution_verified": values["clevrer_substitution"].get("status") == "CLEVRER_ANONYMOUS_HARNESS_SUBSTITUTION_VERIFIED",
        "agqa_existing_anonymous_substitution_verified": values["agqa_substitution"].get("status") == "AGQA_EXISTING_LAYER_B_ANONYMOUS_SUBSTITUTION_VERIFIED",
        "controller_lineage_shared_across_benchmarks": (
            values["clevrer_substitution"].get("controller_artifact_sha256") == controller_sha
            and values["agqa_substitution"].get("controller_artifact_sha256") == controller_sha
        ),
        "agqa_broad_significant_existing_signal": float(values["agqa_broad"]["comparisons"]["neural_only"]["exact_two_sided_p"]) < 0.05,
        "agqa_temporal_significant_existing_signal": float(values["agqa_temporal"]["comparisons"]["neural_only"]["exact_two_sided_p"]) < 0.05,
        "agqa_new_fresh_formal_not_claimed": values["agqa_substitution"].get("fresh_evaluation") is False,
    }
    body = {
        "schema_version": "two-video-transfer-paper-bundle-v1",
        "status": "TWO_VIDEO_TRANSFER_EVIDENCE_BUNDLE_VALIDATED" if all(gates.values()) else "TWO_VIDEO_TRANSFER_EVIDENCE_BUNDLE_FAILED",
        "validation_scope": {
            "clevrer": "FRESH_FULL_BENCHMARK_SELECTIVE_LAYER_B_FORMAL",
            "agqa2": "TWO_INDEPENDENT_EXISTING_RAW_VIDEO_SELECTIVE_REPLICATIONS;NO_NEW_UNTOUCHED_OFFICIAL_TEST_FORMAL",
        },
        "headline_results": {
            "clevrer": {"tasks": 1600, "neural_accuracy": 0.3475, "source_accuracy": 0.623125, "wins": 464, "losses": 23},
            "agqa2_broad": {"tasks": 512, "neural_accuracy": 0.46484375, "source_accuracy": 0.50390625, "wins": 44, "losses": 24},
            "agqa2_temporal": {"tasks": 256, "neural_accuracy": 0.4140625, "source_accuracy": 0.52734375, "wins": 34, "losses": 5},
        },
        "anonymous_controller_artifact_sha256": controller_sha,
        "artifact_file_sha256s": {key: _sha(path) for key, path in paths.items()},
        "gates": gates,
        "remaining_blocker": "AGQA official test has no untouched local video reserve; this prevents a new fresh official-test formal, not validation of the two existing independent replication artifacts.",
    }
    body["bundle_sha256"] = stable_hash(body)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "gates": gates, "bundle_sha256": body["bundle_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
