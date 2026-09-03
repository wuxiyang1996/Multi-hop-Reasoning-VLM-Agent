#!/usr/bin/env python3
"""Compile three consumed matched-model batches into V36 adaptation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.natural_video_recovery import FEATURE_NAMES, build_features  # noqa: E402
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row(
    *, batch: str, benchmark: str, family: str, sample_id: str, video_id: str,
    direct: Mapping[str, Any], proof: Mapping[str, Any], direct_correct: bool,
    proof_correct: bool, lineage: Mapping[str, Any],
) -> dict[str, Any]:
    features = build_features(
        benchmark=benchmark, family=family, primary=direct, proof=proof,
    )
    return {
        "schema_version": 36,
        "batch": batch,
        "benchmark": benchmark,
        "family": family,
        "sample_id": sample_id,
        "video_id": video_id,
        "features": list(map(float, features)),
        "direct_correct": bool(direct_correct),
        "proof_correct": bool(proof_correct),
        "uplift": int(proof_correct) - int(direct_correct),
        "runtime_feature_saw_gold_or_official_structure": False,
        "matched_model": True,
        "matched_frames": True,
        "lineage": dict(lineage),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key, raw_path in config["inputs"].items():
        path = Path(raw_path)
        expected = str(config["frozen_lineage"][f"{key}_sha256"])
        if sha256(path) != expected:
            raise ValueError(f"V36 matched adaptation lineage mismatch: {key}")
    source_path = Path(config["source_receipt"])
    if sha256(source_path) != config["frozen_lineage"]["source_receipt_sha256"]:
        raise ValueError("V36 source receipt lineage mismatch")
    validate_source_receipt(json.loads(source_path.read_text()))

    v19 = json.loads(Path(config["inputs"]["v19_proof_receipts"]).read_text())
    proof19 = {(str(row["benchmark"]), str(row["sample_id"])): row for row in v19}
    v21 = json.loads(Path(config["inputs"]["v21_matched_direct"]).read_text())
    v27 = json.loads(Path(config["inputs"]["v27_matched_factorial"]).read_text())
    v35 = json.loads(Path(config["inputs"]["v35_proof_receipts"]).read_text())
    proof35 = {(str(row["benchmark"]), str(row["sample_id"])): row for row in v35}
    direct35 = json.loads(Path(config["inputs"]["v35_matched_direct"]).read_text())
    rows = []
    for direct in v21:
        key = (str(direct["benchmark"]), str(direct["sample_id"]))
        proof = proof19[key]
        if list(direct["proof_panel_sha256"]) != list(proof["proof_panel_sha256"]):
            raise ValueError(f"V21 matched panels drift: {key}")
        rows.append(_row(
            batch="v21_gemini_consumed", benchmark=key[0], family=str(direct["family"]),
            sample_id=key[1], video_id=str(direct["video_id"]),
            direct=direct["generic_direct"], proof=proof["proof"],
            direct_correct=bool(direct["generic_direct_correct"]),
            proof_correct=bool(direct["typed_proof_correct"]),
            lineage={"direct": content_hash(direct), "proof": content_hash(proof)},
        ))
    for row in v27:
        if not bool(row.get("within_view_direct_and_proof_panels_identical")):
            raise ValueError(f"V27 direct/proof panel mismatch: {row['sample_id']}")
        direct_model = str(row["usage"]["uniform_direct"]["model"])
        proof_model = str(row["usage"]["uniform_typed_proof"]["model"])
        if direct_model != proof_model:
            raise ValueError(f"V27 direct/proof model mismatch: {row['sample_id']}")
        rows.append(_row(
            batch="v27_gpt41mini_consumed", benchmark="star", family=str(row["family"]),
            sample_id=str(row["sample_id"]), video_id=str(row["video_id"]),
            direct=row["uniform_direct"], proof=row["uniform_typed_proof"],
            direct_correct=bool(row["uniform_direct_correct"]),
            proof_correct=bool(row["uniform_typed_proof_correct"]),
            lineage={"factorial": content_hash(row)},
        ))
    for direct in direct35:
        key = (str(direct["benchmark"]), str(direct["sample_id"]))
        proof = proof35[key]
        if list(direct["proof_panel_sha256"]) != list(proof["proof_panel_sha256"]):
            raise ValueError(f"V35 matched panels drift: {key}")
        rows.append(_row(
            batch="v35_gemini_consumed", benchmark=key[0], family=str(direct["family"]),
            sample_id=key[1], video_id=str(direct["video_id"]),
            direct=direct["generic_direct"], proof=proof["proof"],
            direct_correct=bool(direct["generic_direct_correct"]),
            proof_correct=bool(direct["typed_proof_correct"]),
            lineage={"direct": content_hash(direct), "proof": content_hash(proof)},
        ))
    identities = [(row["benchmark"], row["sample_id"]) for row in rows]
    if len(rows) != int(config["expected_rows"]) or len(set(identities)) != len(rows):
        raise ValueError("V36 matched adaptation rows are incomplete or duplicated")
    groups: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        groups.setdefault((row["benchmark"], row["video_id"]), set()).add(row["batch"])
    if any(len(batches) != 1 for batches in groups.values()):
        raise ValueError("a V36 video appears in multiple adaptation batches")
    matrix = np.asarray([row["features"] for row in rows], dtype=float)
    if matrix.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("V36 matched feature matrix is invalid")
    payload = {
        "schema_version": 36,
        "status": "V36_MATCHED_MODEL_ADAPTATION_COMPILED",
        "claim_boundary": config["claim_boundary"],
        "feature_names": list(FEATURE_NAMES),
        "rows": rows,
        "audit": {
            "samples": len(rows),
            "video_groups": len(groups),
            "batch_counts": {
                batch: sum(row["batch"] == batch for row in rows)
                for batch in sorted({row["batch"] for row in rows})
            },
            "benchmark_counts": {
                benchmark: sum(row["benchmark"] == benchmark for row in rows)
                for benchmark in sorted({row["benchmark"] for row in rows})
            },
            "uplift_counts": {
                str(value): sum(row["uplift"] == value for row in rows)
                for value in (-1, 0, 1)
            },
            "direct_correct": sum(row["direct_correct"] for row in rows),
            "proof_correct": sum(row["proof_correct"] for row in rows),
            "sample_overlap_across_batches": 0,
            "video_overlap_across_batches": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["audit"] | {
        "status": payload["status"], "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
