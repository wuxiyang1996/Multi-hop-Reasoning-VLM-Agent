#!/usr/bin/env python3
"""Prospective V35 Sokoban-source CATE evaluation on reserve videos."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping

import numpy as np
from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_natural_video_active_recovery_v18 as transport  # noqa: E402
import collect_natural_video_recovery_v15 as paired  # noqa: E402
import collect_natural_video_v19_formal as v19  # noqa: E402
from motif_transfer.natural_video_proof_cate import (  # noqa: E402
    compile_v19_features,
    proof_binding_rotation,
    validate_v34_artifact,
)
from motif_transfer.natural_video_recovery import BASE_FEATURE_NAMES  # noqa: E402
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _correct(row: Mapping[str, Any], recover: bool) -> bool:
    answer = str(row["proof"]["answer"] if recover else row["primary"]["answer"])
    return answer == str(row["gold_answer"])


def _same_rate_marginal(
    rows: list[Mapping[str, Any]], authentic: np.ndarray, salt: str,
) -> np.ndarray:
    """Match recovery rate within benchmark/family/disagreement cells."""

    output = np.zeros(len(rows), dtype=bool)
    cells = sorted({
        (
            str(row["benchmark"]),
            str(row["family"]),
            str(row["primary"]["answer"] != row["proof"]["answer"]),
        )
        for row in rows
    })
    for cell in cells:
        indices = [
            index for index, row in enumerate(rows)
            if (
                str(row["benchmark"]),
                str(row["family"]),
                str(row["primary"]["answer"] != row["proof"]["answer"]),
            ) == cell
        ]
        count = int(np.sum(authentic[indices]))
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{salt}|{rows[index]['benchmark']}|{rows[index]['sample_id']}".encode()
            ).hexdigest(),
        )
        output[ranked[:count]] = True
    return output


def _finalize(
    rows: list[dict[str, Any]], config: Mapping[str, Any], artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source, base, permuted, shuffled, threshold = validate_v34_artifact(artifact)
    matrix = np.asarray([compile_v19_features(row) for row in rows], dtype=float)
    masks = {
        "source_proof_cate": source.predict(matrix) > threshold,
        "base_only_cate": base.predict(matrix[:, :len(BASE_FEATURE_NAMES)]) > threshold,
        "permuted_uplift_cate": permuted.predict(matrix) > threshold,
        "shuffled_proof_training_cate": shuffled.predict(matrix) > threshold,
    }
    binding = proof_binding_rotation(rows, int(config["controls"]["binding_rotation_seed"]))
    bound_matrix = np.concatenate(
        (matrix[:, :len(BASE_FEATURE_NAMES)], matrix[np.asarray(binding), len(BASE_FEATURE_NAMES):]),
        axis=1,
    )
    masks["binding_rotation_cate"] = source.predict(bound_matrix) > threshold
    masks["inverted_source_contract"] = source.predict(matrix) <= threshold
    masks["same_rate_marginal"] = _same_rate_marginal(
        rows, masks["source_proof_cate"], str(config["controls"]["marginal_salt"]),
    )
    source_heads = source.predict_heads(matrix)
    base_heads = base.predict_heads(matrix[:, :len(BASE_FEATURE_NAMES)])
    source_predictions = source_heads.mean(axis=0)
    base_predictions = base_heads.mean(axis=0)

    finalized = []
    for index, original in enumerate(rows):
        row = dict(original)
        row["schema_version"] = 35
        row["split"] = "prospective_reserve_formal"
        row["cate"] = {
            "decision_threshold": threshold,
            "source_proof_prediction": float(source_predictions[index]),
            "source_proof_head_predictions": list(map(float, source_heads[:, index])),
            "base_only_prediction": float(base_predictions[index]),
            "base_only_head_predictions": list(map(float, base_heads[:, index])),
        }
        row["binding_rotation_target"] = {
            "benchmark": str(rows[binding[index]]["benchmark"]),
            "sample_id": str(rows[binding[index]]["sample_id"]),
            "video_id": str(rows[binding[index]]["video_id"]),
        }
        row["conditions"] = {}
        for name, mask in masks.items():
            recover = bool(mask[index])
            answer = str(row["proof"]["answer"] if recover else row["primary"]["answer"])
            row["conditions"][name] = {
                "recover": recover,
                "answer": answer,
                "correct": answer == str(row["gold_answer"]),
            }
        row["conditions"]["primary"] = {
            "recover": False,
            "answer": str(row["primary"]["answer"]),
            "correct": bool(row["primary_correct"]),
        }
        row["conditions"]["always_proof"] = {
            "recover": True,
            "answer": str(row["proof"]["answer"]),
            "correct": bool(row["proof_correct"]),
        }
        finalized.append(row)
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lineage_paths = {
        "source_receipt_sha256": Path(config["source_receipt"]),
        "formal_manifest_sha256": Path(config["formal_manifest"]),
        "cate_artifact_sha256": Path(config["cate_artifact"]),
        "development_receipts_sha256": Path(config["development_receipts"]),
        "collector_sha256": Path(__file__).resolve(),
        "v19_collector_sha256": REPO / "scripts/collect_natural_video_v19_formal.py",
        "v15_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "transport_module_sha256": REPO / "scripts/collect_natural_video_active_recovery_v18.py",
        "contract_module_sha256": REPO / "src/motif_transfer/natural_video_recovery.py",
        "cate_module_sha256": REPO / "src/motif_transfer/natural_video_proof_cate.py",
    }
    for key, path in lineage_paths.items():
        if sha256(path) != str(config["frozen_lineage"].get(key, "")):
            raise ValueError(f"V35 frozen lineage mismatch: {key}")
    validate_source_receipt(json.loads(Path(config["source_receipt"]).read_text()))
    artifact = json.loads(Path(config["cate_artifact"]).read_text())
    validate_v34_artifact(artifact)
    manifest = json.loads(Path(config["formal_manifest"]).read_text())
    if manifest.get("status") != "FROZEN_BEFORE_V22_RESERVE_RUNTIME_OR_OUTCOMES":
        raise ValueError("V35 reserve manifest is not prospectively sealed")
    ordered_pairs = [
        (benchmark, str(row["sample_id"]))
        for benchmark in ("star", "nextqa")
        for row in manifest["benchmarks"][benchmark]
    ]
    development = json.loads(Path(config["development_receipts"]).read_text())
    dev_ids = {(str(row["benchmark"]), str(row["sample_id"])) for row in development}
    dev_videos = {(str(row["benchmark"]), str(row["video_id"])) for row in development}
    formal_ids = set(ordered_pairs)
    formal_videos = {
        (benchmark, str(row["video_id"]))
        for benchmark in ("star", "nextqa") for row in manifest["benchmarks"][benchmark]
    }
    if dev_ids & formal_ids or dev_videos & formal_videos:
        raise ValueError("V35 prospective reserve overlaps CATE development data")
    samples = {
        benchmark: paired._load_samples(
            benchmark,
            [sample_id for name, sample_id in ordered_pairs if name == benchmark],
            config,
        )
        for benchmark in ("star", "nextqa")
    }
    contract_sha256 = content_hash({
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(Path(config["formal_manifest"])),
        "artifact_sha256": sha256(Path(config["cate_artifact"])),
        "collector_sha256": sha256(Path(__file__).resolve()),
        "ordered_pairs": ordered_pairs,
    })
    keys = runpy.run_path(str(args.keys))
    primary_key = keys.get(config["primary_model"]["api_key_name"])
    proof_key = keys.get(config["proof_model"]["api_key_name"])
    if not primary_key or not proof_key:
        raise SystemExit("V35 primary/proof API key is missing")
    paired.media_helpers._json_call = v19._provider_json_call
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V35 receipt contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    pending = [pair for pair in ordered_pairs if pair not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_partial() -> None:
        args.output.write_text(json.dumps(
            [existing[pair] for pair in ordered_pairs if pair in existing],
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                v19._collect_one,
                samples[benchmark][sample_id],
                benchmark=benchmark,
                config=config,
                primary_api_key=str(primary_key),
                proof_api_key=str(proof_key),
                contract_sha256=contract_sha256,
            ): (benchmark, sample_id)
            for benchmark, sample_id in pending
        }
        for future in as_completed(futures):
            pair = futures[future]
            try:
                existing[pair] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": list(pair), "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            save_partial()
            print(json.dumps({
                "completed": list(pair),
                "progress": f"{len(existing)}/{len(ordered_pairs)}",
            }), flush=True)
    missing = [pair for pair in ordered_pairs if pair not in existing]
    if missing:
        raise SystemExit(f"incomplete V35 formal collection; rerun: {missing}")
    rows = _finalize([existing[pair] for pair in ordered_pairs], config, artifact)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "NATURAL_VIDEO_V35_PROSPECTIVE_FORMAL_COLLECTED",
        "samples": len(rows),
        "benchmark_counts": {
            value: sum(row["benchmark"] == value for row in rows)
            for value in ("star", "nextqa")
        },
        "condition_correct": {
            name: sum(bool(row["conditions"][name]["correct"]) for row in rows)
            for name in rows[0]["conditions"]
        },
        "proof_cost": sum(float(row["usage"]["proof"].get("cost", 0)) for row in rows),
        "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
