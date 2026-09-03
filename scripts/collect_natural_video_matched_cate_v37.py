#!/usr/bin/env python3
"""Collect prospective matched-model V37 STAR/NExT-QA receipts."""

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

import collect_natural_video_generic_control_v21 as generic  # noqa: E402
import collect_natural_video_recovery_v15 as paired  # noqa: E402
import collect_natural_video_v19_formal as transport  # noqa: E402
from motif_transfer.natural_video_matched_cate import (  # noqa: E402
    cross_video_binding_rotation,
    validate_v36_artifact,
)
from motif_transfer.natural_video_recovery import (  # noqa: E402
    BASE_FEATURE_NAMES,
    build_features,
)
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


def _collect_one(
    sample: Any, *, benchmark: str, config: Mapping[str, Any], api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    model = config["model"]
    client = OpenAI(
        api_key=api_key, base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]), max_retries=int(model["max_retries"]),
    )
    _overview, panels, metadata = paired._panels(sample, config)
    direct, direct_raw, direct_usage = generic._generic_call(
        client, sample=sample, panels=panels, config=config,
    )
    proof, proof_raw, proof_usage = paired._proof_call(
        client, sample=sample, panels=panels, config=config,
    )
    family = str(
        getattr(sample, "question_family", None) or getattr(sample, "question_type", "")
    )
    features = build_features(
        benchmark=benchmark, family=family, primary=direct, proof=proof,
    )
    # Outcome access begins only after both matched neural calls and runtime
    # features are immutable.
    gold = str(sample.answer)
    return {
        "schema_version": 37,
        "benchmark": benchmark,
        "split": "prospective_matched_formal",
        "sample_id": str(sample.sample_id), "video_id": str(sample.video_id),
        "family": family, "gold_answer": gold,
        "direct": direct, "proof": proof,
        "direct_correct": str(direct["answer"]) == gold,
        "proof_correct": str(proof["answer"]) == gold,
        "features": list(map(float, features)),
        "runtime_saw_gold_or_official_structure": False,
        "same_model_direct_and_proof": True,
        "same_frames_direct_and_proof": True,
        "direct_raw": direct_raw, "proof_raw": proof_raw,
        "usage": {"direct": direct_usage, "proof": proof_usage},
        "video_metadata": metadata,
        "video_sha256": sha256(Path(sample.video_path)),
        "panel_sha256": [hashlib.sha256(value).hexdigest() for value in panels],
        "collection_contract_sha256": contract_sha256,
    }


def _same_rate_marginal(
    rows: list[Mapping[str, Any]], source_mask: np.ndarray, salt: str,
) -> np.ndarray:
    output = np.zeros(len(rows), dtype=bool)
    cells = sorted({
        (
            str(row["benchmark"]), str(row["family"]),
            str(row["direct"]["answer"] != row["proof"]["answer"]),
        )
        for row in rows
    })
    for cell in cells:
        indices = [
            index for index, row in enumerate(rows)
            if (
                str(row["benchmark"]), str(row["family"]),
                str(row["direct"]["answer"] != row["proof"]["answer"]),
            ) == cell
        ]
        count = int(np.sum(source_mask[indices]))
        ranked = sorted(indices, key=lambda index: hashlib.sha256(
            f"{salt}|{rows[index]['benchmark']}|{rows[index]['sample_id']}".encode()
        ).hexdigest())
        output[ranked[:count]] = True
    return output


def _finalize(
    rows: list[dict[str, Any]], config: Mapping[str, Any], artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source, base, permuted, binding_training, threshold = validate_v36_artifact(artifact)
    matrix = np.asarray([row["features"] for row in rows], dtype=float)
    source_predictions = source.predict(matrix)
    masks = {
        "source_proof_cate": source_predictions > threshold,
        "base_only_cate": base.predict(matrix[:, :len(BASE_FEATURE_NAMES)]) > threshold,
        "permuted_uplift_cate": permuted.predict(matrix) > threshold,
        "binding_training_cate": binding_training.predict(matrix) > threshold,
        "inverted_source_contract": source_predictions <= threshold,
    }
    runtime_binding = cross_video_binding_rotation(rows, cell_fields=("benchmark",))
    bound_matrix = np.concatenate((
        matrix[:, :len(BASE_FEATURE_NAMES)],
        matrix[np.asarray(runtime_binding), len(BASE_FEATURE_NAMES):],
    ), axis=1)
    masks["runtime_cross_video_binding"] = source.predict(bound_matrix) > threshold
    masks["same_rate_marginal"] = _same_rate_marginal(
        rows, masks["source_proof_cate"], str(config["controls"]["marginal_salt"]),
    )
    source_heads = source.predict_heads(matrix)
    final = []
    for index, original in enumerate(rows):
        row = dict(original)
        row["cate"] = {
            "threshold": threshold,
            "source_prediction": float(source_predictions[index]),
            "source_head_predictions": list(map(float, source_heads[:, index])),
        }
        target = rows[runtime_binding[index]]
        row["runtime_binding_target"] = {
            "benchmark": str(target["benchmark"]), "sample_id": str(target["sample_id"]),
            "video_id": str(target["video_id"]), "family": str(target["family"]),
        }
        row["conditions"] = {
            "matched_direct": {
                "recover": False, "answer": str(row["direct"]["answer"]),
                "correct": bool(row["direct_correct"]),
            },
            "raw_typed_proof": {
                "recover": True, "answer": str(row["proof"]["answer"]),
                "correct": bool(row["proof_correct"]),
            },
        }
        for name, mask in masks.items():
            recover = bool(mask[index])
            answer = str(row["proof"]["answer"] if recover else row["direct"]["answer"])
            row["conditions"][name] = {
                "recover": recover, "answer": answer,
                "correct": answer == str(row["gold_answer"]),
            }
        final.append(row)
    return final


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
        "adaptation_receipts_sha256": Path(config["adaptation_receipts"]),
        "collector_sha256": Path(__file__).resolve(),
        "generic_collector_sha256": REPO / "scripts/collect_natural_video_generic_control_v21.py",
        "paired_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "transport_sha256": REPO / "scripts/collect_natural_video_v19_formal.py",
        "feature_module_sha256": REPO / "src/motif_transfer/natural_video_recovery.py",
        "cate_module_sha256": REPO / "src/motif_transfer/natural_video_matched_cate.py",
    }
    for key, path in lineage_paths.items():
        if sha256(path) != str(config["frozen_lineage"].get(key, "")):
            raise ValueError(f"V37 frozen lineage mismatch: {key}")
    validate_source_receipt(json.loads(Path(config["source_receipt"]).read_text()))
    artifact = json.loads(Path(config["cate_artifact"]).read_text())
    validate_v36_artifact(artifact)
    manifest = json.loads(Path(config["formal_manifest"]).read_text())
    if manifest.get("status") != "FROZEN_BEFORE_V37_MATCHED_FORMAL_CALLS_OR_OUTCOMES":
        raise ValueError("V37 formal manifest is not sealed")
    ordered_pairs = [
        (benchmark, str(row["sample_id"]))
        for benchmark in ("star", "nextqa") for row in manifest["benchmarks"][benchmark]
    ]
    adaptation = json.loads(Path(config["adaptation_receipts"]).read_text())["rows"]
    adaptation_ids = {(str(row["benchmark"]), str(row["sample_id"])) for row in adaptation}
    adaptation_videos = {(str(row["benchmark"]), str(row["video_id"])) for row in adaptation}
    formal_videos = {
        (benchmark, str(row["video_id"]))
        for benchmark in ("star", "nextqa") for row in manifest["benchmarks"][benchmark]
    }
    if adaptation_ids & set(ordered_pairs) or adaptation_videos & formal_videos:
        raise ValueError("V37 formal data overlaps V36 adaptation")
    samples = {
        benchmark: paired._load_samples(
            benchmark, [sid for name, sid in ordered_pairs if name == benchmark], config,
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
    api_key = runpy.run_path(str(args.keys)).get(config["model"]["api_key_name"])
    if not api_key:
        raise SystemExit("V37 OpenRouter key is missing")
    paired.media_helpers._json_call = transport._provider_json_call
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text()):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V37 receipt contract mismatch")
            existing[(str(row["benchmark"]), str(row["sample_id"]))] = row
    pending = [pair for pair in ordered_pairs if pair not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[pair] for pair in ordered_pairs if pair in existing],
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one, samples[benchmark][sample_id], benchmark=benchmark,
                config=config, api_key=str(api_key), contract_sha256=contract_sha256,
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
            save()
            print(json.dumps({
                "completed": list(pair), "progress": f"{len(existing)}/{len(ordered_pairs)}",
            }), flush=True)
    missing = [pair for pair in ordered_pairs if pair not in existing]
    if missing:
        raise SystemExit(f"incomplete V37 collection; rerun: {missing}")
    rows = _finalize([existing[pair] for pair in ordered_pairs], config, artifact)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "V37_MATCHED_NATURAL_VIDEO_FORMAL_COLLECTED",
        "rows": len(rows),
        "condition_correct": {
            name: sum(bool(row["conditions"][name]["correct"]) for row in rows)
            for name in rows[0]["conditions"]
        },
        "reported_cost": sum(
            float(value.get("cost", 0) or 0)
            for row in rows for value in row["usage"].values()
        ),
        "output_sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
