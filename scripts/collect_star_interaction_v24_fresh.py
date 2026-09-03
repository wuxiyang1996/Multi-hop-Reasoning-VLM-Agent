#!/usr/bin/env python3
"""Collect prospective fresh-cluster STAR Interaction transfer branches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_natural_video_generic_control_v21 as generic  # noqa: E402
import collect_natural_video_recovery_v15 as paired  # noqa: E402
import collect_natural_video_v19_formal as v19  # noqa: E402
from motif_transfer.natural_video_symbolic_controls import execute_recovery  # noqa: E402
from motif_transfer.sokoban_video_recovery import validate_source_receipt  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_samples(ids: Sequence[str], config: Mapping[str, Any]) -> dict[str, Any]:
    wrapper_root = Path(config["wrapper_root"])
    if str(wrapper_root) not in sys.path:
        sys.path.insert(0, str(wrapper_root))
    from visual_reasoning_wrapper.benchmarks.star import iter_star_samples

    samples = iter_star_samples(
        "val",
        star_root=config["benchmark"]["root"],
        sample_ids=ids,
        require_video=True,
    )
    output = {str(sample.sample_id): sample for sample in samples}
    if set(output) != set(ids):
        raise ValueError(f"missing fresh STAR samples/videos: {sorted(set(ids) - set(output))}")
    return output


def _collect_one(
    sample: Any,
    *,
    config: Mapping[str, Any],
    primary_api_key: str,
    proof_api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    primary_client = OpenAI(
        api_key=primary_api_key,
        base_url=str(config["primary_model"]["base_url"]),
        timeout=float(config["primary_model"]["timeout_seconds"]),
        max_retries=int(config["primary_model"]["max_retries"]),
    )
    proof_client = OpenAI(
        api_key=proof_api_key,
        base_url=str(config["proof_model"]["base_url"]),
        timeout=float(config["proof_model"]["timeout_seconds"]),
        max_retries=int(config["proof_model"]["max_retries"]),
    )
    primary_panel, proof_panels, metadata = paired._panels(sample, config)
    primary_config = {**config, "model": config["primary_model"]}
    proof_config = {**config, "model": config["proof_model"]}
    primary, primary_raw, primary_usage = paired._primary_call(
        primary_client, sample=sample, panel=primary_panel, config=primary_config,
    )
    proof, proof_raw, proof_usage = paired._proof_call(
        proof_client, sample=sample, panels=proof_panels, config=proof_config,
    )
    generic_direct, generic_raw, generic_usage = generic._generic_call(
        proof_client, sample=sample, panels=proof_panels, config=proof_config,
    )
    primary_answer = str(primary["answer"])
    authentic_answer = execute_recovery(primary_answer, proof)
    binding_answer = execute_recovery(
        primary_answer, proof, shuffled_binding=True,
    )
    topology_answer = execute_recovery(
        primary_answer, proof, shuffled_topology=True,
    )
    # Gold is attached only after all blind neural calls, symbolic programs, and
    # destructive controls are fully determined for this row.
    gold = str(sample.answer)
    return {
        "schema_version": 24,
        "benchmark": "star",
        "split": "fresh_prospective_interaction",
        "sample_id": str(sample.sample_id),
        "video_id": str(sample.video_id),
        "family": "Interaction",
        "gold_answer": gold,
        "primary": primary,
        "typed_proof": proof,
        "generic_direct": generic_direct,
        "source_authentic_answer": authentic_answer,
        "shuffled_binding_answer": binding_answer,
        "shuffled_topology_answer": topology_answer,
        "primary_correct": primary_answer == gold,
        "typed_proof_correct": str(proof["answer"]) == gold,
        "generic_direct_correct": str(generic_direct["answer"]) == gold,
        "source_authentic_correct": authentic_answer == gold,
        "shuffled_binding_correct": binding_answer == gold,
        "shuffled_topology_correct": topology_answer == gold,
        "source_recover": authentic_answer != primary_answer,
        "binding_control_recover": binding_answer != primary_answer,
        "topology_control_recover": topology_answer != primary_answer,
        "primary_raw": primary_raw,
        "proof_raw": proof_raw,
        "generic_raw": generic_raw,
        "usage": {
            "primary": primary_usage,
            "typed_proof": proof_usage,
            "generic_direct": generic_usage,
        },
        "video_metadata": metadata,
        "video_sha256": _sha256(Path(sample.video_path)),
        "primary_panel_sha256": hashlib.sha256(primary_panel).hexdigest(),
        "proof_panel_sha256": [hashlib.sha256(value).hexdigest() for value in proof_panels],
        "generic_and_proof_panels_identical": True,
        "collection_contract_sha256": contract_sha256,
        "runtime_saw_gold_or_official_structure": False,
    }


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
        "manifest_sha256": Path(config["manifest"]),
        "download_receipt_sha256": Path(config["download_receipt"]),
        "collector_sha256": Path(__file__).resolve(),
        "paired_collector_sha256": REPO / "scripts/collect_natural_video_recovery_v15.py",
        "generic_collector_sha256": REPO / "scripts/collect_natural_video_generic_control_v21.py",
        "transport_module_sha256": REPO / "scripts/collect_natural_video_v19_formal.py",
        "symbolic_executor_sha256": REPO / "src/motif_transfer/natural_video_symbolic_controls.py",
    }
    for key, path in lineage_paths.items():
        if _sha256(path) != config["frozen_lineage"].get(key):
            raise ValueError(f"V24 fresh lineage mismatch: {key}")
    validate_source_receipt(json.loads(
        Path(config["source_receipt"]).read_text(encoding="utf-8")
    ))
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_V24_VIDEO_DOWNLOAD_OR_RUNTIME_OUTCOMES":
        raise ValueError("V24 manifest is not sealed")
    download = json.loads(Path(config["download_receipt"]).read_text(encoding="utf-8"))
    if download.get("status") != "V24_OFFICIAL_RANGE_DOWNLOAD_COMPLETE":
        raise ValueError("V24 official video download is incomplete")
    ordered_ids = [str(row["sample_id"]) for row in manifest["samples"]]
    samples = _load_samples(ordered_ids, config)
    download_by_video = {str(row["video_id"]): row for row in download["videos"]}
    for sample in samples.values():
        receipt = download_by_video.get(str(sample.video_id))
        if not receipt or _sha256(Path(sample.video_path)) != str(receipt["sha256"]):
            raise ValueError(f"V24 video receipt mismatch: {sample.video_id}")
    contract_sha256 = _content_hash({
        "config_sha256": _sha256(args.config),
        "collector_sha256": _sha256(Path(__file__).resolve()),
        "ordered_ids": ordered_ids,
    })
    key_values = runpy.run_path(str(args.keys))
    primary_api_key = key_values.get(config["primary_model"]["api_key_name"])
    proof_api_key = key_values.get(config["proof_model"]["api_key_name"])
    if not primary_api_key or not proof_api_key:
        raise SystemExit("V24 primary/proof API key is missing")
    paired.media_helpers._json_call = v19._provider_json_call
    existing = {}
    if args.output.is_file():
        for row in json.loads(args.output.read_text(encoding="utf-8")):
            if row.get("collection_contract_sha256") != contract_sha256:
                raise ValueError("cached V24 fresh contract mismatch")
            existing[str(row["sample_id"])] = row
    pending = [sample_id for sample_id in ordered_ids if sample_id not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(
            [existing[sample_id] for sample_id in ordered_ids if sample_id in existing],
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_one,
                samples[sample_id],
                config=config,
                primary_api_key=str(primary_api_key),
                proof_api_key=str(proof_api_key),
                contract_sha256=contract_sha256,
            ): sample_id
            for sample_id in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                existing[sample_id] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": sample_id, "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            save()
            print(json.dumps({
                "completed": sample_id,
                "progress": f"{len(existing)}/{len(ordered_ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in ordered_ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete V24 fresh collection; rerun: {missing}")
    rows = [existing[sample_id] for sample_id in ordered_ids]
    print(json.dumps({
        "status": "STAR_INTERACTION_V24_FRESH_COLLECTED",
        "rows": len(rows),
        "video_clusters": len({row["video_id"] for row in rows}),
        "primary_correct": sum(row["primary_correct"] for row in rows),
        "generic_direct_correct": sum(row["generic_direct_correct"] for row in rows),
        "typed_proof_correct": sum(row["typed_proof_correct"] for row in rows),
        "source_authentic_correct": sum(row["source_authentic_correct"] for row in rows),
        "binding_control_correct": sum(row["shuffled_binding_correct"] for row in rows),
        "topology_control_correct": sum(row["shuffled_topology_correct"] for row in rows),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
