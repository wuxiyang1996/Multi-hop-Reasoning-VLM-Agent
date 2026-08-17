#!/usr/bin/env python3
"""Run induced shared-IR Sokoban-to-TIR maze transfer."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping

from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.relational_structural_induction import (  # noqa: E402
    validate_relational_structural_program,
)
from motif_transfer.tir_maze_topology import (  # noqa: E402
    STRUCTURAL_CONDITIONS,
    evaluate_tir_maze_structural_transfer,
    execute_maze_structural_program,
)
from run_tir_maze_topology_v2 import (  # noqa: E402
    _baseline,
    _bind,
    _image_data,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: Mapping[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _collect_sample(
    sample_id: str,
    *,
    row: Mapping[str, Any],
    dataset_root: Path,
    config: Mapping[str, Any],
    artifact: Mapping[str, Any],
    api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    if row.get("task") != "maze" or row.get("image_2"):
        raise ValueError("TIR structural runner accepts single-image maze only")
    image_path = dataset_root / str(row["image_1"])
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    media = config["media"]
    image_url = _image_data(
        image, max_side=int(media["max_side"]), quality=int(media["jpeg_quality"]),
    )
    model_config = config["model"]
    client = OpenAI(
        api_key=api_key,
        base_url=str(model_config["base_url"]),
        timeout=float(model_config["timeout_seconds"]),
        max_retries=int(model_config["max_retries"]),
    )
    model = str(model_config["id"])
    binding, binding_usage = _bind(
        client,
        model=model,
        prompt=str(row["prompt"]),
        image_url=image_url,
        maximum_tokens=int(model_config["maximum_output_tokens"]),
    )
    baseline, baseline_usage = _baseline(
        client,
        model=model,
        prompt=str(row["prompt"]),
        image_url=image_url,
        maximum_tokens=int(model_config["maximum_output_tokens"]),
    )
    conditions = {
        condition: execute_maze_structural_program(
            image,
            str(row["prompt"]),
            neural_binding=binding,
            source_artifact=artifact,
            condition=condition,
        )
        for condition in STRUCTURAL_CONDITIONS
        if condition != "neural_only"
    }
    body = {
        "schema_version": "tir-maze-structural-receipt-v2",
        "collection_contract_sha256": contract_sha256,
        "sample_id": sample_id,
        "family": "maze",
        "image_path": str(image_path),
        "image_sha256": _sha256(image_path),
        "prompt_sha256": stable_hash(str(row["prompt"])),
        "neural_binding": binding,
        "neural_binding_valid": True,
        "binding_usage": binding_usage,
        "baseline_answer": baseline,
        "baseline_usage": baseline_usage,
        "conditions": conditions,
    }
    # Attach evaluation authority only after every condition is immutable.
    body["gold_answer_evaluator_only"] = str(row["answer"])
    return body | {"receipt_sha256": stable_hash(body)}


def _validate_integrity(config: Mapping[str, Any]) -> None:
    for relative, expected in config["integrity"]["file_sha256"].items():
        path = REPO / str(relative)
        if _sha256(path) != str(expected):
            raise SystemExit(f"frozen TIR structural dependency changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("qualification", "heldout"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _self_hash(config, "manifest_sha256")
    if config.get("status") != "FROZEN_BEFORE_FRESH_QUALIFICATION":
        raise SystemExit("TIR structural config is not frozen")
    _validate_integrity(config)
    target_authority_path = REPO / config["target_interface_authority"]["report_path"]
    target_authority = json.loads(target_authority_path.read_text(encoding="utf-8"))
    _self_hash(target_authority, "report_sha256")
    if target_authority.get("status") != config["target_interface_authority"][
        "required_status"
    ]:
        raise SystemExit("target-native TIR interface was not qualified")
    artifact_path = REPO / config["source"]["artifact_path"]
    confirmation_path = REPO / config["source"]["confirmation_path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    validate_relational_structural_program(artifact)
    _self_hash(confirmation, "report_sha256")
    if confirmation.get("artifact_sha256") != artifact.get("artifact_sha256") or not (
        confirmation.get("source_gate_passed")
    ):
        raise SystemExit("fresh source structural confirmation did not pass")
    if args.split == "heldout":
        qualification_path = REPO / config["qualification_authority"]["report_path"]
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        _self_hash(qualification, "report_sha256")
        if qualification.get("status") != "FRESH_QUALIFICATION_GATE_PASSED":
            raise SystemExit("TIR structural formal split remains locked")
    dataset_path = args.dataset_root / "TIR-Bench.json"
    if _sha256(dataset_path) != config["dataset"]["file_sha256"]:
        raise SystemExit("TIR dataset hash mismatch")
    all_rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in all_rows}
    sample_ids = list(map(str, config["splits"][args.split]))
    code_paths = [
        Path(__file__).resolve(),
        REPO / "scripts/run_tir_maze_topology_v2.py",
        REPO / "src/motif_transfer/tir_maze_topology.py",
        REPO / "src/motif_transfer/relational_structural_induction.py",
    ]
    collection_contract = stable_hash({
        "config": config,
        "split": args.split,
        "code_sha256": {str(path): _sha256(path) for path in code_paths},
        "source_artifact_sha256": artifact["artifact_sha256"],
        "source_confirmation_sha256": confirmation["report_sha256"],
    })
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / f"{args.split}_receipts.json"
    existing: dict[str, Any] = {}
    if receipts_path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(receipts_path.read_text(encoding="utf-8"))
        }
        if any(
            row.get("collection_contract_sha256") != collection_contract
            for row in existing.values()
        ):
            raise SystemExit("resumed TIR structural receipt contract mismatch")
    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_sample,
                sample_id,
                row=index[sample_id],
                dataset_root=args.dataset_root,
                config=config,
                artifact=artifact,
                api_key=str(key),
                contract_sha256=collection_contract,
            ): sample_id
            for sample_id in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                existing[sample_id] = future.result()
            except Exception as exc:
                print(json.dumps({
                    "failed": sample_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                continue
            ordered = [existing[value] for value in sample_ids if value in existing]
            receipts_path.write_text(
                json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "completed": sample_id,
                "progress": f"{len(ordered)}/{len(sample_ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in sample_ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete TIR structural receipts; rerun: {missing}")
    tier = (
        "FRESH_QUALIFICATION" if args.split == "qualification"
        else "FRESH_FORMAL_CONFIRMATION"
    )
    report = evaluate_tir_maze_structural_transfer(
        [existing[sample_id] for sample_id in sample_ids],
        source_artifact=artifact,
        source_confirmation=confirmation,
        expected_ids=sample_ids,
        evidence_tier=tier,
        claim_boundary=str(config["claim_boundary"][args.split]),
    )
    report_body = dict(report)
    report_body.pop("report_sha256")
    report_body["integrity"] = {
        "config_file_sha256": _sha256(config_path),
        "collection_contract_sha256": collection_contract,
        "receipts_file_sha256": _sha256(receipts_path),
        "source_artifact_file_sha256": _sha256(artifact_path),
        "source_confirmation_file_sha256": _sha256(confirmation_path),
    }
    report_body["formal_heldout_consumed"] = args.split == "heldout"
    report_body["report_sha256"] = stable_hash(report_body)
    report_path = args.output_dir / f"{args.split}_report.json"
    report_path.write_text(
        json.dumps(report_body, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report_body["status"],
        "summaries": report_body["summaries"],
        "paired": report_body["paired"],
        "gates": report_body["gates"],
        "report": str(report_path.resolve()),
    }, indent=2))
    return 0 if all(report_body["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
