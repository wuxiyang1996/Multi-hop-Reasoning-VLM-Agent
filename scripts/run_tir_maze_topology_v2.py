#!/usr/bin/env python3
"""Run target-neural/source-symbolic Sokoban topology transfer on TIR maze."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping

from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.sokoban_topology_skill import (  # noqa: E402
    validate_topology_artifact,
)
from motif_transfer.tir_maze_topology import (  # noqa: E402
    CONDITIONS,
    evaluate_tir_maze_transfer,
    execute_maze_topology,
    validate_neural_binding,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: Mapping[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _image_data(image: Image.Image, *, max_side: int, quality: int) -> str:
    rendered = image.convert("RGB").copy()
    rendered.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rendered.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _json_call(
    client: OpenAI,
    *,
    model: str,
    system: str,
    text: str,
    image_url: str,
    maximum_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=maximum_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("target neural model returned empty JSON")
    payload = json.loads(raw)
    usage = response.usage
    return payload, {
        "model": str(response.model),
        "finish_reason": str(response.choices[0].finish_reason),
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "completion_tokens": int(usage.completion_tokens if usage else 0),
        "cost": float(getattr(usage, "cost", 0.0) or 0.0),
        "response_sha256": stable_hash(payload),
    }


def _bind(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    image_url: str,
    maximum_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Answer sequences and evaluator labels are unnecessary for grounding.  The
    # binder receives only the task instruction before the option question.
    instruction = prompt.split("Which of the following", 1)[0].strip()
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "Bind a visual maze interface, not its solution. Return valid JSON "
            "only and never choose an answer option."
        ),
        text=(
            instruction
            + "\nReturn {\"move_deltas\":{\"R\":[dx,dy],\"L\":[dx,dy],"
            "\"U\":[dx,dy],\"D\":[dx,dy]},\"start_color_rgb\":[r,g,b],"
            "\"goal_color_rgb\":[r,g,b],\"start_corner\":string,"
            "\"goal_corner\":string,\"confidence\":number}. Coordinates use "
            "+x right and +y down. RGB values should name the visible marker "
            "colors approximately."
        ),
        image_url=image_url,
        maximum_tokens=maximum_tokens,
    )
    binding = {
        "role": "TARGET_NATIVE_NEURAL_MAZE_BINDING",
        "answer_or_gold_seen": False,
        **payload,
    }
    validate_neural_binding(binding)
    return binding, usage


def _baseline(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    image_url: str,
    maximum_tokens: int,
) -> tuple[str, dict[str, Any]]:
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "Solve the visual maze multiple-choice task. Return concise JSON "
            "{\"answer\":\"A-E\",\"reason\":\"brief\"}."
        ),
        text=prompt,
        image_url=image_url,
        maximum_tokens=maximum_tokens,
    )
    answer = str(payload.get("answer") or "").strip().upper()[:1]
    if answer not in "ABCDE":
        raise ValueError("target baseline did not emit A-E")
    return answer, usage


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
        raise ValueError("TIR topology runner accepts single-image maze only")
    image_path = dataset_root / str(row["image_1"])
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    media = config["media"]
    image_url = _image_data(
        image,
        max_side=int(media["max_side"]),
        quality=int(media["jpeg_quality"]),
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
        client, model=model, prompt=str(row["prompt"]), image_url=image_url,
        maximum_tokens=int(model_config["maximum_output_tokens"]),
    )
    baseline, baseline_usage = _baseline(
        client, model=model, prompt=str(row["prompt"]), image_url=image_url,
        maximum_tokens=int(model_config["maximum_output_tokens"]),
    )
    condition_receipts = {
        condition: execute_maze_topology(
            image,
            str(row["prompt"]),
            neural_binding=binding,
            source_artifact=artifact,
            condition=condition,
        )
        for condition in CONDITIONS if condition != "raw_target_only"
    }
    body = {
        "schema_version": "tir-maze-topology-receipt-v1",
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
        "conditions": condition_receipts,
    }
    # The label is attached only after every condition has completed.
    body["gold_answer_evaluator_only"] = str(row["answer"])
    return body | {"receipt_sha256": stable_hash(body)}


def _validate_integrity(config: Mapping[str, Any]) -> None:
    for relative, expected in config.get("integrity", {}).get("file_sha256", {}).items():
        path = (REPO / relative).resolve()
        if _sha256(path) != str(expected):
            raise SystemExit(f"frozen TIR-maze dependency changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("consumed_development", "qualification", "heldout"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.split != "consumed_development":
        if config.get("status") != "FROZEN_BEFORE_FRESH_QUALIFICATION":
            raise SystemExit("TIR-maze formal config is not frozen")
        _validate_integrity(config)
        development_path = (
            REPO / config["development_authority"]["report_path"]
        ).resolve()
        development = json.loads(development_path.read_text(encoding="utf-8"))
        _self_hash(development, "report_sha256")
        if development.get("status") != "CONSUMED_DEVELOPMENT_GATE_PASSED":
            raise SystemExit("consumed TIR-maze development did not authorize qualification")
    artifact_path = (REPO / config["source"]["artifact_path"]).resolve()
    confirmation_path = (REPO / config["source"]["confirmation_path"]).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    validate_topology_artifact(artifact)
    _self_hash(confirmation, "report_sha256")
    if not confirmation.get("source_gate_passed"):
        raise SystemExit("source topology confirmation did not pass")
    if args.split == "heldout":
        authority_path = (
            REPO / config["qualification_authority"]["report_path"]
        ).resolve()
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        _self_hash(authority, "report_sha256")
        if authority.get("status") != "FRESH_QUALIFICATION_GATE_PASSED":
            raise SystemExit("TIR-maze heldout remains locked after failed qualification")

    dataset_path = args.dataset_root / "TIR-Bench.json"
    if _sha256(dataset_path) != config["dataset"]["file_sha256"]:
        raise SystemExit("TIR dataset hash mismatch")
    all_rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in all_rows}
    sample_ids = list(map(str, config["splits"][args.split]))
    code_paths = [
        Path(__file__).resolve(),
        REPO / "src/motif_transfer/tir_maze_topology.py",
        REPO / "src/motif_transfer/sokoban_topology_skill.py",
    ]
    collection_contract = stable_hash({
        "config": config,
        "split": args.split,
        "code_sha256": {str(path): _sha256(path) for path in code_paths},
        "source_artifact_sha256": str(artifact["artifact_sha256"]),
        "source_confirmation_sha256": str(confirmation["report_sha256"]),
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
            raise SystemExit("resumed TIR-maze receipt contract mismatch")
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
        raise SystemExit(f"incomplete TIR-maze receipts; rerun: {missing}")
    tier = {
        "consumed_development": "CONSUMED_DEVELOPMENT",
        "qualification": "FRESH_QUALIFICATION",
        "heldout": "FRESH_FORMAL_CONFIRMATION",
    }[args.split]
    report = evaluate_tir_maze_transfer(
        [existing[sample_id] for sample_id in sample_ids],
        source_artifact=artifact,
        source_confirmation=confirmation,
        expected_ids=sample_ids,
        evidence_tier=tier,
        claim_boundary=str(config["claim_boundary"][args.split]),
    )
    body = dict(report)
    body.pop("report_sha256")
    body["integrity"] = {
        "config_file_sha256": _sha256(config_path),
        "collection_contract_sha256": collection_contract,
        "receipts_file_sha256": _sha256(receipts_path),
        "source_artifact_file_sha256": _sha256(artifact_path),
        "source_confirmation_file_sha256": _sha256(confirmation_path),
    }
    body["formal_heldout_consumed"] = args.split == "heldout"
    body["report_sha256"] = stable_hash(body)
    report_path = args.output_dir / f"{args.split}_report.json"
    report_path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": body["status"], "summaries": body["summaries"],
        "paired": body["paired"], "gates": body["gates"],
        "report": str(report_path.resolve()),
    }, indent=2))
    return 0 if all(body["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
