#!/usr/bin/env python3
"""TIR parameterized wrapper interventions with matched adaptation forks."""

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
from typing import Any, Mapping, Sequence

from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.active_video_transfer import (  # noqa: E402
    ANSWER_SLOTS,
    normalized_probabilities,
    stable_hash,
)
from motif_transfer.candidate_transfer_experiment import (  # noqa: E402
    evaluate_candidate_adaptation,
)
from motif_transfer.visual_wrapper_bridge import (  # noqa: E402
    TIR_INTERVENTION_TOOLS,
    build_tir_registry,
    execute_tir_intervention,
    route_question,
    tir_tool_schemas,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_contract(config: Mapping[str, Any]) -> str:
    wrapper_root = Path(config["wrapper"]["root"])
    paths = (
        Path(__file__).resolve(),
        REPO / "src/motif_transfer/active_video_transfer.py",
        REPO / "src/motif_transfer/candidate_transfer_experiment.py",
        REPO / "src/motif_transfer/visual_wrapper_bridge.py",
        wrapper_root / "visual_reasoning_wrapper/tools_visual.py",
        wrapper_root / "visual_reasoning_wrapper/question_router.py",
    )
    return stable_hash({
        "config": config,
        "code_sha256": {str(path): file_sha256(path) for path in paths},
    })


def _thumbnail(image: Image.Image, *, max_side: int) -> Image.Image:
    output = image.convert("RGB").copy()
    if max_side:
        output.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return output


def _image_bytes(image: Image.Image, *, max_side: int, quality: int) -> bytes:
    output = _thumbnail(image, max_side=max_side)
    buffer = io.BytesIO()
    output.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _image_content(data: bytes) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        },
    }


def _json_call(
    client: OpenAI,
    *,
    model: str,
    system: str,
    content: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("model returned no JSON content")
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


def _answer(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    overview: bytes,
    evidence: bytes | None,
    wrapper_receipt: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            "Solve this TIR multiple-choice task from visible evidence. Return a "
            "probability for every A--F slot; use near-zero mass for absent "
            "choices. Do the requested spatial/numeric/temporal reasoning rather "
            "than guessing from option frequency. Task: " + prompt
        ),
    }, {"type": "text", "text": "Low-bandwidth overview:"}, _image_content(overview)]
    if evidence is not None and wrapper_receipt is not None:
        compact = {
            "tool": wrapper_receipt["tool"],
            "arguments": wrapper_receipt["arguments"],
            "result": wrapper_receipt["result"],
        }
        content.extend([
            {
                "type": "text",
                "text": (
                    "Target-native wrapper intervention receipt: "
                    + json.dumps(compact, ensure_ascii=False)
                    + "\nMatching high-resolution re-observation:"
                ),
            },
            _image_content(evidence),
        ])
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "Return concise JSON {\"answer\":\"A-F\",\"probabilities\":"
            "{\"A\":number,...,\"F\":number},\"reason\":\"brief grounded "
            "reason\"}. Do not claim to see evidence outside supplied images."
        ),
        content=content,
        max_tokens=1200,
    )
    probabilities = normalized_probabilities(payload.get("probabilities") or {})
    answer = str(payload.get("answer") or "").strip().upper()[:1]
    if answer not in ANSWER_SLOTS:
        answer = ANSWER_SLOTS[int(probabilities.argmax())]
    return {
        "answer": answer,
        "probabilities": {
            slot: float(value) for slot, value in zip(ANSWER_SLOTS, probabilities)
        },
        "reason": str(payload.get("reason") or ""),
    }, usage


def _propose_actions(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    image_size: tuple[int, int],
    overview: bytes,
    tool_schemas: Sequence[Mapping[str, Any]],
    routing: Mapping[str, Any],
    candidate_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "Propose parameterized visual evidence interventions, not answers. "
            "Return JSON {\"actions\":[{\"candidate_id\":\"C0\","
            "\"tool\":\"zoom_region|read_text_region|describe_region\","
            "\"arguments\":{\"x\":int,\"y\":int,\"w\":int,\"h\":int,"
            "\"zoom\":number,\"reason\":string},\"score\":number in [0,1],"
            "\"hypothesis\":\"what this intervention tests\"}]}."
        ),
        content=[{
            "type": "text",
            "text": (
                f"Propose exactly {candidate_count} distinct wrapper tool calls. "
                "They must test different visual hypotheses and use actual pixel "
                f"coordinates inside width={image_size[0]}, height={image_size[1]}. "
                "At least one action should preserve enough global context when the "
                "task requires a global path, orientation, or proportion. Do not "
                "solve the question and do not include an answer letter. "
                f"Question routing receipt: {json.dumps(routing, ensure_ascii=False)} "
                f"Available schemas: {json.dumps(tool_schemas, ensure_ascii=False)} "
                f"Task: {prompt}"
            ),
        }, _image_content(overview)],
        max_tokens=1800,
    )
    actions = list(payload.get("actions") or ())
    if len(actions) != candidate_count:
        raise ValueError("planner did not return the frozen candidate count")
    output = []
    seen_ids: set[str] = set()
    seen_calls: set[str] = set()
    for index, action in enumerate(actions):
        candidate_id = str(action.get("candidate_id") or f"C{index}")
        tool = str(action.get("tool") or "")
        arguments = dict(action.get("arguments") or {})
        if candidate_id in seen_ids or tool not in TIR_INTERVENTION_TOOLS:
            raise ValueError("planner emitted duplicate ID or unsupported wrapper tool")
        call_hash = stable_hash({"tool": tool, "arguments": arguments})
        if call_hash in seen_calls:
            raise ValueError("planner emitted duplicate intervention calls")
        score = float(action.get("score", 0.0))
        if not 0.0 <= score <= 1.0:
            raise ValueError("planner score is outside [0,1]")
        seen_ids.add(candidate_id)
        seen_calls.add(call_hash)
        output.append({
            "candidate_id": candidate_id,
            "tool": tool,
            "arguments": arguments,
            "planner_score": score,
            "hypothesis": str(action.get("hypothesis") or ""),
        })
    return output, usage


def _descriptor(
    tool: str,
    arguments: Mapping[str, Any],
    image_size: tuple[int, int],
) -> list[float]:
    width, height = image_size
    one_hot = [float(tool == value) for value in TIR_INTERVENTION_TOOLS]
    x = float(arguments["x"]) / width
    y = float(arguments["y"]) / height
    w = float(arguments["w"]) / width
    h = float(arguments["h"]) / height
    return one_hot + [x, y, w, h, w * h]


def _scale_overview_arguments(
    arguments: Mapping[str, Any],
    *,
    overview_size: tuple[int, int],
    image_size: tuple[int, int],
    minimum_overview_crop_side: int,
) -> dict[str, Any]:
    """Map planner coordinates on the supplied overview to native pixels."""

    overview_width, overview_height = overview_size
    image_width, image_height = image_size
    minimum_width = min(minimum_overview_crop_side, overview_width)
    minimum_height = min(minimum_overview_crop_side, overview_height)
    x = max(0, min(int(arguments.get("x", 0)), overview_width - minimum_width))
    y = max(0, min(int(arguments.get("y", 0)), overview_height - minimum_height))
    width = max(
        minimum_width,
        min(int(arguments.get("w", overview_width)), overview_width - x),
    )
    height = max(
        minimum_height,
        min(int(arguments.get("h", overview_height)), overview_height - y),
    )
    width = min(width, overview_width - x)
    height = min(height, overview_height - y)
    output = dict(arguments)
    output.update({
        "x": round(x * image_width / overview_width),
        "y": round(y * image_height / overview_height),
        "w": max(1, round(width * image_width / overview_width)),
        "h": max(1, round(height * image_height / overview_height)),
    })
    return output


def _collect_sample(
    sample_id: str,
    *,
    row: Mapping[str, Any],
    dataset_root: Path,
    config: Mapping[str, Any],
    api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    if str(row.get("answer")) not in ANSWER_SLOTS:
        raise ValueError(f"sample is not native A--F MCQ: {sample_id}")
    if row.get("image_2"):
        raise ValueError(f"dual-image sample is outside this contract: {sample_id}")
    image_path = dataset_root / str(row["image_1"])
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    wrapper_root = Path(config["wrapper"]["root"])
    registry = build_tir_registry(image, wrapper_root=wrapper_root)
    routing = route_question(
        str(row["prompt"]), modality="image", wrapper_root=wrapper_root,
    ).as_dict()
    media = config["media"]
    overview_image = _thumbnail(
        image, max_side=int(media["overview_max_side"]),
    )
    overview = _image_bytes(
        overview_image, max_side=0, quality=int(media["jpeg_quality"]),
    )
    client = OpenAI(
        api_key=api_key,
        base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
        max_retries=int(config["model"]["max_retries"]),
    )
    model = str(config["model"]["id"])
    proposals, proposal_usage = _propose_actions(
        client,
        model=model,
        prompt=str(row["prompt"]),
        image_size=overview_image.size,
        overview=overview,
        tool_schemas=tir_tool_schemas(registry),
        routing=routing,
        candidate_count=int(config["interventions"]["candidate_count"]),
    )
    baseline, baseline_usage = _answer(
        client, model=model, prompt=str(row["prompt"]), overview=overview,
        evidence=None, wrapper_receipt=None,
    )
    candidates = []
    for proposal in proposals:
        native_arguments = _scale_overview_arguments(
            proposal["arguments"],
            overview_size=overview_image.size,
            image_size=image.size,
            minimum_overview_crop_side=int(config["interventions"][
                "minimum_overview_crop_side"
            ]),
        )
        crop, wrapper_receipt = execute_tir_intervention(
            registry,
            image,
            tool=str(proposal["tool"]),
            arguments=native_arguments,
        )
        evidence = _image_bytes(
            crop,
            max_side=int(media["evidence_max_side"]),
            quality=int(media["jpeg_quality"]),
        )
        answer, usage = _answer(
            client,
            model=model,
            prompt=str(row["prompt"]),
            overview=overview,
            evidence=evidence,
            wrapper_receipt=wrapper_receipt,
        )
        candidates.append({
            **proposal,
            "overview_arguments": proposal["arguments"],
            "arguments": native_arguments,
            "descriptor": _descriptor(
                str(proposal["tool"]), wrapper_receipt["arguments"], image.size,
            ),
            "wrapper_receipt": wrapper_receipt,
            "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
            "answer": answer,
            "usage": usage,
        })
    return {
        "schema_version": 1,
        "collection_contract_sha256": contract_sha256,
        "sample_id": sample_id,
        "family": str(row["task"]),
        "gold_answer": str(row["answer"]),
        "image_path": str(image_path),
        "image_sha256": file_sha256(image_path),
        "image_size": list(image.size),
        "overview_size": list(overview_image.size),
        "overview_sha256": hashlib.sha256(overview).hexdigest(),
        "wrapper_routing": routing,
        "wrapper_tool_names": registry.tool_names(),
        "proposal_usage": proposal_usage,
        "baseline": {"answer": baseline, "usage": baseline_usage},
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--split", choices=("adaptation",), default="adaptation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract_sha256 = _collection_contract(config)
    sample_ids = list(config["splits"]["adaptation"])
    rows = json.loads((args.dataset_root / "TIR-Bench.json").read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in rows}
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in index]
    if missing_ids:
        raise SystemExit(f"frozen sample IDs are missing: {missing_ids}")
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / "receipts.json"
    existing: dict[str, Any] = {}
    if receipts_path.is_file():
        existing = {
            str(row["sample_id"]): row
            for row in json.loads(receipts_path.read_text(encoding="utf-8"))
        }
        bad = [
            sample_id for sample_id, row in existing.items()
            if row.get("collection_contract_sha256") != contract_sha256
        ]
        if bad:
            raise SystemExit(f"receipt/config contract mismatch: {bad}")
    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_sample,
                sample_id,
                row=index[sample_id],
                dataset_root=args.dataset_root,
                config=config,
                api_key=str(key),
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
        raise SystemExit(f"incomplete receipts; rerun to resume: {missing}")
    receipts = [existing[sample_id] for sample_id in sample_ids]
    controlled_path = Path(config["source"]["controlled_v3_config"])
    controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
    if stable_hash(controlled) != config["source"]["controlled_v3_config_content_sha256"]:
        raise SystemExit("controlled source config content hash mismatch")
    report, artifact = evaluate_candidate_adaptation(
        receipts, config=config, controlled_config=controlled,
    )
    artifact_path = args.output_dir / "target_grounder_candidate.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["receipts"] = {
        "path": str(receipts_path.resolve()), "sha256": file_sha256(receipts_path),
    }
    report["target_grounder_candidate"] = {
        "path": str(artifact_path.resolve()),
        "sha256": file_sha256(artifact_path),
        "content_sha256": artifact["artifact_sha256"],
    }
    report_path = args.output_dir / "adaptation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "baseline_accuracy": report["baseline_accuracy"],
        "cross_fitted_selector_accuracy": report["cross_fitted_selector_accuracy"],
        "oracle_candidate_accuracy": report["oracle_candidate_accuracy"],
        "gates": report["gates"],
        "report": str(report_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
