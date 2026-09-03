#!/usr/bin/env python3
"""Collect matched H1/H4/H8 TIR non-maze evidence-program forks."""

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
from PIL import Image, ImageOps


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.active_video_transfer import (  # noqa: E402
    ANSWER_SLOTS,
    normalized_probabilities,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_tir_nonmaze import (  # noqa: E402
    TYPED_EFFECTS,
    target_native_program_bank,
)
from motif_transfer.visual_wrapper_bridge import (  # noqa: E402
    build_tir_registry,
    execute_tir_intervention,
    route_question,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_contract(config: Mapping[str, Any]) -> str:
    wrapper = Path(config["wrapper"]["root"])
    paths = (
        Path(__file__).resolve(),
        REPO / "src/motif_transfer/phase3_tir_nonmaze.py",
        REPO / "src/motif_transfer/visual_wrapper_bridge.py",
        wrapper / "visual_reasoning_wrapper/tools_visual.py",
        wrapper / "visual_reasoning_wrapper/question_router.py",
    )
    return stable_hash({
        "config_sha256": config["config_sha256"],
        "program_bank": target_native_program_bank(),
        "code_sha256": {str(path): file_sha256(path) for path in paths},
    })


def _thumbnail(image: Image.Image, max_side: int) -> Image.Image:
    output = image.convert("RGB").copy()
    if max_side:
        output.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return output


def _image_bytes(image: Image.Image, *, max_side: int, quality: int) -> bytes:
    output = _thumbnail(image, max_side)
    buffer = io.BytesIO()
    output.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _image_content(data: bytes) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
        },
    }


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(nested) for nested in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_call(
    client: OpenAI, *, model: str, system: str,
    content: list[dict[str, Any]], max_tokens: int,
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
        raise ValueError("target neural model returned no JSON")
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


def _rate_programs(
    client: OpenAI, *, model: str, prompt: str, overview: bytes,
    routing: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    bank = target_native_program_bank()
    compact = [
        {
            "candidate_id": row["candidate_id"],
            "descriptor": row["descriptor"],
            "actions": row["actions"],
        }
        for row in bank
    ]
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "You are a target-native visual evidence grounder. Do not answer the "
            "question and do not use source-game concepts. Rate the four supplied "
            "wrapper evidence programs. Return JSON {\"programs\":[{\"candidate_id\":"
            "string,\"planner_score\":0..1,\"typed_effect_probabilities\":{"
            "\"EFFECT_BY_TRANSITION_1\":0..1,\"EFFECT_BY_TRANSITION_4\":0..1,"
            "\"EFFECT_BY_TRANSITION_8\":0..1,"
            "\"EXECUTABLE_TRANSITION_PERSISTENCE\":0..1}}]}. The first three "
            "values estimate whether evidence accumulated after 1, 4, or 8 real "
            "wrapper calls will support a correct answer. Persistence estimates "
            "whether the full program will keep producing distinct usable evidence."
        ),
        content=[
            {
                "type": "text",
                "text": (
                    "Question: " + prompt
                    + "\nQuestion routing: " + json.dumps(routing, ensure_ascii=False)
                    + "\nTarget-native program bank: "
                    + json.dumps(compact, ensure_ascii=False)
                ),
            },
            _image_content(overview),
        ],
        max_tokens=1800,
    )
    rows = payload.get("programs") or ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("grounder response omitted programs")
    expected = {row["candidate_id"] for row in bank}
    output = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        effects = row.get("typed_effect_probabilities") or {}
        if candidate_id not in expected or candidate_id in output:
            raise ValueError("grounder returned unknown or duplicate candidate")
        if set(effects) != set(TYPED_EFFECTS):
            raise ValueError("grounder returned the wrong typed-effect schema")
        values = {}
        for name in TYPED_EFFECTS:
            value = float(effects[name])
            if not 0 <= value <= 1:
                raise ValueError("grounder effect probability is outside [0,1]")
            values[name] = value
        score = float(row.get("planner_score", 0.0))
        if not 0 <= score <= 1:
            raise ValueError("grounder planner_score is outside [0,1]")
        output[candidate_id] = {
            "planner_score": score,
            "raw_typed_effect_probabilities": values,
        }
    if set(output) != expected:
        raise ValueError("grounder did not rate every target-native program")
    return output, usage


def _answer(
    client: OpenAI, *, model: str, prompt: str, overview: bytes,
    evidence: bytes | None, evidence_receipts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Solve this TIR multiple-choice visual-reasoning task. Return a "
                "probability for every A--F slot, using near-zero mass for absent "
                "choices. Use only the supplied overview and target-native evidence. "
                "Question: " + prompt
            ),
        },
        {"type": "text", "text": "Overview:"},
        _image_content(overview),
    ]
    if evidence is not None:
        compact = [
            {
                "transition": row["transition"],
                "tool": row["action"]["tool"],
                "normalized_box": row["action"]["normalized_box"],
                "result": row["effect"]["tool_result"],
                "nonredundant": row["effect"]["nonredundant"],
            }
            for row in evidence_receipts
        ]
        content.extend([
            {
                "type": "text",
                "text": (
                    "Accumulated wrapper receipts in transition order: "
                    + json.dumps(compact, ensure_ascii=False)
                    + "\nMatching regional evidence collage:"
                ),
            },
            _image_content(evidence),
        ])
    payload, usage = _json_call(
        client,
        model=model,
        system=(
            "Return concise JSON {\"answer\":\"A-F\",\"probabilities\":{"
            "\"A\":number,...,\"F\":number},\"reason\":\"brief grounded reason\"}. "
            "Never claim evidence not supplied."
        ),
        content=content,
        max_tokens=1000,
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


def _native_box(
    normalized: Sequence[float], image_size: tuple[int, int],
) -> dict[str, int]:
    x, y, width, height = map(float, normalized)
    image_width, image_height = image_size
    left = max(0, min(round(x * image_width), image_width - 1))
    top = max(0, min(round(y * image_height), image_height - 1))
    right = max(left + 1, min(round((x + width) * image_width), image_width))
    bottom = max(top + 1, min(round((y + height) * image_height), image_height))
    return {"x": left, "y": top, "w": right - left, "h": bottom - top}


def _perceptual_sha(crop: Image.Image, result: Mapping[str, Any]) -> str:
    proxy = crop.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR)
    compact_result = dict(result)
    compact_result.pop("region", None)
    return stable_hash({
        "pixels_sha256": hashlib.sha256(proxy.tobytes()).hexdigest(),
        "tool_result": compact_result,
    })


def _execute_action(
    registry, image: Image.Image, action: Mapping[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    tool = str(action["tool"])
    native = _native_box(action["normalized_box"], image.size)
    if tool == "zoom_region":
        crop, receipt = execute_tir_intervention(
            registry, image, tool=tool,
            arguments=native | {"zoom": 2.0, "reason": "phase3 evidence state"},
        )
        result = _json_native(receipt["result"])
    elif tool == "extract_colors":
        response = registry.dispatch(tool, native | {"top_k": 6})
        if response.error:
            raise RuntimeError(f"wrapper extract_colors failed: {response.error}")
        result = _json_native(response.result or {})
        if not result.get("colors"):
            raise RuntimeError("wrapper extract_colors returned no color evidence")
        crop = image.crop((
            native["x"], native["y"],
            native["x"] + native["w"], native["y"] + native["h"],
        )).convert("RGB")
    else:
        raise ValueError(f"unsupported Phase-3 TIR native tool: {tool}")
    return crop, {
        "tool": tool,
        "normalized_box": list(map(float, action["normalized_box"])),
        "native_arguments": native,
        "result": result,
    }


def _collage(crops: Sequence[Image.Image]) -> Image.Image:
    if not crops:
        raise ValueError("cannot create empty evidence collage")
    tile = (256, 256)
    columns = 2
    rows = (len(crops) + columns - 1) // columns
    output = Image.new("RGB", (tile[0] * columns, tile[1] * rows), "white")
    for index, crop in enumerate(crops):
        fitted = ImageOps.contain(crop.convert("RGB"), tile, Image.Resampling.LANCZOS)
        left = (index % columns) * tile[0] + (tile[0] - fitted.width) // 2
        top = (index // columns) * tile[1] + (tile[1] - fitted.height) // 2
        output.paste(fitted, (left, top))
    return output


def _collect_sample(
    sample_id: str, *, target_input: Mapping[str, Any], gold_answer: str,
    dataset_root: Path, config: Mapping[str, Any], api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    if target_input.get("task") != "color" or target_input.get("image_2"):
        raise ValueError("Phase-3 TIR non-maze contract requires one color image")
    image_path = dataset_root / str(target_input["image_1"])
    with Image.open(image_path) as handle:
        original_image_size = list(handle.size)
        working_max_side = int(config["media"]["native_working_max_side"])
        # JPEG draft requests decoder-level subsampling before materialisation;
        # the following thumbnail is the format-independent hard bound.
        handle.draft("RGB", (working_max_side, working_max_side))
        image = handle.convert("RGB")
        image.thumbnail(
            (working_max_side, working_max_side), Image.Resampling.LANCZOS,
        )
    wrapper_root = Path(config["wrapper"]["root"])
    registry = build_tir_registry(image, wrapper_root=wrapper_root)
    required = {"zoom_region", "extract_colors"}
    if not required <= set(registry.tool_names()):
        raise RuntimeError("wrapper lacks Phase-3 TIR target-native tools")
    routing = route_question(
        str(target_input["prompt"]), modality="image", wrapper_root=wrapper_root,
    ).as_dict()
    media = config["media"]
    overview_image = _thumbnail(image, int(media["overview_max_side"]))
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
    ratings, rating_usage = _rate_programs(
        client, model=model, prompt=str(target_input["prompt"]),
        overview=overview, routing=routing,
    )
    baseline, baseline_usage = _answer(
        client, model=model, prompt=str(target_input["prompt"]),
        overview=overview, evidence=None, evidence_receipts=(),
    )

    candidates = []
    for program in target_native_program_bank():
        crops = []
        transitions = []
        signatures: set[str] = set()
        state_hash = stable_hash({
            "sample_id": sample_id, "evidence": [], "target_outcome_read": False,
        })
        endpoints = {}
        endpoint_usage = {}
        for index, action in enumerate(program["actions"], start=1):
            crop, native = _execute_action(registry, image, action)
            signature = _perceptual_sha(crop, native["result"])
            nonredundant = signature not in signatures
            signatures.add(signature)
            next_state_hash = stable_hash({
                "previous_state_sha256": state_hash,
                "evidence_signature": signature,
                "transition": index,
            })
            transition = {
                "transition": index,
                "state_sha256": state_hash,
                "action": {
                    "tool": native["tool"],
                    "normalized_box": native["normalized_box"],
                    "native_arguments": native["native_arguments"],
                    "action_sha256": stable_hash({
                        "tool": native["tool"],
                        "normalized_box": native["normalized_box"],
                    }),
                },
                "effect": {
                    "evidence_signature": signature,
                    "nonredundant": nonredundant,
                    "tool_result": native["result"],
                },
                "next_state_sha256": next_state_hash,
                "formal_outcome_read": False,
            }
            transitions.append(transition | {
                "transition_tuple_sha256": stable_hash(transition),
            })
            crops.append(crop)
            state_hash = next_state_hash
            if index in {1, 4, 8}:
                evidence = _image_bytes(
                    _collage(crops), max_side=int(media["evidence_max_side"]),
                    quality=int(media["jpeg_quality"]),
                )
                answer, usage = _answer(
                    client, model=model, prompt=str(target_input["prompt"]),
                    overview=overview, evidence=evidence,
                    evidence_receipts=transitions,
                )
                endpoints[str(index)] = answer | {
                    "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
                    "evidence_state_sha256": state_hash,
                }
                endpoint_usage[str(index)] = usage
        candidate = {
            **program,
            **ratings[program["candidate_id"]],
            "endpoints": endpoints,
            "transitions": transitions,
            "observed_persistence_fraction": len(signatures) / 8.0,
            "endpoint_usage": endpoint_usage,
        }
        candidates.append(candidate)

    # The gold answer is attached only after every planner/answer call has
    # completed.  It was never included in target_input or any model payload.
    body = {
        "schema_version": "phase3-tir-nonmaze-intervention-receipt-v1",
        "collection_contract_sha256": contract_sha256,
        "sample_id": sample_id,
        "family": str(target_input["task"]),
        "image_path": str(image_path),
        "image_sha256": file_sha256(image_path),
        "original_image_size": original_image_size,
        "image_size": list(image.size),
        "overview_size": list(overview_image.size),
        "overview_sha256": hashlib.sha256(overview).hexdigest(),
        "wrapper_routing": routing,
        "wrapper_tool_names": list(registry.tool_names()),
        "target_program_bank_sha256": stable_hash(target_native_program_bank()),
        "rating_usage": rating_usage,
        "baseline": baseline,
        "baseline_usage": baseline_usage,
        "candidates": candidates,
        "formal_outcome_exposed_to_neural_calls": False,
        "source_program_or_identity_exposed_to_neural_calls": False,
        "gold_answer": str(gold_answer),
    }
    return body | {"receipt_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("development_train", "development_validation",
                            "qualification", "formal"), required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_body = dict(config)
    claimed_config = str(config_body.pop("config_sha256", ""))
    if not claimed_config or stable_hash(config_body) != claimed_config:
        raise SystemExit("TIR Phase-3 split manifest hash mismatch")
    if config.get("status") != "FROZEN_BEFORE_ANY_PHASE3_TIR_TARGET_CALL":
        raise SystemExit("TIR Phase-3 split manifest is not frozen")
    Image.MAX_IMAGE_PIXELS = int(config["media"]["maximum_source_pixels"])
    for relative, expected in config["integrity"]["code_sha256"].items():
        if file_sha256(REPO / relative) != str(expected):
            raise SystemExit(f"TIR Phase-3 frozen dependency changed: {relative}")
    if file_sha256(args.dataset_root / "TIR-Bench.json") != config["dataset"]["sha256"]:
        raise SystemExit("TIR dataset drift")
    ids = list(map(str, config["splits"][args.stage]))
    rows = json.loads((args.dataset_root / "TIR-Bench.json").read_text())
    index = {str(row["id"]): row for row in rows}
    if missing := [sample_id for sample_id in ids if sample_id not in index]:
        raise SystemExit(f"frozen TIR IDs are missing: {missing}")
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    contract = _collection_contract(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / f"{args.stage}_receipts.json"
    existing = {}
    if receipts_path.is_file():
        existing = {
            str(row["sample_id"]): row for row in json.loads(receipts_path.read_text())
        }
        bad = [
            sample_id for sample_id, row in existing.items()
            if row.get("collection_contract_sha256") != contract
        ]
        if bad:
            raise SystemExit(f"receipt contract mismatch: {bad}")
    pending = [sample_id for sample_id in ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for sample_id in pending:
            row = index[sample_id]
            target_input = {
                key: value for key, value in row.items() if key != "answer"
            }
            future = executor.submit(
                _collect_sample,
                sample_id,
                target_input=target_input,
                gold_answer=str(row["answer"]),
                dataset_root=args.dataset_root,
                config=config,
                api_key=str(key),
                contract_sha256=contract,
            )
            futures[future] = sample_id
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
            ordered = [existing[value] for value in ids if value in existing]
            receipts_path.write_text(
                json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "completed": sample_id,
                "progress": f"{len(ordered)}/{len(ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete TIR receipts; rerun: {missing}")
    summary = {
        "stage": args.stage,
        "receipts": len(ids),
        "collection_contract_sha256": contract,
        "receipts_file_sha256": file_sha256(receipts_path),
        "formal_outcome_exposed_to_neural_calls": False,
        "source_program_or_identity_exposed_to_neural_calls": False,
        "output": str(receipts_path.resolve()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
