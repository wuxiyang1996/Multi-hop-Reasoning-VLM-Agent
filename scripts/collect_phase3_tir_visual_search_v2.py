#!/usr/bin/env python3
"""Collect neural-anchor H1/H4/H8 forks for TIR visual-search tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_typed_effect_induction import TYPED_EFFECTS  # noqa: E402
from motif_transfer.visual_wrapper_bridge import (  # noqa: E402
    build_tir_registry,
    route_question,
)


def _load_base_collector():
    path = REPO / "scripts/collect_phase3_tir_nonmaze.py"
    spec = importlib.util.spec_from_file_location("phase3_tir_collection_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Phase-3 TIR collection base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_collector()


def _expand_box(
    box: Sequence[float], *, scale: float = 1.0,
    dx: float = 0.0, dy: float = 0.0,
) -> list[float]:
    x, y, width, height = map(float, box)
    cx = x + width / 2 + dx * width
    cy = y + height / 2 + dy * height
    width = min(1.0, max(0.04, width * scale))
    height = min(1.0, max(0.04, height * scale))
    left = min(max(0.0, cx - width / 2), 1.0 - width)
    top = min(max(0.0, cy - height / 2), 1.0 - height)
    return [round(left, 6), round(top, 6), round(width, 6), round(height, 6)]


def expand_neural_anchor(
    anchor: Sequence[float], *, planner_score: float,
    raw_effects: Mapping[str, float], hypothesis: str,
) -> dict[str, Any]:
    """Turn one target-neural region into a real eight-call evidence program."""

    transforms = (
        (1.00, 0.00, 0.00),
        (1.35, 0.00, 0.00),
        (1.80, 0.00, 0.00),
        (2.40, 0.00, 0.00),
        (1.25, -0.55, 0.00),
        (1.25, 0.55, 0.00),
        (1.25, 0.00, -0.55),
        (1.25, 0.00, 0.55),
    )
    actions = [
        {
            "tool": "zoom_region",
            "normalized_box": _expand_box(
                anchor, scale=scale, dx=dx, dy=dy,
            ),
        }
        for scale, dx, dy in transforms
    ]
    body = {
        "schema_version": "phase3-tir-neural-anchor-program-v2",
        "actions": actions,
        "planner_score": float(planner_score),
        "raw_typed_effect_probabilities": {
            name: float(raw_effects[name]) for name in TYPED_EFFECTS
        },
        "target_hypothesis": str(hypothesis),
    }
    return body | {"candidate_id": stable_hash({
        "schema_version": body["schema_version"],
        "actions": actions,
    })}


def _propose_programs(
    client: OpenAI, *, model: str, prompt: str, overview: bytes,
    routing: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system = (
        "You are the target-native neural grounder for visual search. Do not "
        "answer the multiple-choice question. Propose exactly four distinct "
        "normalized image regions that test different referent/location "
        "hypotheses. Return JSON {\"anchors\":[{\"x\":0..1,\"y\":0..1,"
        "\"w\":0..1,\"h\":0..1,\"planner_score\":0..1,\"hypothesis\":"
        "string,\"typed_effect_probabilities\":{\"EFFECT_BY_TRANSITION_1\":"
        "0..1,\"EFFECT_BY_TRANSITION_4\":0..1,\"EFFECT_BY_TRANSITION_8\":"
        "0..1,\"EXECUTABLE_TRANSITION_PERSISTENCE\":0..1}}]}. Coordinates "
        "must be normalized fractions, not pixels. The first three effects "
        "estimate useful answer evidence after 1/4/8 multiscale zoom calls; "
        "persistence estimates whether neighboring refinements remain useful."
    )
    content = [
        {
            "type": "text",
            "text": (
                "Question: " + prompt + "\nRouting: "
                + json.dumps(routing, ensure_ascii=False)
            ),
        },
        BASE._image_content(overview),
    ]

    def parse(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        anchors = payload.get("anchors") or ()
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            raise ValueError("neural anchor response omitted anchors")
        if len(anchors) != 4:
            raise ValueError("neural grounder did not return exactly four anchors")
        output = []
        anchor_keys = set()
        for row in anchors:
            box = [float(row.get(key, -1)) for key in ("x", "y", "w", "h")]
            x, y, width, height = box
            if (
                x < 0 or y < 0 or width < 0.04 or height < 0.04
                or x + width > 1.000001 or y + height > 1.000001
            ):
                raise ValueError(
                    f"neural anchor is outside normalized image: {box}"
                )
            key = tuple(round(value, 5) for value in box)
            if key in anchor_keys:
                raise ValueError("neural grounder returned duplicate anchors")
            anchor_keys.add(key)
            score = float(row.get("planner_score", -1.0))
            effects = row.get("typed_effect_probabilities") or {}
            if not 0 <= score <= 1 or set(effects) != set(TYPED_EFFECTS):
                raise ValueError("neural anchor score/effect schema is invalid")
            if any(not 0 <= float(effects[name]) <= 1 for name in TYPED_EFFECTS):
                raise ValueError("neural anchor effect is outside [0,1]")
            output.append(expand_neural_anchor(
                box, planner_score=score, raw_effects=effects,
                hypothesis=str(row.get("hypothesis") or ""),
            ))
        if len({row["candidate_id"] for row in output}) != 4:
            raise ValueError("expanded target-native programs are not unique")
        return output

    usages = []
    errors = []
    for attempt in range(2):
        repair_suffix = ""
        if attempt:
            repair_suffix = (
                " This is a schema-repair retry. The prior response failed: "
                + errors[-1]
                + ". Return every required field for all four anchors."
            )
        payload, usage = BASE._json_call(
            client, model=model, system=system + repair_suffix,
            content=content, max_tokens=2200,
        )
        usages.append(usage)
        try:
            output = parse(payload)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            if attempt == 0:
                continue
            raise
        return output, {
            "calls": usages,
            "schema_repair_attempts": attempt,
            "schema_errors": errors,
        }
    raise RuntimeError("unreachable neural anchor repair state")


def _collection_contract(config: Mapping[str, Any]) -> str:
    wrapper = Path(config["wrapper"]["root"])
    paths = (
        Path(__file__).resolve(),
        REPO / "scripts/collect_phase3_tir_nonmaze.py",
        REPO / "src/motif_transfer/phase3_tir_nonmaze.py",
        REPO / "src/motif_transfer/visual_wrapper_bridge.py",
        wrapper / "visual_reasoning_wrapper/tools_visual.py",
        wrapper / "visual_reasoning_wrapper/question_router.py",
    )
    return stable_hash({
        "config_sha256": config["config_sha256"],
        "code_sha256": {str(path): BASE.file_sha256(path) for path in paths},
        "target_candidate_generation": "FOUR_NEURAL_ANCHORS_EXPANDED_TO_H1_H4_H8",
    })


def _collect_sample(
    sample_id: str, *, target_input: Mapping[str, Any], gold_answer: str,
    dataset_root: Path, config: Mapping[str, Any], api_key: str,
    contract_sha256: str,
) -> dict[str, Any]:
    if target_input.get("task") != "visual_search" or target_input.get("image_2"):
        raise ValueError("TIR V2 contract requires one visual_search image")
    image_path = dataset_root / str(target_input["image_1"])
    with Image.open(image_path) as handle:
        original_size = list(handle.size)
        maximum = int(config["media"]["native_working_max_side"])
        handle.draft("RGB", (maximum, maximum))
        image = handle.convert("RGB")
        image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
    wrapper_root = Path(config["wrapper"]["root"])
    registry = build_tir_registry(image, wrapper_root=wrapper_root)
    if "zoom_region" not in registry.tool_names():
        raise RuntimeError("wrapper lacks zoom_region")
    routing = route_question(
        str(target_input["prompt"]), modality="image", wrapper_root=wrapper_root,
    ).as_dict()
    media = config["media"]
    overview_image = BASE._thumbnail(image, int(media["overview_max_side"]))
    overview = BASE._image_bytes(
        overview_image, max_side=0, quality=int(media["jpeg_quality"]),
    )
    client = OpenAI(
        api_key=api_key, base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
        max_retries=int(config["model"]["max_retries"]),
    )
    model = str(config["model"]["id"])
    programs, proposal_usage = _propose_programs(
        client, model=model, prompt=str(target_input["prompt"]),
        overview=overview, routing=routing,
    )
    baseline, baseline_usage = BASE._answer(
        client, model=model, prompt=str(target_input["prompt"]),
        overview=overview, evidence=None, evidence_receipts=(),
    )
    candidates = []
    for program in programs:
        crops = []
        transitions = []
        signatures = set()
        state_hash = stable_hash({
            "sample_id": sample_id, "evidence": [], "target_outcome_read": False,
        })
        endpoints = {}
        endpoint_usage = {}
        for index, action in enumerate(program["actions"], start=1):
            crop, native = BASE._execute_action(registry, image, action)
            signature = BASE._perceptual_sha(crop, native["result"])
            nonredundant = signature not in signatures
            signatures.add(signature)
            next_state = stable_hash({
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
                "next_state_sha256": next_state,
                "formal_outcome_read": False,
            }
            transitions.append(transition | {
                "transition_tuple_sha256": stable_hash(transition),
            })
            crops.append(crop)
            state_hash = next_state
            if index in {1, 4, 8}:
                evidence = BASE._image_bytes(
                    BASE._collage(crops),
                    max_side=int(media["evidence_max_side"]),
                    quality=int(media["jpeg_quality"]),
                )
                answer, usage = BASE._answer(
                    client, model=model, prompt=str(target_input["prompt"]),
                    overview=overview, evidence=evidence,
                    evidence_receipts=transitions,
                )
                endpoints[str(index)] = answer | {
                    "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
                    "evidence_state_sha256": state_hash,
                }
                endpoint_usage[str(index)] = usage
        candidates.append(program | {
            "endpoints": endpoints,
            "transitions": transitions,
            "observed_persistence_fraction": len(signatures) / 8.0,
            "endpoint_usage": endpoint_usage,
        })
    body = {
        "schema_version": "phase3-tir-visual-search-receipt-v2",
        "collection_contract_sha256": contract_sha256,
        "sample_id": sample_id,
        "family": "visual_search",
        "image_path": str(image_path),
        "image_sha256": BASE.file_sha256(image_path),
        "original_image_size": original_size,
        "image_size": list(image.size),
        "overview_size": list(overview_image.size),
        "overview_sha256": hashlib.sha256(overview).hexdigest(),
        "wrapper_routing": routing,
        "wrapper_tool_names": list(registry.tool_names()),
        "proposal_usage": proposal_usage,
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
    config = json.loads(args.config.read_text())
    body = dict(config)
    claimed = str(body.pop("config_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise SystemExit("TIR V2 manifest hash mismatch")
    if config.get("status") != "FROZEN_BEFORE_ANY_TIR_V2_TARGET_CALL":
        raise SystemExit("TIR V2 manifest is not frozen")
    for relative, expected in config["integrity"]["code_sha256"].items():
        if BASE.file_sha256(REPO / relative) != str(expected):
            raise SystemExit(f"TIR V2 frozen dependency changed: {relative}")
    dataset_file = args.dataset_root / "TIR-Bench.json"
    if BASE.file_sha256(dataset_file) != config["dataset"]["sha256"]:
        raise SystemExit("TIR dataset drift")
    Image.MAX_IMAGE_PIXELS = int(config["media"]["maximum_source_pixels"])
    ids = list(map(str, config["splits"][args.stage]))
    rows = json.loads(dataset_file.read_text())
    index = {str(row["id"]): row for row in rows}
    if missing := [sample_id for sample_id in ids if sample_id not in index]:
        raise SystemExit(f"frozen IDs missing: {missing}")
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
        if any(
            row.get("collection_contract_sha256") != contract
            for row in existing.values()
        ):
            raise SystemExit("TIR V2 receipt contract mismatch")
    pending = [sample_id for sample_id in ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for sample_id in pending:
            row = index[sample_id]
            target_input = {key: value for key, value in row.items() if key != "answer"}
            futures[executor.submit(
                _collect_sample, sample_id, target_input=target_input,
                gold_answer=str(row["answer"]), dataset_root=args.dataset_root,
                config=config, api_key=str(key), contract_sha256=contract,
            )] = sample_id
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
                json.dumps(ordered, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps({
                "completed": sample_id, "progress": f"{len(ordered)}/{len(ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete TIR V2 receipts; rerun: {missing}")
    print(json.dumps({
        "stage": args.stage,
        "receipts": len(ids),
        "collection_contract_sha256": contract,
        "receipts_file_sha256": BASE.file_sha256(receipts_path),
        "formal_outcome_exposed_to_neural_calls": False,
        "output": str(receipts_path.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
