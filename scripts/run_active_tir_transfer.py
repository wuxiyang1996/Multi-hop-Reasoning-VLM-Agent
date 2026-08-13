#!/usr/bin/env python3
"""Active-crop TIR bridge for intervention-grounded TEST/COMMIT transfer."""

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

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.active_video_transfer import (  # noqa: E402
    ANSWER_SLOTS,
    VIDEO_CONDITIONS,
    CalibrationRow,
    GainRow,
    NeuralGainGrounder,
    SoftmaxCalibrationHead,
    build_source_value_models,
    choose_video_action,
    exact_binomial_two_sided,
    fit_calibration_head,
    fit_gain_grounder,
    normalized_entropy,
    normalized_probabilities,
    stable_hash,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_bytes(image: Image.Image, *, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _thumbnail(image: Image.Image, max_side: int) -> Image.Image:
    output = image.convert("RGB").copy()
    output.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return output


def _grid_crops(
    image: Image.Image,
    *,
    rows: int,
    columns: int,
    max_side: int,
    quality: int,
) -> tuple[list[dict[str, Any]], bytes]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    overview = _thumbnail(rgb, max_side)
    draw = ImageDraw.Draw(overview)
    scale_x = overview.width / width
    scale_y = overview.height / height
    crops = []
    for row in range(rows):
        for column in range(columns):
            left = round(column * width / columns)
            right = round((column + 1) * width / columns)
            top = round(row * height / rows)
            bottom = round((row + 1) * height / rows)
            crop = _thumbnail(rgb.crop((left, top, right, bottom)), max_side)
            crop_id = f"R{row}C{column}"
            draw.rectangle(
                (left * scale_x, top * scale_y, right * scale_x, bottom * scale_y),
                outline="red", width=2,
            )
            draw.text((left * scale_x + 3, top * scale_y + 3), crop_id, fill="red")
            crops.append({
                "id": crop_id,
                "box": [left, top, right, bottom],
                "bytes": _image_bytes(crop, quality=quality),
                "size": list(crop.size),
            })
    return crops, _image_bytes(overview, quality=quality)


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


def _rank_crops(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    crop_ids: Sequence[str],
    overview: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, usage = _json_call(
        client, model=model,
        system=(
            "Return concise JSON {\"ranking\":[all grid IDs],\"scores\":"
            "{each grid ID:number in [0,1]},\"rationale\":\"brief\"}. "
            "Do not answer the task."
        ),
        content=[
            {
                "type": "text",
                "text": (
                    "Using only this labeled low-resolution overview, rank every "
                    "grid cell by the expected value of inspecting its native "
                    "high-resolution crop for this task. Do not solve the task. "
                    f"Task: {prompt} Grid IDs: {list(crop_ids)}"
                ),
            },
            _image_content(overview),
        ],
        max_tokens=1200,
    )
    expected = set(crop_ids)
    ranking = [str(value) for value in payload.get("ranking", ())]
    if len(ranking) != len(expected) or set(ranking) != expected:
        raise ValueError("crop ranking is not a complete permutation")
    scores = {key: float((payload.get("scores") or {})[key]) for key in expected}
    if any(value < 0 for value in scores.values()):
        raise ValueError(f"crop score is negative: {scores}")
    maximum = max(scores.values())
    if maximum > 1:
        if maximum <= 10:
            scores = {key: value / 10 for key, value in scores.items()}
        elif maximum <= 100:
            scores = {key: value / 100 for key, value in scores.items()}
        else:
            raise ValueError(f"crop score scale is ambiguous: {scores}")
    return {
        "ranking": ranking,
        "scores": scores,
        "rationale": str(payload.get("rationale") or ""),
    }, usage


def _answer(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    overview: bytes,
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            "Answer the TIR task using the low-resolution overview and only the "
            "high-resolution crop receipts supplied below. Return probabilities "
            "for A--F even if fewer choices are present; assign absent choices "
            "near-zero probability. Task: " + prompt
        ),
    }, {"type": "text", "text": "Low-resolution overview:"}, _image_content(overview)]
    for crop in selected:
        content.append({
            "type": "text",
            "text": f"High-resolution crop receipt {crop['id']} box={crop['box']}:",
        })
        content.append(_image_content(crop["bytes"]))
    payload, usage = _json_call(
        client, model=model,
        system=(
            "Return concise JSON {\"answer\":\"A-F\",\"probabilities\":"
            "{\"A\":number,...,\"F\":number},\"confidence\":number,"
            "\"evidence_crops\":[...],\"reason\":\"brief\"}."
        ),
        content=content,
        max_tokens=1200,
    )
    probabilities = normalized_probabilities(payload.get("probabilities") or {})
    answer = str(payload.get("answer") or "").strip().upper()[:1]
    if answer not in ANSWER_SLOTS:
        answer = ANSWER_SLOTS[int(np.argmax(probabilities))]
    return {
        "answer": answer,
        "probabilities": {
            slot: float(value) for slot, value in zip(ANSWER_SLOTS, probabilities)
        },
        "confidence": float(payload.get("confidence") or max(probabilities)),
        "evidence_crops": [str(value) for value in payload.get("evidence_crops", ())],
        "reason": str(payload.get("reason") or ""),
    }, usage


def _collect_sample(
    sample_id: str,
    *,
    row: Mapping[str, Any],
    dataset_root: Path,
    config: Mapping[str, Any],
    api_key: str,
) -> dict[str, Any]:
    if str(row.get("answer")) not in ANSWER_SLOTS:
        raise ValueError(f"sample is not native A--F MCQ: {sample_id}")
    if row.get("image_2"):
        raise ValueError(f"dual-image sample is outside v1 contract: {sample_id}")
    image_path = dataset_root / str(row["image_1"])
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    media = config["media"]
    crops, overview = _grid_crops(
        image, rows=int(media["grid_rows"]), columns=int(media["grid_columns"]),
        max_side=int(media["max_side"]), quality=int(media["jpeg_quality"]),
    )
    client = OpenAI(
        api_key=api_key, base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
        max_retries=int(config["model"]["max_retries"]),
    )
    model = str(config["model"]["id"])
    planner, planner_usage = _rank_crops(
        client, model=model, prompt=str(row["prompt"]),
        crop_ids=[crop["id"] for crop in crops], overview=overview,
    )
    by_id = {crop["id"]: crop for crop in crops}
    max_tests = int(config["policy"]["max_tests"])
    answers = []
    for prefix in range(max_tests + 1):
        selected_ids = planner["ranking"][:prefix]
        answer, usage = _answer(
            client, model=model, prompt=str(row["prompt"]), overview=overview,
            selected=[by_id[value] for value in selected_ids],
        )
        answers.append({
            "prefix_length": prefix,
            "selected_crops": selected_ids,
            "answer": answer,
            "usage": usage,
        })
    return {
        "sample_id": sample_id,
        "task_family": str(row["task"]),
        "gold_answer": str(row["answer"]),
        "image_path": str(image_path),
        "image_sha256": file_sha256(image_path),
        "image_size": list(image.size),
        "overview_sha256": hashlib.sha256(overview).hexdigest(),
        "crops": [{
            "id": crop["id"], "box": crop["box"], "size": crop["size"],
            "sha256": hashlib.sha256(crop["bytes"]).hexdigest(),
        } for crop in crops],
        "planner": planner,
        "planner_usage": planner_usage,
        "answers": answers,
    }


def _calibration_rows(receipts: Sequence[Mapping[str, Any]], max_tests: int) -> list[CalibrationRow]:
    output = []
    for receipt in receipts:
        answer_index = ANSWER_SLOTS.index(str(receipt["gold_answer"]))
        ranking = receipt["planner"]["ranking"]
        scores = receipt["planner"]["scores"]
        for answer in receipt["answers"]:
            prefix = int(answer["prefix_length"])
            selected_scores = [float(scores[value]) for value in ranking[:prefix]]
            output.append(CalibrationRow(
                sample_id=str(receipt["sample_id"]), prefix_length=prefix,
                max_tests=max_tests,
                mean_planner_score=float(np.mean(selected_scores)) if selected_scores else 0.0,
                raw_probabilities=tuple(normalized_probabilities(
                    answer["answer"]["probabilities"]
                )),
                answer_index=answer_index,
            ))
    return output


def _predictions(
    rows: Sequence[CalibrationRow], head: SoftmaxCalibrationHead,
) -> dict[tuple[str, int], np.ndarray]:
    return {
        (row.sample_id, row.prefix_length): head.predict(row.features())
        for row in rows
    }


def _gain_rows(
    receipts: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, int], np.ndarray],
    *,
    max_tests: int,
) -> list[GainRow]:
    output = []
    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        ranking = receipt["planner"]["ranking"]
        scores = receipt["planner"]["scores"]
        for prefix in range(max_tests):
            before = predictions[(sample_id, prefix)]
            after = predictions[(sample_id, prefix + 1)]
            output.append(GainRow(
                sample_id=sample_id, current_belief=tuple(map(float, before)),
                next_planner_score=float(scores[ranking[prefix]]),
                prefix_fraction=prefix / max_tests,
                information_gain=normalized_entropy(before) - normalized_entropy(after),
                confidence_gain=float(np.max(after) - np.max(before)),
            ))
    return output


def _trace(
    receipt: Mapping[str, Any],
    *,
    condition: str,
    predictions: Mapping[tuple[str, int], np.ndarray],
    gain_grounder: NeuralGainGrounder,
    source_models,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(receipt["sample_id"])
    max_tests = int(policy["max_tests"])
    ranking = receipt["planner"]["ranking"]
    scores = receipt["planner"]["scores"]
    steps = []
    committed = None
    for prefix in range(max_tests + 1):
        belief = predictions[(sample_id, prefix)]
        next_score = float(scores[ranking[prefix]]) if prefix < max_tests else 0.0
        decision = choose_video_action(
            belief, condition=condition, prefix_length=prefix,
            max_tests=max_tests, next_planner_score=next_score,
            gain_grounder=gain_grounder, source_models=source_models,
            fallback_commit_threshold=float(policy["fallback_commit_threshold"]),
            uncertainty_scale=float(policy["uncertainty_scale"]),
            decision_margin=float(policy["decision_margin"]),
            information_gain_threshold=float(policy["information_gain_threshold"]),
        )
        steps.append({
            "prefix_length": prefix,
            "visible_receipt_ids": [
                f"{sample_id}:{value}" for value in ranking[:prefix]
            ],
            "belief": list(map(float, belief)),
            "decision": decision.__dict__,
        })
        if decision.kind == "COMMIT":
            committed = int(decision.answer_index)
            break
    if committed is None:
        committed = int(np.argmax(predictions[(sample_id, max_tests)]))
    gold = ANSWER_SLOTS.index(str(receipt["gold_answer"]))
    return {
        "sample_id": sample_id,
        "task_family": str(receipt["task_family"]),
        "condition": condition,
        "committed_answer": ANSWER_SLOTS[committed],
        "gold_answer": ANSWER_SLOTS[gold],
        "correct": committed == gold,
        "tests": len(steps) - 1 if steps[-1]["decision"]["kind"] == "COMMIT" else max_tests,
        "steps": steps,
    }


def _summaries(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        condition: {
            "samples": len(rows := [row for row in traces if row["condition"] == condition]),
            "accuracy": float(np.mean([row["correct"] for row in rows])),
            "mean_tests": float(np.mean([row["tests"] for row in rows])),
            "by_family": {
                family: {
                    "samples": len(family_rows := [
                        row for row in rows if row["task_family"] == family
                    ]),
                    "accuracy": float(np.mean([row["correct"] for row in family_rows])),
                    "mean_tests": float(np.mean([row["tests"] for row in family_rows])),
                }
                for family in sorted({row["task_family"] for row in rows})
            },
        }
        for condition in VIDEO_CONDITIONS
    }


def _paired(authentic: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    auth = {row["sample_id"]: row for row in authentic}
    base = {row["sample_id"]: row for row in control}
    if set(auth) != set(base):
        raise ValueError("paired trace identities differ")
    wins = sum(auth[key]["correct"] and not base[key]["correct"] for key in auth)
    losses = sum(base[key]["correct"] and not auth[key]["correct"] for key in auth)
    test_delta = float(np.mean([
        auth[key]["tests"] - base[key]["tests"] for key in auth
    ]))
    return {
        "wins": wins, "losses": losses,
        "ties": len(auth) - wins - losses,
        "accuracy_delta": (wins - losses) / len(auth),
        "mean_test_delta": test_delta,
        "exact_sign_p_two_sided": exact_binomial_two_sided(wins, losses),
    }


def _evaluate(
    receipts: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    head: SoftmaxCalibrationHead,
    gain_grounder: NeuralGainGrounder,
    source_models,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _calibration_rows(receipts, int(config["policy"]["max_tests"]))
    predictions = _predictions(rows, head)
    traces = [
        _trace(
            receipt, condition=condition, predictions=predictions,
            gain_grounder=gain_grounder, source_models=source_models,
            policy=config["policy"],
        )
        for receipt in receipts for condition in VIDEO_CONDITIONS
    ]
    by_condition = {
        condition: [row for row in traces if row["condition"] == condition]
        for condition in VIDEO_CONDITIONS
    }
    report = {
        "conditions": _summaries(traces),
        "authentic_paired": {
            control: _paired(
                by_condition["authentic_source_plus_target"], by_condition[control]
            )
            for control in VIDEO_CONDITIONS if control != "authentic_source_plus_target"
        },
    }
    return traces, report


def _adaptation(
    receipts: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    controlled: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    max_tests = int(config["policy"]["max_tests"])
    rows = _calibration_rows(receipts, max_tests)
    head = fit_calibration_head(rows, seed=int(config["target_grounder"]["seed"]))
    predictions = _predictions(rows, head)
    gain = fit_gain_grounder(
        _gain_rows(receipts, predictions, max_tests=max_tests),
        seed=int(config["target_grounder"]["gain_seed"]),
        hidden_units=int(config["target_grounder"]["gain_hidden_units"]),
    )
    source_models = build_source_value_models(
        controlled, seed=int(config["source"]["model_seed"]),
    )
    traces, evaluation = _evaluate(
        receipts, config=config, head=head, gain_grounder=gain,
        source_models=source_models,
    )
    raw_accuracy = {
        f"k{prefix}": float(np.mean([
            int(np.argmax(row.raw_probabilities)) == row.answer_index
            for row in rows if row.prefix_length == prefix
        ]))
        for prefix in range(max_tests + 1)
    }
    by_condition = {
        condition: {
            row["sample_id"]: tuple(
                step["decision"]["kind"] for step in row["steps"]
            )
            for row in traces if row["condition"] == condition
        }
        for condition in VIDEO_CONDITIONS
    }
    contrasts = sum(
        by_condition["authentic_source_plus_target"][key]
        != by_condition["target_only"][key]
        for key in by_condition["target_only"]
    )
    evidence_gain = max(raw_accuracy[f"k{k}"] for k in range(1, max_tests + 1)) - raw_accuracy["k0"]
    gates = {
        "all_receipts_complete": len(receipts) == len(config["splits"]["adaptation"]),
        "raw_overview_accuracy_above_chance": raw_accuracy["k0"] > 1 / 6,
        "positive_crop_evidence_response": evidence_gain > 0,
        "authentic_source_action_contrast": contrasts >= 2,
    }
    artifact = {
        "schema_version": 1,
        "role": "TARGET_NATIVE_TIR_GROUNDER_FROZEN_FROM_ADAPTATION_ONLY",
        "calibration_head": head.as_dict(),
        "gain_grounder": gain.as_dict(),
        "training_sample_ids": sorted({row.sample_id for row in rows}),
        "source_config_sha256": stable_hash(controlled),
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    report = {
        "schema_version": 1,
        "status": "ADAPTATION_PREFLIGHT_PASS" if all(gates.values()) else "ADAPTATION_PREFLIGHT_FAIL",
        "claim_boundary": "TIR adaptation only; no qualification or held-out IDs consumed.",
        "raw_accuracy_by_prefix": raw_accuracy,
        "best_crop_accuracy_gain_over_overview": evidence_gain,
        "authentic_vs_target_action_contrast_samples": contrasts,
        **evaluation,
        "gates": gates,
        "policy_traces": traces,
    }
    report["report_sha256"] = stable_hash(report)
    return report, artifact


def _formal(
    receipts: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    controlled: Mapping[str, Any],
    artifact: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    if stable_hash({key: value for key, value in artifact.items() if key != "artifact_sha256"}) != artifact["artifact_sha256"]:
        raise ValueError("target grounder artifact content hash mismatch")
    head = SoftmaxCalibrationHead.from_dict(artifact["calibration_head"])
    gain = NeuralGainGrounder.from_dict(artifact["gain_grounder"])
    source_models = build_source_value_models(
        controlled, seed=int(config["source"]["model_seed"]),
    )
    traces, evaluation = _evaluate(
        receipts, config=config, head=head, gain_grounder=gain,
        source_models=source_models,
    )
    comparisons = evaluation["authentic_paired"]
    gate = {
        "authentic_accuracy_strictly_above_target": comparisons["target_only"]["accuracy_delta"] > 0,
        "authentic_accuracy_strictly_above_shuffled": comparisons["shuffled_source_plus_target"]["accuracy_delta"] > 0,
        "authentic_accuracy_strictly_above_marginal": comparisons["source_marginal_plus_target"]["accuracy_delta"] > 0,
        "authentic_accuracy_not_below_target_native_ig": comparisons["target_native_information_gain"]["accuracy_delta"] >= 0,
        "authentic_has_action_contrasts": any(
            row["tests"] != next(item for item in traces if item["sample_id"] == row["sample_id"] and item["condition"] == "target_only")["tests"]
            for row in traces if row["condition"] == "authentic_source_plus_target"
        ),
    }
    report = {
        "schema_version": 1,
        "split": split,
        "status": f"{split.upper()}_PASS" if all(gate.values()) else f"{split.upper()}_FAIL",
        **evaluation,
        "gate": gate,
        "policy_traces": traces,
    }
    report["report_sha256"] = stable_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--split", choices=("adaptation", "qualification", "held_out"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grounder-artifact", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample_ids = list(config["splits"][args.split])
    if not sample_ids:
        raise SystemExit(f"config has no {args.split} IDs")
    rows = json.loads((args.dataset_root / "TIR-Bench.json").read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in rows}
    if any(sample_id not in index for sample_id in sample_ids):
        raise SystemExit("a frozen sample ID is missing")
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_dir / "receipts.json"
    existing = {}
    if receipts_path.is_file():
        existing = {
            row["sample_id"]: row
            for row in json.loads(receipts_path.read_text(encoding="utf-8"))
        }
    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_sample, sample_id, row=index[sample_id],
                dataset_root=args.dataset_root, config=config, api_key=str(key),
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
            receipts_path.write_text(json.dumps(
                [existing[value] for value in sample_ids if value in existing],
                ensure_ascii=False, indent=2,
            ) + "\n", encoding="utf-8")
            print(json.dumps({
                "completed": sample_id,
                "progress": f"{len(existing)}/{len(sample_ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in sample_ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete receipts; rerun to resume: {missing}")
    receipts = [existing[value] for value in sample_ids]
    controlled = json.loads(Path(
        config["source"]["controlled_v3_config"]
    ).read_text(encoding="utf-8"))
    if stable_hash(controlled) != config["source"]["controlled_v3_config_content_sha256"]:
        raise SystemExit("controlled V3 source config hash mismatch")
    if args.split == "adaptation":
        report, artifact = _adaptation(receipts, config=config, controlled=controlled)
        artifact_path = args.output_dir / "target_grounder_candidate.json"
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["target_grounder_candidate"] = {
            "path": str(artifact_path.resolve()),
            "file_sha256": file_sha256(artifact_path),
            "content_sha256": artifact["artifact_sha256"],
        }
        report_path = args.output_dir / "adaptation_report.json"
    else:
        if args.grounder_artifact is None:
            raise SystemExit("formal split requires --grounder-artifact")
        artifact = json.loads(args.grounder_artifact.read_text(encoding="utf-8"))
        report = _formal(
            receipts, config=config, controlled=controlled, artifact=artifact,
            split=args.split,
        )
        report_path = args.output_dir / f"{args.split}_report.json"
    report["receipts"] = {
        "path": str(receipts_path.resolve()), "sha256": file_sha256(receipts_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "conditions": report.get("conditions"),
        "authentic_paired": report.get("authentic_paired"),
        "report": str(report_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
