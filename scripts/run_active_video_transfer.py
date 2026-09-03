#!/usr/bin/env python3
"""Collect and evaluate target-native active Video-Holmes receipts."""

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

import cv2
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
    build_source_value_models,
    choose_video_action,
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


def _questions(dataset_root: Path, split: str) -> dict[str, dict[str, Any]]:
    path = dataset_root / "Benchmark" / f"{split}_Video-Holmes.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        f"{row['video ID']}.Q{int(row['Question ID'])}": row
        for row in rows
    }


def _video_path(dataset_root: Path, video_id: str) -> Path:
    root = dataset_root / "Benchmark"
    candidates = (
        root / "videos_cropped" / f"{video_id}.mp4",
        root / "videos" / "videos_cropped" / f"{video_id}.mp4",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(candidates[0])


def _panels(
    video_path: Path,
    *,
    segment_count: int,
    frames_per_segment: int,
    frame_width: int,
    jpeg_quality: int,
) -> tuple[list[bytes], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: {video_path}")
    duration = total_frames / fps
    columns = 4
    rows = (frames_per_segment + columns - 1) // columns
    frame_height = round(frame_width * 9 / 16)
    output: list[bytes] = []
    sampled_seconds: list[list[float]] = []
    for segment in range(segment_count):
        canvas = Image.new(
            "RGB", (columns * frame_width, 20 + rows * frame_height), "white",
        )
        ImageDraw.Draw(canvas).text(
            (5, 2),
            f"S{segment} {segment * duration / segment_count:.1f}-"
            f"{(segment + 1) * duration / segment_count:.1f}s",
            fill="black",
        )
        seconds = []
        for index in range(frames_per_segment):
            second = (
                segment + (index + 0.5) / frames_per_segment
            ) * duration / segment_count
            capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                raise RuntimeError(f"failed to decode {video_path} at {second:.3f}s")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((frame_width, frame_height))
            x = (index % columns) * frame_width
            y = 20 + (index // columns) * frame_height
            canvas.paste(image, (x, y))
            seconds.append(round(second, 3))
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=jpeg_quality)
        output.append(buffer.getvalue())
        sampled_seconds.append(seconds)
    capture.release()
    return output, {
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": duration,
        "sampled_seconds": sampled_seconds,
        "panel_sha256": [hashlib.sha256(value).hexdigest() for value in output],
    }


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


def _planner(
    client: OpenAI,
    *,
    model: str,
    row: Mapping[str, Any],
    panels: Sequence[bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        "Rank the labeled temporal segments by how useful their visible evidence is "
        "for answering the multiple-choice question. Also describe the visible "
        "events, actions, ordering, and readable text in each segment in enough "
        "detail for a separate reasoner. Do not answer the question. "
        "Return JSON with ranking containing every segment ID exactly once and "
        "scores and descriptions mapping every segment ID. Question: "
        f"{row['Question']} Options: {json.dumps(row['Options'], ensure_ascii=False)}"
    )
    payload, usage = _json_call(
        client, model=model,
        system=(
            "Return concise JSON {\"ranking\":[\"S0\",...],"
            "\"scores\":{\"S0\":number,...},\"descriptions\":"
            "{\"S0\":\"visible evidence only\",...},\"rationale\":\"brief\"}."
        ),
        content=[{"type": "text", "text": prompt}] + [
            _image_content(panel) for panel in panels
        ],
        max_tokens=2500,
    )
    expected = {f"S{index}" for index in range(len(panels))}
    ranking = [str(value) for value in payload.get("ranking", ())]
    scores = payload.get("scores") or {}
    descriptions = payload.get("descriptions") or {}
    if len(ranking) != len(expected) or set(ranking) != expected:
        raise ValueError("planner ranking is not a complete permutation")
    parsed_scores = {key: float(scores[key]) for key in expected}
    if any(value < 0 for value in parsed_scores.values()):
        raise ValueError(f"planner score is negative: {parsed_scores}")
    if any(value > 1 for value in parsed_scores.values()):
        # Some providers render an otherwise valid six-value score vector as
        # explicit percentages.  Accept only the unambiguous 0--100 case; do
        # not silently clip mildly out-of-range or mixed-scale outputs.
        maximum_score = max(parsed_scores.values())
        if maximum_score <= 10:
            parsed_scores = {
                key: value / 10.0 for key, value in parsed_scores.items()
            }
        elif maximum_score <= 100:
            parsed_scores = {
                key: value / 100.0 for key, value in parsed_scores.items()
            }
        else:
            raise ValueError(f"planner score scale is ambiguous: {parsed_scores}")
    parsed_descriptions = {key: str(descriptions.get(key) or "").strip() for key in expected}
    if any(not value for value in parsed_descriptions.values()):
        raise ValueError("planner omitted a segment evidence description")
    return {
        "ranking": ranking,
        "scores": parsed_scores,
        "descriptions": parsed_descriptions,
        "rationale": str(payload.get("rationale") or ""),
    }, usage


def _answer(
    client: OpenAI,
    *,
    model: str,
    row: Mapping[str, Any],
    selected: Sequence[tuple[str, bytes]],
    descriptions: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        "Answer using only the supplied chronological evidence panels. If there "
        "are no panels, use only the question and choices. Return a probability "
        "for every option; probabilities must be nonnegative and sum to one. "
        "Consider the causal and temporal chain and rule out distractors. "
        f"Question: {row['Question']} Options: "
        f"{json.dumps(row['Options'], ensure_ascii=False)}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for segment_id, panel in sorted(selected, key=lambda value: int(value[0][1:])):
        content.append({
            "type": "text",
            "text": (
                f"Evidence receipt {segment_id}: {descriptions[segment_id]}\n"
                f"Matching visual panel {segment_id}:"
            ),
        })
        content.append(_image_content(panel))
    payload, usage = _json_call(
        client, model=model,
        system=(
            "Return concise JSON {\"answer\":\"A-F\",\"probabilities\":"
            "{\"A\":number,...,\"F\":number},\"confidence\":number,"
            "\"evidence_segments\":[...],\"reason\":\"brief\"}."
        ),
        content=content,
        max_tokens=1000,
    )
    probabilities = normalized_probabilities(payload.get("probabilities") or {})
    answer = str(payload.get("answer") or "").strip().upper()
    if answer not in ANSWER_SLOTS:
        answer = ANSWER_SLOTS[int(np.argmax(probabilities))]
    return {
        "answer": answer,
        "probabilities": {
            slot: float(probability)
            for slot, probability in zip(ANSWER_SLOTS, probabilities)
        },
        "confidence": float(payload.get("confidence") or max(probabilities)),
        "evidence_segments": [str(value) for value in payload.get(
            "evidence_segments", ()
        )],
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
    video_id = str(row["video ID"])
    path = _video_path(dataset_root, video_id)
    media = config["media"]
    panels, video_meta = _panels(
        path,
        segment_count=int(media["segment_count"]),
        frames_per_segment=int(media["frames_per_segment"]),
        frame_width=int(media["frame_width"]),
        jpeg_quality=int(media["jpeg_quality"]),
    )
    client = OpenAI(
        api_key=api_key,
        base_url=str(config["model"]["base_url"]),
        timeout=float(config["model"]["timeout_seconds"]),
        max_retries=int(config["model"]["max_retries"]),
    )
    model = str(config["model"]["id"])
    plan, plan_usage = _planner(client, model=model, row=row, panels=panels)
    max_tests = int(config["policy"]["max_tests"])
    answers = []
    for prefix_length in range(max_tests + 1):
        selected_ids = plan["ranking"][:prefix_length]
        selected = [(value, panels[int(value[1:])]) for value in selected_ids]
        answer, usage = _answer(
            client, model=model, row=row, selected=selected,
            descriptions=plan["descriptions"],
        )
        answers.append({
            "prefix_length": prefix_length,
            "selected_segments": selected_ids,
            "answer": answer,
            "usage": usage,
        })
    return {
        "sample_id": sample_id,
        "video_id": video_id,
        "question_type": str(row.get("Question Type") or ""),
        "gold_answer": str(row["Answer"]),
        "video_path": str(path),
        "video_sha256": file_sha256(path),
        "video_meta": video_meta,
        "planner": plan,
        "planner_usage": plan_usage,
        "answers": answers,
    }


def _calibration_rows(receipts: Sequence[Mapping[str, Any]], max_tests: int) -> list[CalibrationRow]:
    output = []
    for receipt in receipts:
        answer_index = ANSWER_SLOTS.index(str(receipt["gold_answer"]))
        ranking = receipt["planner"]["ranking"]
        scores = receipt["planner"]["scores"]
        for row in receipt["answers"]:
            prefix = int(row["prefix_length"])
            selected_scores = [float(scores[value]) for value in ranking[:prefix]]
            output.append(CalibrationRow(
                sample_id=str(receipt["sample_id"]),
                prefix_length=prefix,
                max_tests=max_tests,
                mean_planner_score=float(np.mean(selected_scores)) if selected_scores else 0.0,
                raw_probabilities=tuple(normalized_probabilities(
                    row["answer"]["probabilities"]
                )),
                answer_index=answer_index,
            ))
    return output


def _leave_one_sample_predictions(
    rows: Sequence[CalibrationRow], *, seed: int,
) -> dict[tuple[str, int], np.ndarray]:
    output = {}
    sample_ids = sorted({row.sample_id for row in rows})
    for fold, sample_id in enumerate(sample_ids):
        train = [row for row in rows if row.sample_id != sample_id]
        model = fit_calibration_head(train, seed=seed + fold)
        for row in rows:
            if row.sample_id == sample_id:
                output[(sample_id, row.prefix_length)] = model.predict(row.features())
    return output


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
                sample_id=sample_id,
                current_belief=tuple(map(float, before)),
                next_planner_score=float(scores[ranking[prefix]]),
                prefix_fraction=prefix / max_tests,
                information_gain=normalized_entropy(before) - normalized_entropy(after),
                confidence_gain=float(np.max(after) - np.max(before)),
            ))
    return output


def _policy_trace(
    receipt: Mapping[str, Any],
    *,
    condition: str,
    predictions: Mapping[tuple[str, int], np.ndarray],
    gain_grounder,
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
            belief,
            condition=condition,
            prefix_length=prefix,
            max_tests=max_tests,
            next_planner_score=next_score,
            gain_grounder=gain_grounder,
            source_models=source_models,
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
        "condition": condition,
        "committed_answer": ANSWER_SLOTS[committed],
        "gold_answer": ANSWER_SLOTS[gold],
        "correct": committed == gold,
        "tests": len(steps) - 1 if steps[-1]["decision"]["kind"] == "COMMIT" else max_tests,
        "steps": steps,
    }


def _adaptation_report(
    receipts: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    controlled_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    max_tests = int(config["policy"]["max_tests"])
    rows = _calibration_rows(receipts, max_tests)
    seed = int(config["target_grounder"]["seed"])
    cross_fitted = _leave_one_sample_predictions(rows, seed=seed)
    raw_curves = {}
    calibrated_curves = {}
    for prefix in range(max_tests + 1):
        subset = [row for row in rows if row.prefix_length == prefix]
        raw_curves[f"k{prefix}"] = float(np.mean([
            int(np.argmax(row.raw_probabilities)) == row.answer_index for row in subset
        ]))
        calibrated_curves[f"k{prefix}"] = float(np.mean([
            int(np.argmax(cross_fitted[(row.sample_id, prefix)])) == row.answer_index
            for row in subset
        ]))
    full_head = fit_calibration_head(rows, seed=seed)
    full_predictions = {
        (row.sample_id, row.prefix_length): full_head.predict(row.features())
        for row in rows
    }
    gain_rows = _gain_rows(receipts, full_predictions, max_tests=max_tests)
    gain_grounder = fit_gain_grounder(
        gain_rows,
        seed=int(config["target_grounder"]["gain_seed"]),
        hidden_units=int(config["target_grounder"]["gain_hidden_units"]),
    )
    source_models = build_source_value_models(
        controlled_config, seed=int(config["source"]["model_seed"]),
    )
    traces = [
        _policy_trace(
            receipt, condition=condition, predictions=full_predictions,
            gain_grounder=gain_grounder, source_models=source_models,
            policy=config["policy"],
        )
        for receipt in receipts for condition in VIDEO_CONDITIONS
    ]
    conditions = {
        condition: {
            "samples": sum(row["condition"] == condition for row in traces),
            "accuracy": float(np.mean([
                row["correct"] for row in traces if row["condition"] == condition
            ])),
            "mean_tests": float(np.mean([
                row["tests"] for row in traces if row["condition"] == condition
            ])),
        }
        for condition in VIDEO_CONDITIONS
    }
    changed = sum(
        int(np.argmax(row.raw_probabilities))
        != int(np.argmax(next_row.raw_probabilities))
        for row, next_row in zip(
            sorted((row for row in rows if row.prefix_length == 0), key=lambda x: x.sample_id),
            sorted((row for row in rows if row.prefix_length == max_tests), key=lambda x: x.sample_id),
        )
    )
    by_condition = {
        condition: {
            row["sample_id"]: tuple(
                step["decision"]["kind"] for step in row["steps"]
            )
            for row in traces if row["condition"] == condition
        }
        for condition in VIDEO_CONDITIONS
    }
    authentic_contrasts = sum(
        by_condition["authentic_source_plus_target"][sample_id]
        != by_condition["target_only"][sample_id]
        for sample_id in by_condition["target_only"]
    )
    evidence_response = max(
        raw_curves[f"k{prefix}"] for prefix in range(1, max_tests + 1)
    ) - raw_curves["k0"]
    preflight = {
        "all_receipts_complete": len(receipts) == len(config["splits"]["adaptation"]),
        "visual_evidence_changes_prediction": changed >= 2,
        "raw_full_evidence_above_chance": raw_curves[f"k{max_tests}"] > 1 / 6,
        "cross_fitted_calibrator_above_chance": max(calibrated_curves.values()) > 1 / 6,
        "authentic_source_action_contrast": authentic_contrasts >= 1,
        "positive_evidence_response": evidence_response > 0,
    }
    artifact = {
        "schema_version": 1,
        "role": "TARGET_NATIVE_VIDEO_GROUNDER_FROZEN_FROM_ADAPTATION_ONLY",
        "calibration_head": full_head.as_dict(),
        "gain_grounder": gain_grounder.as_dict(),
        "training_sample_ids": sorted({row.sample_id for row in rows}),
        "source_config_sha256": stable_hash(controlled_config),
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    report = {
        "schema_version": 1,
        "status": "ADAPTATION_PREFLIGHT_PASS" if all(preflight.values()) else "ADAPTATION_PREFLIGHT_FAIL",
        "claim_boundary": "Development/adaptation only; no test transfer outcome consumed.",
        "raw_accuracy_by_prefix": raw_curves,
        "leave_one_video_out_calibrated_accuracy_by_prefix": calibrated_curves,
        "prediction_changed_k0_to_kmax": changed,
        "authentic_vs_target_action_contrast_samples": authentic_contrasts,
        "best_raw_accuracy_gain_over_k0": evidence_response,
        "conditions_in_sample_fit_diagnostic_only": conditions,
        "preflight": preflight,
        "policy_traces": traces,
    }
    report["report_sha256"] = stable_hash(report)
    return report, artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--split", choices=("adaptation", "qualification", "held_out"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample_ids = list(config["splits"][args.split])
    benchmark_split = "train" if args.split == "adaptation" else "test"
    questions = _questions(args.dataset_root, benchmark_split)
    if any(sample_id not in questions for sample_id in sample_ids):
        missing = [sample_id for sample_id in sample_ids if sample_id not in questions]
        raise SystemExit(f"sample IDs are missing: {missing}")
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
                _collect_sample,
                sample_id,
                row=questions[sample_id],
                dataset_root=args.dataset_root,
                config=config,
                api_key=str(key),
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
                "progress": f"{len(existing)}/{len(sample_ids)}",
            }), flush=True)
    missing = [sample_id for sample_id in sample_ids if sample_id not in existing]
    if missing:
        raise SystemExit(f"incomplete receipts; rerun to resume: {missing}")
    receipts = [existing[sample_id] for sample_id in sample_ids]
    controlled_path = Path(config["source"]["controlled_v3_config"])
    controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
    if stable_hash(controlled) != config["source"]["controlled_v3_config_content_sha256"]:
        raise SystemExit("controlled V3 source config content hash mismatch")
    if args.split != "adaptation":
        raise SystemExit("formal split execution is blocked until an adaptation artifact is frozen")
    report, artifact = _adaptation_report(
        receipts, config=config, controlled_config=controlled,
    )
    artifact_path = args.output_dir / "target_grounder_candidate.json"
    report_path = args.output_dir / "adaptation_report.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    report["receipts"] = {
        "path": str(receipts_path.resolve()),
        "sha256": file_sha256(receipts_path),
    }
    report["target_grounder_candidate"] = {
        "path": str(artifact_path.resolve()),
        "sha256": file_sha256(artifact_path),
        "content_sha256": artifact["artifact_sha256"],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "raw_accuracy_by_prefix": report["raw_accuracy_by_prefix"],
        "calibrated_accuracy_by_prefix": report[
            "leave_one_video_out_calibrated_accuracy_by_prefix"
        ],
        "conditions": report["conditions_in_sample_fit_diagnostic_only"],
        "report": str(report_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
