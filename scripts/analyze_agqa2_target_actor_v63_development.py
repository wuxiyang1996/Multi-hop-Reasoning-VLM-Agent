#!/usr/bin/env python3
"""Compare target-native AGQA actors on the sealed consumed development split."""

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


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from motif_transfer.contracts import stable_hash  # noqa: E402
import scripts.collect_agqa2_active_grounding_v3 as active  # noqa: E402
from scripts.collect_agqa2_temporal_localized_query_v59 import (  # noqa: E402
    _load_metadata,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _verified_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(str(config["manifest"]))
    if _sha(path) != config["manifest_file_sha256"]:
        raise ValueError("development manifest drifted")
    manifest = _read(path)
    body = dict(manifest)
    claimed = body.pop("manifest_sha256")
    if stable_hash(body) != claimed or not manifest.get("samples"):
        raise ValueError("development manifest is not self-consistent")
    return manifest


def _collect_one(
    sample: Mapping[str, Any], *, model: Mapping[str, Any], media: Mapping[str, Any],
    api_key: str, cache_root: Path,
) -> dict[str, Any]:
    video_path = Path(str(sample["video_path"]))
    frames, seconds, metadata = active._sample_video_range(
        video_path,
        frame_count=int(media["frame_count"]),
        max_side=int(media["frame_max_side"]),
    )
    panels = active._panels(
        frames, seconds,
        frames_per_panel=int(media["frames_per_panel"]),
        frame_width=int(media["panel_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    client = OpenAI(
        api_key=api_key,
        base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]),
        max_retries=int(model["max_retries"]),
    )
    model_slug = str(model["id"]).replace("/", "__")
    response, payload, usage, reused = active._direct_call(
        client,
        question=str(sample["question"]),
        panels=panels,
        model=model,
        cache_dir=cache_root / model_slug / str(sample["task_id"]),
    )
    core = {
        "task_id": str(sample["task_id"]),
        "video_id": str(sample["video_id"]),
        "video_sha256": str(sample["video_sha256"]),
        "question_sha256": stable_hash(str(sample["question"])),
        "model": str(model["id"]),
        "response": response,
        "payload": payload,
        "usage": usage,
        "cache_reused": reused,
        "media": metadata,
        "runtime_answer_read": False,
        "runtime_program_read": False,
        "runtime_scene_graph_read": False,
    }
    return core | {"runtime_receipt_sha256": stable_hash(core)}


def run(
    *, config_path: Path, keys_path: Path, output_path: Path, workers: int,
) -> dict[str, Any]:
    config = _read(config_path)
    manifest = _verified_manifest(config)
    baseline_report_path = config.get("baseline_report")
    if baseline_report_path and _sha(_resolve(baseline_report_path)) != config[
        "baseline_report_file_sha256"
    ]:
        raise ValueError("baseline report drifted")
    dataset_config = {"dataset": config["dataset"]}
    metadata = _load_metadata(dataset_config, manifest)
    runtime_inputs = []
    for frozen in manifest["samples"]:
        target = metadata[str(frozen["task_id"])]
        question = str(target.get("question", ""))
        if stable_hash(question) != frozen["question_sha256"]:
            raise ValueError("question hash mismatch")
        runtime_inputs.append(dict(frozen) | {"question": question})

    keys = runpy.run_path(str(keys_path))
    runtime_root = output_path.parent / "runtime_receipts"
    cache_root = output_path.parent / "call_cache"
    runtime_root.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for model in config["models"]:
        api_key = keys.get(model["api_key_name"])
        if not api_key:
            raise ValueError("OpenRouter key unavailable")
        model_slug = str(model["id"]).replace("/", "__")
        model_dir = runtime_root / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)
        completed: dict[str, dict[str, Any]] = {}
        pending = []
        for sample in runtime_inputs:
            path = model_dir / f"{sample['task_id']}.json"
            if path.is_file():
                row = _read(path)
                if (
                    row.get("model") == model["id"]
                    and row.get("question_sha256") == stable_hash(sample["question"])
                    and row.get("video_sha256") == sample["video_sha256"]
                ):
                    completed[str(sample["task_id"])] = row
                    continue
            pending.append(sample)
        errors = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    _collect_one, sample, model=model, media=config["media"],
                    api_key=api_key, cache_root=cache_root,
                ): str(sample["task_id"])
                for sample in pending
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    errors[task_id] = f"{type(exc).__name__}: {exc}"
                    continue
                completed[task_id] = row
                (model_dir / f"{task_id}.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                )
        if errors or len(completed) != len(runtime_inputs):
            raise RuntimeError(json.dumps({"model": model["id"], "errors": errors}))
        all_rows[str(model["id"])] = [completed[str(row["task_id"])] for row in runtime_inputs]

    # Gold is consulted only after every candidate runtime receipt exists.
    baseline_model_id = config.get("baseline_model_id")
    if baseline_model_id:
        if baseline_model_id not in all_rows:
            raise ValueError("declared baseline model was not evaluated")
        baseline_by_id = {
            str(row["task_id"]): row for row in all_rows[baseline_model_id]
        }
    else:
        baseline = _read(_resolve(str(config["baseline_report"])))
        baseline_by_id = {str(row["task_id"]): row for row in baseline["rows"]}
    summaries = {}
    gates = config["selection_gates"]
    for model_id, rows in all_rows.items():
        evaluated = []
        for row in rows:
            task_id = str(row["task_id"])
            frozen = baseline_by_id[task_id]
            gold = str(metadata[task_id].get("answer", ""))
            correct = active._answer_matches(
                row["response"], gold,
            )
            baseline_response = frozen.get("response", frozen.get("direct_response"))
            baseline_correct = active._answer_matches(baseline_response, gold)
            evaluated.append(dict(row) | {
                "correct_evaluator_only": correct,
                "gold_read_after_all_runtime_receipts_froze": True,
                "baseline_correct_evaluator_only": baseline_correct,
            })
        wins = sum(x["correct_evaluator_only"] and not x["baseline_correct_evaluator_only"] for x in evaluated)
        losses = sum(x["baseline_correct_evaluator_only"] and not x["correct_evaluator_only"] for x in evaluated)
        correct = sum(x["correct_evaluator_only"] for x in evaluated)
        baseline_correct = sum(x["baseline_correct_evaluator_only"] for x in evaluated)
        cost = sum(float((x["usage"] or {}).get("reported_cost_usd", 0.0)) for x in evaluated)
        summaries[model_id] = {
            "rows": len(evaluated),
            "correct": correct,
            "accuracy": correct / len(evaluated),
            "baseline_correct": baseline_correct,
            "baseline_accuracy": baseline_correct / len(evaluated),
            "accuracy_gain": (correct - baseline_correct) / len(evaluated),
            "paired_wins": wins,
            "paired_losses": losses,
            "net_paired_wins": wins - losses,
            "reported_provider_cost_usd": cost,
            "rows_outcome_blind": all(not x["runtime_answer_read"] for x in evaluated),
        }
    eligible = [
        model_id for model_id, row in summaries.items()
        if model_id != baseline_model_id
        and row["rows"] == int(gates["required_rows_per_model"])
        and row["accuracy_gain"] >= float(gates["minimum_accuracy_gain_over_baseline"])
        and row["net_paired_wins"] >= int(gates["minimum_net_paired_wins"])
        and row["reported_provider_cost_usd"] <= float(gates["maximum_reported_provider_cost_usd"])
        and row["rows_outcome_blind"]
    ]
    eligible.sort(key=lambda model_id: (-summaries[model_id]["accuracy"], model_id))
    body = {
        "schema_version": "agqa2-target-actor-development-report-v63",
        "status": "TARGET_ACTOR_DEVELOPMENT_WINNER_SELECTED" if eligible else "NO_TARGET_ACTOR_PASSED_DEVELOPMENT_GATE",
        "config_file_sha256": _sha(config_path),
        "claim_boundary": config["claim_boundary"],
        "summaries": summaries,
        "selected_model": eligible[0] if eligible else None,
        "next_step": "FREEZE_NEW_VIDEO_DISJOINT_QUALIFICATION_BEFORE_ANY_CALL" if eligible else "DO_NOT_OPEN_NEW_RESERVE",
    }
    result = body | {"report_sha256": stable_hash(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/agqa2_target_actor_v63_development.json")
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--output", type=Path, default=REPO / "runs/agqa2_target_actor_v63_development/report.json")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = run(config_path=args.config.resolve(), keys_path=args.keys.resolve(), output_path=args.output.resolve(), workers=args.workers)
    print(json.dumps({key: result[key] for key in ("status", "summaries", "selected_model", "next_step")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
