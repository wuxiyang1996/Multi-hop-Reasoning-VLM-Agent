#!/usr/bin/env python3
"""Collect independent full-timeline adjudication for V17 AGQA overrides."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping
import zipfile

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_override_adjudicator import (  # noqa: E402
    adjudication_supports_typed_override,
    parse_override_adjudication,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _answer_matches, _cached_provider_call, _image_content, _panels,
    _provider_json_call, _sample_video_range, _sha256,
)


PROMPT_VERSION = "AGQA_INDEPENDENT_OVERRIDE_ADJUDICATOR_V18_0"


def _response_format(allowed: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agqa_override_adjudication_v18",
            "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "decision": {
                        "type": "string", "enum": allowed + ["unknown"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_frames": {
                        "type": "array", "items": {"type": "integer"},
                        "maxItems": 6,
                    },
                    "observed_events": {
                        "type": "array", "items": {"type": "string"},
                        "maxItems": 6,
                    },
                    "ambiguity": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "decision", "confidence", "evidence_frames",
                    "observed_events", "ambiguity", "reason",
                ],
            },
        },
    }


def _system(frame_count: int, allowed: list[str]) -> str:
    return (
        "You are an independent video adjudicator. Answer one public video question "
        "from the complete chronological proxy timeline. You never see a prior model "
        "answer, a symbolic prediction, gold, functional program, scene graph, source "
        "identity, or competing candidate. Ground the exact requested predicate and "
        "entity roles: a related action, prerequisite, consequence, gaze direction, "
        "proximity, or object interaction is not sufficient. For temporal questions, "
        "inspect all visible occurrences and return unknown if recurrence makes the "
        "requested order ambiguous. For duration, compare only directly observed "
        "intervals. Prefer unknown to semantic inference. Cite at most six frame IDs "
        f"from F0..F{frame_count - 1}. Closed decisions: {allowed}; unknown is allowed. "
        "Return only the requested JSON."
    )


def _question_rows(manifest: Mapping[str, Any]) -> dict[str, str]:
    wanted = {str(row["task_id"]): row for row in manifest["samples"]}
    output: dict[str, str] = {}
    with zipfile.ZipFile(manifest["archive_path"]) as bundle, bundle.open(
        manifest["entry"], "r"
    ) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                if task_id not in wanted:
                    continue
                question = str(row["question"])
                if stable_hash(question) != wanted[task_id]["question_sha256"]:
                    raise ValueError(f"V18 question hash mismatch: {task_id}")
                output[task_id] = question
                if len(output) == len(wanted):
                    break
    if set(output) != set(wanted):
        raise ValueError("V18 manifest questions are missing")
    return output


def _call_one(
    sample: Mapping[str, Any], *, question: str, config: Mapping[str, Any],
    api_key: str, cache_root: Path,
) -> dict[str, Any]:
    video_path = Path(sample["video_path"])
    if _sha256(video_path) != sample["video_sha256"]:
        raise ValueError(f"V18 video hash mismatch: {sample['task_id']}")
    media, model = config["media"], config["model"]
    frames, seconds, metadata = _sample_video_range(
        video_path, frame_count=int(media["frame_count"]),
        max_side=int(media["max_side"]),
    )
    panels = _panels(
        frames, seconds, frames_per_panel=int(media["frames_per_panel"]),
        frame_width=int(media["panel_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    allowed = [str(value).strip().casefold() for value in sample["allowed_decisions"]]
    system = _system(len(frames), allowed)
    client = OpenAI(
        api_key=api_key, base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]),
        max_retries=int(model["max_retries"]),
    )
    attempts = []
    last_error = ""
    parsed = None
    for attempt in range(int(model["schema_retries"])):
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"Public question: {question.strip()}\n"
                "Independently answer from the full timeline."
            ),
        }]
        for index, panel in enumerate(panels):
            content.extend([
                {"type": "text", "text": f"Chronological panel {index + 1}:"},
                _image_content(panel),
            ])
        if last_error:
            content.append({
                "type": "text", "text": "Fix this schema error: " + last_error,
            })
        input_core = {
            "stage": "independent_override_adjudication",
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "system": system,
            "question_sha256": stable_hash(question),
            "allowed_decisions": allowed,
            "panel_sha256": [hashlib.sha256(panel).hexdigest() for panel in panels],
            "retry_error": last_error,
        }
        payload, usage, reused = _cached_provider_call(
            cache_dir=cache_root / str(sample["task_id"]),
            call_name=f"adjudicator_{attempt}", input_core=input_core,
            invoke=lambda: _provider_json_call(
                client, model=model, system=system, content=content,
                max_tokens=int(model["max_tokens"]),
                response_format=_response_format(allowed),
            ),
        )
        attempts.append({"payload": payload, "usage": usage, "cache_reused": reused})
        try:
            parsed = parse_override_adjudication(
                payload, allowed_decisions=allowed, frame_count=len(frames),
            )
            break
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    if parsed is None:
        raise ValueError("V18 adjudicator schema retries exhausted: " + last_error)
    body = {
        "schema_version": "agqa2-override-adjudication-runtime-v18",
        "task_id": sample["task_id"], "video_id": sample["video_id"],
        "question_sha256": stable_hash(question),
        "video_sha256": sample["video_sha256"],
        "panel_sha256": [hashlib.sha256(panel).hexdigest() for panel in panels],
        "video_metadata": metadata,
        "allowed_decisions": allowed,
        "adjudication": parsed.as_dict(), "attempts": attempts,
        "typed_decision_visible_to_model": False,
        "direct_response_visible_to_model": False,
        "gold_or_correctness_visible_to_model": False,
        "functional_program_or_scene_graph_visible_to_model": False,
        "source_identity_visible_to_model": False,
    }
    return body | {"runtime_receipt_sha256": stable_hash(body)}


def collect(
    *, config_path: Path, keys_path: Path, output_path: Path, workers: int,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    manifest_path = REPO_ROOT / config["manifest"]
    if _sha256(manifest_path) != config["manifest_file_sha256"]:
        raise ValueError("V18 manifest file hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    body = dict(manifest)
    claimed = body.pop("manifest_sha256")
    if stable_hash(body) != claimed:
        raise ValueError("V18 manifest content hash mismatch")
    if manifest["status"] != config["expected_manifest_status"]:
        raise ValueError("V18 manifest is not frozen")
    parent_path = REPO_ROOT / config["parent_v17_report"]
    if _sha256(parent_path) != config["parent_v17_report_file_sha256"]:
        raise ValueError("V18 parent report changed")
    if _sha256(REPO_ROOT / config["module"]) != config["module_sha256"]:
        raise ValueError("V18 adjudicator module changed")
    if _sha256(REPO_ROOT / config["collector"]) != config["collector_sha256"]:
        raise ValueError("V18 collector changed")
    questions = _question_rows(manifest)
    parent = json.loads(parent_path.read_text())
    parent_by_task = {str(row["task_id"]): row for row in parent["rows"]}
    runtime_parent = {}
    for sample in manifest["samples"]:
        row = parent_by_task[str(sample["task_id"])]
        if row["calibrated_authorization_class"] != "SOURCE_TYPED_OVERRIDE":
            raise ValueError("V18 sample is not a frozen source override")
        if stable_hash(row["direct_response"]) != sample["direct_response_sha256"]:
            raise ValueError("V18 direct receipt changed")
        typed = row["calibrated_target_native_execution"]["decision"]
        if stable_hash(typed) != sample["typed_decision_sha256"]:
            raise ValueError("V18 typed receipt changed")
        runtime_parent[str(sample["task_id"])] = {
            "typed_decision": typed,
            "direct_response": row["direct_response"],
        }

    key = runpy.run_path(str(keys_path)).get(config["model"]["api_key_name"])
    if not key:
        raise ValueError("OpenRouter key is unavailable")
    cache_root = output_path.parent / "call_cache"
    receipt_root = output_path.parent / "runtime_receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    runtime: dict[str, dict[str, Any]] = {}
    pending = []
    for sample in manifest["samples"]:
        task_id = str(sample["task_id"])
        path = receipt_root / f"{task_id}.json"
        if path.is_file():
            cached = json.loads(path.read_text())
            if (
                cached.get("question_sha256") == sample["question_sha256"]
                and cached.get("video_sha256") == sample["video_sha256"]
            ):
                runtime[task_id] = cached
                continue
        pending.append(sample)
    errors = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _call_one, sample, question=questions[str(sample["task_id"])],
                config=config, api_key=key, cache_root=cache_root,
            ): str(sample["task_id"])
            for sample in pending
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                receipt = future.result()
            except Exception as exc:
                errors[task_id] = f"{type(exc).__name__}: {exc}"
                print(f"failed {task_id}: {errors[task_id]}", flush=True)
                continue
            runtime[task_id] = receipt
            (receipt_root / f"{task_id}.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            print(f"completed {task_id}", flush=True)
    if errors:
        raise RuntimeError(f"V18 adjudicator workers failed: {errors}")

    # Evaluator-only outcomes enter only after every adjudication is immutable.
    threshold = float(config["authorization"]["minimum_confidence"])
    evaluated = []
    usage = []
    for sample in manifest["samples"]:
        task_id = str(sample["task_id"])
        receipt = runtime[task_id]
        row = parent_by_task[task_id]
        typed = runtime_parent[task_id]["typed_decision"]
        direct = runtime_parent[task_id]["direct_response"]
        adjudication = parse_override_adjudication(
            receipt["adjudication"],
            allowed_decisions=sample["allowed_decisions"], frame_count=48,
        )
        authorized = adjudication_supports_typed_override(
            adjudication, typed_decision=str(typed),
            minimum_confidence=threshold,
        )
        final = typed if authorized else direct
        gold = row["gold_answer_evaluator_only"]
        for attempt in receipt["attempts"]:
            usage.append(attempt["usage"])
        evaluated.append({
            "task_id": task_id, "video_id": sample["video_id"],
            "route": row["oracle_route_evaluator_only"],
            "comparison": row["query_plan"]["comparison"],
            "adjudication": adjudication.as_dict(),
            "typed_override": typed, "direct_response": direct,
            "gold_answer_evaluator_only": gold,
            "adjudicator_correct": (
                adjudication.decision != "unknown"
                and _answer_matches(adjudication.decision, gold)
            ),
            "typed_override_authorized": authorized,
            "final_prediction": final,
            "final_correct": _answer_matches(final, gold),
            "direct_correct": _answer_matches(direct, gold),
            "typed_correct": _answer_matches(typed, gold),
            "runtime_receipt_sha256": receipt["runtime_receipt_sha256"],
            "gold_first_read_after_all_adjudications_froze": True,
        })
    wins = sum(row["final_correct"] and not row["direct_correct"] for row in evaluated)
    losses = sum(not row["final_correct"] and row["direct_correct"] for row in evaluated)
    direct_correct = sum(row["direct_correct"] for row in evaluated)
    final_correct = sum(row["final_correct"] for row in evaluated)
    judge_correct = sum(row["adjudicator_correct"] for row in evaluated)
    cost = sum(float(row["reported_cost_usd"]) for row in usage)
    gates_spec = config["qualification_gates"]
    gates = {
        "required_rows": len(evaluated) == gates_spec["required_rows"],
        "minimum_adjudicator_correct": (
            judge_correct >= gates_spec["minimum_adjudicator_correct"]
        ),
        "minimum_retained_typed_vs_direct_wins": (
            wins >= gates_spec["minimum_retained_typed_vs_direct_wins"]
        ),
        "maximum_authorized_typed_vs_direct_losses": (
            losses <= gates_spec["maximum_authorized_typed_vs_direct_losses"]
        ),
        "minimum_final_vs_direct_delta": (
            final_correct - direct_correct >= gates_spec["minimum_final_vs_direct_delta"]
        ),
        "provider_cost_within_cap": (
            cost <= gates_spec["maximum_reported_provider_cost_usd"]
        ),
        "runtime_candidate_and_evaluator_fields_hidden": all(
            not receipt[flag]
            for receipt in runtime.values()
            for flag in (
                "typed_decision_visible_to_model", "direct_response_visible_to_model",
                "gold_or_correctness_visible_to_model",
                "functional_program_or_scene_graph_visible_to_model",
                "source_identity_visible_to_model",
            )
        ),
    }
    qualified = all(gates.values())
    result = {
        "schema_version": "agqa2-override-adjudicator-report-v18-development",
        "status": (
            "AGQA2_OVERRIDE_ADJUDICATOR_V18_DEVELOPMENT_QUALIFIED"
            if qualified else
            "AGQA2_OVERRIDE_ADJUDICATOR_V18_DEVELOPMENT_NOT_QUALIFIED"
        ),
        "claim_boundary": config["claim_boundary"],
        "config_sha256": _sha256(config_path),
        "manifest_sha256": claimed,
        "model": config["model"]["id"],
        "sample_count": len(evaluated),
        "metrics": {
            "adjudicator_correct": judge_correct,
            "typed_overrides_authorized": sum(
                row["typed_override_authorized"] for row in evaluated
            ),
            "direct_correct": direct_correct, "final_correct": final_correct,
            "final_vs_direct_delta": final_correct - direct_correct,
            "retained_typed_vs_direct_wins": wins,
            "authorized_typed_vs_direct_losses": losses,
        },
        "provider_calls": len(usage),
        "reported_provider_cost_usd": cost,
        "qualification_gates": gates,
        "grounder_qualified": qualified,
        "rows": evaluated,
        "fresh_benchmark_claim": False,
    }
    result["report_sha256"] = stable_hash(result)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.output.resolve(), workers=args.workers,
    )
    print(json.dumps({
        "status": result["status"], "metrics": result["metrics"],
        "qualification_gates": result["qualification_gates"],
        "provider_calls": result["provider_calls"],
        "reported_provider_cost_usd": result["reported_provider_cost_usd"],
        "report_sha256": result["report_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
