#!/usr/bin/env python3
"""Collect an isolated, ontology-aware AGQA QUERY_OBJECT qualification."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping, Sequence

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import (  # noqa: E402
    parse_public_question_plan,
)
from motif_transfer.agqa_frame_grounder import (  # noqa: E402
    parse_frame_grounding_receipt, select_source_for_grounding,
)
from motif_transfer.agqa_program_transfer import (  # noqa: E402
    RELATION_ROUTE, profile_program,
)
from motif_transfer.agqa_query_object_grounder import (  # noqa: E402
    AGQA_OBJECT_ONTOLOGY, atomic_query_object_plan,
    calibrate_query_object_execution, parse_object_ontology_receipt,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_ir_applicability import SourceIRContract  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _load_sources  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _answer_matches, _cached_provider_call, _collect_runtime,
    _cumulative_cache_usage, _grounder_semantic_core, _load_selected_rows,
    _panel_content, _panels, _provider_json_call, _sample_video_range, _sha256,
    _usage_rows,
)


PROMPT_VERSION = "AGQA_QUERY_OBJECT_ONTOLOGY_V20_0"


def _ontology_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agqa_query_object_ontology_v20",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": list(AGQA_OBJECT_ONTOLOGY) + ["unknown"],
                    },
                    "relation_observed": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_frames": {
                        "type": "array", "maxItems": 6,
                        "items": {"type": "integer", "minimum": 0, "maximum": 47},
                    },
                    "visual_description": {"type": "string"},
                    "uncertainty": {"type": "string"},
                },
                "required": [
                    "decision", "relation_observed", "confidence",
                    "evidence_frames", "visual_description", "uncertainty",
                ],
            },
        },
    }


def _ontology_system() -> str:
    ontology = ", ".join(AGQA_OBJECT_ONTOLOGY)
    return (
        "You are a candidate-blind relation-to-object video grounder. You receive "
        "one requested visual relation and chronological frames, never the original "
        "question, answer, functional program, scene graph, per-question candidates, "
        "direct response, or source identity. Decide which single object is directly "
        "linked to the person by that relation. Do not choose a salient nearby object. "
        "For actions such as taking, putting down, holding, opening, touching, or "
        "watching, require visible interaction; for sitting/standing/lying/leaning, "
        "require the specified support/contact relation. Return unknown when the exact "
        "relation is not visually grounded. The fixed dataset-level object taxonomy is: "
        f"{ontology}. Return only the requested JSON with chronological F0..F47 evidence."
    )


def _ontology_call(
    *, plan, video_path: Path, model: Mapping[str, Any],
    media: Mapping[str, Any], api_key: str, cache_dir: Path,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    frames, seconds, metadata = _sample_video_range(
        video_path,
        frame_count=int(media["dense_proxy_frame_count"]),
        max_side=int(media["proxy_frame_max_side"]),
    )
    panels = _panels(
        frames, seconds,
        frames_per_panel=int(media["frames_per_panel"]),
        frame_width=int(media["panel_frame_width"]),
        quality=int(media["jpeg_quality"]),
    )
    client = OpenAI(
        api_key=api_key, base_url=str(model["base_url"]),
        timeout=float(model["timeout_seconds"]),
        max_retries=int(model["max_retries"]),
    )
    attempts = []
    last_error = ""
    for attempt in range(int(model["schema_retries"])):
        content = [{"type": "text", "text": (
            f"requested_relation: {plan.operand_a}\n"
            "Ground only this relation and return its related object."
        )}] + _panel_content(panels)
        if last_error:
            content.append({
                "type": "text", "text": "Fix this schema error: " + last_error,
            })
        input_core = {
            "stage": "query_object_ontology",
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "requested_relation": plan.operand_a,
            "ontology": list(AGQA_OBJECT_ONTOLOGY),
            "system": _ontology_system(),
            "panel_sha256": [hashlib.sha256(value).hexdigest() for value in panels],
            "retry_error": last_error,
        }
        payload, usage, reused = _cached_provider_call(
            cache_dir=cache_dir,
            call_name=f"object_ontology_{attempt}",
            input_core=input_core,
            invoke=lambda: _provider_json_call(
                client, model=model, system=_ontology_system(), content=content,
                max_tokens=int(model["max_ontology_tokens"]),
                response_format=_ontology_response_format(),
            ),
        )
        attempts.append({"payload": payload, "usage": usage, "cache_reused": reused})
        try:
            return (
                parse_object_ontology_receipt(payload, frame_count=len(frames)),
                attempts,
                metadata,
            )
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
    raise ValueError("QUERY_OBJECT ontology schema retries exhausted: " + last_error)


def _semantic_core(
    config: Mapping[str, Any], sources: Sequence[SourceIRContract],
) -> dict[str, Any]:
    return {
        "base_active_grounder": _grounder_semantic_core(config, sources),
        "query_object_prompt_version": PROMPT_VERSION,
        "query_object_grounder": config["query_object_grounder"],
        "query_object_calibration": config["query_object_calibration"],
        "object_ontology": list(AGQA_OBJECT_ONTOLOGY),
    }


def _evaluation_core(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "qualification_gates": config["qualification_gates"],
        "split": config["split"],
        "manifest": config["manifest"],
    }


def _collect_query_object_runtime(
    sample: Mapping[str, Any], *, config: Mapping[str, Any], api_key: str,
    sources: Sequence[SourceIRContract], grounder_sha256: str,
    cache_root: Path,
) -> dict[str, Any]:
    question = str(sample["question"])
    plan = parse_public_question_plan(question)
    if plan is None or not atomic_query_object_plan(plan):
        raise ValueError("sample is not an atomic QUERY_OBJECT public question")

    # Freeze the ontology view before the base runtime creates its matched direct
    # response. Neither call receives the other one's output.
    ontology, ontology_attempts, ontology_video_metadata = _ontology_call(
        plan=plan, video_path=Path(sample["video_path"]),
        model=config["query_object_grounder"]["model"], media=config["media"],
        api_key=api_key, cache_dir=cache_root / str(sample["task_id"]),
    )
    base = _collect_runtime(
        sample, config=config, api_key=api_key, sources=sources,
        grounder_sha256=grounder_sha256, cache_root=cache_root,
    )
    if base["query_plan"]["comparison"] != "QUERY_OBJECT":
        raise ValueError("base runtime changed the QUERY_OBJECT route")
    calibrated = calibrate_query_object_execution(
        base_decision=base["target_native_execution"]["decision"],
        direct_response=base["direct_response"],
        ontology_receipt=ontology,
        minimum_confidence=float(
            config["query_object_calibration"]["minimum_ontology_confidence"]
        ),
    )
    body = deepcopy(base)
    body.pop("runtime_receipt_sha256", None)
    body.update({
        "object_ontology_receipt": ontology.as_dict(),
        "object_ontology_attempts": ontology_attempts,
        "object_ontology_video_metadata": ontology_video_metadata,
        "object_ontology_call_started_before_direct": True,
        "object_ontology_original_question_read": False,
        "object_ontology_answer_candidates_read": False,
        "calibrated_target_native_execution": calibrated,
        "calibration_started_after_typed_and_direct_receipts_froze": True,
        "grounder_sha256": grounder_sha256,
    })
    return body | {"runtime_receipt_sha256": stable_hash(body)}


def _validate_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text())
    for field, hash_field in (
        ("manifest", "manifest_file_sha256"),
        ("preregistration", "preregistration_file_sha256"),
    ):
        path = REPO_ROOT / config[field]
        if _sha256(path) != config[hash_field]:
            raise ValueError(f"QUERY_OBJECT {field} file hash mismatch")
    manifest = json.loads((REPO_ROOT / config["manifest"]).read_text())
    body = dict(manifest)
    claimed = body.pop("manifest_sha256")
    if stable_hash(body) != claimed:
        raise ValueError("QUERY_OBJECT manifest content hash mismatch")
    if manifest["split"] != config["split"]:
        raise ValueError("QUERY_OBJECT config/manifest split mismatch")
    if manifest["status"] != config["expected_manifest_status"]:
        raise ValueError("QUERY_OBJECT manifest status mismatch")
    prereg = json.loads((REPO_ROOT / config["preregistration"]).read_text())
    if prereg["status"] != config["expected_preregistration_status"]:
        raise ValueError("QUERY_OBJECT preregistration status mismatch")
    for section, labels in (
        ("grounder", ("module", "collector", "executor")),
        ("local_object_grounder", ("module", "model_path")),
        ("query_object_grounder", ("module", "base_collector")),
    ):
        spec = config[section]
        for label in labels:
            path = Path(spec[label]) if label == "model_path" else REPO_ROOT / spec[label]
            expected_key = "model_sha256" if label == "model_path" else f"{label}_sha256"
            if _sha256(path) != spec[expected_key]:
                raise ValueError(f"QUERY_OBJECT {section}.{label} hash mismatch")
    return config, manifest


def collect(
    *, config_path: Path, keys_path: Path, output_path: Path,
    workers: int,
) -> dict[str, Any]:
    config, manifest = _validate_inputs(config_path)
    keys = runpy.run_path(str(keys_path))
    api_key = keys.get(config["model"]["api_key_name"])
    if not api_key:
        raise ValueError("OpenRouter API key is unavailable")
    sources, arcade_report = _load_sources(config)
    grounder_sha256 = stable_hash(_semantic_core(config, sources))
    evaluation_protocol_sha256 = stable_hash(_evaluation_core(config))
    if config.get("expected_grounder_sha256") not in {None, grounder_sha256}:
        raise ValueError("QUERY_OBJECT grounder differs from preregistration")
    if config.get("expected_evaluation_protocol_sha256") not in {
        None, evaluation_protocol_sha256,
    }:
        raise ValueError("QUERY_OBJECT evaluation protocol differs from preregistration")
    dependency = None
    if config["split"] == "reserve":
        dependency_path = REPO_ROOT / config["development_qualification_report"]
        if _sha256(dependency_path) != config["development_qualification_file_sha256"]:
            raise ValueError("QUERY_OBJECT development dependency hash mismatch")
        dependency = json.loads(dependency_path.read_text())
        if not dependency.get("grounder_qualified"):
            raise ValueError("QUERY_OBJECT reserve requires a qualified development grounder")
        if dependency["grounder_sha256"] != grounder_sha256:
            raise ValueError("QUERY_OBJECT reserve changed the qualified grounder")

    metadata = _load_selected_rows(manifest)
    runtime_inputs = []
    for frozen in manifest["samples"]:
        task_id = str(frozen["task_id"])
        question = str(metadata[task_id]["question"])
        if stable_hash(question) != frozen["question_sha256"]:
            raise ValueError(f"question hash mismatch: {task_id}")
        runtime_inputs.append(dict(frozen) | {"question": question})
    cache_root = output_path.parent / "call_cache"
    runtime_dir = output_path.parent / "runtime_receipts"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows: dict[str, dict[str, Any]] = {}
    pending = []
    for sample in runtime_inputs:
        path = runtime_dir / f"{sample['task_id']}.json"
        if path.is_file():
            cached = json.loads(path.read_text())
            if (
                cached.get("grounder_sha256") == grounder_sha256
                and cached.get("question_sha256") == stable_hash(sample["question"])
                and cached.get("video_sha256") == sample["video_sha256"]
            ):
                runtime_rows[str(sample["task_id"])] = cached
                print(f"reused {sample['task_id']}", flush=True)
                continue
        pending.append(sample)
    errors = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _collect_query_object_runtime, sample, config=config,
                api_key=api_key, sources=sources, grounder_sha256=grounder_sha256,
                cache_root=cache_root,
            ): str(sample["task_id"])
            for sample in pending
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                errors[task_id] = f"{type(exc).__name__}: {exc}"
                print(f"failed {task_id}: {errors[task_id]}", flush=True)
                continue
            runtime_rows[task_id] = row
            (runtime_dir / f"{task_id}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n"
            )
            print(f"completed {task_id}", flush=True)
    if errors:
        (output_path.parent / "worker_errors.json").write_text(
            json.dumps({"grounder_sha256": grounder_sha256, "errors": errors},
                       indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError("QUERY_OBJECT workers failed; completed calls were cached")

    # Evaluator-only answer and program access starts after all runtime receipts freeze.
    evaluated = []
    for frozen in manifest["samples"]:
        task_id = str(frozen["task_id"])
        target, runtime = metadata[task_id], runtime_rows[task_id]
        program = str(target["program"])
        if stable_hash(program) != frozen["program_sha256"]:
            raise ValueError(f"program hash mismatch: {task_id}")
        oracle_route = profile_program(task_id=task_id, program=program).route_kind
        gold = str(target["answer"])
        calibrated = runtime["calibrated_target_native_execution"]
        decision = calibrated["decision"]
        direct = runtime["direct_response"]
        decisive = decision is not None
        fallback = decision if decisive else direct
        evaluated.append(runtime | {
            "oracle_route_evaluator_only": oracle_route,
            "gold_answer_evaluator_only": gold,
            "predicted_route_correct": oracle_route == RELATION_ROUTE,
            "decisive_execution": decisive,
            "decisive_correct": _answer_matches(decision, gold) if decisive else None,
            "direct_correct": _answer_matches(direct, gold),
            "typed_fallback_prediction": fallback,
            "typed_fallback_correct": _answer_matches(fallback, gold),
            "official_answer_first_read_after_all_runtime_rows_froze": True,
            "official_scene_graph_read_by_evaluator": False,
        })

    metrics = {
        "valid_runtime_rows": len(evaluated),
        "route_correct": sum(row["predicted_route_correct"] for row in evaluated),
        "decisive_executions": sum(row["decisive_execution"] for row in evaluated),
        "decisive_correct": sum(
            bool(row["decisive_correct"]) for row in evaluated
            if row["decisive_execution"]
        ),
        "direct_correct": sum(row["direct_correct"] for row in evaluated),
        "typed_fallback_correct": sum(row["typed_fallback_correct"] for row in evaluated),
        "typed_vs_direct_wins": sum(
            row["typed_fallback_correct"] and not row["direct_correct"]
            for row in evaluated
        ),
        "typed_vs_direct_losses": sum(
            row["direct_correct"] and not row["typed_fallback_correct"]
            for row in evaluated
        ),
        "source_typed_overrides": sum(
            row["calibrated_target_native_execution"]["authorization_class"]
            == "SOURCE_TYPED_OVERRIDE" for row in evaluated
        ),
    }
    metrics["decisive_accuracy"] = (
        metrics["decisive_correct"] / metrics["decisive_executions"]
        if metrics["decisive_executions"] else 0.0
    )
    controls = {
        "source_permuted_abstentions": sum(
            row["source_permuted_wrong_type_abstained"] for row in evaluated
        ),
        "target_written_equivalent_matches": sum(
            row["target_written_equivalent_dynamics_match"] for row in evaluated
        ),
    }
    runtime_usage = []
    for row in runtime_rows.values():
        runtime_usage.extend(_usage_rows(row))
        runtime_usage.extend(
            attempt["usage"] for attempt in row["object_ontology_attempts"]
        )
    cumulative_usage = _cumulative_cache_usage(cache_root)
    runtime_cost = sum(float(row["reported_cost_usd"]) for row in runtime_usage)
    total_cost = sum(float(row["reported_cost_usd"]) for row in cumulative_usage)
    gate = config["qualification_gates"]
    gates = {
        "required_valid_runtime_rows": metrics["valid_runtime_rows"]
        >= gate["required_valid_runtime_rows"],
        "minimum_route_correct": metrics["route_correct"] >= gate["minimum_route_correct"],
        "minimum_decisive_executions": metrics["decisive_executions"]
        >= gate["minimum_decisive_executions"],
        "minimum_decisive_accuracy": metrics["decisive_accuracy"]
        >= gate["minimum_decisive_accuracy"],
        "minimum_typed_vs_direct_wins": metrics["typed_vs_direct_wins"]
        >= gate["minimum_typed_vs_direct_wins"],
        "no_typed_vs_direct_losses": metrics["typed_vs_direct_losses"]
        <= gate["maximum_typed_vs_direct_losses"],
        "source_permuted_wrong_type_abstains": controls["source_permuted_abstentions"]
        >= gate["required_source_permuted_abstentions"],
        "target_written_equivalent_matches": controls["target_written_equivalent_matches"]
        >= gate["required_target_written_equivalent_matches"],
        "runtime_no_answer_program_scene_graph_candidates_or_source_identity": all(
            not row[key] for row in evaluated for key in (
                "runtime_answer_read", "runtime_functional_program_read",
                "runtime_scene_graph_read", "runtime_source_identity_read",
                "operand_grounder_question_read", "operand_grounder_competing_operand_read",
                "object_ontology_original_question_read",
                "object_ontology_answer_candidates_read",
            )
        ),
        "object_ontology_precedes_direct": all(
            row["object_ontology_call_started_before_direct"] for row in evaluated
        ),
        "prequalification_harness_abstained": all(
            row["prequalification_source_selection"]["selected_program_sha256"] is None
            for row in evaluated
        ),
        "provider_cost_within_cap": total_cost <= gate["maximum_reported_provider_cost_usd"],
    }
    qualified = all(gates.values())
    for row in evaluated:
        receipt = parse_frame_grounding_receipt(
            row["grounding_receipt"],
            frame_count=int(config["media"]["dense_proxy_frame_count"]),
        )
        selection = select_source_for_grounding(
            sources, task_id=row["task_id"], receipt=receipt,
            target_grounder_sha256=grounder_sha256, grounder_qualified=qualified,
        )
        row["postqualification_source_selection"] = selection
        authorized = (
            selection["selected_program_sha256"] is not None
            and row["calibrated_target_native_execution"]["decision"] is not None
        )
        row["unified_harness_executor_authorized"] = authorized
        row["unified_harness_prediction"] = (
            row["calibrated_target_native_execution"]["decision"]
            if authorized else row["direct_response"]
        )
        row["unified_harness_correct"] = _answer_matches(
            row["unified_harness_prediction"], row["gold_answer_evaluator_only"]
        )
    metrics["unified_harness_executor_authorizations"] = sum(
        row["unified_harness_executor_authorized"] for row in evaluated
    )
    metrics["unified_harness_correct"] = sum(
        row["unified_harness_correct"] for row in evaluated
    )
    metrics["unified_harness_vs_direct_delta"] = (
        metrics["unified_harness_correct"] - metrics["direct_correct"]
    )
    body = {
        "schema_version": "agqa2-query-object-qualification-report-v20",
        "status": (
            f"AGQA2_QUERY_OBJECT_V20_{config['split'].upper()}_QUALIFIED"
            if qualified else
            f"AGQA2_QUERY_OBJECT_V20_{config['split'].upper()}_NOT_QUALIFIED"
        ),
        "split": config["split"],
        "claim_boundary": config["claim_boundary"],
        "grounder_sha256": grounder_sha256,
        "evaluation_protocol_sha256": evaluation_protocol_sha256,
        "config_file_sha256": _sha256(config_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "model": config["model"]["id"],
        "object_ontology_model": config["query_object_grounder"]["model"]["id"],
        "sample_count": len(evaluated),
        "metrics": metrics,
        "controls": controls,
        "qualification_gates": gates,
        "grounder_qualified": qualified,
        "accepted_runtime_provider_calls": len(runtime_usage),
        "accepted_runtime_reported_provider_cost_usd": runtime_cost,
        "provider_calls": len(cumulative_usage),
        "reported_provider_cost_usd": total_cost,
        "development_qualification_dependency": (
            dependency["report_sha256"] if dependency else None
        ),
        "source_portfolio_caveat": {
            "status": arcade_report["status"],
            "source_specific_claim_passed": False,
        },
        "rows": sorted(evaluated, key=lambda row: row["task_id"]),
        "untouched_benchmark_claim": False,
        "source_provenance_claim": False,
    }
    result = body | {"report_sha256": stable_hash(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", type=Path,
                        default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.output.resolve(), workers=args.workers,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "metrics", "controls", "qualification_gates",
        "reported_provider_cost_usd", "report_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
