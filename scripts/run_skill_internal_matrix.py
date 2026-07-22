#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import runpy
from typing import Any

from motif_transfer.frozen_motif_agent import OpenAICompatibleBackend
from motif_transfer.instrumented_import import import_native_source_batch
from motif_transfer.skill_internal import (
    SourcePromptCondition,
    SkillInternalGraphAgent,
    _conditioned_payload,
    audit_frozen_alignment,
    audit_internal_graph,
    build_execution_sets,
    build_skill_off_control_set,
    load_skill_hypotheses,
    structural_fingerprint,
)


CONDITIONS = tuple(SourcePromptCondition)


def _records(episodes):
    return {
        record.transition.receipt_id: record
        for episode in episodes for record in episode.records
    }


def _three_way(execution_set) -> bool:
    return all(any(row.split == split for row in execution_set.executions) for split in (
        "discovery", "qualification", "held_out"
    ))


def _proposal_run(agent, execution_set, records, hypothesis, condition):
    try:
        candidates = agent.propose(
            execution_set, records, hypothesis, condition=condition
        )
        proposal_call = dict(agent.last_call)
        intervention_requests = [asdict(row) for row in agent.intervention_requests]
        audits = [
            audit_internal_graph(candidate, execution_set, records)
            for candidate in candidates
        ]
        qualification = []
        held_out = []
        for candidate, audit in zip(candidates, audits):
            if not audit.backbone_eligible:
                continue
            try:
                q_alignment = agent.align_frozen_graph(
                    candidate, execution_set, records,
                    split="qualification", hypothesis=hypothesis,
                )
                q_audit = audit_frozen_alignment(q_alignment, candidate, execution_set)
                qualification.append({
                    "graph_id": candidate.graph_id,
                    "alignment": asdict(q_alignment),
                    "audit": asdict(q_audit),
                })
            except Exception as exc:
                qualification.append({
                    "graph_id": candidate.graph_id,
                    "error": f"{type(exc).__name__}:{exc}",
                })
            try:
                h_alignment = agent.align_frozen_graph(
                    candidate, execution_set, records,
                    split="held_out", hypothesis=hypothesis,
                )
                h_audit = audit_frozen_alignment(h_alignment, candidate, execution_set)
                held_out.append({
                    "graph_id": candidate.graph_id,
                    "alignment": asdict(h_alignment),
                    "audit": asdict(h_audit),
                })
            except Exception as exc:
                held_out.append({
                    "graph_id": candidate.graph_id,
                    "error": f"{type(exc).__name__}:{exc}",
                })
        return {
            "condition": condition.value,
            "model_error": None,
            "agent_call": proposal_call,
            "intervention_requests": intervention_requests,
            "candidates": [asdict(row) for row in candidates],
            "audits": [asdict(row) for row in audits],
            "structural_fingerprints": [structural_fingerprint(row) for row in candidates],
            "qualification": qualification,
            "held_out": held_out,
        }
    except Exception as exc:
        return {
            "condition": condition.value,
            "model_error": f"{type(exc).__name__}:{exc}",
            "candidates": [],
            "audits": [],
            "qualification": [],
            "held_out": [],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen skill-internal control matrix")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--api-key-name", default="OPENAI_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--conditions", nargs="+", choices=[row.value for row in CONDITIONS],
        default=[row.value for row in CONDITIONS],
    )
    args = parser.parse_args()
    if not args.dry_run and (not args.endpoint or not args.model):
        parser.error("--endpoint and --model are required unless --dry-run is used")
    if args.key_file is not None and not args.dry_run:
        value = runpy.run_path(str(args.key_file)).get(args.api_key_name)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"missing {args.api_key_name} in key file")
        os.environ["SKILL_MATRIX_API_KEY"] = value
    selected_conditions = tuple(SourcePromptCondition(row) for row in args.conditions)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    backend = None if args.dry_run else OpenAICompatibleBackend(
        args.endpoint,
        {
            "skill_internal_graph": args.model,
            "skill_internal_alignment": args.model,
        },
        api_key_env="SKILL_MATRIX_API_KEY" if args.key_file else args.api_key_name,
        json_mode=True,
        temperature=None,
    )
    rows: list[dict[str, Any]] = []
    for spec in config["games"]:
        authentic_episodes = import_native_source_batch(spec["authentic_evidence"])
        skill_off_episodes = import_native_source_batch(spec["skill_off_evidence"])
        hypotheses = load_skill_hypotheses(spec["bank"])
        records = _records(authentic_episodes)
        skill_off_records = _records(skill_off_episodes)
        for execution_set in build_execution_sets(spec["game"], authentic_episodes, hypotheses):
            if not _three_way(execution_set):
                continue
            hypothesis = hypotheses.get(execution_set.skill_id)
            row: dict[str, Any] = {
                "game": spec["game"],
                "skill_id": execution_set.skill_id,
                "execution_set_id": execution_set.execution_set_id,
                "executions": len(execution_set.executions),
                "transitions": len(execution_set.transition_receipt_ids),
                "conditions": [],
            }
            if args.dry_run:
                discovery = tuple(
                    item for item in execution_set.executions if item.split == "discovery"
                )
                for condition in selected_conditions:
                    payload = _conditioned_payload(
                        execution_set, discovery, records, hypothesis, condition
                    )
                    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    row["conditions"].append({
                        "condition": condition.value,
                        "request_bytes": len(encoded),
                        "approx_tokens_upper_bound": (len(encoded) + 2) // 3,
                    })
            else:
                for condition in selected_conditions:
                    result = _proposal_run(
                        SkillInternalGraphAgent(backend), execution_set,
                        records, hypothesis, condition,
                    )
                    row["conditions"].append(result)
                    print(json.dumps({
                        "game": spec["game"],
                        "skill_id": execution_set.skill_id,
                        "condition": condition.value,
                        "candidates": len(result["candidates"]),
                        "backbone_eligible": sum(
                            audit.get("backbone_eligible", False)
                            for audit in result["audits"]
                        ),
                        "error": result["model_error"],
                    }, sort_keys=True), flush=True)
                skill_off_set = build_skill_off_control_set(
                    spec["game"], skill_off_episodes, execution_set.skill_id
                )
                row["skill_off_control"] = _proposal_run(
                    SkillInternalGraphAgent(backend), skill_off_set,
                    skill_off_records, None, SourcePromptCondition.AUTHENTIC,
                )
                print(json.dumps({
                    "game": spec["game"],
                    "skill_id": execution_set.skill_id,
                    "condition": "skill_off",
                    "candidates": len(row["skill_off_control"]["candidates"]),
                    "error": row["skill_off_control"]["model_error"],
                }, sort_keys=True), flush=True)
            rows.append(row)
    all_runs = [
        condition
        for row in rows
        for condition in row["conditions"] + (
            [row["skill_off_control"]] if "skill_off_control" in row else []
        )
    ]
    usage_totals: dict[str, int] = {}
    for result in all_runs:
        for key, value in result.get("agent_call", {}).get("usage", {}).items():
            if isinstance(value, int):
                usage_totals[key] = usage_totals.get(key, 0) + value
    report = {
        "schema_version": "SKILL_INTERNAL_MATRIX_V1_FROZEN",
        "dry_run": args.dry_run,
        "model_identity": dict(backend.identity) if backend is not None else None,
        "conditions": [row.value for row in selected_conditions],
        "skills": rows,
        "totals": {
            "skills": len(rows),
            "main_conditions": len(rows) * len(selected_conditions),
            "skill_off_controls": sum("skill_off_control" in row for row in rows),
            "model_calls": len(all_runs),
            "request_bytes": sum(
                condition.get("request_bytes", 0)
                for row in rows for condition in row["conditions"]
            ),
            "model_errors": sum(
                result.get("model_error") is not None for result in all_runs
            ) if not args.dry_run else 0,
            "backbone_eligible": sum(
                audit.get("backbone_eligible", False)
                for result in all_runs for audit in result.get("audits", [])
            ) if not args.dry_run else 0,
            "usage": usage_totals,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
