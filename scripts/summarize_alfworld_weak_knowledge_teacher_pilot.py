#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


EXPECTED_CONDITIONS = (
    "target_only",
    "generic_reasoning",
    "source_receipts_only",
    "authentic_weak_control_prior",
    "shuffled_evidence_prior",
    "other_game_abstain",
)


def _uncached_usage(calls: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for call in calls:
        usage = call.get("usage") or {}
        if usage.get("cache_hit") is True:
            continue
        yield usage.get("original_usage") or usage


def _teacher_cost(
    usage_rows: Iterable[Mapping[str, Any]],
    *,
    input_per_million: float,
    cached_input_per_million: float,
    output_per_million: float,
) -> tuple[dict[str, int], float]:
    prompt = cached = completion = 0
    for usage in usage_rows:
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached += int(prompt_details.get("cached_tokens") or 0)
    uncached = prompt - cached
    cost = (
        uncached * input_per_million
        + cached * cached_input_per_million
        + completion * output_per_million
    ) / 1_000_000
    return {
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "completion_tokens": completion,
    }, cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="CONDITION=RUN_DIR",
        help="replace one condition's rows with task-matched rows from another run",
    )
    parser.add_argument("--teacher-input-per-million", type=float, default=0.25)
    parser.add_argument(
        "--teacher-cached-input-per-million", type=float, default=0.025
    )
    parser.add_argument("--teacher-output-per-million", type=float, default=2.0)
    args = parser.parse_args()

    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("task_*.json"))
    ]
    override_rows: dict[tuple[int, str], dict[str, Any]] = {}
    override_sources = {}
    for raw_override in args.override:
        condition, separator, raw_path = raw_override.partition("=")
        if not separator or condition not in EXPECTED_CONDITIONS:
            raise SystemExit(f"invalid --override: {raw_override}")
        override_dir = Path(raw_path)
        override_sources[condition] = str(override_dir.resolve())
        for path in sorted(override_dir.glob("task_*.json")):
            artifact = json.loads(path.read_text(encoding="utf-8"))
            matching = [
                row for row in artifact.get("rows") or []
                if row["condition"] == condition
            ]
            if len(matching) != 1:
                raise SystemExit(
                    f"{path} does not contain exactly one {condition} row"
                )
            override_rows[(int(artifact["task_offset"]), condition)] = matching[0]
    for artifact in artifacts:
        artifact["rows"] = [
            override_rows.get(
                (int(artifact["task_offset"]), row["condition"]), row
            )
            for row in artifact.get("rows") or []
        ]
    rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    integrity = {
        "task_files": len(artifacts),
        "complete_condition_sets": True,
        "matched_initial_state_within_task": True,
        "errors": [],
    }
    teacher_usage: list[Mapping[str, Any]] = []
    decision_cost = 0.0
    for artifact in artifacts:
        rows = artifact.get("rows") or []
        conditions = tuple(row["condition"] for row in rows)
        if conditions != EXPECTED_CONDITIONS:
            integrity["complete_condition_sets"] = False
        state_hashes = {
            row.get("initial_state_hash") for row in rows if row.get("initial_state_hash")
        }
        if len(state_hashes) > 1:
            integrity["matched_initial_state_within_task"] = False
        for row in rows:
            enriched = {**row, "task_offset": artifact["task_offset"]}
            rows_by_condition[row["condition"]].append(enriched)
            if row.get("error"):
                integrity["errors"].append({
                    "task_offset": artifact["task_offset"],
                    "condition": row["condition"],
                    "error": row["error"],
                })
            teacher_usage.extend(_uncached_usage(row.get("teacher_calls") or []))
            for usage in _uncached_usage(row.get("decision_calls") or []):
                decision_cost += float(usage.get("cost") or 0.0)

    aggregates = {}
    for condition in EXPECTED_CONDITIONS:
        rows = rows_by_condition.get(condition, [])
        metrics = [row["metrics"] for row in rows if row.get("metrics")]
        aggregates[condition] = {
            "tasks": len(rows),
            "successes": sum(bool(row["official_success"]) for row in metrics),
            "success_rate": (
                mean(bool(row["official_success"]) for row in metrics)
                if metrics else None
            ),
            "mean_official_score": (
                mean(float(row["official_score"]) for row in metrics)
                if metrics else None
            ),
            "mean_steps": (
                mean(int(row["steps"]) for row in metrics) if metrics else None
            ),
            "mean_repeated_actions": (
                mean(int(row["repeated_actions"]) for row in metrics)
                if metrics else None
            ),
            "mean_no_observable_progress": (
                mean(int(row["no_observable_progress"]) for row in metrics)
                if metrics else None
            ),
            "total_source_replans": sum(
                int(row.get("source_replans") or 0) for row in rows
            ),
            "source_fallbacks": sum(
                row.get("source_fallback_step") is not None for row in rows
            ),
            "initialization_abstentions": sum(
                bool((row.get("target_hypothesis") or {}).get("abstain"))
                for row in rows
            ),
            "initialization_validation_errors": sum(
                row.get("initialization_validation_error") is not None for row in rows
            ),
            "teacher_protocol_violations": sum(
                len(row.get("teacher_protocol_violations") or []) for row in rows
            ),
        }

    paired = []
    for artifact in artifacts:
        by_condition = {row["condition"]: row for row in artifact.get("rows") or []}
        if not all(condition in by_condition for condition in EXPECTED_CONDITIONS):
            continue
        authentic = by_condition["authentic_weak_control_prior"]["metrics"]
        paired.append({
            "task_offset": artifact["task_offset"],
            "authentic_success": authentic["official_success"],
            "authentic_steps": authentic["steps"],
            "deltas": {
                condition: {
                    "success": (
                        int(authentic["official_success"])
                        - int(by_condition[condition]["metrics"]["official_success"])
                    ),
                    "steps": (
                        int(authentic["steps"])
                        - int(by_condition[condition]["metrics"]["steps"])
                    ),
                }
                for condition in EXPECTED_CONDITIONS
                if condition != "authentic_weak_control_prior"
            },
        })

    teacher_tokens, teacher_cost = _teacher_cost(
        teacher_usage,
        input_per_million=args.teacher_input_per_million,
        cached_input_per_million=args.teacher_cached_input_per_million,
        output_per_million=args.teacher_output_per_million,
    )
    summary = {
        "schema_version": 1,
        "claim_limit": "PILOT_DISCOVERY_PROVISIONAL_NOT_SOURCE_SUPPORTED",
        "run_dir": str(args.run_dir.resolve()),
        "condition_overrides": override_sources,
        "integrity": integrity,
        "aggregates": aggregates,
        "paired_authentic_deltas": paired,
        "cost": {
            "teacher_model": "gpt-5-mini",
            **teacher_tokens,
            "teacher_estimated_usd": teacher_cost,
            "decision_openrouter_reported_usd": decision_cost,
            "total_estimated_usd": teacher_cost + decision_cost,
        },
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
