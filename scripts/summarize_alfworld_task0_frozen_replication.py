#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping


CONDITIONS = (
    "target_only",
    "generic_reasoning",
    "source_receipts_only",
    "authentic_weak_control_prior",
    "shuffled_evidence_prior",
    "other_game_abstain",
)
AUTHENTIC = "authentic_weak_control_prior"


def _hash(value: Any) -> str:
    return sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _binomial_tail(successes: int, trials: int) -> float | None:
    if trials == 0:
        return None
    return sum(comb(trials, value) for value in range(successes, trials + 1)) / (
        2 ** trials
    )


def _uncached_usage(calls: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for call in calls:
        usage = call.get("usage") or {}
        if usage.get("cache_hit") is True:
            continue
        yield usage.get("original_usage") or usage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-replicates", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="CONDITION=RUN_DIR",
    )
    args = parser.parse_args()

    paths = sorted(
        args.run_dir.glob("replicate_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    override_rows: dict[tuple[int, str], dict[str, Any]] = {}
    override_sources = {}
    for raw_override in args.override:
        condition, separator, raw_path = raw_override.partition("=")
        if not separator or condition not in CONDITIONS:
            raise SystemExit(f"invalid --override: {raw_override}")
        override_dir = Path(raw_path)
        override_sources[condition] = str(override_dir.resolve())
        for path in override_dir.glob("replicate_*.json"):
            artifact = json.loads(path.read_text(encoding="utf-8"))
            matches = [
                row for row in artifact.get("rows") or []
                if row["condition"] == condition
            ]
            if len(matches) != 1:
                raise SystemExit(
                    f"{path} does not contain exactly one {condition} row"
                )
            override_rows[(int(artifact["run_seed"]), condition)] = matches[0]
    for artifact in artifacts:
        artifact["rows"] = [
            override_rows.get(
                (int(artifact["run_seed"]), row["condition"]), row
            )
            for row in artifact.get("rows") or []
        ]
    rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seed_rows = []
    errors = []
    teacher_prompt = teacher_cached = teacher_completion = 0
    decision_cost = 0.0
    initial_hashes = set()
    frozen_hashes = set()
    hypothesis_hashes: dict[str, set[str]] = defaultdict(set)

    for artifact in artifacts:
        rows = artifact.get("rows") or []
        by_condition = {row["condition"]: row for row in rows}
        if tuple(row["condition"] for row in rows) != CONDITIONS:
            errors.append({
                "run_seed": artifact.get("run_seed"),
                "error": "incomplete_or_reordered_conditions",
            })
        row_summary = {
            "run_seed": artifact.get("run_seed"),
            "environment_seed": artifact.get("environment_seed"),
            "conditions": {},
        }
        for condition, row in by_condition.items():
            rows_by_condition[condition].append(row)
            if row.get("initial_state_hash"):
                initial_hashes.add(row["initial_state_hash"])
            hypothesis_hashes[condition].add(_hash(row["target_hypothesis"]))
            if row.get("error"):
                errors.append({
                    "run_seed": artifact.get("run_seed"),
                    "condition": condition,
                    "error": row["error"],
                })
            metrics = row.get("metrics") or {}
            row_summary["conditions"][condition] = {
                "success": metrics.get("official_success"),
                "steps": metrics.get("steps"),
                "repeated_actions": metrics.get("repeated_actions"),
                "source_replans": row.get("source_replans"),
                "source_fallback_step": row.get("source_fallback_step"),
                "protocol_violations": row.get(
                    "teacher_protocol_violations"
                ) or [],
            }
            for usage in _uncached_usage(row.get("teacher_calls") or []):
                teacher_prompt += int(usage.get("prompt_tokens") or 0)
                teacher_completion += int(usage.get("completion_tokens") or 0)
                details = usage.get("prompt_tokens_details") or {}
                teacher_cached += int(details.get("cached_tokens") or 0)
            for usage in _uncached_usage(row.get("decision_calls") or []):
                decision_cost += float(usage.get("cost") or 0.0)
        frozen = artifact.get("frozen_hypotheses") or {}
        if frozen.get("sha256"):
            frozen_hashes.add(frozen["sha256"])
        seed_rows.append(row_summary)

    aggregates = {}
    for condition in CONDITIONS:
        rows = rows_by_condition.get(condition, [])
        metrics = [row["metrics"] for row in rows if row.get("metrics")]
        aggregates[condition] = {
            "replicates": len(rows),
            "successes": sum(bool(row["official_success"]) for row in metrics),
            "success_rate": (
                mean(bool(row["official_success"]) for row in metrics)
                if metrics else None
            ),
            "mean_steps": (
                mean(int(row["steps"]) for row in metrics) if metrics else None
            ),
            "median_steps": (
                median(int(row["steps"]) for row in metrics) if metrics else None
            ),
            "mean_repeated_actions": (
                mean(int(row["repeated_actions"]) for row in metrics)
                if metrics else None
            ),
            "source_replans": sum(
                int(row.get("source_replans") or 0) for row in rows
            ),
            "source_fallbacks": sum(
                row.get("source_fallback_step") is not None for row in rows
            ),
            "protocol_violations": sum(
                len(row.get("teacher_protocol_violations") or [])
                for row in rows
            ),
        }

    contrasts = {}
    complete_by_seed = {
        row["run_seed"]: row["conditions"]
        for row in seed_rows
        if set(row["conditions"]) == set(CONDITIONS)
    }
    for control in CONDITIONS:
        if control == AUTHENTIC:
            continue
        wins = losses = ties = 0
        capped_step_deltas = []
        for by_condition in complete_by_seed.values():
            authentic = by_condition[AUTHENTIC]
            baseline = by_condition[control]
            authentic_success = bool(authentic["success"])
            baseline_success = bool(baseline["success"])
            if authentic_success and not baseline_success:
                wins += 1
            elif baseline_success and not authentic_success:
                losses += 1
            else:
                ties += 1
            capped_step_deltas.append(
                int(authentic["steps"]) - int(baseline["steps"])
            )
        discordant = wins + losses
        one_sided = _binomial_tail(wins, discordant)
        contrasts[control] = {
            "authentic_wins": wins,
            "authentic_losses": losses,
            "success_ties": ties,
            "discordant_pairs": discordant,
            "exact_one_sided_p": one_sided,
            "exact_two_sided_p": (
                min(1.0, 2 * one_sided) if one_sided is not None else None
            ),
            "mean_capped_step_delta": (
                mean(capped_step_deltas) if capped_step_deltas else None
            ),
        }

    uncached_prompt = teacher_prompt - teacher_cached
    teacher_cost = (
        uncached_prompt * 0.25
        + teacher_cached * 0.025
        + teacher_completion * 2.0
    ) / 1_000_000
    summary = {
        "schema_version": 1,
        "claim_limit": "TASK0_FROZEN_MULTI_SEED_REPLICATION",
        "run_dir": str(args.run_dir.resolve()),
        "condition_overrides": override_sources,
        "integrity": {
            "expected_replicates": args.expected_replicates,
            "completed_replicates": len(artifacts),
            "all_complete": (
                len(artifacts) == args.expected_replicates and not errors
            ),
            "unique_initial_state_hashes": sorted(initial_hashes),
            "unique_frozen_hypothesis_artifact_hashes": sorted(frozen_hashes),
            "unique_hypothesis_hash_count_by_condition": {
                condition: len(hypothesis_hashes[condition])
                for condition in CONDITIONS
            },
            "errors": errors,
        },
        "aggregates": aggregates,
        "paired_authentic_contrasts": contrasts,
        "replicates": seed_rows,
        "cost": {
            "teacher_prompt_tokens": teacher_prompt,
            "teacher_cached_prompt_tokens": teacher_cached,
            "teacher_completion_tokens": teacher_completion,
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
