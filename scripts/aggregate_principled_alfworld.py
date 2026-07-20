#!/usr/bin/env python3
"""Validate, pair, and aggregate every shard in the preregistered study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_frozen_inputs(spec: Mapping[str, Any]) -> None:
    entries = dict(spec.get("frozen_inputs") or {})
    entries["source_sft"] = spec["source_sft"]
    for name, item in entries.items():
        path = REPO_ROOT / str(item["path"])
        if not path.is_file():
            raise RuntimeError(f"missing frozen input {name}: {path}")
        actual = _sha256(path)
        if actual != str(item["sha256"]):
            raise RuntimeError(
                f"frozen input drift for {name}: expected {item['sha256']}, got {actual}"
            )


def _bootstrap_delta(values: Sequence[float], *, seed: int, samples: int = 10000) -> List[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def _mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, Any]:
    b_only = sum((not x) and y for x, y in zip(a, b))
    a_only = sum(x and (not y) for x, y in zip(a, b))
    discordant = b_only + a_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(b_only, a_only)
        tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "a_only_success": a_only,
        "b_only_success": b_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failures = Counter()
    prompt_tokens = 0
    completion_tokens = 0
    request_latency = 0.0
    n_requests = 0
    for row in rows:
        if row.get("abstain_reason"):
            failures[str(row["abstain_reason"]).split(":", 1)[0]] += 1
        for trace in row.get("traces", []):
            for key in ("skill_usage", "action_usage"):
                usage = trace.get(key)
                if isinstance(usage, dict):
                    prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    completion_tokens += int(usage.get("completion_tokens") or 0)
                    request_latency += float(usage.get("latency_s") or 0.0)
                    n_requests += 1
    n = len(rows)
    return {
        "n": n,
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / n if n else 0.0,
        "abstentions": sum(bool(row["abstained"]) for row in rows),
        "abstention_rate": sum(bool(row["abstained"]) for row in rows) / n if n else 0.0,
        "errors": sum(row.get("error") is not None for row in rows),
        "mean_steps": mean(int(row["steps"]) for row in rows) if rows else 0.0,
        "episode_wall_time_s": sum(float(row["wall_time_s"]) for row in rows),
        "failure_families": dict(failures),
        "requests": n_requests,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "mean_request_latency_s": request_latency / n_requests if n_requests else 0.0,
    }


def _require_error_free(
    rows: Sequence[Mapping[str, Any]], *, condition: str, split: str,
) -> None:
    failures = [row for row in rows if row.get("error") is not None]
    if failures:
        examples = [str(row.get("error")) for row in failures[:3]]
        raise RuntimeError(
            f"evaluation errors in {condition}/{split}: "
            f"{len(failures)} row(s); examples={examples}"
        )


def _comparison(
    index: Mapping[tuple[str, str], Mapping[int, Mapping[str, Any]]],
    *, a: str, b: str, splits: Sequence[str], seed: int,
) -> Dict[str, Any]:
    a_values: List[bool] = []
    b_values: List[bool] = []
    by_split: Dict[str, Any] = {}
    for split in splits:
        a_rows = index[(a, split)]
        b_rows = index[(b, split)]
        ids = sorted(a_rows)
        av = [bool(a_rows[item]["success"]) for item in ids]
        bv = [bool(b_rows[item]["success"]) for item in ids]
        deltas = [float(y) - float(x) for x, y in zip(av, bv)]
        by_split[split] = {
            "n": len(ids),
            "a_success_rate": mean(av) if av else 0.0,
            "b_success_rate": mean(bv) if bv else 0.0,
            "delta_b_minus_a": mean(deltas) if deltas else 0.0,
            "paired_bootstrap_95ci": _bootstrap_delta(deltas, seed=seed + len(a_values)),
            "mcnemar": _mcnemar_exact(av, bv),
        }
        a_values.extend(av)
        b_values.extend(bv)
    deltas = [float(y) - float(x) for x, y in zip(a_values, b_values)]
    return {
        "a": a,
        "b": b,
        "n": len(deltas),
        "a_success_rate": mean(a_values) if a_values else 0.0,
        "b_success_rate": mean(b_values) if b_values else 0.0,
        "delta_b_minus_a": mean(deltas) if deltas else 0.0,
        "paired_bootstrap_95ci": _bootstrap_delta(deltas, seed=seed),
        "mcnemar": _mcnemar_exact(a_values, b_values),
        "by_split": by_split,
    }


def aggregate(run_root: Path, spec_path: Path) -> Dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _verify_frozen_inputs(spec)
    conditions = [str(item) for item in spec["conditions"]]
    split_counts = {str(key): int(value) for key, value in spec["splits"].items()}
    num_shards = int(spec["num_rollout_shards"])
    index: Dict[tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    summaries: Dict[str, Any] = {}
    task_identity: Dict[tuple[str, int], str] = {}
    artifact_hashes = None
    for condition in conditions:
        summaries[condition] = {}
        for split, expected_n in split_counts.items():
            rows: List[Dict[str, Any]] = []
            for shard in range(num_shards):
                path = run_root / "eval" / condition / split / f"shard_{shard}.json"
                if not path.is_file():
                    raise RuntimeError(f"missing result shard: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                for key, expected in (
                    ("condition", condition), ("split", split),
                    ("num_shards", num_shards), ("shard_index", shard),
                    ("episodes_planned_total", expected_n),
                ):
                    if payload.get(key) != expected:
                        raise RuntimeError(f"shard metadata mismatch {path}: {key}")
                if payload.get("target_gradient_updates") != 0:
                    raise RuntimeError(f"target update leakage in {path}")
                hashes = tuple(payload.get("artifact_hashes") or [])
                if artifact_hashes is None:
                    artifact_hashes = hashes
                elif hashes != artifact_hashes:
                    raise RuntimeError(f"artifact scope drift in {path}")
                rows.extend(dict(row) for row in payload.get("rows", []))
            if len(rows) != expected_n:
                raise RuntimeError(
                    f"incomplete cell {condition}/{split}: {len(rows)} != {expected_n}"
                )
            _require_error_free(rows, condition=condition, split=split)
            by_id = {int(row["global_episode_index"]): row for row in rows}
            if len(by_id) != expected_n or set(by_id) != set(range(expected_n)):
                raise RuntimeError(f"duplicate/missing episode indices: {condition}/{split}")
            if len({str(row["task_id"]) for row in rows}) != expected_n:
                raise RuntimeError(f"official task IDs are not unique: {condition}/{split}")
            for episode_index, row in by_id.items():
                identity_key = (split, episode_index)
                task = str(row["task_id"])
                if identity_key in task_identity and task_identity[identity_key] != task:
                    raise RuntimeError(
                        f"unpaired task order at {condition}/{split}/{episode_index}"
                    )
                task_identity[identity_key] = task
            index[(condition, split)] = by_id
            summaries[condition][split] = _summarize(rows)
        combined = [
            row for split in split_counts for row in index[(condition, split)].values()
        ]
        summaries[condition]["overall"] = _summarize(combined)

    comparisons = {
        "game_sft_harness_vs_game_sft": _comparison(
            index, a="game_sft", b="game_sft_harness",
            splits=list(split_counts), seed=91001,
        ),
        "game_sft_harness_vs_base": _comparison(
            index, a="base", b="game_sft_harness",
            splits=list(split_counts), seed=91002,
        ),
        "base_harness_vs_base": _comparison(
            index, a="base", b="base_harness",
            splits=list(split_counts), seed=91003,
        ),
    }
    return {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "experiment_spec": str(spec_path.resolve()),
        "experiment_spec_sha256": _sha256(spec_path),
        "complete": True,
        "paired_unique_official_tasks": sum(split_counts.values()),
        "total_episodes": sum(split_counts.values()) * len(conditions),
        "artifact_hashes": list(artifact_hashes or []),
        "summaries": summaries,
        "comparisons": comparisons,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Principled ALFWorld 2×4 L40S Results",
        "",
        f"Complete: **{result['complete']}**; paired official tasks: "
        f"**{result['paired_unique_official_tasks']}**; total episodes: "
        f"**{result['total_episodes']}**.",
        "",
        "| condition | seen success | unseen success | overall success | abstention |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, cells in result["summaries"].items():
        lines.append(
            f"| {condition} | {cells['eval_in_distribution']['success_rate']:.3f} | "
            f"{cells['eval_out_of_distribution']['success_rate']:.3f} | "
            f"{cells['overall']['success_rate']:.3f} | "
            f"{cells['overall']['abstention_rate']:.3f} |"
        )
    lines += ["", "## Paired comparisons", ""]
    for name, item in result["comparisons"].items():
        ci = item["paired_bootstrap_95ci"]
        lines.append(
            f"- `{name}`: Δ={item['delta_b_minus_a']:+.3f}, "
            f"paired 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
            f"McNemar exact p={item['mcnemar']['two_sided_exact_p']:.4g}."
        )
    lines += [
        "",
        "Success is ALFWorld official `won` only. Invalid model output and "
        "out-of-scope operators are abstentions; there is no random fallback.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--spec", type=Path,
        default=REPO_ROOT / "configs/principled_alfworld_2x4_experiment.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = aggregate(args.run_root, args.spec)
    output = args.output or (args.run_root / "aggregate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "complete": result["complete"],
        "tasks": result["paired_unique_official_tasks"],
        "episodes": result["total_episodes"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
