#!/usr/bin/env python3
"""Run frozen Sokoban-effect to WebShop neural-symbolic transfer V12."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.webshop_neural_symbolic_v9 import TargetOutcomeMLP  # noqa: E402
from motif_transfer.webshop_sokoban_effect_transfer import (  # noqa: E402
    CONDITIONS,
    choose_sokoban_effect_action,
    validate_source_gate,
)
import scripts.run_webshop_neural_symbolic_v9 as v9_runner  # noqa: E402


# Reuse the already-validated environment loop while replacing only its
# selector.  Python resolves this name in the V9 function's module globals.
v9_runner.choose_transfer_action = choose_sokoban_effect_action


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _retryable_provider_failure(receipts: list[dict]) -> bool:
    tokens = ("schema", "completion", "provider", "timeout", "timed out", "http")
    return any(
        row.get("failure") is not None
        and any(token in str(row["failure"]).lower() for token in tokens)
        for row in receipts
    )


def _make_backend(args: argparse.Namespace, cache_path: Path) -> MemoizedCompletionBackend:
    return MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url,
            {"decision": args.model},
            api_key_env="V12_OPENROUTER_API_KEY",
            json_mode=True,
            temperature=0,
            timeout_seconds=180,
            request_overrides={"max_tokens": args.maximum_output_tokens},
        ),
        cache_path=cache_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--role", choices=("qualification", "confirmation"), required=True)
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument(
        "--target-grounder",
        type=Path,
        default=REPO / "docs/results/webshop_neural_symbolic_v9_frozen_grounder.json",
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=REPO / "runs/sokoban_effect_program_v2/discovery_artifact.json",
    )
    parser.add_argument(
        "--source-confirmation",
        type=Path,
        default=REPO / "runs/sokoban_effect_program_v2/fresh_confirmation_report.json",
    )
    parser.add_argument("--goal-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--maximum-output-tokens", type=int, default=3200)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--whole-group-provider-retries", type=int, default=1)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--maximum-steps", type=int, default=12)
    parser.add_argument("--number-of-goals", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    target_artifact = _read(args.target_grounder)
    if not target_artifact.get("preflight_passed"):
        raise SystemExit("frozen V9 target grounder did not pass preflight")
    grounder = TargetOutcomeMLP.from_dict(target_artifact["grounder"])
    source_artifact = _read(args.source_artifact)
    source_confirmation = _read(args.source_confirmation)
    validate_source_gate(source_artifact, source_confirmation)
    source_models = {"artifact": source_artifact}

    values = __import__("runpy").run_path(str(args.keys))
    api_key = values.get("OPENROUTER_API_KEY") or values.get("openrouter_api_key")
    if not api_key:
        raise SystemExit("OpenRouter API key is missing")
    os.environ["V12_OPENROUTER_API_KEY"] = str(api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    goals = _read(args.goal_manifest) if args.goal_manifest else {}
    final_receipts: list[dict] = []
    operational_retries = []
    cache_hashes = {}

    for task_id in args.task_ids:
        final_paths = {
            condition: args.output_dir / f"{task_id}.{condition}.json"
            for condition in CONDITIONS
        }
        if args.resume and all(path.exists() for path in final_paths.values()):
            task_receipts = [_read(final_paths[condition]) for condition in CONDITIONS]
            final_receipts.extend(task_receipts)
            print(json.dumps({"task_id": task_id, "status": "resumed"}), flush=True)
            continue

        selected_receipts = None
        for attempt in range(args.whole_group_provider_retries + 1):
            attempt_dir = args.output_dir / "attempts" / task_id / f"attempt_{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            cache_path = attempt_dir / "decision_cache.json"
            backend = _make_backend(args, cache_path)
            expected_goal = goals.get("goals", {}).get(task_id, {}).get("instruction_text")
            receipts = []
            for condition in CONDITIONS:
                receipt = v9_runner._run_condition(
                    task_id=task_id,
                    condition=condition,
                    backend=backend,
                    grounder=grounder,
                    source_models=source_models,
                    # The reused V9 environment loop forwards these legacy
                    # interface scalars.  The Sokoban selector explicitly
                    # discards both; values are compatibility-only.
                    source_policy={"uncertainty_scale": 0.0, "decision_margin": 0.0},
                    expected_goal=expected_goal,
                    wrapper_root=args.wrapper_root,
                    session_namespace=(
                        f"{args.run_id}.a{attempt}.{condition.replace('_', '-')}"
                    ),
                    number_of_goals=args.number_of_goals,
                    maximum_steps=args.maximum_steps,
                    candidate_count=args.candidate_count,
                    schema_retries=args.schema_retries,
                )
                receipt["role"] = args.role
                receipt["attempt"] = attempt
                receipt["runtime_hashes"] = {
                    "runner": file_sha256(Path(__file__)),
                    "core": file_sha256(
                        REPO / "src/motif_transfer/webshop_sokoban_effect_transfer.py"
                    ),
                    "target_grounder": file_sha256(args.target_grounder),
                    "source_artifact": file_sha256(args.source_artifact),
                    "source_confirmation": file_sha256(args.source_confirmation),
                }
                receipt["receipt_sha256"] = stable_hash(receipt)
                (attempt_dir / f"{task_id}.{condition}.json").write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                receipts.append(receipt)
                if expected_goal is None and receipt["failure"] is None:
                    expected_goal = receipt["goal"]
                print(json.dumps({
                    "task_id": task_id,
                    "attempt": attempt,
                    "condition": condition,
                    "reward": receipt["official_reward"],
                    "strict": receipt["strict_success"],
                    "steps": receipt["step_count"],
                    "changes": receipt["changed_from_target_rank_zero_count"],
                    "source_decisions": receipt["source_decision_count"],
                    "failure": receipt["failure"],
                }), flush=True)
            if cache_path.exists():
                cache_hashes[f"{task_id}.attempt_{attempt}"] = file_sha256(cache_path)
            retryable = _retryable_provider_failure(receipts)
            if retryable and attempt < args.whole_group_provider_retries:
                operational_retries.append({
                    "task_id": task_id,
                    "attempt": attempt,
                    "reason": "symmetric_whole_group_provider_or_schema_failure",
                    "conditions_retried": list(CONDITIONS),
                })
                continue
            selected_receipts = receipts
            break
        assert selected_receipts is not None
        for receipt in selected_receipts:
            destination = final_paths[receipt["condition"]]
            source = (
                args.output_dir / "attempts" / task_id
                / f"attempt_{receipt['attempt']}" / destination.name
            )
            shutil.copyfile(source, destination)
        final_receipts.extend(selected_receipts)

    summary = {
        "schema_version": 1,
        "experiment": "webshop_sokoban_effect_transfer_v12",
        "role": args.role,
        "claim_limit": (
            f"Consumed fresh WebShop {args.role} tasks only; no broader-domain claim."
        ),
        "tasks": list(args.task_ids),
        "conditions": {
            condition: {
                "strict_successes": sum(
                    row["strict_success"] for row in final_receipts
                    if row["condition"] == condition
                ),
                "pass_successes": sum(
                    row["pass_success"] for row in final_receipts
                    if row["condition"] == condition
                ),
                "mean_reward": float(np.mean([
                    row["official_reward"] for row in final_receipts
                    if row["condition"] == condition
                ])),
                "mean_steps": float(np.mean([
                    row["step_count"] for row in final_receipts
                    if row["condition"] == condition
                ])),
                "changed_from_target_rank_zero": sum(
                    row["changed_from_target_rank_zero_count"] for row in final_receipts
                    if row["condition"] == condition
                ),
                "source_decisions": sum(
                    row["source_decision_count"] for row in final_receipts
                    if row["condition"] == condition
                ),
                "failures": sum(
                    row["failure"] is not None for row in final_receipts
                    if row["condition"] == condition
                ),
            }
            for condition in CONDITIONS
        },
        "matched_initial_state_hashes": all(
            len({
                row["initial_state_hash"] for row in final_receipts
                if row["task_id"] == task_id
            }) == 1
            for task_id in args.task_ids
        ),
        "operational_retries": operational_retries,
        "model": args.model,
        "number_of_goals": args.number_of_goals,
        "run_id": args.run_id,
        "runtime_hashes": {
            "runner": file_sha256(Path(__file__)),
            "core": file_sha256(
                REPO / "src/motif_transfer/webshop_sokoban_effect_transfer.py"
            ),
            "target_grounder": file_sha256(args.target_grounder),
            "source_artifact": file_sha256(args.source_artifact),
            "source_confirmation": file_sha256(args.source_confirmation),
            "decision_caches": cache_hashes,
        },
        "held_out_read_or_run": False,
    }
    summary["summary_sha256"] = stable_hash(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
