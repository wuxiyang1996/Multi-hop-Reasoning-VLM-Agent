#!/usr/bin/env python3
"""Run a matched V16 search-automaton probe on WebShop development tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.active_video_transfer import (  # noqa: E402
    exact_binomial_two_sided,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
)
from motif_transfer.webshop_constraint_coverage_v14 import (  # noqa: E402
    augment_with_constraint_labels,
    augment_with_product_backtrack,
    audit_receipt_commits,
)
from motif_transfer.webshop_neural_symbolic_v9 import (  # noqa: E402
    TargetOutcomeMLP,
)
from motif_transfer.webshop_search_automaton_v16 import (  # noqa: E402
    AUTHENTIC,
    CEILING,
    CONDITIONS,
    LEDGER_BLIND,
    PERMUTED,
    RAW,
    WebShopSearchAutomatonController,
)
import scripts.run_webshop_neural_symbolic_v9 as v9_runner  # noqa: E402


ORIGINAL_DECISION_CANDIDATES = v9_runner._decision_candidates


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_formal_protocol(
    *,
    path: Path,
    task_ids: list[str],
    qualification_report: Path,
    source_artifact: Path,
    target_grounder: Path,
    goal_manifest: Path,
    model: str,
    maximum_output_tokens: int,
    maximum_steps: int,
) -> dict[str, Any]:
    protocol = _read(path)
    if protocol.get("status") != "FROZEN_BEFORE_FORMAL_EXECUTION":
        raise SystemExit("WebShop V16 formal protocol is not frozen")
    expected = {
        "tasks": task_ids,
        "conditions": list(CONDITIONS),
        "qualification_report_file_sha256": _sha256(qualification_report),
        "source_artifact_file_sha256": _sha256(source_artifact),
        "target_grounder_file_sha256": _sha256(target_grounder),
        "goal_manifest_file_sha256": _sha256(goal_manifest),
        "controller_file_sha256": _sha256(
            REPO / "src/motif_transfer/webshop_search_automaton_v16.py"
        ),
        "coverage_controller_file_sha256": _sha256(
            REPO / "src/motif_transfer/webshop_coverage_transfer_v14.py"
        ),
        "runner_file_sha256": _sha256(Path(__file__)),
        "model": model,
        "maximum_output_tokens": maximum_output_tokens,
        "maximum_steps": maximum_steps,
    }
    mismatches = {
        key: {"expected": value, "observed": protocol.get(key)}
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"WebShop V16 formal protocol mismatch: {mismatches}")
    qualification = _read(qualification_report)
    if qualification.get("status") != "CONSUMED_DEVELOPMENT_TRANSFER_GATE_PASSED":
        raise SystemExit("WebShop V16 development qualification did not pass")
    if not all(qualification.get("gates", {}).values()):
        raise SystemExit("WebShop V16 development qualification gates are incomplete")
    return protocol


def _candidate_augmenter(goal_options: Mapping[str, Any]):
    def decide(**kwargs: Any):
        candidates, raw, attempts = ORIGINAL_DECISION_CANDIDATES(**kwargs)
        augmented = augment_with_constraint_labels(
            candidates,
            axtree=kwargs["axtree"],
            goal=str(kwargs["payload"]["goal"]),
            goal_options=goal_options,
        )
        augmented = augment_with_product_backtrack(
            augmented, url=str(kwargs["payload"].get("url") or ""),
        )
        return augmented, raw, attempts

    return decide

def _summary(receipts: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    rows = [row for row in receipts if row["condition"] == condition]
    return {
        "tasks": len(rows),
        "strict_successes": sum(bool(row["strict_success"]) for row in rows),
        "pass_successes": sum(bool(row["pass_success"]) for row in rows),
        "mean_reward": float(np.mean([row["official_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["step_count"] for row in rows])),
        "changed_from_target_rank_zero": sum(
            int(row["changed_from_target_rank_zero_count"]) for row in rows
        ),
        "source_decisions": sum(
            int(row["v16_controller"]["source_decisions"]) for row in rows
        ),
        "source_action_counts": dict(sorted({
            action: sum(
                int(row["v16_controller"]["source_action_counts"].get(action, 0))
                for row in rows
            )
            for action in {
                action
                for row in rows
                for action in row["v16_controller"]["source_action_counts"]
            }
        }.items())),
        "unsafe_commits": sum(len(row["unsafe_commits"]) for row in rows),
        "failures": sum(row["failure"] is not None for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", nargs="+", default=["webshop.4"])
    parser.add_argument(
        "--role", choices=("development", "formal_reserve"),
        default="development",
    )
    parser.add_argument(
        "--wrapper-root", type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent"
        ),
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--goal-manifest", type=Path,
        default=REPO / "configs/webshop_synthetic_unique_v14_frozen.json",
    )
    parser.add_argument(
        "--target-grounder", type=Path,
        default=REPO / "docs/results/webshop_neural_symbolic_v9_frozen_grounder.json",
    )
    parser.add_argument(
        "--source-artifact", type=Path,
        default=REPO / "runs/sokoban_search_automaton_v16/artifact.json",
    )
    parser.add_argument(
        "--cache-seed", type=Path,
        default=(
            REPO / "runs/webshop_coverage_transfer_v14_backtrack_probe_v4/"
            "attempts/webshop.4/attempt_0/decision_cache.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/webshop_search_automaton_v16_development",
    )
    parser.add_argument(
        "--qualification-report", type=Path,
        default=(
            REPO / "runs/webshop_search_automaton_v16_development_"
            "gpt41mini_anytime/report.json"
        ),
    )
    parser.add_argument("--formal-protocol", type=Path)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--maximum-output-tokens", type=int, default=3200)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--maximum-steps", type=int, default=12)
    parser.add_argument("--run-id", default="webshop-search-automaton-v16-dev")
    args = parser.parse_args()

    manifest = _read(args.goal_manifest)
    if manifest.get("status") != "FROZEN_BEFORE_ANY_PROVIDER_CALL_OR_OUTCOME":
        raise SystemExit("WebShop synthetic manifest is not frozen")
    selected_role = {
        str(row["task_id"]): row
        for row in manifest["roles"][args.role]
    }
    invalid = sorted(set(args.task_ids) - set(selected_role))
    if invalid:
        raise SystemExit(f"tasks outside WebShop {args.role} forbidden: {invalid}")
    protocol = None
    if args.role == "formal_reserve":
        if args.formal_protocol is None:
            raise SystemExit("--formal-protocol is required for formal reserve")
        protocol = _require_formal_protocol(
            path=args.formal_protocol,
            task_ids=list(args.task_ids),
            qualification_report=args.qualification_report,
            source_artifact=args.source_artifact,
            target_grounder=args.target_grounder,
            goal_manifest=args.goal_manifest,
            model=args.model,
            maximum_output_tokens=args.maximum_output_tokens,
            maximum_steps=args.maximum_steps,
        )
    target_artifact = _read(args.target_grounder)
    if not target_artifact.get("preflight_passed"):
        raise SystemExit("target neural grounder did not pass")
    grounder = TargetOutcomeMLP.from_dict(target_artifact["grounder"])
    source = SourceSearchAutomaton(_read(args.source_artifact))
    values = runpy.run_path(str(args.keys))
    key = values.get("OPENROUTER_API_KEY") or values.get("openrouter_api_key")
    if not key:
        raise SystemExit("OpenRouter API key is missing")
    os.environ["V16_WEBSHOP_OPENROUTER_KEY"] = str(key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "decision_cache.json"
    if not cache_path.exists() and args.cache_seed.is_file():
        shutil.copyfile(args.cache_seed, cache_path)
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url,
            {"decision": args.model},
            api_key_env="V16_WEBSHOP_OPENROUTER_KEY",
            json_mode=True,
            temperature=0,
            timeout_seconds=180,
            request_overrides={"max_tokens": args.maximum_output_tokens},
        ),
        cache_path=cache_path,
    )
    receipts = []
    number_of_goals = int(manifest["number_of_registered_tasks_required"])
    for task_id in args.task_ids:
        goal_row = selected_role[task_id]
        goal_options = dict(goal_row.get("goal", {}).get("goal_options", {}))
        for condition in CONDITIONS:
            controller = WebShopSearchAutomatonController(
                condition=condition,
                source=source,
                episode_id=task_id,
                goal_options=goal_options,
                maximum_steps=args.maximum_steps,
            )
            v9_runner._decision_candidates = _candidate_augmenter(goal_options)
            v9_runner.choose_transfer_action = controller
            receipt = v9_runner._run_condition(
                task_id=task_id,
                condition=condition,
                backend=backend,
                grounder=grounder,
                source_models={"artifact": {}},
                source_policy={"uncertainty_scale": 0.0, "decision_margin": 0.0},
                expected_goal=str(goal_row["instruction_text"]),
                wrapper_root=args.wrapper_root,
                session_namespace=(
                    f"{args.run_id}.{condition.replace('_', '-')}"
                ),
                number_of_goals=number_of_goals,
                maximum_steps=args.maximum_steps,
                candidate_count=args.candidate_count,
                schema_retries=args.schema_retries,
            )
            receipt["v16_controller"] = controller.as_dict()
            receipt["commit_audit"] = audit_receipt_commits(receipt)
            receipt["unsafe_commits"] = [
                row for row in receipt["commit_audit"]
                if goal_options
                and not row["authorized"]
                and not bool(receipt["steps"][int(row["step"])]["source_abstained"])
            ] if condition in {AUTHENTIC, LEDGER_BLIND, CEILING} else []
            receipt["runtime_hashes"] = {
                "runner": file_sha256(Path(__file__)),
                "controller": file_sha256(
                    REPO / "src/motif_transfer/webshop_search_automaton_v16.py"
                ),
                "source_artifact": file_sha256(args.source_artifact),
                "target_grounder": file_sha256(args.target_grounder),
                "goal_manifest": file_sha256(args.goal_manifest),
            }
            receipt["receipt_sha256"] = stable_hash(receipt)
            (args.output_dir / f"{task_id}.{condition}.json").write_text(
                json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            receipts.append(receipt)
            print(json.dumps({
                "task_id": task_id,
                "condition": condition,
                "strict": receipt["strict_success"],
                "reward": receipt["official_reward"],
                "steps": receipt["step_count"],
                "source_actions": controller.as_dict()["source_action_counts"],
                "unsafe_commits": len(receipt["unsafe_commits"]),
                "failure": receipt["failure"],
            }), flush=True)

    summaries = {condition: _summary(receipts, condition) for condition in CONDITIONS}
    authentic = summaries[AUTHENTIC]
    paired = {}
    by_condition = {
        condition: {
            row["task_id"]: row
            for row in receipts if row["condition"] == condition
        }
        for condition in CONDITIONS
    }
    for comparator in CONDITIONS:
        if comparator == AUTHENTIC:
            continue
        wins = losses = reward_wins = reward_losses = 0
        for task_id in args.task_ids:
            a = bool(by_condition[AUTHENTIC][task_id]["strict_success"])
            b = bool(by_condition[comparator][task_id]["strict_success"])
            wins += a and not b
            losses += b and not a
            a_reward = float(by_condition[AUTHENTIC][task_id]["official_reward"])
            b_reward = float(by_condition[comparator][task_id]["official_reward"])
            reward_wins += a_reward > b_reward + 1e-12
            reward_losses += b_reward > a_reward + 1e-12
        paired[comparator] = {
            "wins": wins,
            "losses": losses,
            "ties": len(args.task_ids) - wins - losses,
            "net_wins": wins - losses,
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
            "reward_wins": reward_wins,
            "reward_losses": reward_losses,
            "reward_ties": len(args.task_ids) - reward_wins - reward_losses,
            "reward_net_wins": reward_wins - reward_losses,
            "reward_exact_two_sided_p": exact_binomial_two_sided(
                reward_wins, reward_losses,
            ),
        }
    gates = {
        "all_receipts_complete": all(row["failure"] is None for row in receipts),
        "matched_initial_state_hashes": all(
            len({
                row["initial_state_hash"] for row in receipts
                if row["task_id"] == task_id
            }) == 1
            for task_id in args.task_ids
        ),
        "all_three_source_actions_exercised": set(
            authentic["source_action_counts"]
        ) == {"BACKTRACK_REPLAN", "COMMIT_VERIFY", "EXPLORE_UNTRIED"},
        "zero_authentic_unsafe_commits": authentic["unsafe_commits"] == 0,
        "authentic_success_gain_over_raw": (
            authentic["strict_successes"] > summaries[RAW]["strict_successes"]
        ),
        "zero_negative_transfer_vs_raw": paired[RAW]["losses"] == 0,
        "pass_success_not_below_raw": (
            authentic["pass_successes"] >= summaries[RAW]["pass_successes"]
        ),
        "mean_reward_not_below_raw": (
            authentic["mean_reward"] + 1e-12 >= summaries[RAW]["mean_reward"]
        ),
        "reward_pairing_not_net_negative": (
            paired[RAW]["reward_net_wins"] >= 0
        ),
        "strictly_beats_destructive_controls": all(
            authentic["strict_successes"] > summaries[name]["strict_successes"]
            for name in (PERMUTED, LEDGER_BLIND)
        ),
        "matches_isomorphic_target_search_ceiling": (
            authentic["strict_successes"] == summaries[CEILING]["strict_successes"]
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "webshop-search-automaton-transfer-v16",
        "status": (
            (
                "FRESH_FORMAL_TRANSFER_GATE_PASSED"
                if passed else "FRESH_FORMAL_TRANSFER_GATE_FAILED"
            )
            if args.role == "formal_reserve"
            else (
                "CONSUMED_DEVELOPMENT_TRANSFER_GATE_PASSED"
                if passed else "CONSUMED_DEVELOPMENT_TRANSFER_GATE_FAILED"
            )
        ),
        "claim_boundary": (
            "Prospectively frozen synthetic WebShop formal reserve; opened once "
            "after the hash-locked V16 development gate passed."
            if args.role == "formal_reserve"
            else (
                "Previously consumed synthetic WebShop development tasks only; "
                "formal reserve remains sealed."
            )
        ),
        "role": args.role,
        "formal_protocol_file_sha256": (
            _sha256(args.formal_protocol) if args.formal_protocol else None
        ),
        "qualification_report_file_sha256": (
            _sha256(args.qualification_report)
            if args.role == "formal_reserve" else None
        ),
        "formal_protocol": protocol,
        "source_artifact_sha256": source.artifact_sha256,
        "tasks": list(args.task_ids),
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "decision_cache_file_sha256": (
            file_sha256(cache_path) if cache_path.is_file() else None
        ),
    }
    report = body | {"report_sha256": stable_hash(body)}
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
