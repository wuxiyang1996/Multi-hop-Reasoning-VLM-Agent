#!/usr/bin/env python3
"""Run sealed-development Sokoban-to-WebShop coverage transfer V14."""

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
from motif_transfer.webshop_constraint_coverage_v14 import (  # noqa: E402
    augment_with_constraint_labels,
    augment_with_product_backtrack,
    audit_receipt_commits,
)
from motif_transfer.webshop_coverage_transfer_v14 import (  # noqa: E402
    AUTHENTIC_COVERAGE,
    COMMIT_AVAILABILITY_COVERAGE,
    CONDITIONS,
    CoverageTransferController,
    INVERTED_COVERAGE,
    POSITION_PRIOR_COVERAGE,
    TARGET_COVERAGE,
    TARGET_ONLY,
)
from motif_transfer.webshop_neural_symbolic_v9 import TargetOutcomeMLP  # noqa: E402
from motif_transfer.webshop_sokoban_effect_transfer import (  # noqa: E402
    validate_source_gate,
)
import scripts.run_webshop_neural_symbolic_v9 as v9_runner  # noqa: E402


ORIGINAL_DECISION_CANDIDATES = v9_runner._decision_candidates


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _development_rows(manifest: dict) -> dict[str, dict]:
    rows = manifest.get("roles", {}).get("development", [])
    return {
        str(row["task_id"]): row
        for row in rows
    }


def _candidate_augmenter(goal_options: dict):
    def decide(**kwargs):
        candidates, raw, attempts = ORIGINAL_DECISION_CANDIDATES(**kwargs)
        payload = kwargs["payload"]
        augmented = augment_with_constraint_labels(
            candidates,
            axtree=kwargs["axtree"],
            goal=str(payload["goal"]),
            goal_options=goal_options,
        )
        augmented = augment_with_product_backtrack(
            augmented, url=str(payload.get("url") or ""),
        )
        return augmented, raw, attempts

    return decide


def _retryable_provider_failure(receipts: list[dict]) -> bool:
    tokens = ("schema", "completion", "provider", "timeout", "timed out", "http")
    return any(
        row.get("failure") is not None
        and any(token in str(row["failure"]).lower() for token in tokens)
        for row in receipts
    )


def _make_backend(args: argparse.Namespace, cache_path: Path):
    return MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url,
            {"decision": args.model},
            api_key_env="V14_OPENROUTER_API_KEY",
            json_mode=True,
            temperature=0,
            timeout_seconds=180,
            request_overrides={"max_tokens": args.maximum_output_tokens},
        ),
        cache_path=cache_path,
    )


def _condition_summary(receipts: list[dict], condition: str) -> dict:
    rows = [row for row in receipts if row["condition"] == condition]
    return {
        "episodes": len(rows),
        "strict_successes": sum(row["strict_success"] for row in rows),
        "pass_successes": sum(row["pass_success"] for row in rows),
        "mean_reward": float(np.mean([row["official_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["step_count"] for row in rows])),
        "changed_from_target_rank_zero": sum(
            row["changed_from_target_rank_zero_count"] for row in rows
        ),
        "coverage_interventions": sum(
            row["coverage_controller"]["coverage_interventions"] for row in rows
        ),
        "source_decisions": sum(row["source_decision_count"] for row in rows),
        "source_commit_decisions": sum(
            not step["source_abstained"] and step["abstract_kind"] == "COMMIT"
            for row in rows for step in row["steps"]
        ),
        "source_position_decisions": sum(
            not step["source_abstained"] and step["abstract_kind"] == "POSITION"
            for row in rows for step in row["steps"]
        ),
        "unsafe_commits": sum(len(row["unsafe_commits"]) for row in rows),
        "failures": sum(row["failure"] is not None for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--role", choices=("development",), required=True)
    parser.add_argument("--wrapper-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument(
        "--target-grounder", type=Path,
        default=REPO / "docs/results/webshop_neural_symbolic_v9_frozen_grounder.json",
    )
    parser.add_argument(
        "--source-artifact", type=Path,
        default=REPO / "runs/sokoban_effect_program_v2/discovery_artifact.json",
    )
    parser.add_argument(
        "--source-confirmation", type=Path,
        default=REPO / "runs/sokoban_effect_program_v2/fresh_confirmation_report.json",
    )
    parser.add_argument(
        "--goal-manifest", type=Path,
        default=REPO / "configs/webshop_synthetic_unique_v14_frozen.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--maximum-output-tokens", type=int, default=3200)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--whole-group-provider-retries", type=int, default=1)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--maximum-steps", type=int, default=12)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = _read(args.goal_manifest)
    if manifest.get("status") != "FROZEN_BEFORE_ANY_PROVIDER_CALL_OR_OUTCOME":
        raise SystemExit("V14 manifest is not sealed at the required boundary")
    development_rows = _development_rows(manifest)
    invalid = sorted(set(args.task_ids) - set(development_rows))
    if invalid:
        raise SystemExit(f"formal or unknown task IDs are forbidden: {invalid}")
    number_of_goals = int(manifest["number_of_registered_tasks_required"])

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
    os.environ["V14_OPENROUTER_API_KEY"] = str(api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_receipts: list[dict] = []
    cache_hashes: dict[str, str] = {}
    operational_retries: list[dict] = []
    for task_id in args.task_ids:
        final_paths = {
            condition: args.output_dir / f"{task_id}.{condition}.json"
            for condition in CONDITIONS
        }
        if args.resume and all(path.exists() for path in final_paths.values()):
            rows = [_read(final_paths[condition]) for condition in CONDITIONS]
            final_receipts.extend(rows)
            print(json.dumps({"task_id": task_id, "status": "resumed"}), flush=True)
            continue

        selected_receipts = None
        for attempt in range(args.whole_group_provider_retries + 1):
            attempt_dir = args.output_dir / "attempts" / task_id / f"attempt_{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            cache_path = attempt_dir / "decision_cache.json"
            backend = _make_backend(args, cache_path)
            receipts = []
            for condition in CONDITIONS:
                goal_options = dict(
                    development_rows[task_id].get("goal", {}).get("goal_options", {})
                )
                controller = CoverageTransferController(
                    condition, goal_options=goal_options,
                )
                v9_runner._decision_candidates = _candidate_augmenter(goal_options)
                v9_runner.choose_transfer_action = controller
                receipt = v9_runner._run_condition(
                    task_id=task_id,
                    condition=condition,
                    backend=backend,
                    grounder=grounder,
                    source_models=source_models,
                    source_policy={"uncertainty_scale": 0.0, "decision_margin": 0.0},
                    expected_goal=str(development_rows[task_id]["instruction_text"]),
                    wrapper_root=args.wrapper_root,
                    session_namespace=(
                        f"{args.run_id}.a{attempt}.{condition.replace('_', '-')}"
                    ),
                    number_of_goals=number_of_goals,
                    maximum_steps=args.maximum_steps,
                    candidate_count=args.candidate_count,
                    schema_retries=args.schema_retries,
                )
                receipt["role"] = args.role
                receipt["attempt"] = attempt
                receipt["coverage_controller"] = controller.as_dict()
                commit_audit = audit_receipt_commits(receipt)
                receipt["commit_audit"] = commit_audit
                receipt["unsafe_commits"] = [
                    row for row in commit_audit if not row["authorized"]
                ] if controller.coverage_enabled else []
                receipt["paired_label_candidate_steps"] = sum(
                    any(row.get("paired_constraint_bid") is not None
                        for row in step["candidate_semantics"])
                    for step in receipt["steps"]
                )
                receipt["paired_label_selected_count"] = sum(
                    step["candidate_semantics"][step["selected_index"]].get(
                        "paired_constraint_bid"
                    ) is not None
                    for step in receipt["steps"]
                )
                receipt["runtime_hashes"] = {
                    "runner": file_sha256(Path(__file__)),
                    "coverage_core": file_sha256(
                        REPO / "src/motif_transfer/webshop_coverage_transfer_v14.py"
                    ),
                    "coverage_ledger": file_sha256(
                        REPO / "src/motif_transfer/webshop_constraint_coverage_v14.py"
                    ),
                    "target_grounder": file_sha256(args.target_grounder),
                    "source_artifact": file_sha256(args.source_artifact),
                    "source_confirmation": file_sha256(args.source_confirmation),
                    "goal_manifest": file_sha256(args.goal_manifest),
                }
                receipt["receipt_sha256"] = stable_hash(receipt)
                (attempt_dir / f"{task_id}.{condition}.json").write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                receipts.append(receipt)
                print(json.dumps({
                    "task_id": task_id,
                    "attempt": attempt,
                    "condition": condition,
                    "reward": receipt["official_reward"],
                    "strict": receipt["strict_success"],
                    "steps": receipt["step_count"],
                    "coverage_interventions": controller.coverage_interventions,
                    "source_decisions": receipt["source_decision_count"],
                    "unsafe_commits": len(receipt["unsafe_commits"]),
                    "failure": receipt["failure"],
                }), flush=True)
            if cache_path.exists():
                cache_hashes[f"{task_id}.attempt_{attempt}"] = file_sha256(cache_path)
            if _retryable_provider_failure(receipts) and attempt < args.whole_group_provider_retries:
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

    condition_summaries = {
        condition: _condition_summary(final_receipts, condition)
        for condition in CONDITIONS
    }
    target = condition_summaries[TARGET_ONLY]
    coverage = condition_summaries[TARGET_COVERAGE]
    authentic = condition_summaries[AUTHENTIC_COVERAGE]
    source_controls = {
        condition: condition_summaries[condition]
        for condition in (
            COMMIT_AVAILABILITY_COVERAGE,
            INVERTED_COVERAGE,
            POSITION_PRIOR_COVERAGE,
        )
    }
    target_bridge_recovered = bool(
        coverage["strict_successes"] > target["strict_successes"]
        or coverage["mean_reward"] > target["mean_reward"] + 1e-9
    )
    source_authority_exercised = authentic["source_commit_decisions"] > 0
    authentic_increment = bool(
        authentic["strict_successes"] > coverage["strict_successes"]
        or authentic["mean_reward"] > coverage["mean_reward"] + 1e-9
    )
    authentic_control_superiority = all(
        (
            authentic["strict_successes"] > row["strict_successes"]
            or (
                authentic["strict_successes"] == row["strict_successes"]
                and authentic["mean_reward"] > row["mean_reward"] + 1e-9
            )
        )
        for row in source_controls.values()
    )
    if (
        target_bridge_recovered
        and source_authority_exercised
        and authentic_increment
        and authentic_control_superiority
    ):
        status = "DEVELOPMENT_SOURCE_TRANSFER_SIGNAL"
    elif target_bridge_recovered:
        status = "TARGET_BRIDGE_RECOVERED_SOURCE_INCREMENT_NOT_IDENTIFIED"
    else:
        status = "NO_TARGET_BRIDGE_RECOVERY"
    summary = {
        "schema_version": 1,
        "experiment": "webshop_coverage_transfer_v14_development",
        "status": status,
        "claim_limit": (
            "Development-only mechanism diagnostic; formal reserve remains unopened."
        ),
        "tasks": list(args.task_ids),
        "conditions": condition_summaries,
        "gates": {
            "all_receipts_complete": all(
                row["failure"] is None for row in final_receipts
            ),
            "matched_initial_state_hashes": all(
                len({row["initial_state_hash"] for row in final_receipts
                     if row["task_id"] == task_id}) == 1
                for task_id in args.task_ids
            ),
            "coverage_conditions_have_no_unsafe_commit": (
                coverage["unsafe_commits"] == 0
                and authentic["unsafe_commits"] == 0
            ),
            "target_bridge_recovered": target_bridge_recovered,
            "source_authority_exercised": source_authority_exercised,
            "authentic_increment_over_coverage_only": authentic_increment,
            "authentic_strictly_exceeds_each_source_control": (
                authentic_control_superiority
            ),
        },
        "operational_retries": operational_retries,
        "model": args.model,
        "number_of_goals": number_of_goals,
        "run_id": args.run_id,
        "runtime_hashes": {
            "runner": file_sha256(Path(__file__)),
            "coverage_core": file_sha256(
                REPO / "src/motif_transfer/webshop_coverage_transfer_v14.py"
            ),
            "coverage_ledger": file_sha256(
                REPO / "src/motif_transfer/webshop_constraint_coverage_v14.py"
            ),
            "goal_manifest": file_sha256(args.goal_manifest),
            "decision_caches": cache_hashes,
        },
        "formal_reserve_read_or_run": False,
    }
    summary["summary_sha256"] = stable_hash(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
