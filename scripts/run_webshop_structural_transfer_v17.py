#!/usr/bin/env python3
"""Run matched source-induced structural-IR transfer on WebShop V17."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import re
import shutil
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.active_video_transfer import exact_binomial_two_sided  # noqa: E402
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
from motif_transfer.webshop_neural_symbolic_v9 import TargetOutcomeMLP  # noqa: E402
from motif_transfer.webshop_structural_transfer_v17 import (  # noqa: E402
    CONDITIONS,
    GENERIC_SCAFFOLD,
    NEURAL_ONLY,
    SOURCE_INDUCED,
    SOURCE_PERMUTED,
    TARGET_NATIVE_CEILING,
    WebShopStructuralController,
)
import scripts.run_webshop_neural_symbolic_v9 as v9_runner  # noqa: E402


ORIGINAL_DECISION_CANDIDATES = v9_runner._decision_candidates


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _candidate_augmenter(goal_options: Mapping[str, Any]):
    def decide(**kwargs: Any):
        candidates, raw, attempts = ORIGINAL_DECISION_CANDIDATES(**kwargs)
        output = augment_with_constraint_labels(
            candidates, axtree=kwargs["axtree"],
            goal=str(kwargs["payload"]["goal"]), goal_options=goal_options,
        )
        output = augment_with_product_backtrack(
            output, url=str(kwargs["payload"].get("url") or ""),
        )
        return output, raw, attempts
    return decide


def _summary(receipts: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    rows = [row for row in receipts if row["condition"] == condition]
    return {
        "tasks": len(rows),
        "strict_successes": sum(bool(row["strict_success"]) for row in rows),
        "pass_successes": sum(bool(row["pass_success"]) for row in rows),
        "mean_reward": float(np.mean([row["official_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["step_count"] for row in rows])),
        "source_authorized_decisions": sum(
            row["structural_controller"]["source_authorized_decisions"]
            for row in rows
        ),
        "source_admitted_episodes": sum(
            bool(row["structural_controller"]["source_admitted"])
            for row in rows
        ),
        "unsafe_commits": sum(len(row["unsafe_commits"]) for row in rows),
        "failures": sum(row["failure"] is not None for row in rows),
    }


def _paired(
    receipts: list[dict[str, Any]], tasks: list[str], comparator: str,
) -> dict[str, Any]:
    index = {
        condition: {
            row["task_id"]: row for row in receipts
            if row["condition"] == condition
        }
        for condition in CONDITIONS
    }
    wins = losses = reward_wins = reward_losses = 0
    for task in tasks:
        left = index[SOURCE_INDUCED][task]
        right = index[comparator][task]
        a, b = bool(left["strict_success"]), bool(right["strict_success"])
        wins += a and not b
        losses += b and not a
        ar, br = float(left["official_reward"]), float(right["official_reward"])
        reward_wins += ar > br + 1e-12
        reward_losses += br > ar + 1e-12
    return {
        "wins": wins, "losses": losses,
        "ties": len(tasks) - wins - losses,
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        "reward_wins": reward_wins, "reward_losses": reward_losses,
        "reward_ties": len(tasks) - reward_wins - reward_losses,
    }


def _trajectory_signature(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "prompt_sha256": row["prompt_sha256"],
            "before_hash": row["before_hash"],
            "selected_action": row["selected_action"],
            "after_hash": row["after_hash"],
            "reward": row["reward"],
            "terminated": row["terminated"],
        }
        for row in receipt.get("steps") or ()
    ]


def _require_formal_protocol(
    path: Path, *, task_ids: list[str], args: argparse.Namespace,
    reserve_version: str,
) -> dict[str, Any]:
    protocol = _read(path)
    qualification = _read(args.qualification_report)
    expected = {
        "status": f"FROZEN_BEFORE_{reserve_version}_FORMAL_EXECUTION",
        "tasks": task_ids,
        "conditions": list(CONDITIONS),
        "model": args.model,
        "maximum_steps": args.maximum_steps,
        "maximum_output_tokens": args.maximum_output_tokens,
        "candidate_count": args.candidate_count,
        "manifest_file_sha256": file_sha256(args.manifest),
        "source_artifact_file_sha256": file_sha256(args.source_artifact),
        "source_confirmation_file_sha256": file_sha256(args.source_confirmation),
        "target_function_file_sha256": file_sha256(args.target_function),
        "target_grounder_file_sha256": file_sha256(args.target_grounder),
        "qualification_report_file_sha256": file_sha256(args.qualification_report),
        "controller_file_sha256": file_sha256(
            REPO / "src/motif_transfer/webshop_structural_transfer_v17.py"
        ),
        "runner_file_sha256": file_sha256(Path(__file__)),
    }
    mismatches = {
        key: {"expected": value, "observed": protocol.get(key)}
        for key, value in expected.items() if protocol.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"V17 formal protocol mismatch: {mismatches}")
    if qualification.get("status") != (
        f"{reserve_version}_TRANSPORT_QUALIFICATION_PASSED"
    ):
        raise SystemExit(f"{reserve_version} transport qualification did not pass")
    if not all((qualification.get("gates") or {}).values()):
        raise SystemExit("V17 qualification gates are incomplete")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("transport_qualification", "formal_reserve"),
        default="transport_qualification",
    )
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--wrapper-root", type=Path, default=Path(
        "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent"
    ))
    parser.add_argument("--keys", type=Path, default=Path(
        "/fs/gamma-projects/vlm-robot/keys.py"
    ))
    parser.add_argument("--manifest", type=Path, default=REPO / (
        "configs/webshop_structural_v17_frozen.json"
    ))
    parser.add_argument("--source-artifact", type=Path, default=REPO / (
        "runs/sokoban_relational_structural_v2/artifact.json"
    ))
    parser.add_argument("--source-confirmation", type=Path, default=REPO / (
        "runs/sokoban_relational_structural_v2/fresh_confirmation_report.json"
    ))
    parser.add_argument("--target-function", type=Path, default=REPO / (
        "runs/webshop_structural_transfer_v17_development/target_function.json"
    ))
    parser.add_argument("--target-grounder", type=Path, default=REPO / (
        "runs/webshop_structural_transfer_v17_development/low_sample_grounder.json"
    ))
    parser.add_argument("--development-report", type=Path, default=REPO / (
        "runs/webshop_structural_transfer_v17_development/development_report.json"
    ))
    parser.add_argument("--qualification-report", type=Path, default=REPO / (
        "runs/webshop_structural_transfer_v17_qualification/report.json"
    ))
    parser.add_argument("--formal-protocol", type=Path)
    parser.add_argument("--cache-seed", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--maximum-output-tokens", type=int, default=1200)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--maximum-steps", type=int, default=12)
    parser.add_argument("--run-id", default="webshop-structural-v17")
    args = parser.parse_args()

    manifest = _read(args.manifest)
    schema_match = re.fullmatch(
        r"webshop-structural-v(\d+)-reserve-v1",
        str(manifest.get("schema_version")),
    )
    if schema_match is None:
        raise SystemExit("unsupported structural reserve schema")
    reserve_version = f"V{schema_match.group(1)}"
    if manifest.get("status") != (
        f"FROZEN_BEFORE_ANY_{reserve_version}_PROVIDER_CALL_OR_OUTCOME"
    ):
        raise SystemExit(f"{reserve_version} WebShop manifest is not frozen")
    development = _read(args.development_report)
    if development.get("status") != "PHASE4_WEBSHOP_DEVELOPMENT_GATE_PASSED":
        raise SystemExit("Phase-4 WebShop development gate did not pass")
    if not all((development.get("gates") or {}).values()):
        raise SystemExit("Phase-4 development gates are incomplete")
    role_rows = {row["task_id"]: row for row in manifest["roles"][args.role]}
    tasks = list(args.task_ids or role_rows)
    invalid = sorted(set(tasks) - set(role_rows))
    if invalid:
        raise SystemExit(f"tasks outside {reserve_version} {args.role}: {invalid}")
    if args.role == "formal_reserve":
        if args.formal_protocol is None:
            raise SystemExit("--formal-protocol is required for formal execution")
        _require_formal_protocol(
            args.formal_protocol, task_ids=tasks, args=args,
            reserve_version=reserve_version,
        )

    source = _read(args.source_artifact)
    confirmation = _read(args.source_confirmation)
    target = _read(args.target_function)
    grounder_artifact = _read(args.target_grounder)
    if grounder_artifact.get("status") != (
        "TARGET_NATIVE_LOW_SAMPLE_GROUNDER_QUALIFIED"
    ):
        raise SystemExit("low-sample target grounder is not qualified")
    if target.get("target_grounder_sha256") != grounder_artifact.get("artifact_sha256"):
        raise SystemExit("target function and low-sample grounder lineage mismatch")
    grounder = TargetOutcomeMLP.from_dict(grounder_artifact["grounder"])

    values = runpy.run_path(str(args.keys))
    key = values.get("OPENROUTER_API_KEY") or values.get("openrouter_api_key")
    if not key:
        raise SystemExit("OpenRouter API key is missing")
    os.environ["V17_WEBSHOP_OPENROUTER_KEY"] = str(key)
    output = args.output_dir or REPO / (
        "runs/webshop_structural_transfer_v17_qualification"
        if args.role == "transport_qualification" else
        "runs/webshop_structural_transfer_v17_formal"
    )
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "decision_cache.json"
    if args.cache_seed and args.cache_seed.is_file() and not cache.exists():
        shutil.copyfile(args.cache_seed, cache)
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url, {"decision": args.model},
            api_key_env="V17_WEBSHOP_OPENROUTER_KEY", json_mode=True,
            temperature=0, timeout_seconds=180,
            request_overrides={"max_tokens": args.maximum_output_tokens},
        ),
        cache_path=cache,
    )

    receipts = []
    for task_id in tasks:
        goal_row = role_rows[task_id]
        goal_options = dict((goal_row.get("goal") or {}).get("goal_options") or {})
        for condition in CONDITIONS:
            controller = WebShopStructuralController(
                condition=condition, source=source,
                source_confirmation=confirmation, target_function=target,
                goal_options=goal_options, maximum_steps=args.maximum_steps,
            )
            v9_runner._decision_candidates = _candidate_augmenter(goal_options)
            v9_runner.choose_transfer_action = controller
            receipt = v9_runner._run_condition(
                task_id=task_id, condition=condition, backend=backend,
                grounder=grounder, source_models={"artifact": {}},
                source_policy={"uncertainty_scale": 0.0, "decision_margin": 0.0},
                expected_goal=str(goal_row["instruction_text"]),
                wrapper_root=args.wrapper_root,
                session_namespace=(
                    f"{args.run_id}.{condition.replace('_', '-')}"
                ),
                number_of_goals=int(manifest["number_of_registered_tasks_required"]),
                maximum_steps=args.maximum_steps,
                candidate_count=args.candidate_count,
                schema_retries=args.schema_retries,
            )
            receipt["role"] = args.role
            receipt["structural_controller"] = controller.as_dict()
            receipt["commit_audit"] = audit_receipt_commits(receipt)
            receipt["all_coverage_unsafe_commits"] = [
                row for row in receipt["commit_audit"] if not row["authorized"]
            ]
            receipt["unsafe_commits"] = [
                row for row in receipt["commit_audit"]
                if not row["authorized"]
                and condition == SOURCE_INDUCED
                and not bool(receipt["steps"][int(row["step"])]["source_abstained"])
            ]
            receipt["runtime_hashes"] = {
                "runner": file_sha256(Path(__file__)),
                "controller": file_sha256(REPO / (
                    "src/motif_transfer/webshop_structural_transfer_v17.py"
                )),
                "manifest": file_sha256(args.manifest),
                "source_artifact": file_sha256(args.source_artifact),
                "source_confirmation": file_sha256(args.source_confirmation),
                "target_function": file_sha256(args.target_function),
                "target_grounder": file_sha256(args.target_grounder),
            }
            receipt["receipt_sha256"] = stable_hash(receipt)
            (output / f"{task_id}.{condition}.json").write_text(
                json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            receipts.append(receipt)
            print(json.dumps({
                "task_id": task_id, "condition": condition,
                "strict": receipt["strict_success"],
                "reward": receipt["official_reward"],
                "steps": receipt["step_count"],
                "source_admitted": controller.source_admitted,
                "unsafe_commits": len(receipt["unsafe_commits"]),
                "failure": receipt["failure"],
            }), flush=True)

    summaries = {name: _summary(receipts, name) for name in CONDITIONS}
    paired = {
        name: _paired(receipts, tasks, name)
        for name in CONDITIONS if name != SOURCE_INDUCED
    }
    authentic = summaries[SOURCE_INDUCED]
    indexed = {
        condition: {
            row["task_id"]: row for row in receipts
            if row["condition"] == condition
        }
        for condition in CONDITIONS
    }
    source_free_exact = all(
        _trajectory_signature(indexed[NEURAL_ONLY][task])
        == _trajectory_signature(indexed[condition][task])
        for task in tasks
        for condition in (SOURCE_PERMUTED, GENERIC_SCAFFOLD)
    )
    ceiling_exact = all(
        _trajectory_signature(indexed[SOURCE_INDUCED][task])
        == _trajectory_signature(indexed[TARGET_NATIVE_CEILING][task])
        for task in tasks
    )
    gates = {
        "all_receipts_complete": all(row["failure"] is None for row in receipts),
        "matched_initial_state_hashes": all(
            len({row["initial_state_hash"] for row in receipts
                 if row["task_id"] == task}) == 1 for task in tasks
        ),
        "source_free_control_trajectories_exactly_matched": source_free_exact,
        "authentic_and_target_ceiling_trajectories_exactly_matched": ceiling_exact,
        "authentic_source_admitted_every_episode": (
            authentic["source_admitted_episodes"] == len(tasks)
        ),
        "controls_never_receive_source_authority": all(
            summaries[name]["source_authorized_decisions"] == 0
            for name in (NEURAL_ONLY, SOURCE_PERMUTED, GENERIC_SCAFFOLD,
                         TARGET_NATIVE_CEILING)
        ),
        "zero_authentic_unsafe_commits": authentic["unsafe_commits"] == 0,
        "authentic_success_gain_over_neural": (
            authentic["strict_successes"] > summaries[NEURAL_ONLY]["strict_successes"]
        ),
        "zero_negative_transfer_vs_neural": paired[NEURAL_ONLY]["losses"] == 0,
        "pass_success_not_below_neural": (
            authentic["pass_successes"] >= summaries[NEURAL_ONLY]["pass_successes"]
        ),
        "mean_reward_not_below_neural": (
            authentic["mean_reward"] + 1e-12
            >= summaries[NEURAL_ONLY]["mean_reward"]
        ),
        "reward_pairing_not_net_negative": (
            paired[NEURAL_ONLY]["reward_wins"]
            >= paired[NEURAL_ONLY]["reward_losses"]
        ),
        "authentic_beats_structural_controls": all(
            authentic["strict_successes"] > summaries[name]["strict_successes"]
            for name in (SOURCE_PERMUTED, GENERIC_SCAFFOLD)
        ),
        "authentic_matches_source_free_target_ceiling": (
            authentic["strict_successes"]
            == summaries[TARGET_NATIVE_CEILING]["strict_successes"]
        ),
    }
    if args.role == "formal_reserve":
        gates.update({
            "source_vs_neural_exact_p_at_most_0p05": (
                paired[NEURAL_ONLY]["exact_two_sided_p"] <= 0.05
            ),
            "source_vs_permuted_exact_p_at_most_0p05": (
                paired[SOURCE_PERMUTED]["exact_two_sided_p"] <= 0.05
            ),
        })
    passed = all(gates.values())
    body = {
        "schema_version": "webshop-source-structural-transfer-v17-run-v1",
        "status": (
            f"{reserve_version}_TRANSPORT_QUALIFICATION_PASSED" if passed else
            f"{reserve_version}_TRANSPORT_QUALIFICATION_FAILED"
        ) if args.role == "transport_qualification" else (
            f"{reserve_version}_FRESH_FORMAL_STRUCTURAL_TRANSFER_VALIDATED"
            if passed else
            f"{reserve_version}_FRESH_FORMAL_STRUCTURAL_TRANSFER_FAILED"
        ),
        "claim_boundary": (
            f"Fresh product-disjoint {reserve_version} transport qualification."
            if args.role == "transport_qualification" else
            f"One-shot fresh product-disjoint {reserve_version} formal reserve."
        ),
        "role": args.role,
        "tasks": tasks,
        "model": args.model,
        "maximum_steps": args.maximum_steps,
        "maximum_output_tokens": args.maximum_output_tokens,
        "candidate_count": args.candidate_count,
        "schema_retries": args.schema_retries,
        "target_grounder_training_rows": grounder_artifact["training_rows"],
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "formal_protocol_file_sha256": (
            file_sha256(args.formal_protocol) if args.formal_protocol else None
        ),
        "runtime_hashes": {
            "runner": file_sha256(Path(__file__)),
            "controller": file_sha256(REPO / (
                "src/motif_transfer/webshop_structural_transfer_v17.py"
            )),
            "manifest": file_sha256(args.manifest),
            "decision_cache": file_sha256(cache),
        },
    }
    report = body | {"report_sha256": stable_hash(body)}
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
