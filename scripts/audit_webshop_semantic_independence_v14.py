#!/usr/bin/env python3
"""Audit WebShop replication independence before spending more model calls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import comb
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.webshop_constraint_coverage_v14 import (  # noqa: E402
    audit_receipt_commits,
)
from motif_transfer.webshop_semantic_reserve import (  # noqa: E402
    audit_semantic_reserve,
    canonical_instruction,
)


AUTHENTIC = "authentic_sokoban_effect_plus_target"
BASELINE = "target_only"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_sign_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def load_pairs(run_dir: Path) -> list[dict[str, Any]]:
    pairs = []
    suffix = f".{AUTHENTIC}.json"
    for authentic_path in sorted(run_dir.glob(f"webshop.*{suffix}")):
        task_id = authentic_path.name[: -len(suffix)]
        baseline_path = run_dir / f"{task_id}.{BASELINE}.json"
        if not baseline_path.exists():
            raise ValueError(f"missing paired baseline: {baseline_path}")
        authentic = json.loads(authentic_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if authentic["goal"] != baseline["goal"]:
            raise ValueError(f"unmatched goal for {task_id}")
        if authentic["initial_state_hash"] != baseline["initial_state_hash"]:
            raise ValueError(f"unmatched initial state for {task_id}")
        pairs.append({
            "task_id": task_id,
            "goal": authentic["goal"],
            "goal_key": canonical_instruction(authentic["goal"]),
            "initial_state_hash": authentic["initial_state_hash"],
            "authentic_strict": bool(authentic["strict_success"]),
            "baseline_strict": bool(baseline["strict_success"]),
            "authentic_reward": float(authentic["official_reward"]),
            "baseline_reward": float(baseline["official_reward"]),
            "authentic_path": str(authentic_path),
            "baseline_path": str(baseline_path),
        })
    if not pairs:
        raise ValueError(f"no paired WebShop receipts under {run_dir}")
    return pairs


def paired_metrics(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wins = sum(row["authentic_strict"] and not row["baseline_strict"] for row in pairs)
    losses = sum(row["baseline_strict"] and not row["authentic_strict"] for row in pairs)
    return {
        "tasks": len(pairs),
        "authentic_strict": sum(row["authentic_strict"] for row in pairs),
        "baseline_strict": sum(row["baseline_strict"] for row in pairs),
        "wins": wins,
        "losses": losses,
        "ties": len(pairs) - wins - losses,
        "net_wins": wins - losses,
        "exact_two_sided_p": exact_sign_p(wins, losses),
    }


def semantic_cluster_metrics(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        clusters[str(row["goal_key"])].append(row)
    details = []
    for goal_key, rows in sorted(clusters.items()):
        delta = sum(
            int(row["authentic_strict"]) - int(row["baseline_strict"])
            for row in rows
        )
        details.append({
            "goal": goal_key,
            "tasks": len(rows),
            "task_ids": [row["task_id"] for row in rows],
            "strict_success_delta": delta,
        })
    wins = sum(row["strict_success_delta"] > 0 for row in details)
    losses = sum(row["strict_success_delta"] < 0 for row in details)
    return {
        "semantic_clusters": len(details),
        "positive_clusters": wins,
        "negative_clusters": losses,
        "tied_clusters": len(details) - wins - losses,
        "positive_minus_negative": wins - losses,
        "cluster_sign_exact_two_sided_p": exact_sign_p(wins, losses),
        "clusters": details,
    }


def goal_pool_inventory(data_dir: Path) -> dict[str, Any]:
    products_path = data_dir / "items_shuffle_1000.json"
    human_path = data_dir / "items_human_ins.json"
    synthetic_path = data_dir / "items_ins_v2_1000.json"
    products = json.loads(products_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))

    unique_products = []
    seen_asins = set()
    for product in products:
        asin = str(product.get("asin", ""))
        if not asin or asin == "nan" or len(asin) > 10 or asin in seen_asins:
            continue
        seen_asins.add(asin)
        unique_products.append(product)

    human_asins = {product["asin"] for product in unique_products if product["asin"] in human}
    human_goals = sum(
        bool(instruction.get("instruction_attributes"))
        for product in unique_products
        for instruction in human.get(product["asin"], [])
    )
    combination_counts = []
    for product in unique_products:
        row = synthetic.get(product["asin"], {})
        if not row.get("instruction") or not row.get("instruction_attributes"):
            continue
        combinations = 1
        for values in (product.get("customization_options") or {}).values():
            if values is None:
                continue
            valid_values = [value for value in values if str(value.get("value", "")).strip()]
            combinations *= len(valid_values)
        combination_counts.append(combinations)
    return {
        "product_rows": len(unique_products),
        "human_goal_asins_in_product_set": len(human_asins),
        "human_goals_generated": human_goals,
        "synthetic_eligible_products": len(combination_counts),
        "synthetic_option_specific_goals": sum(combination_counts),
        "synthetic_goal_median_per_product": statistics.median(combination_counts),
        "synthetic_goal_maximum_per_product": max(combination_counts),
        "synthetic_products_over_100_option_combinations": sum(
            count > 100 for count in combination_counts
        ),
        "data_hashes": {
            "products": file_sha256(products_path),
            "human_instructions": file_sha256(human_path),
            "synthetic_instructions": file_sha256(synthetic_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-run",
        type=Path,
        default=REPO / "runs/webshop_sokoban_effect_transfer_v13/fresh_replication",
    )
    parser.add_argument(
        "--replication-run",
        type=Path,
        default=REPO / "runs/webshop_sokoban_effect_replication_v1",
    )
    parser.add_argument(
        "--webshop-data",
        type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/"
            "WebShop/data"
        ),
    )
    parser.add_argument(
        "--server-app",
        type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/"
            "WebShop/web_agent_site/app.py"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs/results/webshop_semantic_independence_v14_audit.json",
    )
    args = parser.parse_args()

    original = load_pairs(args.original_run)
    replication = load_pairs(args.replication_run)
    original_rows = [{"goal": row["goal"]} for row in original]
    replication_rows = [{"goal": row["goal"]} for row in replication]
    independence = audit_semantic_reserve(
        replication_rows,
        consumed_rows=original_rows,
        required_unique_goals=len(replication_rows),
    )
    original_counts = Counter(row["goal_key"] for row in original)
    replication_counts = Counter(row["goal_key"] for row in replication)

    regression_path = args.replication_run / f"webshop.171.{AUTHENTIC}.json"
    regression = json.loads(regression_path.read_text(encoding="utf-8"))
    regression_commits = audit_receipt_commits(regression)
    artifact = {
        "schema_version": 1,
        "status": "WEBSHOP_SEMANTICALLY_INDEPENDENT_REPLICATION_NOT_ESTABLISHED",
        "claim_correction": (
            "Task-level paired results remain historical observations, but IDs 114-177 "
            "wrap over the same 13 human goal semantics. They are pseudoreplicates and "
            "cannot establish an independent 32-goal WebShop confirmation."
        ),
        "original_v13": {
            "task_level": paired_metrics(original),
            "semantic_cluster_level": semantic_cluster_metrics(original),
            "unique_goal_semantics": len(original_counts),
            "goal_multiplicities": dict(sorted(original_counts.items())),
        },
        "replication_v1": {
            "task_level": paired_metrics(replication),
            "semantic_cluster_level": semantic_cluster_metrics(replication),
            "unique_goal_semantics": len(replication_counts),
            "goal_multiplicities": dict(sorted(replication_counts.items())),
        },
        "combined_descriptive_same_clusters": semantic_cluster_metrics(
            [*original, *replication]
        ),
        "semantic_independence_preflight": independence,
        "same_unique_semantic_goal_set": set(original_counts) == set(replication_counts),
        "overlapping_unique_goal_semantics": len(set(original_counts) & set(replication_counts)),
        "distinct_initial_state_hashes": {
            "original": len({row["initial_state_hash"] for row in original}),
            "replication": len({row["initial_state_hash"] for row in replication}),
            "note": (
                "Different state hashes do not imply different goals; canonical task-specific "
                "URLs/prompts differ while instruction semantics repeat."
            ),
        },
        "server_mechanism": {
            "bridge_rule": "goal_idx = numeric_session_suffix % len(goals)",
            "server_app": str(args.server_app),
            "server_app_sha256": file_sha256(args.server_app),
        },
        "available_goal_pool": goal_pool_inventory(args.webshop_data),
        "known_regression_webshop_171": {
            "goal": regression["goal"],
            "authentic_reward": regression["official_reward"],
            "authentic_strict_success": regression["strict_success"],
            "coverage_commit_audit": regression_commits,
            "diagnosis": (
                "The binary ready-state summary lost prerequisite multiplicity. Only the "
                "black constraint had a verified state-changing action before commit; size "
                "10.5 remained unverified. A target-native set-coverage gate would reject it."
            ),
        },
        "required_next_protocol": [
            "Switch the server to its synthetic goal generator; the installed human subset has only 13 goals.",
            "Enumerate and freeze goal objects before model calls; require one task per instruction hash.",
            "Require instruction and ASIN disjointness from every consumed WebShop split.",
            "Run target-only, target-native coverage-only, authentic source plus coverage, and destructive controls.",
            "Use semantic-goal clusters (not task IDs) as the primary inference unit.",
        ],
        "cost_decision": "DO_NOT_SPEND_PROVIDER_CALLS_UNTIL_SEMANTIC_RESERVE_PREFLIGHT_PASSES",
        "inputs": {
            "original_run": str(args.original_run),
            "replication_run": str(args.replication_run),
            "original_summary_sha256": file_sha256(args.original_run / "summary.json"),
            "replication_summary_sha256": file_sha256(args.replication_run / "summary.json"),
            "regression_receipt_sha256": file_sha256(regression_path),
        },
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": artifact["status"],
        "original_task_level": artifact["original_v13"]["task_level"],
        "replication_task_level": artifact["replication_v1"]["task_level"],
        "original_cluster_level": artifact["original_v13"]["semantic_cluster_level"],
        "replication_cluster_level": artifact["replication_v1"]["semantic_cluster_level"],
        "semantic_preflight_passed": independence["passed"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
