#!/usr/bin/env python3
"""Build the paper table only from a fully passed six-benchmark 9B run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO
    / "runs/harness_controller_qwen35_9b_mixed_v3/source_only_sft_seed20260901"
)


def _read(path: Path) -> dict[str, Any]:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_pass(value: dict[str, Any], status: str, label: str) -> None:
    if value.get("status") != status or not all((value.get("gates") or {}).values()):
        raise ValueError(f"{label} is not authoritative and gate-clean")


def _write_if_absent_or_identical(path: Path, content: str) -> None:
    """Keep repeated downstream paper jobs idempotent without hiding drift."""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite divergent artifact: {path}")
        return
    path.write_text(content, encoding="utf-8")


def formal_success_rows() -> dict[str, dict[str, Any]]:
    webshop = _read(REPO / "runs/webshop_structural_transfer_v21_formal/report.json")
    discoveryworld = _read(
        REPO / "runs/discoveryworld_structural_transfer_v1_matched/report.json"
    )
    tirbench = _read(REPO / "runs/tir_maze_structural_transfer_v3/heldout_report.json")
    alfworld = _read(
        REPO / "runs/alfworld_unified_goal_acquisition_v13_formal/report.json.gz"
    )
    clevrer = _read(
        REPO / "runs/clevrer_unified_goal_relation_v15_reserve/formal_report.json"
    )
    agqa2 = _read(REPO / "runs/agqa2_full_distribution_v62/report.json")

    clevrer_neural = "neural_only_explicit_relation"
    clevrer_source = "authentic_source_induced_goal_relation"
    rows = {
        "webshop": {
            "label": "WebShop", "semantic_domain": "web interaction",
            "tasks": webshop["summaries"]["neural_only"]["tasks"],
            "neural_correct": webshop["summaries"]["neural_only"]["strict_successes"],
            "source_correct": webshop["summaries"]["source_induced_structural_ir"]["strict_successes"],
            "formal_status": webshop["status"],
        },
        "alfworld": {
            "label": "ALFWorld", "semantic_domain": "embodied text interaction",
            "tasks": alfworld["summaries"]["raw_target_only"]["tasks"],
            "neural_correct": alfworld["summaries"]["raw_target_only"]["successes"],
            "source_correct": alfworld["summaries"]["authentic_source_goal_relation_macro"]["successes"],
            "formal_status": alfworld["status"],
        },
        "discoveryworld": {
            "label": "DiscoveryWorld", "semantic_domain": "scientific discovery",
            "tasks": discoveryworld["applicable_tasks"],
            "neural_correct": discoveryworld["condition_successes"]["neural_only"],
            "source_correct": discoveryworld["condition_successes"]["source_induced"],
            "formal_status": discoveryworld["status"],
        },
        "tirbench": {
            "label": "TIRBench", "semantic_domain": "visual reasoning",
            "tasks": tirbench["summaries"]["neural_only"]["tasks"],
            "neural_correct": tirbench["summaries"]["neural_only"]["successes"],
            "source_correct": tirbench["summaries"]["source_induced"]["successes"],
            "formal_status": tirbench["status"],
        },
        "clevrer": {
            "label": "CLEVRER", "semantic_domain": "video understanding",
            "tasks": len(clevrer["rows"]),
            "neural_correct": sum(
                bool(row["conditions"][clevrer_neural]["correct"])
                for row in clevrer["rows"]
            ),
            "source_correct": sum(
                bool(row["conditions"][clevrer_source]["correct"])
                for row in clevrer["rows"]
            ),
            "formal_status": clevrer["status"],
        },
        "agqa2": {
            "label": "AGQA2", "semantic_domain": "video understanding",
            "tasks": agqa2["sample_count"],
            "neural_correct": agqa2["source_vs_direct_rows"]["direct_correct"],
            "source_correct": agqa2["source_vs_direct_rows"]["source_correct"],
            "formal_status": agqa2["status"],
        },
    }
    for row in rows.values():
        row["delta_correct"] = row["source_correct"] - row["neural_correct"]
        row["neural_rate"] = row["neural_correct"] / row["tasks"]
        row["source_rate"] = row["source_correct"] / row["tasks"]
        row["delta_percentage_points"] = 100 * (
            row["source_rate"] - row["neural_rate"]
        )
    return rows


def build(
    *, protocol_path: Path, receipt_path: Path, qualification_path: Path,
    route_path: Path, action_path: Path, markdown_path: Path, json_path: Path,
) -> dict[str, Any]:
    protocol = _read(protocol_path)
    receipt = _read(receipt_path)
    qualification = _read(qualification_path)
    route = _read(route_path)
    action = _read(action_path)
    _require_pass(
        qualification, "SOURCE_MIXED_HARNESS_GATE_PASSED", "source qualification",
    )
    _require_pass(
        route, "SIX_BENCHMARK_MODEL_SUBSTITUTION_ROUTE_GATE_PASSED", "route report",
    )
    _require_pass(
        action,
        "SIX_BENCHMARK_9B_SUBSTITUTION_ACTION_EQUIVALENCE_VALIDATED",
        "action-equivalence audit",
    )
    if protocol.get("status") != "FROZEN_BEFORE_MIXED_SOURCE_WEIGHT_UPDATES":
        raise ValueError("training protocol is not frozen")
    if (qualification.get("protocol") or {}).get("sha256") != _sha(protocol_path):
        raise ValueError("qualification is not bound to this protocol")
    if (action.get("route_report") or {}).get("sha256") != _sha(route_path):
        raise ValueError("action audit is not bound to this route report")
    forbidden = (
        "target_data_used", "target_outcome_used_for_controller_labels",
        "formal_or_qualification_targets_used", "video_target_data_used",
        "target_grounder_training_used_target_outcomes",
    )
    if any(receipt.get(field) is not False for field in forbidden):
        raise ValueError("9B adapter training was not source-only")
    summary = action["summary"]
    if not (
        summary["formal_tasks"] == 1346
        and summary["route_decisions"] == 2246
        and summary["action_equivalence"] == 1.0
        and summary["divergence_episode_count"] == 0
        and route["regimes"]["CONTROLLER_LORA"]["overall"]["exact_json_accuracy"] == 1.0
    ):
        raise ValueError("six-benchmark result is incomplete")

    fresh_presentation = (
        protocol.get("six_benchmark_substitution", {}).get(
            "preregistration_status"
        ) == "FROZEN_FRESH_PRESENTATION_BEFORE_SOURCE_ONLY_PERMUTATION_UPDATE"
    )

    formal = formal_success_rows()
    benchmark_order = (
        "webshop", "alfworld", "discoveryworld", "tirbench", "clevrer", "agqa2",
    )
    route_counts = route["selection"]["balance_group_counts"]
    table = []
    for benchmark in benchmark_order:
        native = action["by_benchmark"][benchmark]
        row = dict(formal[benchmark])
        row.update({
            "benchmark": benchmark,
            "route_decisions": int(route_counts[benchmark]),
            "success_critical_decisions": int(native["success_critical_decisions"]),
            "equivalent_success_critical_decisions": int(
                native["equivalent_success_critical_decisions"]
            ),
            "action_equivalence": float(native["action_equivalence"]),
        })
        if row["tasks"] != native["tasks"] or row["action_equivalence"] != 1.0:
            raise ValueError(f"formal/native mismatch for {benchmark}")
        table.append(row)

    total_tasks = sum(row["tasks"] for row in table)
    total_neural = sum(row["neural_correct"] for row in table)
    total_source = sum(row["source_correct"] for row in table)
    paper = {
        "schema_version": (
            "harness-controller-qwen35-9b-six-benchmark-paper-v2-fresh-presentation"
            if fresh_presentation
            else "harness-controller-qwen35-9b-six-benchmark-paper-v1"
        ),
        "status": "PAPER_TABLE_READY_SIX_BENCHMARK_9B_SUBSTITUTION_VALIDATED",
        "scope": {
            "target_benchmarks": 6,
            "semantic_domains": 5,
            "video_benchmarks_share_one_domain": ["CLEVRER", "AGQA2"],
            "formal_tasks": total_tasks,
            "route_decisions": summary["route_decisions"],
            "success_critical_decisions": summary["success_critical_decisions"],
        },
        "training": {
            "model": receipt["model"],
            "source_only": True,
            "train_examples": receipt["train_examples"],
            "validation_examples": receipt["validation_examples"],
            "max_steps": receipt["max_steps"],
            "learning_rate": receipt["learning_rate"],
            "seed": receipt["seed"],
            "train_runtime_seconds": receipt["train_metrics"].get("train_runtime"),
            "initial_adapter_role": protocol["initial_adapter"]["role"],
            "permutation_closure_update": fresh_presentation,
        },
        "source_gates": {
            "scalar_executor": qualification["scalar_executor"]["overall"],
            "multi_ir_selector": qualification["multi_ir_selector"]["overall"],
        },
        "model_substitution": {
            "base_exact_route_accuracy": route["regimes"]["BASE"]["overall"]["exact_json_accuracy"],
            "lora_exact_route_accuracy": route["regimes"]["CONTROLLER_LORA"]["overall"]["exact_json_accuracy"],
            "native_action_equivalence": summary["action_equivalence"],
            "divergence_episodes": summary["divergence_episode_count"],
            "fresh_anonymous_presentation_reserve": fresh_presentation,
            "new_target_task_sample": False,
        },
        "benchmarks": table,
        "descriptive_micro_average": {
            "tasks": total_tasks,
            "neural_correct": total_neural,
            "source_correct": total_source,
            "delta_correct": total_source - total_neural,
            "neural_rate": total_neural / total_tasks,
            "source_rate": total_source / total_tasks,
            "delta_percentage_points": 100 * (
                (total_source - total_neural) / total_tasks
            ),
            "primary_inference": False,
            "reason": "heterogeneous benchmarks; report per-benchmark effects as primary",
        },
        "artifacts": {
            "protocol": {"path": str(protocol_path.resolve()), "sha256": _sha(protocol_path)},
            "training_receipt": {"path": str(receipt_path.resolve()), "sha256": _sha(receipt_path)},
            "source_qualification": {"path": str(qualification_path.resolve()), "sha256": _sha(qualification_path)},
            "route_report": {"path": str(route_path.resolve()), "sha256": _sha(route_path)},
            "action_equivalence": {"path": str(action_path.resolve()), "sha256": _sha(action_path)},
        },
        "claim_boundary": (
            (
                "After a consumed diagnostic, a fresh anonymous catalog-presentation "
                "reserve was frozen before a symmetric source-only permutation-closure "
                "update. The Qwen3.5-9B LoRA exactly replaces every frozen symbolic "
                "route decision and preserves success-critical native actions on the "
                "content-addressed traces of six previously validated benchmarks. "
            )
            if fresh_presentation else
            (
                "A source-only Qwen3.5-9B LoRA exactly replaces every frozen symbolic "
                "route decision and preserves success-critical native actions on the "
                "content-addressed traces of six previously validated benchmarks. "
            )
        ) + (
            "The official success outcomes are inherited, not fresh live 9B reruns; "
            "the fresh presentation is not a new target-task sample; target-native "
            "grounding/execution remains domain-specific, and CLEVRER plus AGQA2 "
            "constitute one semantic video domain."
        ),
    }

    lines = [
        (
            "# Source-only Qwen3.5-9B Harness: Fresh Six-Benchmark Model Substitution"
            if fresh_presentation else
            "# Source-only Qwen3.5-9B Harness: Six-Benchmark Model Substitution"
        ),
        "",
        "## Paper-ready result",
        "",
        (
            "A source-only Qwen3.5-9B LoRA made **2,246/2,246 exact symbolic route "
            "decisions** over **1,346 tasks** from six target benchmarks. Under locked "
            "replay, all content-addressed target-native receipts were available and "
            f"success-critical action equivalence was **{summary['action_equivalence']:.1%}** "
            f"over **{summary['success_critical_decisions']:,} decisions**, with zero "
            "divergence episodes."
        ),
        (
            "The evaluated catalog presentations and opaque aliases were frozen after "
            "the earlier diagnostic but before the V3 source-only weight update. They "
            "are disjoint from the consumed diagnostic presentations; the underlying "
            "pre-outcome target task identities and native traces are intentionally the "
            "same, so this is not presented as a new target-task sample."
            if fresh_presentation else ""
        ),
        "",
        "The six benchmarks cover five semantic domains; CLEVRER and AGQA2 are two "
        "benchmarks in the same video-understanding domain.",
        "",
        "| Benchmark | Semantic domain | Tasks | 9B routes | Neural-only | Source-induced | Δ correct | Δ pp | Action eq. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['label']} | {row['semantic_domain']} | {row['tasks']} | "
            f"{row['route_decisions']} | {row['neural_correct']}/{row['tasks']} "
            f"({row['neural_rate']:.1%}) | {row['source_correct']}/{row['tasks']} "
            f"({row['source_rate']:.1%}) | +{row['delta_correct']} | "
            f"{row['delta_percentage_points']:+.1f} | {row['action_equivalence']:.1%} |"
        )
    micro = paper["descriptive_micro_average"]
    lines.extend([
        "",
        (
            f"Descriptive micro-average: {total_neural}/{total_tasks} ({micro['neural_rate']:.1%}) "
            f"to {total_source}/{total_tasks} ({micro['source_rate']:.1%}), "
            f"+{micro['delta_correct']} correct ({micro['delta_percentage_points']:+.1f} pp). "
            "This pooled number is not the primary inferential statistic because the "
            "benchmarks are heterogeneous."
        ),
        "",
        "## What was trained",
        "",
        (
            f"Qwen3.5-9B was continued for {receipt['max_steps']} steps on "
            f"{receipt['train_examples']:,} source-only examples (seed {receipt['seed']}). "
            "No target prompt, completion, action, success label, formal outcome, or video "
            "example was used for a weight update. The target Decision Agents, neural "
            "grounders, utility/verifier modules, composers, and native executors remained frozen."
        ),
        (
            "The V3 update used balanced, source-only anonymous catalog permutation "
            "and alias closure for all seven source-induced programs; it did not train "
            "on diagnostic errors or target examples."
            if fresh_presentation else ""
        ),
        "",
        "## Claim boundary",
        "",
        paper["claim_boundary"],
        "",
        "This result supports locked model substitution on existing formal traces. It "
        "does not relabel those outcomes as a fresh end-to-end live 9B evaluation, does "
        "not claim that target-native grounding is domain-agnostic, and does not place "
        "raw video perception inside the 9B text/IR controller.",
        "",
        "## Reproducibility artifacts",
        "",
    ])
    for name, spec in paper["artifacts"].items():
        lines.append(f"- `{name}`: `{spec['path']}` (`{spec['sha256']}`)")
    lines.append("")

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _write_if_absent_or_identical(markdown_path, "\n".join(lines))
    _write_if_absent_or_identical(
        json_path, json.dumps(paper, indent=2, sort_keys=True) + "\n",
    )
    return paper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path,
        default=REPO / "runs/harness_controller_qwen35_9b_mixed_v3_protocol/protocol.json",
    )
    parser.add_argument("--training-receipt", type=Path, default=DEFAULT_RUN / "training_receipt.json")
    parser.add_argument("--source-qualification", type=Path, default=DEFAULT_RUN / "source_mixed_qualification.json")
    parser.add_argument("--route-report", type=Path, default=DEFAULT_RUN / "six_benchmark_route_report.json")
    parser.add_argument("--action-audit", type=Path, default=DEFAULT_RUN / "six_benchmark_action_equivalence.json")
    parser.add_argument(
        "--markdown", type=Path,
        default=REPO / "docs/HARNESS_CONTROLLER_QWEN35_9B_SIX_BENCHMARK_V3_RESULTS.md",
    )
    parser.add_argument(
        "--json", type=Path,
        default=REPO / "docs/results/harness_controller_qwen35_9b_six_benchmark_v3.json",
    )
    args = parser.parse_args()
    result = build(
        protocol_path=args.protocol, receipt_path=args.training_receipt,
        qualification_path=args.source_qualification, route_path=args.route_report,
        action_path=args.action_audit, markdown_path=args.markdown, json_path=args.json,
    )
    print(json.dumps({
        "status": result["status"], "scope": result["scope"],
        "model_substitution": result["model_substitution"],
        "markdown": str(args.markdown), "json": str(args.json),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
