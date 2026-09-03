#!/usr/bin/env python3
"""Analyze V30--V33 typed grounding without promoting a weak-baseline gain."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


BENCHMARKS = ("clevrer", "star", "nextqa")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired(
    rows: Sequence[Mapping[str, Any]], left: str, right: str,
) -> dict[str, int | float]:
    wins = sum(bool(row[left]) and not bool(row[right]) for row in rows)
    losses = sum(bool(row[right]) and not bool(row[left]) for row in rows)
    return {
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "ties": len(rows) - wins - losses,
        "accuracy_delta": (wins - losses) / len(rows),
    }


def _candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[tuple[bool, str]] = Counter()
    for row in rows:
        for candidate in row["candidates"]:
            counts[(bool(row["candidate_gold"][candidate["slot"]]), candidate["receipt"]["claim_status"])] += 1
    covered = sum(value for (gold, status), value in counts.items() if status != "UNKNOWN")
    correct = counts[(True, "SUPPORTED")] + counts[(False, "REFUTED")]
    total = sum(counts.values())
    return {
        "candidates": total,
        "coverage": covered / total,
        "covered_accuracy": correct / covered if covered else 0.0,
        "full_accuracy_unknown_is_wrong": correct / total,
        "true_supported": counts[(True, "SUPPORTED")],
        "true_refuted": counts[(True, "REFUTED")],
        "true_unknown": counts[(True, "UNKNOWN")],
        "false_supported": counts[(False, "SUPPORTED")],
        "false_refuted": counts[(False, "REFUTED")],
        "false_unknown": counts[(False, "UNKNOWN")],
    }


def _condition(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    correct = sum(bool(row[key]) for row in rows)
    return {"correct": correct, "samples": len(rows), "accuracy": correct / len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v30", required=True, type=Path)
    parser.add_argument("--v31", required=True, type=Path)
    parser.add_argument("--v33", required=True, type=Path)
    parser.add_argument("--clevrer-v14", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    v30 = json.loads(args.v30.read_text(encoding="utf-8"))
    v31 = json.loads(args.v31.read_text(encoding="utf-8"))
    v33 = json.loads(args.v33.read_text(encoding="utf-8"))
    v14 = json.loads(args.clevrer_v14.read_text(encoding="utf-8"))
    for name, rows in (("V30", v30), ("V31", v31), ("V33", v33)):
        if len(rows) != 12:
            raise ValueError(f"{name} must contain 12 consumed-development rows")
    ids = {
        name: {(row["benchmark"], row["sample_id"]) for row in rows}
        for name, rows in (("V30", v30), ("V31", v31), ("V33", v33))
    }
    if len({frozenset(value) for value in ids.values()}) != 1:
        raise ValueError("V30/V31/V33 identities differ")
    if any(row.get("source_skill_or_structure_available_at_runtime") for row in v30 + v31 + v33):
        raise ValueError("V30--V33 must remain source-free grounding qualification")
    if v14.get("status") != "SOKOBAN_TO_CLEVRER_PROOF_NEUROSYMBOLIC_TRANSFER_FORMAL_VALIDATED":
        raise ValueError("referenced CLEVRER V14 formal route is not validated")

    benchmark_metrics = {}
    for benchmark in BENCHMARKS:
        t30 = [row for row in v30 if row["benchmark"] == benchmark]
        t31 = [row for row in v31 if row["benchmark"] == benchmark]
        t33 = [row for row in v33 if row["benchmark"] == benchmark]
        benchmark_metrics[benchmark] = {
            "candidate_classification_v30": _candidate_metrics(t30),
            "gemini_direct_v31": _condition(t31, "direct_correct"),
            "typed_on_gemini_v31": _condition(t31, "authentic_correct"),
            "gemini_binding_rotation_v31": _condition(t31, "binding_control_correct"),
            "typed_vs_gemini": _paired(t31, "authentic_correct", "direct_correct"),
            "claude_direct_v33": _condition(t33, "direct_correct"),
            "typed_on_claude_v33": _condition(t33, "authentic_correct"),
            "claude_binding_rotation_v33": _condition(t33, "binding_control_correct"),
            "typed_vs_claude": _paired(t33, "authentic_correct", "direct_correct"),
        }
    natural_v30 = [row for row in v30 if row["benchmark"] in {"star", "nextqa"}]
    natural_v31 = [row for row in v31 if row["benchmark"] in {"star", "nextqa"}]
    natural_v33 = [row for row in v33 if row["benchmark"] in {"star", "nextqa"}]
    candidate_natural = _candidate_metrics(natural_v30)
    gates = {
        "complete_matched_consumed_development": len(v30) == len(v31) == len(v33) == 12,
        "candidate_isolated_and_source_free": all(
            row.get("each_neural_call_saw_exactly_one_candidate") is True
            and row.get("source_skill_or_structure_available_at_runtime") is False
            for row in v30
        ),
        "natural_candidate_covered_accuracy_at_least_0p85": candidate_natural["covered_accuracy"] >= 0.85,
        "natural_candidate_coverage_at_least_0p90": candidate_natural["coverage"] >= 0.90,
        "typed_strictly_above_independent_claude_on_star_and_nextqa": all(
            benchmark_metrics[name]["typed_vs_claude"]["net_wins"] > 0
            for name in ("star", "nextqa")
        ),
        "binding_rotation_below_authentic_on_star_and_nextqa": all(
            benchmark_metrics[name]["typed_on_claude_v33"]["correct"]
            > benchmark_metrics[name]["claude_binding_rotation_v33"]["correct"]
            for name in ("star", "nextqa")
        ),
        "typed_strictly_above_strong_matched_gemini_natural": _paired(
            natural_v31, "authentic_correct", "direct_correct"
        )["net_wins"] > 0,
        "typed_non_degrading_on_each_strong_matched_benchmark": all(
            benchmark_metrics[name]["typed_vs_gemini"]["net_wins"] >= 0
            for name in BENCHMARKS
        ),
        "fully_video_disjoint_nextqa_confirmation_available_locally": False,
    }
    strong_target_pass = (
        gates["typed_strictly_above_strong_matched_gemini_natural"]
        and gates["typed_non_degrading_on_each_strong_matched_benchmark"]
    )
    report = {
        "schema_version": 33,
        "status": (
            "THREE_VIDEO_GROUNDING_QUALIFIED_FOR_FRESH_CONFIRMATION"
            if strong_target_pass
            else "CROSS_MODEL_CASCADE_SIGNAL_STRONG_TARGET_GATE_FAILED"
        ),
        "claim_boundary": (
            "V30--V33 are source-free grounding diagnostics on an already-consumed "
            "12-row development set. A gain over an independent weaker baseline is "
            "not promoted when the same typed executor fails to exceed the strongest "
            "matched target-only direct model. CLEVRER V14 is reported as a separate "
            "validated NS-DR/proof route and is not pooled with these 12 rows."
        ),
        "samples": 12,
        "candidate_calls_v30": sum(len(row["candidates"]) for row in v30),
        "candidate_classification_natural": candidate_natural,
        "benchmark_metrics": benchmark_metrics,
        "pooled": {
            "gemini_direct_v31": _condition(v31, "direct_correct"),
            "typed_on_gemini_v31": _condition(v31, "authentic_correct"),
            "typed_vs_gemini_v31": _paired(v31, "authentic_correct", "direct_correct"),
            "claude_direct_v33": _condition(v33, "direct_correct"),
            "typed_on_claude_v33": _condition(v33, "authentic_correct"),
            "typed_vs_claude_v33": _paired(v33, "authentic_correct", "direct_correct"),
        },
        "clevrer_separate_validated_route": {
            "status": v14["status"],
            "samples": v14["samples"],
            "target_explicit_no_recovery": v14["conditions"]["target_explicit_no_recovery"],
            "authentic_sokoban_proof_cate_recover": v14["conditions"]["authentic_sokoban_proof_cate_recover"],
            "gates": v14["gates"],
            "runtime_kind": "neural dynamics predictions + symbolic executor + learned proof-grounded recovery",
        },
        "gates": gates,
        "fresh_confirmation_authorized": strong_target_pass,
        "lineage": {
            "v30_sha256": _sha256(args.v30),
            "v31_sha256": _sha256(args.v31),
            "v33_sha256": _sha256(args.v33),
            "clevrer_v14_sha256": _sha256(args.clevrer_v14),
        },
        "next_action": (
            "Train benchmark-native STAR/NExT action-relation grounders on disjoint "
            "adaptation annotations, keep CLEVRER on the validated NS-DR route, and "
            "rerun the strong matched target gate before opening fresh outcomes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "pooled": report["pooled"],
        "gates": gates,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
