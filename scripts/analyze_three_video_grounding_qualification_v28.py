#!/usr/bin/env python3
"""Analyze the source-free three-video semantic-grounding qualification pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CONDITIONS = (
    "uniform_direct",
    "uniform_receipt",
    "localized_direct",
    "localized_receipt",
    "shifted_receipt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    wins = sum(bool(row["correct"][left]) and not bool(row["correct"][right]) for row in rows)
    losses = sum(not bool(row["correct"][left]) and bool(row["correct"][right]) for row in rows)
    ties = len(rows) - wins - losses
    return {
        "left": left,
        "right": right,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_correct": wins - losses,
        "percentage_point_difference": 100.0 * (wins - losses) / max(1, len(rows)),
    }


def _class_map(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        code, text = line.split(" ", 1)
        output[code] = text.strip()
    return output


def _program_values(program: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(value).casefold()
        for step in program
        for value in (step.get("value_input") or ())
    }


def _star_intrinsic(
    row: Mapping[str, Any],
    annotation: Mapping[str, Any],
    *,
    star_root: Path,
) -> dict[str, Any]:
    class_root = star_root / "annotations/STAR_classes"
    verbs = _class_map(class_root / "verb_classes.txt")
    objects = _class_map(class_root / "object_classes.txt")
    action_map = _class_map(class_root / "action_mapping.txt")
    choices = sorted(annotation["choices"], key=lambda value: int(value["choice_id"]))
    gold_index = next(
        index for index, choice in enumerate(choices)
        if str(choice["choice"]) == str(annotation["answer"])
    )
    gold_values = _program_values(choices[gold_index].get("choice_program") or ())
    question_values = _program_values(annotation.get("question_program") or ())
    target_values = gold_values | question_values
    verb_ids = {code for code, text in verbs.items() if text.casefold() in target_values}
    object_ids = {
        code for code, text in objects.items()
        if text.casefold() in target_values
    }
    action_ids = set()
    for action_id, specification in action_map.items():
        parts = specification.split()
        if len(parts) != 2:
            continue
        verb_id, object_id = parts
        if verb_ids and verb_id not in verb_ids:
            continue
        if object_ids and object_id not in object_ids:
            continue
        if verb_ids or object_ids:
            action_ids.add(action_id)
    gold_frames = [
        int(frame_id)
        for frame_id, situation in annotation.get("situations", {}).items()
        if set(map(str, situation.get("actions") or ())) & action_ids
    ]
    metadata = row["video_metadata"]
    fps = float(metadata["fps"])
    clip_start = float(metadata["clip_start_seconds"])
    duration = float(metadata["duration_seconds"])
    gold_seconds = sorted({frame / fps - clip_start for frame in gold_frames})
    gold_seconds = [value for value in gold_seconds if -0.1 <= value <= duration + 0.1]
    receipt = row["event_grounding_receipt"]
    proxy_seconds = list(map(float, metadata["proxy_sample_seconds"]))
    evidence_seconds = [proxy_seconds[int(index)] for index in receipt["evidence_frames"]]
    tolerance = max(0.5, duration / max(1, len(proxy_seconds) - 1))
    hit = bool(gold_seconds and evidence_seconds and any(
        abs(evidence - gold) <= tolerance
        for evidence in evidence_seconds for gold in gold_seconds
    ))
    interval_iou = None
    if gold_seconds and receipt["start_frame"] is not None:
        predicted_start = proxy_seconds[int(receipt["start_frame"])]
        predicted_end = proxy_seconds[int(receipt["end_frame"])]
        gold_start = max(0.0, min(gold_seconds) - tolerance)
        gold_end = min(duration, max(gold_seconds) + tolerance)
        intersection = max(0.0, min(predicted_end, gold_end) - max(predicted_start, gold_start))
        union = max(predicted_end, gold_end) - min(predicted_start, gold_start)
        interval_iou = intersection / union if union > 0 else float(hit)
    predicate = str(receipt["predicate"]).casefold()
    predicate_tokens = set(re.findall(r"[a-z]+", predicate))
    gold_verbs = {verbs[code].casefold() for code in verb_ids}
    verb_hit = any(
        verb in predicate_tokens
        or any(token.startswith(verb) or verb.startswith(token) for token in predicate_tokens)
        for verb in gold_verbs
    )
    return {
        "sample_id": str(row["sample_id"]),
        "gold_choice_values": sorted(gold_values),
        "gold_verb_ids": sorted(verb_ids),
        "gold_object_ids": sorted(object_ids),
        "gold_action_ids": sorted(action_ids),
        "gold_action_seconds": gold_seconds,
        "evidence_seconds": evidence_seconds,
        "tolerance_seconds": tolerance,
        "gold_action_timestamp_available": bool(gold_seconds),
        "gold_action_timestamp_hit": hit,
        "interval_iou": interval_iou,
        "predicate_gold_verb_hit": verb_hit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = json.loads(args.receipts.read_text(encoding="utf-8"))
    if len(rows) != 12:
        raise ValueError("V28 pilot requires all 12 frozen rows")
    if any(row.get("source_skill_or_structure_available_at_runtime") for row in rows):
        raise ValueError("V28 receipt unexpectedly contains source transfer")
    by_benchmark = {
        benchmark: [row for row in rows if row["benchmark"] == benchmark]
        for benchmark in ("clevrer", "star", "nextqa")
    }
    condition_metrics = {
        condition: {
            "correct": sum(bool(row["correct"][condition]) for row in rows),
            "accuracy": sum(bool(row["correct"][condition]) for row in rows) / len(rows),
        }
        for condition in CONDITIONS
    }
    benchmark_metrics = {}
    for benchmark, subset in by_benchmark.items():
        benchmark_metrics[benchmark] = {
            "samples": len(subset),
            "conditions": {
                condition: {
                    "correct": sum(bool(row["correct"][condition]) for row in subset),
                    "accuracy": sum(bool(row["correct"][condition]) for row in subset) / len(subset),
                }
                for condition in CONDITIONS
            },
            "combined_vs_uniform": _paired(subset, "localized_receipt", "uniform_direct"),
            "localized_vs_shifted": _paired(subset, "localized_receipt", "shifted_receipt"),
            "localization_only": _paired(subset, "localized_direct", "uniform_direct"),
            "receipt_on_localized": _paired(subset, "localized_receipt", "localized_direct"),
        }

    star_root = Path(config["benchmarks"]["star"]["root"])
    star_annotations = {
        str(value["question_id"]): value
        for value in json.loads(
            (star_root / "annotations/STAR_val.json").read_text(encoding="utf-8")
        )
    }
    star_intrinsic = [
        _star_intrinsic(row, star_annotations[str(row["sample_id"])], star_root=star_root)
        for row in by_benchmark["star"]
    ]
    star_available = [
        value for value in star_intrinsic if value["gold_action_timestamp_available"]
    ]
    clevrer_observability = []
    for row in by_benchmark["clevrer"]:
        family = str(row["family"]).casefold()
        observed = str(row["event_grounding_receipt"]["observability"])
        requires_unseen = family in {"predictive", "counterfactual"}
        clevrer_observability.append({
            "sample_id": str(row["sample_id"]),
            "family": family,
            "observability": observed,
            "requires_unseen_dynamics": requires_unseen,
            "calibrated": not requires_unseen or observed != "OBSERVED",
        })

    combined = _paired(rows, "localized_receipt", "uniform_direct")
    evidence_necessity = _paired(rows, "localized_receipt", "shifted_receipt")
    localization_only = _paired(rows, "localized_direct", "uniform_direct")
    receipt_localized = _paired(rows, "localized_receipt", "localized_direct")
    receipt_uniform = _paired(rows, "uniform_receipt", "uniform_direct")
    costs = {
        "reported_usd": sum(
            float(value.get("cost", 0.0) or 0.0)
            for row in rows for value in row["usage"].values()
        ),
        "calls": sum(len(row["usage"]) for row in rows),
    }
    gates = {
        "complete_transport": len(rows) == 12,
        "source_free": all(
            row["source_skill_or_structure_available_at_runtime"] is False for row in rows
        ),
        "matched_answer_frame_budget": all(
            all(len(row["frame_indices"][condition]) == row["frame_budget"] for condition in CONDITIONS)
            for row in rows
        ),
        "destructive_view_nonidentity": all(row["localized_and_shifted_differ"] for row in rows),
        "star_intrinsic_timestamp_coverage": len(star_available) == len(by_benchmark["star"]),
        "star_intrinsic_hit_majority": bool(star_available) and sum(
            value["gold_action_timestamp_hit"] for value in star_available
        ) > len(star_available) / 2,
        "clevrer_unseen_observability_calibrated": all(
            value["calibrated"] for value in clevrer_observability
        ),
        "combined_strictly_above_uniform": combined["net_correct"] > 0,
        "localized_strictly_above_shifted": evidence_necessity["net_correct"] > 0,
        "at_least_two_benchmarks_positive_combined": sum(
            benchmark_metrics[name]["combined_vs_uniform"]["net_correct"] > 0
            for name in benchmark_metrics
        ) >= 2,
    }
    qualification_keys = (
        "complete_transport",
        "source_free",
        "matched_answer_frame_budget",
        "destructive_view_nonidentity",
        "star_intrinsic_timestamp_coverage",
        "star_intrinsic_hit_majority",
        "clevrer_unseen_observability_calibrated",
        "combined_strictly_above_uniform",
        "localized_strictly_above_shifted",
        "at_least_two_benchmarks_positive_combined",
    )
    qualified = all(gates[key] for key in qualification_keys)
    report = {
        "schema_version": 28,
        "status": (
            "SEMANTIC_GROUNDING_PILOT_QUALIFIED_FOR_SCALE"
            if qualified else "SEMANTIC_GROUNDING_PILOT_NOT_YET_QUALIFIED"
        ),
        "claim_boundary": config["claim_boundary"],
        "samples": len(rows),
        "benchmarks": benchmark_metrics,
        "conditions": condition_metrics,
        "factorial_effects": {
            "combined_vs_uniform": combined,
            "evidence_necessity_localized_vs_shifted": evidence_necessity,
            "localization_only": localization_only,
            "receipt_on_localized": receipt_localized,
            "receipt_on_uniform": receipt_uniform,
        },
        "receipt_observability_counts": dict(Counter(
            str(row["event_grounding_receipt"]["observability"]) for row in rows
        )),
        "localized_view_differs_from_uniform": sum(
            bool(row["uniform_and_localized_differ"]) for row in rows
        ),
        "star_intrinsic": {
            "rows": star_intrinsic,
            "timestamp_available": len(star_available),
            "timestamp_hits": sum(value["gold_action_timestamp_hit"] for value in star_available),
            "predicate_verb_hits": sum(value["predicate_gold_verb_hit"] for value in star_intrinsic),
            "mean_interval_iou": (
                sum(value["interval_iou"] for value in star_intrinsic if value["interval_iou"] is not None)
                / max(1, sum(value["interval_iou"] is not None for value in star_intrinsic))
            ),
        },
        "clevrer_observability": clevrer_observability,
        "cost": costs,
        "gates": gates,
        "qualified_for_transfer_scale": qualified,
        "lineage": {
            "config_sha256": _sha256(args.config),
            "receipts_sha256": _sha256(args.receipts),
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "conditions": condition_metrics,
        "factorial_effects": report["factorial_effects"],
        "star_intrinsic": report["star_intrinsic"],
        "gates": gates,
        "cost": costs,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
