#!/usr/bin/env python3
"""Analyze the candidate-blind multi-event grounding repair."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import analyze_three_video_grounding_qualification_v28 as v28  # noqa: E402


CONDITIONS = v28.CONDITIONS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(label: str, text: str) -> bool:
    variants = [value.strip().casefold() for value in label.split("/")]
    normalized = text.casefold().replace("_", " ")
    for value in variants:
        if value in normalized:
            return True
        if len(value) >= 3 and value[:3] in normalized:
            return True
    return False


def _star_oracle(
    row: Mapping[str, Any],
    annotation: Mapping[str, Any],
    *,
    star_root: Path,
) -> dict[str, Any]:
    classes = star_root / "annotations/STAR_classes"
    verbs = v28._class_map(classes / "verb_classes.txt")
    objects = v28._class_map(classes / "object_classes.txt")
    action_map = v28._class_map(classes / "action_mapping.txt")
    choices = sorted(annotation["choices"], key=lambda value: int(value["choice_id"]))
    gold_choice = next(
        choice for choice in choices if str(choice["choice"]) == str(annotation["answer"])
    )
    values = (
        v28._program_values(annotation.get("question_program") or ())
        | v28._program_values(gold_choice.get("choice_program") or ())
    )
    verb_ids = {code for code, text in verbs.items() if text.casefold() in values}
    object_ids = {code for code, text in objects.items() if text.casefold() in values}
    action_ids = set()
    for action_id, specification in action_map.items():
        parts = specification.split()
        if len(parts) == 2 and parts[0] in verb_ids and parts[1] in object_ids:
            action_ids.add(action_id)
    metadata = row["video_metadata"]
    fps = float(metadata["fps"])
    clip_start = float(metadata["clip_start_seconds"])
    duration = float(metadata["duration_seconds"])
    action_seconds = sorted({
        int(frame_id) / fps - clip_start
        for frame_id, situation in annotation.get("situations", {}).items()
        if set(map(str, situation.get("actions") or ())) & action_ids
    })
    action_seconds = [
        value for value in action_seconds if -0.1 <= value <= duration + 0.1
    ]
    proxy_seconds = list(map(float, metadata["proxy_sample_seconds"]))
    tolerance = max(0.5, duration / max(1, len(proxy_seconds) - 1))
    event_scores = []
    for event in row["event_grounding_receipt"]["events"]:
        event_text = " ".join(str(event[key]) for key in (
            "subject", "predicate", "object", "before_state", "after_state", "reason",
        ))
        verb_hit = bool(verb_ids) and any(
            _matches(verbs[code], event_text) for code in verb_ids
        )
        object_hit = bool(object_ids) and any(
            _matches(objects[code], event_text) for code in object_ids
        )
        evidence_seconds = [
            proxy_seconds[int(index)] for index in event["evidence_frames"]
        ]
        timestamp_hit = bool(action_seconds) and any(
            abs(evidence - gold) <= tolerance
            for evidence in evidence_seconds for gold in action_seconds
        )
        interval_iou = None
        if action_seconds:
            predicted_start = proxy_seconds[int(event["start_frame"])]
            predicted_end = proxy_seconds[int(event["end_frame"])]
            gold_start = max(0.0, min(action_seconds) - tolerance)
            gold_end = min(duration, max(action_seconds) + tolerance)
            intersection = max(
                0.0,
                min(predicted_end, gold_end) - max(predicted_start, gold_start),
            )
            union = max(predicted_end, gold_end) - min(predicted_start, gold_start)
            interval_iou = intersection / union if union > 0 else float(timestamp_hit)
        event_scores.append({
            "event_id": str(event["event_id"]),
            "verb_hit": verb_hit,
            "object_hit": object_hit,
            "timestamp_hit": timestamp_hit,
            "semantic_action_hit": verb_hit and object_hit and timestamp_hit,
            "interval_iou": interval_iou,
            "evidence_seconds": evidence_seconds,
        })
    return {
        "sample_id": str(row["sample_id"]),
        "gold_verb_labels": [verbs[code] for code in sorted(verb_ids)],
        "gold_object_labels": [objects[code] for code in sorted(object_ids)],
        "gold_action_ids": sorted(action_ids),
        "gold_action_seconds": action_seconds,
        "tolerance_seconds": tolerance,
        "events": event_scores,
        "semantic_action_hit": any(value["semantic_action_hit"] for value in event_scores),
        "timestamp_hit": any(value["timestamp_hit"] for value in event_scores),
        "maximum_interval_iou": max(
            (value["interval_iou"] for value in event_scores if value["interval_iou"] is not None),
            default=None,
        ),
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
        raise ValueError("V29 requires the full 12-row development repair")
    if config["grounding"]["candidate_access"] is not False:
        raise ValueError("V29 grounder must remain candidate-blind")
    if any(row.get("source_skill_or_structure_available_at_runtime") for row in rows):
        raise ValueError("V29 unexpectedly contains source transfer")
    by_benchmark = {
        benchmark: [row for row in rows if row["benchmark"] == benchmark]
        for benchmark in ("clevrer", "star", "nextqa")
    }
    conditions = {
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
                condition: sum(bool(row["correct"][condition]) for row in subset)
                for condition in CONDITIONS
            },
            "combined_vs_uniform": v28._paired(
                subset, "localized_receipt", "uniform_direct",
            ),
            "localized_vs_shifted": v28._paired(
                subset, "localized_receipt", "shifted_receipt",
            ),
            "localization_only": v28._paired(
                subset, "localized_direct", "uniform_direct",
            ),
            "ledger_on_localized": v28._paired(
                subset, "localized_receipt", "localized_direct",
            ),
        }
    star_root = Path(config["benchmarks"]["star"]["root"])
    annotations = {
        str(value["question_id"]): value
        for value in json.loads(
            (star_root / "annotations/STAR_val.json").read_text(encoding="utf-8")
        )
    }
    star_rows = [
        _star_oracle(row, annotations[str(row["sample_id"])], star_root=star_root)
        for row in by_benchmark["star"]
    ]
    combined = v28._paired(rows, "localized_receipt", "uniform_direct")
    necessity = v28._paired(rows, "localized_receipt", "shifted_receipt")
    gates = {
        "complete_transport": len(rows) == 12,
        "source_free": all(
            row["source_skill_or_structure_available_at_runtime"] is False for row in rows
        ),
        "candidate_blind_grounder": config["grounding"]["candidate_access"] is False,
        "matched_frame_budget": all(
            all(len(row["frame_indices"][name]) == int(row["frame_budget"]) for name in CONDITIONS)
            for row in rows
        ),
        "sparse_event_citations": all(
            all(1 <= len(event["evidence_frames"]) <= 3 for event in row["event_grounding_receipt"]["events"])
            for row in rows
        ),
        "star_semantic_action_majority": sum(
            value["semantic_action_hit"] for value in star_rows
        ) > len(star_rows) / 2,
        "combined_strictly_above_uniform": combined["net_correct"] > 0,
        "localized_strictly_above_shifted": necessity["net_correct"] > 0,
        "at_least_two_benchmarks_positive_combined": sum(
            value["combined_vs_uniform"]["net_correct"] > 0
            for value in benchmark_metrics.values()
        ) >= 2,
    }
    qualified = all(gates.values())
    report = {
        "schema_version": 29,
        "status": (
            "CANDIDATE_BLIND_EVENT_LEDGER_QUALIFIED_FOR_FRESH_SCALE"
            if qualified else "CANDIDATE_BLIND_EVENT_LEDGER_NOT_QUALIFIED"
        ),
        "claim_boundary": config["claim_boundary"],
        "samples": len(rows),
        "conditions": conditions,
        "benchmarks": benchmark_metrics,
        "factorial_effects": {
            "combined_vs_uniform": combined,
            "localized_vs_shifted_evidence_necessity": necessity,
            "localization_only": v28._paired(rows, "localized_direct", "uniform_direct"),
            "ledger_on_localized": v28._paired(rows, "localized_receipt", "localized_direct"),
            "ledger_on_uniform": v28._paired(rows, "uniform_receipt", "uniform_direct"),
        },
        "event_counts": dict(Counter(
            len(row["event_grounding_receipt"]["events"]) for row in rows
        )),
        "coverage_counts": dict(Counter(
            str(row["event_grounding_receipt"]["coverage"]) for row in rows
        )),
        "star_intrinsic": {
            "rows": star_rows,
            "semantic_action_hits": sum(value["semantic_action_hit"] for value in star_rows),
            "timestamp_hits": sum(value["timestamp_hit"] for value in star_rows),
            "mean_maximum_interval_iou": sum(
                float(value["maximum_interval_iou"] or 0.0) for value in star_rows
            ) / len(star_rows),
        },
        "cost": {
            "reported_usd": sum(
                float(value.get("cost", 0.0) or 0.0)
                for row in rows for value in row["usage"].values()
            ),
            "calls": sum(len(row["usage"]) for row in rows),
        },
        "gates": gates,
        "qualified_for_fresh_scale": qualified,
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
        "conditions": conditions,
        "benchmarks": benchmark_metrics,
        "factorial_effects": report["factorial_effects"],
        "star_intrinsic": report["star_intrinsic"],
        "gates": gates,
        "cost": report["cost"],
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
