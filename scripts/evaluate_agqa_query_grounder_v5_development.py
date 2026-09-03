#!/usr/bin/env python3
"""Evaluate frozen AGQA V5 grounding after development outcomes are unsealed."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from motif_transfer.agqa_query_grounder_v2 import query_grounding_v2_from_dict
from motif_transfer.agqa_query_object_grounder import canonical_object_label
from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _answers(archive: Path, entry: str, wanted: set[str]) -> dict[str, str]:
    output = {}
    with zipfile.ZipFile(archive) as bundle, bundle.open(entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            key = str(task_id)
            if key in wanted:
                output[key] = canonical_object_label(str(row["answer"]))
                if len(output) == len(wanted):
                    break
    if set(output) != wanted:
        raise ValueError(f"missing {len(wanted - set(output))} development answers")
    return output


def _wilson_lower(correct: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - radius) / denominator


def _interval_iou(left, right) -> float:
    intersection = max(
        0, min(left.end_frame, right.end_frame)
        - max(left.start_frame, right.start_frame) + 1,
    )
    union = max(left.end_frame, right.end_frame) - min(
        left.start_frame, right.start_frame
    ) + 1
    return intersection / union if union else 0.0


def _gate_requirement(
    required: dict[str, object], canonical_key: str, legacy_key: str | None = None,
) -> float:
    """Read a frozen gate without changing its pre-outcome numerical value."""
    if canonical_key in required:
        return float(required[canonical_key])
    if legacy_key is not None and legacy_key in required:
        return float(required[legacy_key])
    alternatives = f" or {legacy_key!r}" if legacy_key is not None else ""
    raise KeyError(f"missing frozen grounding gate {canonical_key!r}{alternatives}")


def _expected_grounding_status(consumed_development: bool) -> str:
    if consumed_development:
        return "CONSUMED_DEVELOPMENT_QUERY_GROUNDING_V2_NOT_TRANSFER_EVIDENCE"
    return "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument(
        "--consumed-development-grounding", action="store_true",
        help=(
            "Accept a V2 artifact explicitly labeled as consumed development; "
            "this never upgrades it to transfer evidence."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("V5 development evaluation is immutable")

    grounding = json.loads(args.grounding.read_text())
    protocol = json.loads(args.protocol.read_text())
    expected_status = _expected_grounding_status(
        args.consumed_development_grounding
    )
    if grounding.get("status") != expected_status:
        raise ValueError("grounding was not frozen before development outcomes")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(grounding.get(key) for key in forbidden):
        raise ValueError("grounding crossed the authority boundary")
    if any(
        float(row["candidate_support_threshold"]) != 0.0
        for row in grounding["rows"]
    ):
        raise ValueError("development candidate generation was not threshold zero")

    task_ids = {str(row["task_id"]) for row in grounding["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    row_metrics = []
    inventory_hits = 0
    candidate_count = 0
    role_consistent = 0
    total_event_pairs = 0
    residual_duplicates = 0
    provider_success = 0
    authority_safe = 0
    predictions = []
    for raw in grounding["rows"]:
        receipt = query_grounding_v2_from_dict(raw["receipt"])
        authority_safe += 1
        provider_success += int(raw.get("provider_error") is None)
        track_labels = {track.track_id: track.canonical_label for track in receipt.tracks}
        gold = answers[receipt.task_id]
        inventory_hit = gold in set(track_labels.values())
        inventory_hits += int(inventory_hit)

        for candidate in receipt.candidates:
            candidate_count += 1
            matching_events = [
                event for event in receipt.events
                if event.role_map.get(candidate.requested_role) == candidate.track_id
                and set(event.evidence_frames) & set(candidate.evidence_frames)
            ]
            role_consistent += int(
                candidate.requested_role == raw.get("requested_role")
                and bool(matching_events)
            )
        for index, left in enumerate(receipt.events):
            for right in receipt.events[index + 1:]:
                total_event_pairs += 1
                if (
                    left.predicate.casefold().replace("_", " ")
                    == right.predicate.casefold().replace("_", " ")
                    and left.roles == right.roles
                    and _interval_iou(left, right) >= 0.5
                ):
                    residual_duplicates += 1

        candidate = receipt.candidates[0] if receipt.candidates else None
        predicted = track_labels.get(candidate.track_id) if candidate else None
        confidence = float(raw.get("candidate_confidence", -1.0))
        predictions.append({
            "task_id": receipt.task_id,
            "predicted": predicted,
            "confidence": confidence,
            "correct": predicted == gold if predicted is not None else False,
        })
        row_metrics.append({
            "task_id": receipt.task_id,
            "gold_entity_evaluator_only": gold,
            "gold_present_in_answer_blind_track_inventory": inventory_hit,
            "candidate_emitted": candidate is not None,
            "candidate_confidence": confidence,
        })

    selection = protocol["threshold_selection"]
    constraints = selection["constraints"]
    curve = []
    for threshold in selection["candidate_grid"]:
        selected = [row for row in predictions if row["confidence"] >= float(threshold)]
        correct = sum(bool(row["correct"]) for row in selected)
        total = len(selected)
        curve.append({
            "threshold": float(threshold),
            "supported": total,
            "correct": correct,
            "precision": correct / total if total else 0.0,
            "precision_wilson_95_lower": _wilson_lower(correct, total),
            "coverage": total / len(predictions) if predictions else 0.0,
        })
    eligible = [row for row in curve if (
        row["supported"] >= int(constraints["unique_supported_count_minimum"])
        and row["precision_wilson_95_lower"] >= float(
            constraints["unique_supported_precision_wilson_95_lower_bound_minimum"]
        )
        and row["coverage"] >= float(
            constraints["unique_supported_coverage_minimum"]
        )
    )]
    selected_threshold = max(
        eligible, key=lambda row: (row["coverage"], row["threshold"]), default=None,
    )
    n = len(predictions)
    metrics = {
        "entity_candidate_pool_recall": inventory_hits / n if n else 0.0,
        "typed_role_binding_fidelity": role_consistent / candidate_count if candidate_count else 0.0,
        "cross_frame_dedup_fidelity": (
            1.0 - residual_duplicates / total_event_pairs if total_event_pairs else 1.0
        ),
        "residual_duplicate_typed_event_pairs": residual_duplicates,
        "typed_event_pair_count": total_event_pairs,
        "provider_and_contract_success_fraction": float(
            grounding.get(
                "provider_and_contract_success_fraction",
                provider_success / n if n else 0.0,
            )
        ),
        "authority_safe_receipt_fraction": authority_safe / n if n else 0.0,
    }
    required = protocol["grounding_gates"]
    gates = {
        "entity_candidate_pool_recall": metrics["entity_candidate_pool_recall"]
        >= _gate_requirement(required, "entity_candidate_pool_recall_minimum"),
        "typed_role_binding_fidelity": metrics["typed_role_binding_fidelity"]
        >= _gate_requirement(
            required,
            "typed_role_fidelity_minimum",
            "typed_role_binding_fidelity_minimum",
        ),
        "cross_frame_dedup_fidelity": metrics["cross_frame_dedup_fidelity"]
        >= _gate_requirement(required, "cross_frame_dedup_fidelity_minimum"),
        "provider_and_contract_success": metrics["provider_and_contract_success_fraction"]
        >= _gate_requirement(required, "provider_and_contract_success_fraction_minimum"),
        "authority_safe_receipts": metrics["authority_safe_receipt_fraction"]
        == _gate_requirement(required, "authority_safe_receipt_fraction"),
        "one_global_threshold_qualified": selected_threshold is not None,
    }
    body = {
        "schema_version": "agqa-query-grounder-v5-question-blind-event-development-evaluation-v2",
        "status": "V5_STAGE_A_DEVELOPMENT_QUALIFIED" if all(gates.values()) else "V5_STAGE_A_DEVELOPMENT_NOT_QUALIFIED",
        "grounding_report_sha256": grounding["report_sha256"],
        "grounding_file_sha256": _sha256(args.grounding),
        "protocol_file_sha256": _sha256(args.protocol),
        "development_tasks": n,
        "metrics": metrics,
        "threshold_curve": curve,
        "selected_global_threshold": selected_threshold,
        "gates": gates,
        "rows": row_metrics,
        "development_outcomes_opened_only_after_grounding_freeze": True,
        "consumed_development_grounding": bool(
            args.consumed_development_grounding
        ),
        "transfer_evidence": False,
        "answers_available_to_grounder": False,
        "official_scene_graph_or_functional_program_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "development_tasks": n,
        "metrics": metrics, "selected_global_threshold": selected_threshold,
        "gates": gates, "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
