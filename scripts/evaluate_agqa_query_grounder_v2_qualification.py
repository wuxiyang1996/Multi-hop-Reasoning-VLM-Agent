#!/usr/bin/env python3
"""Evaluate frozen V2 grounding on development labels after acquisition."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from motif_transfer.agqa_query_grounder_v2 import query_grounding_v2_from_dict
from motif_transfer.agqa_query_object_grounder import canonical_object_label
from motif_transfer.contracts import stable_hash
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _answers(archive: Path, entry: str, task_ids: set[str]) -> dict[str, str]:
    found = {}
    with zipfile.ZipFile(archive) as bundle, bundle.open(entry) as raw:
        for task_id, row in _iter_top_level_object(io.TextIOWrapper(raw, encoding="utf-8")):
            key = str(task_id)
            if key in task_ids:
                value = row.get("answer")
                if value is None:
                    raise ValueError(f"development row {key} has no answer")
                found[key] = canonical_object_label(str(value))
    if set(found) != task_ids:
        raise ValueError(f"missing {len(task_ids - set(found))} development answers")
    return found


def _support_threshold(protocol: dict) -> float:
    """Read legacy nested or explicit scalar frozen threshold schemas."""
    legacy = protocol.get("grounder", {}).get("minimum_candidate_confidence")
    if legacy is not None:
        return float(legacy)
    frozen = protocol.get("frozen_grounder", {})
    explicit = frozen.get("candidate_support_threshold")
    if explicit is not None:
        return float(explicit)
    candidate = frozen.get("candidate_confidence", {})
    if isinstance(candidate, dict):
        return float(candidate.get("support_threshold", -1.0))
    return -1.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--grounding", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    p.add_argument("--split-audit", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("qualification output is immutable")
    report = json.loads(args.grounding.read_text()); protocol = json.loads(args.protocol.read_text())
    if report["status"] != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("grounding was not frozen before development outcomes")
    if any(report.get(key) for key in ("answer_read", "official_scene_graph_read",
                                       "functional_program_read", "source_controller_read",
                                       "target_outcome_read")):
        raise ValueError("grounding crossed the authority boundary")
    rows = report["rows"]; task_ids = {str(row["task_id"]) for row in rows}
    answers = _answers(args.archive, args.entry, task_ids)
    threshold = _support_threshold(protocol)
    if not 0 <= threshold <= 1:
        raise ValueError("qualification protocol lacks a valid support threshold")
    metrics_rows = []; provider_success = 0; present = 0; unique_count = 0; unique_correct = 0
    residual_duplicates = 0; authority_safe = 0
    for row in rows:
        receipt = query_grounding_v2_from_dict(row["receipt"]); authority_safe += 1
        gold = answers[receipt.task_id]
        labels = {track.canonical_label for track in receipt.tracks}
        has_gold = gold in labels; present += int(has_gold)
        provider_ok = row.get("provider_error") is None; provider_success += int(provider_ok)
        event_supported_tracks = {
            track_id for event in receipt.events
            for role, track_id in event.roles if role == row.get("requested_role")
        }
        supported = [candidate.track_id for candidate in receipt.candidates
                     if candidate.status == "SUPPORTED" and candidate.confidence >= threshold]
        supported = [track_id for track_id in supported if track_id in event_supported_tracks]
        supported_labels = sorted({next(track.canonical_label for track in receipt.tracks
                                        if track.track_id == track_id) for track_id in supported})
        unique = len(supported_labels) == 1
        correct = unique and supported_labels[0] == gold
        unique_count += int(unique); unique_correct += int(correct)
        events = receipt.events
        for i, left in enumerate(events):
            for right in events[i + 1:]:
                if left.predicate.casefold().replace("_", " ") != right.predicate.casefold().replace("_", " ") or left.roles != right.roles:
                    continue
                intersection = max(0, min(left.end_frame, right.end_frame) - max(left.start_frame, right.start_frame) + 1)
                union = max(left.end_frame, right.end_frame) - min(left.start_frame, right.start_frame) + 1
                residual_duplicates += int(intersection / union >= .5)
        metrics_rows.append({"task_id": receipt.task_id, "gold_entity_evaluator_only": gold,
                             "gold_present_in_inventory": has_gold, "supported_labels": supported_labels,
                             "unique_supported": unique, "unique_supported_correct": correct,
                             "provider_contract_success": provider_ok})
    n = len(rows)
    values = {
        "provider_and_contract_success_fraction": provider_success / n if n else 0.0,
        "gold_entity_present_in_inventory_fraction": present / n if n else 0.0,
        "unique_supported_requested_role_precision": unique_correct / unique_count if unique_count else 0.0,
        "unique_supported_coverage_fraction": unique_count / n if n else 0.0,
        "residual_duplicate_typed_event_key_count": residual_duplicates,
        "authority_safe_receipt_fraction": authority_safe / n if n else 0.0,
    }
    expected = protocol["qualification_gates"]
    gates = {
        "provider_and_contract_success_fraction_minimum": values["provider_and_contract_success_fraction"] >= expected["provider_and_contract_success_fraction_minimum"],
        "gold_entity_present_in_inventory_fraction_minimum": values["gold_entity_present_in_inventory_fraction"] >= expected["gold_entity_present_in_inventory_fraction_minimum"],
        "unique_supported_requested_role_precision_minimum": values["unique_supported_requested_role_precision"] >= expected["unique_supported_requested_role_precision_minimum"],
        "residual_duplicate_typed_event_key_count_maximum": residual_duplicates <= expected["residual_duplicate_typed_event_key_count_maximum"],
        "authority_safe_receipt_fraction": values["authority_safe_receipt_fraction"] == expected["authority_safe_receipt_fraction"],
        "full_preregistered_query_cohort": n == protocol.get(
            "cohort", protocol.get("qualification_cohort", {}),
        )["query_tasks"],
    }
    if "unique_supported_coverage_fraction_minimum" in expected:
        gates["unique_supported_coverage_fraction_minimum"] = (
            values["unique_supported_coverage_fraction"]
            >= expected["unique_supported_coverage_fraction_minimum"]
        )
    split_audit = None
    if "held_out_action_genome_split_fraction" in expected:
        if args.split_audit is None:
            raise ValueError("held-out Action Genome split audit is required")
        split_audit = json.loads(args.split_audit.read_text())
        if split_audit.get("cohort_sha256") != report.get("cohort_sha256"):
            raise ValueError("split audit and grounding refer to different cohorts")
        gates["held_out_action_genome_split_fraction"] = (
            split_audit.get("status") == "HELD_OUT_SPLIT_AUDIT_PASSED"
            and split_audit.get("missing_video_count") == 0
            and split_audit.get("wrong_split_video_count") == 0
        )
    body = {"schema_version": "agqa-query-grounder-v2-qualification-v1",
            "status": "QUERY_GROUNDER_V2_QUALIFIED" if all(gates.values()) else "QUERY_GROUNDER_V2_NOT_QUALIFIED",
            "grounding_report_sha256": report["report_sha256"], "grounding_file_sha256": _sha(args.grounding),
            "protocol_file_sha256": _sha(args.protocol), "development_rows": n,
            "split_audit_report_sha256": split_audit.get("report_sha256") if split_audit else None,
            "metrics": values, "gates": gates, "rows": metrics_rows,
            "development_outcome_opened_only_after_grounding_freeze": True}
    body["report_sha256"] = stable_hash(body); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: body[k] for k in ("status", "development_rows", "metrics", "gates", "report_sha256")}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
