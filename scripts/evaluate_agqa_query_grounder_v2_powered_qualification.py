#!/usr/bin/env python3
"""Open outcomes once and evaluate a fixed Query Grounder V2 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.agqa_query_grounder_v2 import query_grounding_v2_from_dict
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_query_grounder_v5_development import (
    _answers,
    _interval_iou,
    _sha256,
    _wilson_lower,
)


def _qualification_gates(metrics: dict, required: dict, *, rows: int) -> dict:
    return {
        "provider_and_contract_success_fraction_minimum": (
            metrics["provider_and_contract_success_fraction"]
            >= float(required["provider_and_contract_success_fraction_minimum"])
        ),
        "entity_candidate_pool_recall_minimum": (
            metrics["entity_candidate_pool_recall"]
            >= float(required["entity_candidate_pool_recall_minimum"])
        ),
        "typed_role_binding_fidelity_minimum": (
            metrics["typed_role_binding_fidelity"]
            >= float(required["typed_role_binding_fidelity_minimum"])
        ),
        "cross_frame_dedup_fidelity_minimum": (
            metrics["cross_frame_dedup_fidelity"]
            >= float(required["cross_frame_dedup_fidelity_minimum"])
        ),
        "unique_supported_count_minimum": (
            metrics["unique_supported_count"]
            >= int(required["unique_supported_count_minimum"])
        ),
        "unique_supported_precision_wilson_95_lower_bound_minimum": (
            metrics["unique_supported_precision_wilson_95_lower"]
            >= float(
                required[
                    "unique_supported_precision_wilson_95_lower_bound_minimum"
                ]
            )
        ),
        "unique_supported_coverage_fraction_minimum": (
            metrics["unique_supported_coverage_fraction"]
            >= float(required["unique_supported_coverage_fraction_minimum"])
        ),
        "residual_duplicate_typed_event_key_count_maximum": (
            metrics["residual_duplicate_typed_event_key_count"]
            <= int(required["residual_duplicate_typed_event_key_count_maximum"])
        ),
        "authority_safe_receipt_fraction": (
            metrics["authority_safe_receipt_fraction"]
            == float(required["authority_safe_receipt_fraction"])
        ),
        "full_preregistered_query_cohort": rows
        == int(metrics["expected_query_tasks"]),
        "one_fixed_global_threshold": metrics["fixed_threshold_verified"],
        "grounder_contract": metrics["grounder_contract_verified"],
    }


def _validate_preoutcome_amendment(amendment: dict, protocol: dict, *, root: Path) -> None:
    if amendment.get("status") != (
        "FROZEN_AFTER_OUTCOME_BLIND_INFRASTRUCTURE_FAILURE_BEFORE_QUALIFICATION_OUTCOMES"
    ):
        raise ValueError("invalid pre-outcome amendment status")
    if amendment.get("protocol_file_sha256") != _sha256(root / amendment["protocol_file"]):
        raise ValueError("pre-outcome amendment names a different protocol")
    frozen_sha = protocol["frozen_grounder"]["component_sha256s"]["query_grounder_compiler"]
    if amendment.get("original_component_sha256") != frozen_sha:
        raise ValueError("pre-outcome amendment does not preserve the frozen component hash")
    if amendment.get("amended_component_sha256") != _sha256(root / amendment["component_path"]):
        raise ValueError("amended component hash does not match runtime code")
    required_false = (
        "qualification_outcomes_opened_before_amendment",
        "authoritative_grounding_output_existed_before_amendment",
        "selection_changed", "videos_or_tasks_changed",
        "model_or_checkpoint_changed", "frame_budget_changed", "ontology_changed",
        "candidate_ranking_changed", "support_threshold_changed", "target_answer_read",
        "official_scene_graph_read", "functional_program_read", "source_controller_read",
        "target_outcome_read",
    )
    if any(amendment.get(key) is not False for key in required_false):
        raise ValueError("pre-outcome amendment crossed its declared repair boundary")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--preoutcome-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("powered qualification evaluation is immutable")

    grounding = json.loads(args.grounding.read_text())
    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "QUERY_GROUNDER_V2_POWERED_QUALIFICATION_FROZEN":
        raise ValueError("qualification cohort was not frozen")
    if manifest.get("protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("qualification protocol changed after cohort freeze")
    amendment_file_sha256 = None
    if args.preoutcome_amendment is not None:
        amendment = json.loads(args.preoutcome_amendment.read_text())
        _validate_preoutcome_amendment(amendment, protocol, root=Path.cwd())
        amendment_file_sha256 = _sha256(args.preoutcome_amendment)
    if grounding.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("grounding was not frozen before qualification outcomes")
    if grounding.get("cohort_sha256") != manifest.get("cohort_sha256"):
        raise ValueError("grounding and frozen qualification cohort differ")
    forbidden = (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )
    if any(grounding.get(key) for key in forbidden):
        raise ValueError("grounding crossed the frozen authority boundary")

    frozen = protocol["frozen_grounder"]
    threshold = float(frozen["candidate_support_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid frozen support threshold")
    if any(
        float(row["candidate_support_threshold"])
        != float(frozen["candidate_generation_threshold"])
        for row in grounding["rows"]
    ):
        raise ValueError("candidate generation threshold changed")
    budgets = grounding.get("component_frame_budgets", {})
    contract_verified = all((
        grounding.get("public_ontology_sha256")
        == frozen["public_ontology_sha256"],
        grounding.get("stable_entity_tracking") is True,
        grounding.get("typed_semantic_roles") is True,
        grounding.get("cross_frame_typed_event_deduplication") is True,
        grounding.get("answer_blind_query_candidate_verification") is True,
        grounding.get("all_harness_arms_share_exact_receipts") is True,
        int(budgets.get("sgdet_unique_and_model_presentations", -1))
        == int(frozen["frame_budget_sgdet"]),
        int(budgets.get("slowfast_unique_sampled_frames", -1))
        == int(frozen["frame_budget_slowfast"]),
    ))

    task_ids = {str(row["task_id"]) for row in grounding["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    row_metrics = []
    inventory_hits = provider_success = authority_safe = 0
    candidate_count = role_consistent = 0
    event_pair_count = residual_duplicates = 0
    supported_count = supported_correct = 0
    for raw in grounding["rows"]:
        receipt = query_grounding_v2_from_dict(raw["receipt"])
        authority_safe += 1
        provider_success += int(raw.get("provider_error") is None)
        gold = answers[receipt.task_id]
        track_labels = {
            track.track_id: track.canonical_label for track in receipt.tracks
        }
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
                event_pair_count += 1
                if (
                    left.predicate.casefold().replace("_", " ")
                    == right.predicate.casefold().replace("_", " ")
                    and left.roles == right.roles
                    and _interval_iou(left, right) >= 0.5
                ):
                    residual_duplicates += 1

        candidate = receipt.candidates[0] if receipt.candidates else None
        confidence = float(raw.get("candidate_confidence", -1.0))
        predicted = track_labels.get(candidate.track_id) if candidate else None
        supported = candidate is not None and confidence >= threshold
        correct = supported and predicted == gold
        supported_count += int(supported)
        supported_correct += int(correct)
        row_metrics.append({
            "task_id": receipt.task_id,
            "gold_entity_evaluator_only": gold,
            "gold_present_in_answer_blind_track_inventory": inventory_hit,
            "candidate_confidence": confidence,
            "supported_at_fixed_threshold": supported,
            "supported_prediction": predicted if supported else None,
            "supported_correct": correct,
        })

    n = len(grounding["rows"])
    precision = supported_correct / supported_count if supported_count else 0.0
    metrics = {
        "expected_query_tasks": int(protocol["qualification_cohort"]["query_tasks"]),
        "provider_and_contract_success_fraction": provider_success / n if n else 0.0,
        "entity_candidate_pool_recall": inventory_hits / n if n else 0.0,
        "typed_role_binding_fidelity": (
            role_consistent / candidate_count if candidate_count else 0.0
        ),
        "cross_frame_dedup_fidelity": (
            1.0 - residual_duplicates / event_pair_count
            if event_pair_count else 1.0
        ),
        "residual_duplicate_typed_event_key_count": residual_duplicates,
        "typed_event_pair_count": event_pair_count,
        "unique_supported_count": supported_count,
        "unique_supported_correct": supported_correct,
        "unique_supported_precision": precision,
        "unique_supported_precision_wilson_95_lower": _wilson_lower(
            supported_correct, supported_count
        ),
        "unique_supported_coverage_fraction": supported_count / n if n else 0.0,
        "authority_safe_receipt_fraction": authority_safe / n if n else 0.0,
        "fixed_candidate_support_threshold": threshold,
        "fixed_threshold_verified": True,
        "grounder_contract_verified": contract_verified,
    }
    gates = _qualification_gates(
        metrics, protocol["qualification_gates"], rows=n,
    )
    body = {
        "schema_version": "agqa-query-grounder-v2-powered-qualification-v1",
        "status": (
            "QUERY_GROUNDER_V2_POWERED_QUALIFIED"
            if all(gates.values())
            else "QUERY_GROUNDER_V2_POWERED_NOT_QUALIFIED"
        ),
        "grounding_report_sha256": grounding["report_sha256"],
        "grounding_file_sha256": _sha256(args.grounding),
        "protocol_file_sha256": _sha256(args.protocol),
        "manifest_file_sha256": _sha256(args.manifest),
        "preoutcome_amendment_file_sha256": amendment_file_sha256,
        "qualification_rows": n,
        "metrics": metrics,
        "gates": gates,
        "rows": row_metrics,
        "qualification_outcomes_opened_only_after_grounding_freeze": True,
        "answers_available_to_grounder": False,
        "official_scene_graph_or_functional_program_read": False,
        "transfer_evidence": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "qualification_rows": n,
        "metrics": metrics, "gates": gates,
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
