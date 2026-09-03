#!/usr/bin/env python3
"""Evaluate strict-boundary AGQA grounder qualification after route freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.agqa_query_grounder_v2 import query_grounding_v2_from_dict
from motif_transfer.contracts import stable_hash
from scripts.evaluate_agqa_query_grounder_v2_powered_qualification import (
    _qualification_gates,
)
from scripts.evaluate_agqa_query_grounder_v5_development import (
    _answers,
    _interval_iou,
    _sha256,
    _wilson_lower,
)


COMPONENT_PATHS = {
    "sgdet_collector": "scripts/pilot_agqa_action_genome_sgdet.py",
    "slowfast_probe": "scripts/probe_agqa_layer_b_charades_action_model.py",
    "query_grounder_compiler_parent": "scripts/compile_agqa_action_genome_query_grounder_v2.py",
    "strict_temporal_compiler": "scripts/compile_agqa_action_genome_query_grounder_v2_strict_temporal.py",
    "strict_temporal_projection": "src/motif_transfer/agqa_strict_temporal_projection.py",
    "typed_executor_v2": "src/motif_transfer/agqa_layer_b_executor_v2.py",
    "preoutcome_coverage_audit": "scripts/audit_agqa_query_grounder_preoutcome_coverage.py",
}


def _preoutcome_gates(protocol: dict, audit: dict, grounding: dict) -> dict:
    required = protocol["qualification_gates"]
    expected_tasks = int(protocol["qualification_cohort"]["query_tasks"])
    rows = audit.get("rows", ())
    every_commit_supported = all(
        not (row.get("source_commit") or row.get("permuted_commit"))
        or bool(row.get("candidate_supported"))
        for row in rows
    )
    return {
        "outcome_blind_full_cohort": (
            int(audit.get("tasks", -1)) == expected_tasks == len(rows)
        ),
        "outcome_blind_source_symbolic_commit_fraction_minimum": (
            float(audit.get("source_commit_fraction", -1.0))
            >= float(required["outcome_blind_source_symbolic_commit_fraction_minimum"])
        ),
        "outcome_blind_source_permuted_commit_fraction_maximum": (
            float(audit.get("permuted_commit_fraction", 2.0))
            <= float(required["outcome_blind_source_permuted_commit_fraction_maximum"])
        ),
        "candidate_support_required_for_every_symbolic_commit": (
            every_commit_supported
            and bool(required["candidate_support_required_for_every_symbolic_commit"])
        ),
        "audit_binds_frozen_grounding": (
            audit.get("query_grounding_report_sha256") == grounding.get("report_sha256")
        ),
        "source_capability_frozen": (
            audit.get("source_capability_sha256")
            == protocol["source_harness"]["source_capability_sha256"]
        ),
        "anonymous_controller_frozen": (
            audit.get("anonymous_controller_sha256")
            == protocol["source_harness"]["anonymous_controller_sha256"]
        ),
    }


def _grounder_contract(
    protocol: dict, grounding: dict, action_grounding: dict,
    slowfast_bindings: dict, *, action_grounding_path: Path,
    slowfast_bindings_path: Path, parent_grounding: dict | None = None,
    parent_grounding_path: Path | None = None,
) -> tuple[bool, dict]:
    frozen = protocol["frozen_grounder"]
    budgets = grounding.get("component_frame_budgets", {})
    checks = {
        "public_ontology": (
            grounding.get("public_ontology_sha256") == frozen["public_ontology_sha256"]
        ),
        "sgdet_budget": (
            int(budgets.get("sgdet_unique_and_model_presentations", -1))
            == int(frozen["frame_budget_sgdet"])
        ),
        "slowfast_unique_budget": (
            int(budgets.get("slowfast_unique_sampled_frames", -1))
            == int(frozen["frame_budget_slowfast_unique"])
        ),
        "slowfast_presentation_budget": (
            int(action_grounding.get("frame_presentation_budget", -1))
            == int(frozen["frame_budget_slowfast_presentations"])
        ),
        "slowfast_sampling": action_grounding.get("sampling") == frozen["slowfast_sampling"],
        "slowfast_checkpoint": (
            action_grounding.get("checkpoint_sha256") == frozen["slowfast_checkpoint_sha256"]
        ),
        "action_probe_authority": not any(
            action_grounding.get(key) for key in (
                "answers_read", "official_program_read", "official_scene_graph_read",
            )
        ),
        "binding_authority": not any(
            slowfast_bindings.get(key) for key in (
                "answer_read", "official_scene_graph_read", "functional_program_read",
                "source_controller_read", "target_outcome_read",
            )
        ),
        "action_probe_to_binding_hash_chain": (
            slowfast_bindings.get("action_grounding_file_sha256")
            == _sha256(action_grounding_path)
        ),
        "binding_to_query_hash_chain": (
            grounding.get("inputs", {}).get("slowfast_bindings_sha256")
            == _sha256(slowfast_bindings_path)
        ),
        "strict_temporal_projection": grounding.get("strict_temporal_projection") is True,
        "in_window_track_evidence": grounding.get("in_window_track_evidence_required") is True,
        "nested_action_coreference": grounding.get("nested_action_patient_coreference") is True,
        "boundary_temporal_representation": (
            grounding.get("action_event_temporal_representation")
            == frozen["action_event_temporal_representation"]
        ),
        "shared_receipts": grounding.get("all_harness_arms_share_exact_receipts") is True,
        "stable_tracks": grounding.get("stable_entity_tracking") is True,
        "typed_roles": grounding.get("typed_semantic_roles") is True,
        "typed_event_dedup": grounding.get("cross_frame_typed_event_deduplication") is True,
        "answer_blind_verification": grounding.get("answer_blind_query_candidate_verification") is True,
    }
    expected_components = frozen["component_sha256s"]
    for name, path in COMPONENT_PATHS.items():
        checks[f"component:{name}"] = _sha256(Path(path)) == expected_components[name]
    checks["ontology_file"] = (
        _sha256(Path(frozen["public_ontology"])) == frozen["public_ontology_sha256"]
    )
    checks["source_capability_file"] = (
        _sha256(Path(protocol["source_harness"]["source_capability_file"]))
        == protocol["source_harness"]["source_capability_file_sha256"]
    )
    checks["anonymous_controller_file"] = (
        _sha256(Path(protocol["source_harness"]["anonymous_controller_file"]))
        == protocol["source_harness"]["anonymous_controller_file_sha256"]
    )
    verifier = frozen.get("candidate_verifier")
    if verifier is not None:
        metadata = grounding.get("candidate_verification", {})
        checks.update({
            "candidate_verifier_schema": (
                grounding.get("schema_version")
                == verifier.get(
                    "grounding_schema_version",
                    "agqa-query-grounder-v2-stable-track-verified-v1",
                )
            ),
            "candidate_verifier_formula": metadata.get("formula") == verifier["formula"],
            "candidate_verifier_no_fitted_weights": metadata.get("fitted_weights") is False,
            "candidate_verifier_source_blind": (
                metadata.get("source_controller_read") is False
                and grounding.get("source_controller_read") is False
            ),
            "candidate_verifier_frame_budget": (
                int(metadata.get("sgdet_frame_budget", -1))
                == int(verifier["sgdet_frame_budget"])
            ),
            "candidate_verifier_parent_present": (
                parent_grounding is not None and parent_grounding_path is not None
            ),
        })
        if parent_grounding is not None and parent_grounding_path is not None:
            parent_rows = {
                str(row["task_id"]): row for row in parent_grounding.get("rows", ())
            }
            checks.update({
                "candidate_verifier_parent_file_hash_chain": (
                    grounding.get("parent_grounding_file_sha256")
                    == _sha256(parent_grounding_path)
                ),
                "candidate_verifier_parent_report_hash_chain": (
                    grounding.get("parent_grounding_report_sha256")
                    == parent_grounding.get("report_sha256")
                ),
                "candidate_verifier_parent_backend_hash_chain": (
                    grounding.get("parent_grounder_backend_sha256")
                    == parent_grounding.get("grounder_backend_sha256")
                ),
                "candidate_verifier_row_receipt_hash_chain": (
                    len(parent_rows) == len(grounding.get("rows", ()))
                    and all(
                        row.get("parent_query_grounding_receipt_sha256")
                        == parent_rows.get(str(row["task_id"]), {}).get("receipt", {}).get(
                            "receipt_sha256"
                        )
                        for row in grounding.get("rows", ())
                    )
                ),
            })
        for name, path in verifier["component_paths"].items():
            checks[f"candidate_verifier_component:{name}"] = (
                _sha256(Path(path)) == verifier["component_sha256s"][name]
            )
    return all(checks.values()), checks


def _validate_amendment(amendment: dict, *, protocol_path: Path,
                        action_grounding_path: Path,
                        slowfast_bindings_path: Path) -> None:
    if amendment.get("status") != (
        "FROZEN_AFTER_OUTCOME_BLIND_CONTRACT_FAILURE_BEFORE_QUALIFICATION_OUTCOMES"
    ):
        raise ValueError("invalid preoutcome amendment status")
    if amendment.get("protocol_file_sha256") != _sha256(protocol_path):
        raise ValueError("amendment names a different protocol")
    if amendment.get("slowfast_probe_file_sha256") != _sha256(action_grounding_path):
        raise ValueError("amendment names a different SlowFast probe")
    if amendment.get("slowfast_binding_file_sha256") != _sha256(slowfast_bindings_path):
        raise ValueError("amendment names a different SlowFast binding")
    if amendment.get("amended_evaluator_sha256") != _sha256(Path(__file__)):
        raise ValueError("amended evaluator differs from amendment")
    required_false = (
        "qualification_outcomes_opened_before_amendment", "target_answers_read",
        "target_outcomes_read", "official_scene_graph_read", "functional_program_read",
        "selection_changed", "videos_or_tasks_changed", "frames_changed",
        "model_or_checkpoint_changed", "grounding_predictions_changed",
        "candidate_ranking_changed", "support_threshold_changed",
        "source_or_permuted_routes_changed", "qualification_gates_changed",
    )
    if any(amendment.get(key) is not False for key in required_false):
        raise ValueError("preoutcome amendment crossed its repair boundary")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--preoutcome-audit", type=Path, required=True)
    parser.add_argument("--action-grounding", type=Path, required=True)
    parser.add_argument("--slowfast-bindings", type=Path, required=True)
    parser.add_argument("--parent-grounding", type=Path)
    parser.add_argument("--preoutcome-amendment", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--entry", default="AGQA_balanced/train_balanced.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("strict-boundary qualification is immutable")

    grounding = json.loads(args.grounding.read_text())
    audit = json.loads(args.preoutcome_audit.read_text())
    action_grounding = json.loads(args.action_grounding.read_text())
    slowfast_bindings = json.loads(args.slowfast_bindings.read_text())
    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    parent_grounding = (
        json.loads(args.parent_grounding.read_text())
        if args.parent_grounding is not None else None
    )
    if args.preoutcome_amendment is not None:
        amendment = json.loads(args.preoutcome_amendment.read_text())
        _validate_amendment(
            amendment, protocol_path=args.protocol,
            action_grounding_path=args.action_grounding,
            slowfast_bindings_path=args.slowfast_bindings,
        )
    amendment_sha256 = (
        _sha256(args.preoutcome_amendment)
        if args.preoutcome_amendment is not None else None
    )
    if manifest.get("status") != "QUERY_GROUNDER_V2_POWERED_QUALIFICATION_FROZEN":
        raise ValueError("qualification cohort was not frozen")
    if manifest.get("protocol_file_sha256") != _sha256(args.protocol):
        raise ValueError("qualification protocol changed after cohort freeze")
    if grounding.get("status") != "QUERY_GROUNDING_V2_FROZEN_BEFORE_OUTCOME":
        raise ValueError("grounding was not frozen before qualification outcomes")
    if grounding.get("cohort_sha256") != manifest.get("cohort_sha256"):
        raise ValueError("grounding and qualification cohort differ")
    if any(grounding.get(key) for key in (
        "answer_read", "official_scene_graph_read", "functional_program_read",
        "source_controller_read", "target_outcome_read",
    )):
        raise ValueError("grounding crossed its authority boundary")
    frozen = protocol["frozen_grounder"]
    if any(
        float(row["candidate_support_threshold"])
        != float(frozen["candidate_generation_threshold"])
        for row in grounding["rows"]
    ):
        raise ValueError("candidate generation threshold changed")

    contract_verified, contract_checks = _grounder_contract(
        protocol, grounding, action_grounding, slowfast_bindings,
        action_grounding_path=args.action_grounding,
        slowfast_bindings_path=args.slowfast_bindings,
        parent_grounding=parent_grounding,
        parent_grounding_path=args.parent_grounding,
    )
    preoutcome_gates = _preoutcome_gates(protocol, audit, grounding)
    if not contract_verified or not all(preoutcome_gates.values()):
        body = {
            "schema_version": "agqa-query-grounder-v2-strict-boundary-qualification-v2",
            "status": "PREOUTCOME_QUALIFICATION_GATE_FAILED",
            "grounding_file_sha256": _sha256(args.grounding),
            "preoutcome_audit_file_sha256": _sha256(args.preoutcome_audit),
            "action_grounding_file_sha256": _sha256(args.action_grounding),
            "slowfast_bindings_file_sha256": _sha256(args.slowfast_bindings),
            "protocol_file_sha256": _sha256(args.protocol),
            "manifest_file_sha256": _sha256(args.manifest),
            "preoutcome_amendment_file_sha256": amendment_sha256,
            "grounder_contract_checks": contract_checks,
            "preoutcome_gates": preoutcome_gates,
            "qualification_outcomes_opened": False,
            "target_outcome_read": False,
        }
        body["report_sha256"] = stable_hash(body)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        print(json.dumps(body, indent=2))
        return 1

    # This is intentionally the first and only outcome-opening operation.
    task_ids = {str(row["task_id"]) for row in grounding["rows"]}
    answers = _answers(args.archive, args.entry, task_ids)
    threshold = float(frozen["candidate_support_threshold"])
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
        track_labels = {track.track_id: track.canonical_label for track in receipt.tracks}
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
        "typed_role_binding_fidelity": role_consistent / candidate_count if candidate_count else 0.0,
        "cross_frame_dedup_fidelity": (
            1.0 - residual_duplicates / event_pair_count if event_pair_count else 1.0
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
    candidate_gates = _qualification_gates(
        metrics, protocol["qualification_gates"], rows=n,
    )
    gates = {**contract_checks, **preoutcome_gates, **candidate_gates}
    body = {
        "schema_version": "agqa-query-grounder-v2-strict-boundary-qualification-v2",
        "status": (
            "QUERY_GROUNDER_V2_STRICT_BOUNDARY_QUALIFIED"
            if all(gates.values()) else "QUERY_GROUNDER_V2_STRICT_BOUNDARY_NOT_QUALIFIED"
        ),
        "grounding_report_sha256": grounding["report_sha256"],
        "grounding_file_sha256": _sha256(args.grounding),
        "preoutcome_audit_file_sha256": _sha256(args.preoutcome_audit),
        "action_grounding_file_sha256": _sha256(args.action_grounding),
        "slowfast_bindings_file_sha256": _sha256(args.slowfast_bindings),
        "protocol_file_sha256": _sha256(args.protocol),
        "manifest_file_sha256": _sha256(args.manifest),
        "preoutcome_amendment_file_sha256": amendment_sha256,
        "qualification_rows": n,
        "metrics": metrics,
        "preoutcome_metrics": {
            "source_symbolic_commits": audit["source_commits"],
            "source_symbolic_commit_fraction": audit["source_commit_fraction"],
            "source_permuted_commits": audit["permuted_commits"],
            "source_permuted_commit_fraction": audit["permuted_commit_fraction"],
        },
        "gates": gates,
        "rows": row_metrics,
        "qualification_outcomes_opened_only_after_grounding_and_routes_frozen": True,
        "answers_available_to_grounder_or_harness": False,
        "official_scene_graph_or_functional_program_read": False,
        "transfer_evidence": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "qualification_rows": n,
        "metrics": metrics, "preoutcome_metrics": body["preoutcome_metrics"],
        "gates": gates, "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
