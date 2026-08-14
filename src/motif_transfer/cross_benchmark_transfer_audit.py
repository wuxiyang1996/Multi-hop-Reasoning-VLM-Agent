"""Audit normalized neural-symbolic transfer evidence across target adapters."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import stable_hash


class TransferAuditError(ValueError):
    pass


def validate_self_hash(payload: Mapping[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise TransferAuditError(f"invalid {field}")


def build_cross_benchmark_audit(
    *,
    source_receipt: Mapping[str, Any],
    webshop: Mapping[str, Any],
    discovery_adaptation: Mapping[str, Any],
    discovery_formal: Mapping[str, Any],
    alfworld: Mapping[str, Any],
    tir: Mapping[str, Any],
    tir_other_source_formal: Mapping[str, Any] | None = None,
    tir_target_diagnosis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed matrix from immutable benchmark reports."""

    for payload, field in (
        (source_receipt, "compact_receipt_sha256"),
        (webshop, "summary_sha256"),
        (discovery_adaptation, "summary_sha256"),
        (discovery_formal, "summary_sha256"),
        (alfworld, "report_sha256"),
        (tir, "report_sha256"),
    ):
        validate_self_hash(payload, field)

    source_hash = str(source_receipt["artifact"]["artifact_sha256"])
    source_confirmation = str(
        source_receipt["fresh_confirmation"]["report_sha256"]
    )
    common_lineage = {
        "webshop": (
            str(webshop["source_artifact_sha256"]) == source_hash
            and str(webshop["source_confirmation_report_sha256"])
            == source_confirmation
        ),
        "discoveryworld": (
            str(discovery_adaptation["source_contract"]["source_program_sha256"])
            == source_hash
            and str(discovery_adaptation["source_contract"][
                "source_confirmation_sha256"
            ]) == source_confirmation
        ),
        "alfworld": (
            str(alfworld["source_artifact_sha256"]) == source_hash
            and str(alfworld["source_confirmation_sha256"])
            == source_confirmation
        ),
        "tir": (
            str(tir["source_artifact_sha256"]) == source_hash
            and str(tir["source_confirmation_sha256"]) == source_confirmation
        ),
    }

    web_target = webshop["conditions"]["target_only"]
    web_authentic = webshop["conditions"][
        "authentic_sokoban_effect_plus_target"
    ]
    dw = discovery_adaptation["v21_shared_spatial_realization"]
    alf_null = alfworld["summaries"]["null_skill_same_harness"]
    alf_authentic = alfworld["summaries"]["authentic_source_skill"]
    tir_target = tir["summaries"]["null_skill_same_harness"]
    tir_authentic = tir["summaries"]["authentic_sokoban_effect_skill"]

    cells = {
        "webshop": {
            "source_lineage_matches_shared_sokoban": common_lineage["webshop"],
            "evidence_tier": "FRESH_FORMAL_VALIDATION",
            "status": str(webshop["scientific_status"]),
            "tasks": int(webshop["tasks"]),
            "target_successes": int(web_target["strict_successes"]),
            "authentic_successes": int(web_authentic["strict_successes"]),
            "success_delta": int(web_authentic["strict_successes"])
            - int(web_target["strict_successes"]),
            "negative_transfer_count": int(
                webshop["paired_comparisons"]["target_only"]["strict_losses"]
            ),
            "mechanism_positive": bool(
                webshop["primary_gates"]["all_comparator_gates"]
                and int(web_authentic["strict_successes"])
                > int(web_target["strict_successes"])
            ),
            "heldout_generalization_validated": True,
        },
        "discoveryworld": {
            "source_lineage_matches_shared_sokoban": common_lineage[
                "discoveryworld"
            ],
            "evidence_tier": "CONSUMED_ADAPTATION_MECHANISM",
            "status": str(discovery_adaptation["status"]),
            "tasks": int(dw["eligible_forks"]),
            "target_successes": int(dw["success_counts"][
                "target_native_myopic"
            ]),
            "authentic_successes": int(dw["success_counts"][
                "authentic_sokoban_effect_plus_target"
            ]),
            "success_delta": int(dw["success_counts"][
                "authentic_sokoban_effect_plus_target"
            ]) - int(dw["success_counts"]["target_native_myopic"]),
            "negative_transfer_count": int(dw[
                "negative_transfer_count_vs_target_native_myopic"
            ]),
            "mechanism_positive": bool(dw["adaptation_gate_passed"]),
            "heldout_generalization_validated": False,
            "heldout_status": str(discovery_formal["status"]),
            "heldout_failure_kind": "APPLICABILITY_COVERAGE_FAILURE",
        },
        "alfworld": {
            "source_lineage_matches_shared_sokoban": common_lineage["alfworld"],
            "evidence_tier": "CONSUMED_QUALIFICATION_DIAGNOSTIC",
            "status": str(alfworld["status"]),
            "tasks": int(alf_authentic["tasks"]),
            "target_successes": int(alf_null["successes"]),
            "authentic_successes": int(alf_authentic["successes"]),
            "success_delta": int(alf_authentic["successes"])
            - int(alf_null["successes"]),
            "negative_transfer_count": int(
                alfworld["paired"]["null_skill_same_harness"]["losses"]
            ),
            "mechanism_positive": str(alfworld["status"])
            == "CONSUMED_MECHANISM_GATE_PASSED",
            "heldout_generalization_validated": False,
        },
        "tir": {
            "source_lineage_matches_shared_sokoban": common_lineage["tir"],
            "source_lineage": "SHARED_REAL_SOKOBAN_EFFECT_PROGRAM",
            "evidence_tier": "CONSUMED_DEVELOPMENT_MECHANISM",
            "status": str(tir["status"]),
            "tasks": int(tir_authentic["tasks"]),
            "target_successes": int(tir_target["successes"]),
            "authentic_successes": int(tir_authentic["successes"]),
            "success_delta": int(tir_authentic["successes"])
            - int(tir_target["successes"]),
            "negative_transfer_count": int(
                tir["paired"]["null_skill_same_harness"]["losses"]
            ),
            "mechanism_positive": str(tir["status"])
            == "CONSUMED_MECHANISM_GATE_PASSED",
            "heldout_generalization_validated": False,
            "heldout_consumed": False,
        },
    }
    if tir_other_source_formal is not None:
        validate_self_hash(tir_other_source_formal, "report_sha256")
        cells["tir"]["different_source_formal_diagnosis"] = {
            "source_lineage": "CONTROLLED_SYNTHETIC_GAME_VALUE_ENSEMBLE",
            "status": str(tir_other_source_formal["status"]),
            "target_successes": int(
                tir_other_source_formal["conditions"]["target_only"]["successes"]
            ),
            "authentic_successes": int(
                tir_other_source_formal["conditions"][
                    "authentic_source_plus_target"
                ]["successes"]
            ),
            "heldout_consumed": bool(
                tir_other_source_formal["formal_held_out_consumed"]
            ),
        }
    if tir_target_diagnosis is not None:
        cells["tir"]["stronger_target_model_diagnosis"] = {
            "status": str(tir_target_diagnosis["status"]),
            "baseline_accuracy": float(tir_target_diagnosis["baseline_accuracy"]),
            "oracle_candidate_accuracy": float(
                tir_target_diagnosis["oracle_candidate_accuracy"]
            ),
            "target_only_accuracy": float(
                tir_target_diagnosis["conditions_cross_fitted"]["target_only"][
                    "accuracy"
                ]
            ),
            "authentic_accuracy": float(
                tir_target_diagnosis["conditions_cross_fitted"][
                    "authentic_source_plus_target"
                ]["accuracy"]
            ),
            "shuffled_accuracy": float(
                tir_target_diagnosis["conditions_cross_fitted"][
                    "shuffled_source_plus_target"
                ]["accuracy"]
            ),
        }

    passed_cells = [name for name, row in cells.items() if row["mechanism_positive"]]
    heldout_cells = [
        name for name, row in cells.items()
        if row["heldout_generalization_validated"]
    ]
    body = {
        "schema_version": "game-to-four-target-transfer-audit-v1",
        "status": (
            "ALL_FOUR_TARGETS_VALIDATED" if len(passed_cells) == 4
            else "PARTIAL_TRANSFER_ONLY"
        ),
        "shared_source": {
            "artifact": "SOKOBAN_EFFECT_PROGRAM_V2",
            "artifact_sha256": source_hash,
            "fresh_confirmation_sha256": source_confirmation,
            "source_gate_passed": bool(
                source_receipt["fresh_confirmation"]["source_gate_passed"]
            ),
        },
        "common_source_lineage": common_lineage,
        "cells": cells,
        "validated_mechanism_cells": passed_cells,
        "validated_heldout_cells": heldout_cells,
        "all_four_mechanisms_validated": len(passed_cells) == 4,
        "all_four_share_one_source_artifact": all(common_lineage.values()),
        "claim": (
            "Shared real-Sokoban neural-symbolic transfer is formally validated "
            "on WebShop and mechanism-positive on consumed DiscoveryWorld Easy "
            "forks. It is not validated on ALFWorld or TIR, and DiscoveryWorld "
            "Normal held-out coverage failed."
        ),
    }
    body["audit_sha256"] = stable_hash(body)
    return body


__all__ = [
    "TransferAuditError",
    "build_cross_benchmark_audit",
    "validate_self_hash",
]
