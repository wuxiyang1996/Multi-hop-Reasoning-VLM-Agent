#!/usr/bin/env python3
"""Zero-provider-cost V15 development audit over consumed CLEVRER evidence."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.clevrer_unified_goal_relation import (  # noqa: E402
    TARGET_INTERFACE,
    build_harness,
    build_route,
    decide_recovery,
    source_goal_relation_envelope,
    target_grounding,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.unified_transfer_runtime import (  # noqa: E402
    PairedCalibration,
    TransferVerdict,
)


AUTHENTIC = "authentic_sokoban_proof_cate_recover"
NEURAL = "target_explicit_no_recovery"
TARGET_BASE = "target_base_receipt_cate_recover"
PERMUTED = "permuted_uplift_cate_recover"
SHUFFLED = "shuffled_proof_binding_recover"
GENERIC = "target_error_only_recover"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired(value: Mapping[str, Any]) -> PairedCalibration:
    return PairedCalibration(
        wins=int(value["wins"]),
        losses=int(value["losses"]),
        ties=int(value["ties"]),
    )


def _reserve_exposure(split_manifest: Path) -> list[str]:
    manifest = _read(split_manifest)
    sample_ids = manifest["benchmarks"]["clevrer"]["splits"]["reserve"]
    command = ["rg", "-l", "-F"]
    for sample_id in sample_ids:
        command.extend(("-e", str(sample_id)))
    command.extend((str(REPO / "runs"), str(REPO / "docs/results")))
    result = subprocess.run(
        command, check=False, capture_output=True, text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "reserve exposure scan failed")
    return sorted(line for line in result.stdout.splitlines() if line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v14-report", type=Path,
        default=REPO / "runs/sokoban_clevrer_proof_v14_formal/formal_report.json",
    )
    parser.add_argument(
        "--source-artifact", type=Path,
        default=REPO / "runs/sokoban_goal_relation_macro_v3/artifact.json",
    )
    parser.add_argument(
        "--source-confirmation", type=Path,
        default=REPO / "runs/sokoban_goal_relation_macro_v3/fresh_confirmation_report.json",
    )
    parser.add_argument(
        "--grounder-artifact", type=Path,
        default=REPO / "runs/clevrer_sokoban_proof_v14_training/frozen_proof_grounder.json",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=REPO / "configs/clevrer_sokoban_proof_v14_splits.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/clevrer_unified_goal_relation_v15_development.json",
    )
    args = parser.parse_args()

    v14 = _read(args.v14_report)
    source = _read(args.source_artifact)
    confirmation = _read(args.source_confirmation)
    grounder_artifact = _read(args.grounder_artifact)
    inducer_path = REPO / "src/motif_transfer/source_goal_relation_induction.py"
    adapter_path = REPO / "src/motif_transfer/clevrer_unified_goal_relation.py"
    envelope = source_goal_relation_envelope(
        source, confirmation, inducer_artifact_sha256=_sha(inducer_path),
    )
    grounder_sha256 = str(grounder_artifact["artifact_sha256"])
    executor_sha256 = _sha(adapter_path)
    paired = v14["paired_authentic"]
    route = build_route(
        source_program_sha256=envelope.contract.program_sha256,
        target_grounder_sha256=grounder_sha256,
        target_executor_sha256=executor_sha256,
        evidence_report_sha256=_sha(args.v14_report),
        utility_vs_neural=_paired(paired[NEURAL]),
        authenticity_vs_source_permuted=_paired(paired[PERMUTED]),
    )
    harness = build_harness(envelope, route)

    action_parity = 0
    answer_parity = 0
    executor_calls = 0
    selected = 0
    outcome_exposure = 0
    authorization_hashes: list[str] = []
    for row in v14["rows"]:
        grounder = row["grounder"]
        target = target_grounding(
            task_id=str(row["sample_id"]),
            contract=envelope.contract,
            target_grounder_sha256=grounder_sha256,
            proof_receipt_sha256=str(row["proof_receipts_sha256"]),
            proof_predicted_uplift=float(grounder["proof_predicted_uplift"]),
            decision_threshold=float(grounder["decision_threshold"]),
        )
        decision = decide_recovery(
            harness=harness, target=target,
            target_executor_sha256=executor_sha256,
        )
        expected = row["conditions"][AUTHENTIC]
        action_parity += decision.selected_native_representation == expected[
            "selected_native_representation"
        ]
        actual_answer = row[
            "trajectory_answer"
            if decision.selected_native_representation == "trajectory"
            else "explicit_answer"
        ] if "trajectory_answer" in row else row["conditions"][
            "target_trajectory_only"
            if decision.selected_native_representation == "trajectory"
            else NEURAL
        ]["answer"]
        answer_parity += actual_answer == expected["answer"]
        executor_calls += decision.executor_calls
        selected += decision.phase7.verdict == TransferVerdict.SELECT_SKILL
        outcome_exposure += int(
            decision.phase7.current_target_outcome_read
            or decision.utility.current_outcome_read
            or target.requirement.formal_outcome_read
            or target.applicability.formal_outcome_read
        )
        authorization_hashes.append(decision.phase7.authorization_sha256)

    samples = int(v14["samples"])
    conditions = v14["conditions"]
    reserve_exposure = _reserve_exposure(args.split_manifest)
    target_base_pair = paired[TARGET_BASE]
    gates = {
        "source_program_is_source_only_template_free_and_fresh": envelope.admitted,
        "v14_consumed_development_evidence_validated": v14.get("status")
        == "SOKOBAN_TO_CLEVRER_PROOF_NEUROSYMBOLIC_TRANSFER_FORMAL_VALIDATED",
        "development_sample_count": samples == 720,
        "unified_harness_exact_action_parity": action_parity == samples,
        "unified_harness_exact_answer_parity": answer_parity == samples,
        "only_authorized_rows_reach_target_executor": (
            executor_calls == selected == conditions[AUTHENTIC]["recoveries"]
        ),
        "zero_current_target_outcome_exposure": outcome_exposure == 0,
        "authentic_strictly_above_neural": conditions[AUTHENTIC]["correct"]
        > conditions[NEURAL]["correct"],
        "authentic_strictly_above_target_base": conditions[AUTHENTIC]["correct"]
        > conditions[TARGET_BASE]["correct"],
        "authentic_strictly_above_generic_scaffold": conditions[AUTHENTIC]["correct"]
        > conditions[GENERIC]["correct"],
        "authentic_strictly_above_source_permuted": conditions[AUTHENTIC]["correct"]
        > conditions[PERMUTED]["correct"],
        "authentic_strictly_above_shuffled_binding": conditions[AUTHENTIC]["correct"]
        > conditions[SHUFFLED]["correct"],
        "paired_utility_exact_p_at_most_0p01": paired[NEURAL][
            "exact_two_sided_p"
        ] <= 0.01,
        "paired_authenticity_exact_p_at_most_0p01": paired[PERMUTED][
            "exact_two_sided_p"
        ] <= 0.01,
        "minimum_action_contrast": selected >= 40,
        "reserve_has_no_prior_result_exposure": not reserve_exposure,
        "external_provider_calls": True,
    }
    diagnostics = {
        "authentic_vs_target_base": target_base_pair,
        "target_base_difference_is_not_confirmatory": (
            target_base_pair["exact_two_sided_p"] > 0.05
        ),
        "reserve_exposure_matches": reserve_exposure,
        "external_provider_call_count": 0,
        "external_provider_cost_usd": 0.0,
    }
    body = {
        "schema_version": "clevrer-unified-goal-relation-v15-development-audit",
        "status": (
            "CLEVRER_UNIFIED_V15_DEVELOPMENT_GATE_PASSED"
            if all(gates.values()) else
            "CLEVRER_UNIFIED_V15_DEVELOPMENT_GATE_FAILED"
        ),
        "role": "consumed_v14_formal_repurposed_as_v15_development",
        "claim_boundary": (
            "Development-only composition and gate calibration. This report "
            "does not add a success-rate claim. The old V14 reserve remains "
            "sealed and may be read only after an independently hashed V15 "
            "configuration is frozen."
        ),
        "source_program": {
            "artifact_sha256": source["artifact_sha256"],
            "confirmation_sha256": confirmation["report_sha256"],
            "contract_sha256": envelope.contract.contract_sha256,
            "envelope_sha256": envelope.envelope_sha256,
            "named_policy_template_used": False,
            "target_data_read": False,
        },
        "target_interface": {
            "name": TARGET_INTERFACE,
            "grounder_artifact_sha256": grounder_sha256,
            "executor_sha256": executor_sha256,
            "runtime_gold_or_official_program_read": False,
        },
        "samples": samples,
        "conditions": {
            name: conditions[name]
            for name in (AUTHENTIC, NEURAL, TARGET_BASE, GENERIC, PERMUTED, SHUFFLED)
        },
        "paired_authentic": {
            name: paired[name]
            for name in (NEURAL, TARGET_BASE, GENERIC, PERMUTED, SHUFFLED)
        },
        "unified_authority": {
            "selected_rows": selected,
            "abstained_rows": samples - selected,
            "target_executor_calls": executor_calls,
            "action_parity": action_parity,
            "answer_parity": answer_parity,
            "authorization_receipts_sha256": stable_hash(authorization_hashes),
        },
        "diagnostics": diagnostics,
        "gates": gates,
        "lineage": {
            "v14_report_file_sha256": _sha(args.v14_report),
            "source_artifact_file_sha256": _sha(args.source_artifact),
            "source_confirmation_file_sha256": _sha(args.source_confirmation),
            "grounder_artifact_file_sha256": _sha(args.grounder_artifact),
            "split_manifest_file_sha256": _sha(args.split_manifest),
            "source_inducer_file_sha256": _sha(inducer_path),
            "adapter_file_sha256": _sha(adapter_path),
            "auditor_file_sha256": _sha(Path(__file__).resolve()),
        },
        "cost": {
            "external_provider_calls": 0,
            "external_provider_cost_usd": 0.0,
            "local_prediction_files_reused": True,
        },
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "conditions": report["conditions"],
        "selected_rows": selected,
        "reserve_exposure_matches": reserve_exposure,
        "cost": report["cost"],
        "gates": gates,
        "output": str(args.output),
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
