#!/usr/bin/env python3
"""Run the zero-provider-cost STAR annotation-side V38 preflight.

The input STAR receipts and outcomes were consumed by earlier development
runs.  This script therefore produces a retrospective development diagnostic,
not a new formal result.  It freezes all neural/source decisions before the
evaluation loop reads answer labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.sokoban_video_recovery import (  # noqa: E402
    exact_binomial_two_sided,
)
from motif_transfer.star_annotation_goal_relation import (  # noqa: E402
    build_harness,
    build_route,
    decide_recovery,
    relation_coverage_receipt,
    source_goal_relation_envelope,
    target_grounding,
)
from motif_transfer.unified_transfer_runtime import (  # noqa: E402
    PairedCalibration,
    TransferVerdict,
)


NEURAL = "neural_only_uniform_direct"
UNIFIED = "unified_fail_closed_source_induced"
AUTHENTIC_DIAGNOSTIC = "authentic_source_semantics_counterfactual"
TARGET_ONLY = "target_native_relation_rule"
GENERIC = "generic_always_use_typed_proof"
PERMUTED = "source_permuted_candidate_binding"
INVERTED = "source_inverted_effect"
UNIFORM_CEILING = "uniform_direct_proof_oracle_ceiling"
ALL_VIEW_CEILING = "four_policy_oracle_ceiling"
OFFICIAL_CEILING = "official_star_symbolic_executor_ceiling"
CONDITIONS = (
    NEURAL, UNIFIED, AUTHENTIC_DIAGNOSTIC, TARGET_ONLY, GENERIC,
    PERMUTED, INVERTED, UNIFORM_CEILING, ALL_VIEW_CEILING,
    OFFICIAL_CEILING,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str) -> Path:
    result = Path(value)
    return result if result.is_absolute() else REPO / result


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_lineage(config: Mapping[str, Any]) -> dict[str, str]:
    if config.get("status") != "CONSUMED_DEVELOPMENT_DIAGNOSTIC_ONLY":
        raise ValueError("STAR V38 is not marked consumed-development-only")
    observed = {
        key: _sha256(_path(str(value)))
        for key, value in config["lineage_paths"].items()
    }
    if observed != config.get("frozen_lineage"):
        mismatch = {
            key: {"expected": config.get("frozen_lineage", {}).get(key),
                  "observed": value}
            for key, value in observed.items()
            if value != config.get("frozen_lineage", {}).get(key)
        }
        raise ValueError(f"STAR V38 lineage mismatch: {mismatch}")
    return observed


def _calibration(value: Mapping[str, Any]) -> PairedCalibration:
    return PairedCalibration(
        wins=int(value["wins"]), losses=int(value["losses"]),
        ties=int(value["ties"]),
    )


def _paired(candidate: Sequence[bool], baseline: Sequence[bool]) -> dict[str, Any]:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired vectors must be nonempty and aligned")
    wins = sum(left and not right for left, right in zip(candidate, baseline))
    losses = sum(right and not left for left, right in zip(candidate, baseline))
    return {
        "n": len(candidate),
        "correct": sum(candidate),
        "accuracy": sum(candidate) / len(candidate),
        "baseline_correct": sum(baseline),
        "baseline_accuracy": sum(baseline) / len(baseline),
        "wins": wins,
        "losses": losses,
        "ties": len(candidate) - wins - losses,
        "net_wins": wins - losses,
        "accuracy_delta": (sum(candidate) - sum(baseline)) / len(candidate),
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
    }


def _policy_answer(
    direct: Mapping[str, Any], proof: Mapping[str, Any], *, use_proof: bool,
) -> str:
    return str(proof["answer"] if use_proof else direct["answer"])


def _official_executor_answers(
    annotations: Mapping[str, Mapping[str, Any]], sample_ids: Sequence[str],
    *, official_root: Path,
) -> dict[str, str]:
    executor_root = official_root / "code/program_executor"
    sys.path.insert(0, str(executor_root))
    from executor import Executor  # type: ignore  # noqa: E402

    label_root = str(official_root / "annotations/STAR_classes") + "/"
    output = {}
    for sample_id in sample_ids:
        annotation = annotations[sample_id]
        executor = Executor(annotation["situations"], label_dir=label_root)
        correct_slots = []
        for index, choice in enumerate(annotation["choices"]):
            result = executor.run(
                annotation["question_program"] + choice["choice_program"]
            )
            if result == "Correct":
                correct_slots.append(chr(ord("A") + index))
        if len(correct_slots) != 1:
            raise ValueError(
                f"official STAR program did not return one answer: {sample_id}"
            )
        output[sample_id] = correct_slots[0]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _read_object(args.config)
    lineage = _verify_lineage(config)

    source = _read_object(_path(config["source"]["artifact"]))
    confirmation = _read_object(_path(config["source"]["confirmation"]))
    envelope = source_goal_relation_envelope(
        source, confirmation,
        inducer_artifact_sha256=lineage["source_inducer"],
    )
    if envelope.contract.contract_sha256 != config["source"]["contract_sha256"]:
        raise ValueError("STAR V38 source contract drift")
    if envelope.envelope_sha256 != config["source"]["envelope_sha256"]:
        raise ValueError("STAR V38 source envelope drift")

    target_grounder_sha256 = lineage["star_adapter"]
    target_executor_sha256 = lineage["star_adapter"]
    calibration = config["prior_star_calibration"]
    route = build_route(
        source_program_sha256=envelope.contract.program_sha256,
        target_grounder_sha256=target_grounder_sha256,
        target_executor_sha256=target_executor_sha256,
        evidence_report_sha256=lineage["prior_star_report"],
        utility_vs_neural=_calibration(calibration["utility_vs_neural"]),
        authenticity_vs_source_permuted=_calibration(
            calibration["authenticity_vs_source_permuted"]
        ),
    )
    harness = build_harness(envelope, route)

    receipts_path = _path(config["target"]["receipts"])
    raw_rows = json.loads(receipts_path.read_text(encoding="utf-8"))
    annotations_path = _path(config["target"]["annotations"])
    raw_annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations = {str(row["question_id"]): row for row in raw_annotations}
    expected_rows = int(config["target"]["expected_rows"])
    expected_clusters = int(config["target"]["expected_video_clusters"])
    if len(raw_rows) != expected_rows:
        raise ValueError("STAR V38 row count drift")
    sample_ids = [str(row["sample_id"]) for row in raw_rows]
    if len(set(sample_ids)) != expected_rows or any(
        sample_id not in annotations for sample_id in sample_ids
    ):
        raise ValueError("STAR V38 sample identity mismatch")
    if len({str(row["video_id"]) for row in raw_rows}) != expected_clusters:
        raise ValueError("STAR V38 video cluster count drift")
    if any(
        bool(row.get("runtime_saw_gold_or_official_structure", True))
        or not bool(row.get("within_view_direct_and_proof_panels_identical"))
        for row in raw_rows
    ):
        raise ValueError("input STAR receipt violated its runtime boundary")

    # Build a deliberately outcome-free view.  Decisions and their aggregate
    # content hash are frozen before the evaluation loop below reads labels.
    runtime_rows = [
        {
            "sample_id": str(row["sample_id"]),
            "video_id": str(row["video_id"]),
            "family": str(row["family"]),
            "direct": row["uniform_direct"],
            "proof": row["uniform_typed_proof"],
            "active_direct": row["active_direct"],
            "active_proof": row["active_typed_proof"],
            "question_program": annotations[str(row["sample_id"])][
                "question_program"
            ],
        }
        for row in raw_rows
    ]
    predecisions = []
    for row in runtime_rows:
        proof_sha256 = stable_hash(row["proof"])
        authentic_coverage = relation_coverage_receipt(
            task_id=row["sample_id"], direct=row["direct"], proof=row["proof"],
            question_program=row["question_program"], binding_rotation=0,
        )
        permuted_coverage = relation_coverage_receipt(
            task_id=row["sample_id"], direct=row["direct"], proof=row["proof"],
            question_program=row["question_program"], binding_rotation=1,
        )
        target = target_grounding(
            contract=envelope.contract,
            target_grounder_sha256=target_grounder_sha256,
            coverage=authentic_coverage,
            proof_receipt_sha256=proof_sha256,
        )
        decision = decide_recovery(
            harness=harness, target=target,
            target_executor_sha256=target_executor_sha256,
        )
        authentic_applicable = target.applicability.structural_applicable
        permuted_applicable = all((
            permuted_coverage.functional_program_supported,
            permuted_coverage.observed_relation_delta > 0.0,
            permuted_coverage.recurrent_update_count > 0,
            permuted_coverage.terminal_relation_coverage,
            permuted_coverage.unique_native_policy_binding,
        ))
        inverted_applicable = all((
            authentic_coverage.functional_program_supported,
            authentic_coverage.observed_relation_delta < 0.0,
            authentic_coverage.terminal_relation_coverage,
            authentic_coverage.unique_native_policy_binding,
        ))
        predecisions.append({
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "direct_answer": str(row["direct"]["answer"]),
            "proof_answer": str(row["proof"]["answer"]),
            "active_direct_answer": str(row["active_direct"]["answer"]),
            "active_proof_answer": str(row["active_proof"]["answer"]),
            "authentic_coverage": asdict(authentic_coverage),
            "permuted_coverage": asdict(permuted_coverage),
            "authentic_semantic_answer": _policy_answer(
                row["direct"], row["proof"], use_proof=authentic_applicable,
            ),
            "target_only_answer": _policy_answer(
                row["direct"], row["proof"], use_proof=authentic_applicable,
            ),
            "permuted_answer": _policy_answer(
                row["direct"], row["proof"], use_proof=permuted_applicable,
            ),
            "inverted_answer": _policy_answer(
                row["direct"], row["proof"], use_proof=inverted_applicable,
            ),
            "unified_answer": _policy_answer(
                row["direct"], row["proof"],
                use_proof=decision.selected_native_policy == "uniform_typed_proof",
            ),
            "phase7_verdict": decision.phase7.verdict.value,
            "phase7_reason": decision.phase7.reason,
            "structurally_applicable": authentic_applicable,
            "current_target_outcome_read": decision.phase7.current_target_outcome_read,
            "executor_calls": decision.executor_calls,
            "target_grounding_receipt_sha256": target.receipt_sha256,
        })
    predecision_sha256 = stable_hash(predecisions)

    # Evaluation-only phase.  The public situation graph establishes the
    # symbolic ceiling; the already-consumed labels score frozen policies.
    official_answers = _official_executor_answers(
        annotations, sample_ids,
        official_root=_path(config["target"]["official_root"]),
    )
    outcomes = {str(row["sample_id"]): str(row["gold_answer"]) for row in raw_rows}
    if any(
        str(annotations[sample_id]["answer"]) != str(next(
            choice["choice"] for index, choice in enumerate(
                annotations[sample_id]["choices"]
            ) if chr(ord("A") + index) == outcomes[sample_id]
        ))
        for sample_id in sample_ids
    ):
        raise ValueError("STAR receipt labels disagree with official annotations")

    answers: dict[str, list[str]] = {name: [] for name in CONDITIONS}
    for row in predecisions:
        sample_id = row["sample_id"]
        gold = outcomes[sample_id]
        direct = row["direct_answer"]
        proof = row["proof_answer"]
        all_view = (
            direct, proof, row["active_direct_answer"], row["active_proof_answer"],
        )
        answers[NEURAL].append(direct)
        answers[UNIFIED].append(row["unified_answer"])
        answers[AUTHENTIC_DIAGNOSTIC].append(row["authentic_semantic_answer"])
        answers[TARGET_ONLY].append(row["target_only_answer"])
        answers[GENERIC].append(proof)
        answers[PERMUTED].append(row["permuted_answer"])
        answers[INVERTED].append(row["inverted_answer"])
        answers[UNIFORM_CEILING].append(gold if gold in (direct, proof) else direct)
        answers[ALL_VIEW_CEILING].append(gold if gold in all_view else direct)
        answers[OFFICIAL_CEILING].append(official_answers[sample_id])
    correct = {
        name: [answer == outcomes[sample_id] for answer, sample_id in zip(values, sample_ids)]
        for name, values in answers.items()
    }
    baseline = correct[NEURAL]
    metrics = {name: _paired(values, baseline) for name, values in correct.items()}
    source_vs_controls = {
        name: _paired(correct[AUTHENTIC_DIAGNOSTIC], correct[name])
        for name in (TARGET_ONLY, GENERIC, PERMUTED, INVERTED)
    }
    gates = config["development_gates"]
    authentic = metrics[AUTHENTIC_DIAGNOSTIC]
    gate_results = {
        "official_symbolic_executor_exact": (
            metrics[OFFICIAL_CEILING]["accuracy"]
            >= float(gates["minimum_official_executor_accuracy"])
        ),
        "target_policy_headroom_exists": (
            metrics[UNIFORM_CEILING]["net_wins"]
            >= int(gates["minimum_uniform_policy_ceiling_net_wins"])
        ),
        "authentic_minimum_net_wins": (
            authentic["net_wins"] >= int(gates["minimum_authentic_net_wins"])
        ),
        "authentic_maximum_exact_p": (
            authentic["exact_two_sided_p"] <= float(gates["maximum_authentic_exact_p"])
        ),
        "authentic_above_permuted": (
            source_vs_controls[PERMUTED]["net_wins"]
            >= int(gates["minimum_authenticity_net_wins"])
        ),
        "authentic_strictly_above_generic": (
            authentic["correct"] > metrics[GENERIC]["correct"]
        ),
        "authentic_strictly_above_target_only": (
            authentic["correct"] > metrics[TARGET_ONLY]["correct"]
        ),
        "unified_harness_authorized_execution": (
            sum(int(row["executor_calls"]) for row in predecisions)
            >= int(gates["minimum_unified_executor_calls"])
        ),
    }
    provider_cost = sum(
        float(usage.get("cost", 0.0))
        for row in raw_rows for usage in (row.get("usage") or {}).values()
    )
    failed = [name for name, passed in gate_results.items() if not passed]
    report = {
        "schema_version": "star-annotation-goal-relation-preflight-v38",
        "status": (
            "STAR_V38_DEVELOPMENT_QUALIFIED"
            if all(gate_results.values())
            else "STAR_V38_NOT_QUALIFIED_STOP"
        ),
        "conclusion": (
            "ANNOTATED_SYMBOLIC_DYNAMICS_HAVE_HEADROOM_BUT_NEURAL_SELECTION_AND_"
            "SOURCE_SPECIFICITY_ARE_NOT_VALIDATED"
        ),
        "claim_boundary": config["claim_boundary"],
        "data_status": "ALREADY_CONSUMED_RETROSPECTIVE_DEVELOPMENT",
        "rows": len(raw_rows),
        "video_clusters": len({str(row["video_id"]) for row in raw_rows}),
        "source": {
            "program_sha256": envelope.contract.program_sha256,
            "contract_sha256": envelope.contract.contract_sha256,
            "envelope_sha256": envelope.envelope_sha256,
            "ir_kind": envelope.contract.ir_kind,
            "operator_sequence": [
                asdict(value) for value in envelope.contract.operator_sequence
            ],
            "recurrent": envelope.contract.recurrent,
            "target_data_read_during_source_induction": envelope.target_data_read,
            "named_policy_template_used": envelope.named_policy_template_used,
        },
        "target_adapter": {
            "interface": route.target_interface,
            "official_functional_program_used": True,
            "official_situation_graph_used_by_runtime": False,
            "official_situation_graph_used_for_ceiling_only": True,
            "neural_grounding_source": "frozen_same-model_same-frame_V27_typed_proof_receipts",
            "predecision_sha256": predecision_sha256,
            "runtime_current_outcome_reads": sum(
                bool(row["current_target_outcome_read"]) for row in predecisions
            ),
            "structurally_applicable_rows": sum(
                bool(row["structurally_applicable"]) for row in predecisions
            ),
        },
        "prior_star_calibration": calibration,
        "condition_metrics_vs_neural": metrics,
        "authentic_source_semantics_vs_controls": source_vs_controls,
        "unified_runtime": {
            "selected_skill": sum(
                row["phase7_verdict"] == TransferVerdict.SELECT_SKILL.value
                for row in predecisions
            ),
            "executor_calls": sum(int(row["executor_calls"]) for row in predecisions),
            "reasons": dict(Counter(row["phase7_reason"] for row in predecisions)),
        },
        "development_gate_results": gate_results,
        "failed_gates": failed,
        "all_development_gates_passed": all(gate_results.values()),
        "cost": {
            "incremental_external_provider_calls": 0,
            "incremental_external_provider_cost_usd": 0.0,
            "historical_receipt_cost_usd_not_recharged": provider_cost,
        },
        "next_action": (
            "STOP_NATURAL_VIDEO_EXTENSION; do not open fresh STAR or spend more "
            "API budget until an oracle-free target-native grounder passes a "
            "new consumed-development qualification."
        ),
        "artifacts": {
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "source_artifact_file_sha256": lineage["source_artifact"],
            "source_confirmation_file_sha256": lineage["source_confirmation"],
            "source_inducer_sha256": lineage["source_inducer"],
            "star_adapter_sha256": lineage["star_adapter"],
            "runner_sha256": lineage["runner"],
            "prior_star_report_sha256": lineage["prior_star_report"],
            "input_receipts_sha256": lineage["star_receipts"],
            "official_annotations_sha256": lineage["star_annotations"],
        },
    }
    report["report_sha256"] = stable_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "condition_correct": {
            name: value["correct"] for name, value in metrics.items()
        },
        "failed_gates": failed,
        "unified_runtime": report["unified_runtime"],
        "incremental_cost_usd": 0.0,
        "output": str(args.output.resolve()),
        "output_file_sha256": _sha256(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
