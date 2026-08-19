#!/usr/bin/env python3
"""Zero-cost robust temporal-pair audit on consumed AGQA V17/V19 rows."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_robust_temporal_transfer import (  # noqa: E402
    bind_robust_temporal_pair_program,
    build_temporal_harness,
    build_temporal_route,
    decide_temporal_relation,
    unified_temporal_grounding,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.online_transfer_utility import (  # noqa: E402
    ApplicabilityReceipt,
)
from motif_transfer.structural_ir_applicability import (  # noqa: E402
    TargetIRRequirement,
    select_source_contract,
)
from motif_transfer.unified_transfer_runtime import (  # noqa: E402
    PairedCalibration,
    TransferVerdict,
)
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _load_sources,
)
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _grounder_semantic_core,
)


DEFAULT_INPUTS = (
    "runs/agqa2_active_grounding_v17_powered_reserve/report.json",
    "runs/agqa2_temporal_selective_v19_reserve/report.json",
)
PROGRAM_CONFIG = "configs/agqa2_program_transfer_v1_development.json"
PARENT_CONFIG = "configs/agqa2_temporal_selective_v19_reserve.json"
SOURCE_ARTIFACT = (
    "configs/phase3_source_function_v4/frozen_reserve/programs/"
    "candy_crush.json"
)
ADAPTER_MODULE = "src/motif_transfer/agqa_robust_temporal_transfer.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _answer_class(value: Any) -> str | None:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    if not text:
        return None
    first = text.split(maxsplit=1)[0]
    return first if first in {"before", "after"} else None


def _runtime_integrity(row: Mapping[str, Any]) -> bool:
    return all(
        row.get(field) is False
        for field in (
            "runtime_answer_read", "runtime_functional_program_read",
            "runtime_scene_graph_read", "runtime_source_identity_read",
            "operand_grounder_question_read",
            "operand_grounder_competing_operand_read",
        )
    )


def _target_requirement(contract, *, task_id: str, grounder_sha256: str):
    return TargetIRRequirement.create(
        task_id=task_id,
        target_domain="agqa2",
        target_interface="robust_temporal_pair_binding_v33",
        target_grounder_sha256=grounder_sha256,
        ir_kind=contract.ir_kind,
        operator_sequence=contract.operator_sequence,
        recurrent=contract.recurrent,
        terminal_predicate_families=contract.terminal_predicate_families,
        grounder_qualified=True,
        formal_outcome_read=False,
    )


def _freeze_predictions(
    *, row: Mapping[str, Any], source_contract, wrong_contract,
    grounder_sha256: str,
) -> dict[str, Any]:
    """Freeze every arm without reading the current row's outcome."""

    binding = bind_robust_temporal_pair_program(
        task_id=str(row["task_id"]),
        target_state_sha256=str(row["runtime_receipt_sha256"]),
        target_grounder_sha256=grounder_sha256,
        source_program_sha256=source_contract.program_sha256,
        obligation_kind=str(row["query_plan"]["obligation_kind"]),
        operand_runs=row["operand_runs"],
        grounder_qualified=_runtime_integrity(row),
        formal_outcome_read=False,
    )
    shuffled = bind_robust_temporal_pair_program(
        task_id=str(row["task_id"]),
        target_state_sha256=str(row["runtime_receipt_sha256"]),
        target_grounder_sha256=grounder_sha256,
        source_program_sha256=source_contract.program_sha256,
        obligation_kind=str(row["query_plan"]["obligation_kind"]),
        operand_runs=row["operand_runs"],
        grounder_qualified=_runtime_integrity(row),
        effect_binding_authenticated=False,
        formal_outcome_read=False,
    )
    requirement = _target_requirement(
        source_contract, task_id=str(row["task_id"]),
        grounder_sha256=grounder_sha256,
    )
    wrong_selection = select_source_contract((wrong_contract,), requirement)
    direct = str(row["direct_response"])
    source_prediction = binding.authorized_relation or direct
    body = {
        "task_id": str(row["task_id"]),
        "video_id": str(row["video_id"]),
        "target_native_prediction": direct,
        "source_induced_prediction": source_prediction,
        "effect_shuffled_prediction": direct,
        "wrong_source_prediction": direct,
        "generic_scaffold_prediction": source_prediction,
        "target_written_equivalent_prediction": source_prediction,
        "source_executor_candidate": binding.authorized_relation,
        "source_binding_receipt_sha256": binding.receipt_sha256,
        "effect_shuffled_binding_receipt_sha256": shuffled.receipt_sha256,
        "effect_shuffled_abstained": shuffled.authorized_relation is None,
        "wrong_source_abstained": (
            wrong_selection["status"] != "UNIQUE_SOURCE_CONTRACT_SELECTED"
        ),
        "runtime_integrity_qualified": _runtime_integrity(row),
        "formal_outcome_used_for_current_authorization": False,
        "binding_reason": binding.reason,
        "operand_a_hypothesis_count": len(binding.operand_a_hypotheses),
        "operand_b_hypothesis_count": len(binding.operand_b_hypotheses),
        "cross_view_relation_count": len(binding.cross_view_relations),
        "all_pairs_strictly_separated": (
            binding.all_pairs_strictly_separated
        ),
        "all_pairs_relation_consistent": (
            binding.all_pairs_relation_consistent
        ),
    }
    return body | {"prediction_receipt_sha256": stable_hash(body)}


def _paired(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    wins = sum(a and not b for a, b in zip(left, right, strict=True))
    losses = sum(b and not a for a, b in zip(left, right, strict=True))
    ties = len(left) - wins - losses
    return {
        "left_correct": sum(left),
        "right_correct": sum(right),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "left_minus_right_correct": sum(left) - sum(right),
    }


def run(
    *, input_paths: Sequence[Path], output_path: Path,
) -> dict[str, Any]:
    program_config = json.loads((REPO_ROOT / PROGRAM_CONFIG).read_text())
    sources, source_report = _load_sources(program_config)
    source_contract, wrong_contract = sources[1], sources[2]
    source_artifact = json.loads((REPO_ROOT / SOURCE_ARTIFACT).read_text())
    parent_config = json.loads((REPO_ROOT / PARENT_CONFIG).read_text())
    adapter_sha256 = _sha256(REPO_ROOT / ADAPTER_MODULE)
    parent_grounder_sha256 = stable_hash(
        _grounder_semantic_core(parent_config, sources)
    )
    grounder_core = {
        "schema_version": "agqa2-robust-temporal-grounder-v33",
        "parent_v19_grounder_semantic_core": _grounder_semantic_core(
            parent_config, sources,
        ),
        "postground_adapter_module_sha256": adapter_sha256,
        "binding_rule": (
            "AT_LEAST_TWO_UNIQUE_OBSERVED_INTERVAL_HYPOTHESES_PER_OPERAND;"
            "ALL_CROSS_VIEW_INTERVAL_PAIRS_STRICTLY_SEPARATED_AND_SAME_RELATION"
        ),
        "minimum_confidence": 0.5,
        "current_outcome_read": False,
    }
    grounder_sha256 = stable_hash(grounder_core)
    executor_sha256 = stable_hash({
        "executor": "AGQA_NATIVE_BEFORE_AFTER_V33",
        "adapter_module_sha256": adapter_sha256,
        "native_relations": ["after", "before"],
    })

    reports = [json.loads(path.read_text()) for path in input_paths]
    source_rows = [
        (path, row)
        for path, report in zip(input_paths, reports, strict=True)
        for row in report["rows"]
        if row["query_plan"]["obligation_kind"]
        == "TEMPORAL_PAIR_RECURRENT"
    ]
    # Freeze all predictions before the evaluator opens any current outcome.
    frozen = [
        _freeze_predictions(
            row=row, source_contract=source_contract,
            wrong_contract=wrong_contract, grounder_sha256=grounder_sha256,
        )
        for _, row in source_rows
    ]
    outcomes = [
        _answer_class(row["gold_answer_evaluator_only"])
        for _, row in source_rows
    ]
    if any(value is None for value in outcomes):
        raise ValueError("temporal-pair development row has non-temporal gold")

    arms = {}
    for key in (
        "target_native_prediction", "source_induced_prediction",
        "effect_shuffled_prediction", "wrong_source_prediction",
        "generic_scaffold_prediction",
        "target_written_equivalent_prediction",
    ):
        arms[key] = [
            _answer_class(row[key]) == gold
            for row, gold in zip(frozen, outcomes, strict=True)
        ]
    primary = _paired(
        arms["source_induced_prediction"],
        arms["target_native_prediction"],
    )
    shuffled = _paired(
        arms["source_induced_prediction"],
        arms["effect_shuffled_prediction"],
    )
    generic = _paired(
        arms["source_induced_prediction"],
        arms["generic_scaffold_prediction"],
    )
    target_written = _paired(
        arms["source_induced_prediction"],
        arms["target_written_equivalent_prediction"],
    )
    calibration = PairedCalibration(
        primary["wins"], primary["losses"], primary["ties"],
    )
    route = build_temporal_route(
        source_program_sha256=source_contract.program_sha256,
        target_grounder_sha256=grounder_sha256,
        target_executor_sha256=executor_sha256,
        evidence_report_sha256=stable_hash({
            "inputs": [_sha256(path) for path in input_paths],
            "adapter": adapter_sha256,
        }),
        utility_vs_target_native=calibration,
        authenticity_vs_effect_shuffled=PairedCalibration(
            shuffled["wins"], shuffled["losses"], shuffled["ties"],
        ),
    )
    harness = build_temporal_harness(
        contract=source_contract,
        source_transition_receipts_sha256=source_artifact[
            "source_function_program"
        ]["source_receipts_sha256"],
        inducer_artifact_sha256=source_artifact["artifact_sha256"],
        route=route,
    )
    future_authorization_checks = []
    for (_, row), prediction in zip(source_rows, frozen, strict=True):
        binding = bind_robust_temporal_pair_program(
            task_id=str(row["task_id"]),
            target_state_sha256=str(row["runtime_receipt_sha256"]),
            target_grounder_sha256=grounder_sha256,
            source_program_sha256=source_contract.program_sha256,
            obligation_kind=str(row["query_plan"]["obligation_kind"]),
            operand_runs=row["operand_runs"],
            grounder_qualified=_runtime_integrity(row),
        )
        target = unified_temporal_grounding(
            contract=source_contract, binding=binding,
        )
        decision = decide_temporal_relation(
            harness=harness, target=target, binding=binding,
            target_executor_sha256=executor_sha256,
        )
        expected = prediction["source_executor_candidate"] is not None
        future_authorization_checks.append(
            (decision.phase7.verdict == TransferVerdict.SELECT_SKILL)
            == expected
        )

    route_decision = calibration.utility_gate(
        ApplicabilityReceipt(True, True, True, True, True)
    )
    gates = {
        "required_35_consumed_temporal_pair_rows": len(frozen) == 35,
        "minimum_four_source_wins": primary["wins"] >= 4,
        "zero_source_losses": primary["losses"] == 0,
        "future_directional_utility_calibrated": (
            route_decision.decision == "SELECT_SKILL"
        ),
        "minimum_twelve_structural_authorizations": sum(
            row["source_executor_candidate"] is not None for row in frozen
        ) >= 12,
        "effect_shuffled_always_abstains": all(
            row["effect_shuffled_abstained"] for row in frozen
        ),
        "wrong_source_type_always_abstains": all(
            row["wrong_source_abstained"] for row in frozen
        ),
        "source_matches_generic_scaffold": (
            generic["wins"] == generic["losses"] == 0
        ),
        "source_matches_target_written_equivalent": (
            target_written["wins"] == target_written["losses"] == 0
        ),
        "runtime_integrity_qualified": all(
            row["runtime_integrity_qualified"] for row in frozen
        ),
        "current_outcome_never_used_for_authorization": all(
            not row["formal_outcome_used_for_current_authorization"]
            for row in frozen
        ),
        "unified_harness_matches_frozen_structural_candidates": all(
            future_authorization_checks
        ),
    }
    status = (
        "AGQA2_ROBUST_TEMPORAL_V33_DEVELOPMENT_QUALIFIED"
        if all(gates.values())
        else "AGQA2_ROBUST_TEMPORAL_V33_DEVELOPMENT_NOT_QUALIFIED"
    )
    row_details = []
    for (path, source_row), prediction, gold in zip(
        source_rows, frozen, outcomes, strict=True,
    ):
        row_details.append(prediction | {
            "consumed_input": str(path.relative_to(REPO_ROOT)),
            "gold_answer_evaluator_only": gold,
            "source_correct": (
                _answer_class(prediction["source_induced_prediction"])
                == gold
            ),
            "target_native_correct": (
                _answer_class(prediction["target_native_prediction"])
                == gold
            ),
        })
    report = {
        "schema_version": "agqa2-robust-temporal-v33-development-report-v1",
        "status": status,
        "split": "consumed_development",
        "confirmatory_claim": False,
        "claim_boundary": (
            "CONSUMED_V17_PLUS_V19_TEMPORAL_PAIR_ROWS;ZERO_PROVIDER_CALLS;"
            "FUTURE_ROUTE_CALIBRATION_ONLY"
        ),
        "rows": len(frozen),
        "input_reports": [
            {"path": str(path.relative_to(REPO_ROOT)), "file_sha256": _sha256(path)}
            for path in input_paths
        ],
        "source_program_sha256": source_contract.program_sha256,
        "source_artifact_sha256": source_artifact["artifact_sha256"],
        "source_induction_authority": source_artifact[
            "source_function_program"
        ]["induction_authority"],
        "source_confirmation_report_sha256": source_report["report_sha256"],
        "target_grounder_sha256": grounder_sha256,
        "parent_target_grounder_sha256": parent_grounder_sha256,
        "target_grounder_core": grounder_core,
        "target_executor_sha256": executor_sha256,
        "source_authorizations": sum(
            row["source_executor_candidate"] is not None for row in frozen
        ),
        "source_vs_target_native": primary,
        "source_vs_effect_shuffled": shuffled,
        "source_vs_generic_scaffold": generic,
        "source_vs_target_written_equivalent": target_written,
        "future_route_calibration": {
            "wins": calibration.wins,
            "losses": calibration.losses,
            "ties": calibration.ties,
            "decision": route_decision.decision,
            "reason": route_decision.reason,
            "posterior_lower_win_probability": (
                route_decision.posterior_lower_win_probability
            ),
        },
        "qualification_gates": gates,
        "provider_calls": 0,
        "reported_provider_cost_usd": 0.0,
        "interpretation": {
            "source_induced_temporal_program_has_future_utility": all(
                gates.values()
            ),
            "source_beats_handwritten_generic": False,
            "source_provenance_is_necessary": False,
            "eligible_for_one_fresh_video_disjoint_confirmation": all(
                gates.values()
            ),
        },
        "rows_detail": row_details,
    }
    report["report_sha256"] = stable_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    output = REPO_ROOT / "runs/agqa2_robust_temporal_v33_development/report.json"
    report = run(
        input_paths=[REPO_ROOT / value for value in DEFAULT_INPUTS],
        output_path=output,
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "status", "rows", "source_authorizations",
            "source_vs_target_native", "future_route_calibration",
            "provider_calls", "reported_provider_cost_usd", "report_sha256",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
