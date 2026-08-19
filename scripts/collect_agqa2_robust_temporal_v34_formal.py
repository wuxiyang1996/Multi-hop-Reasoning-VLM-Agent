#!/usr/bin/env python3
"""Collect and evaluate the frozen AGQA V34 robust temporal formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    exact_one_sided_pvalue,
)
from motif_transfer.agqa_robust_temporal_transfer import (  # noqa: E402
    bind_robust_temporal_pair_program,
    build_temporal_harness,
    build_temporal_route,
    decide_temporal_relation,
    unified_temporal_grounding,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_ir_applicability import (  # noqa: E402
    select_source_contract,
)
from motif_transfer.unified_transfer_runtime import (  # noqa: E402
    PairedCalibration,
)
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _load_sources,
)
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    collect as collect_base,
)


SOURCE_ARTIFACT = (
    "configs/phase3_source_function_v4/frozen_reserve/programs/"
    "candy_crush.json"
)


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
        "exact_one_sided_pvalue": exact_one_sided_pvalue(
            source_wins=wins, source_losses=losses,
        ),
    }


def _freeze_row(
    *, row: Mapping[str, Any], source_contract, wrong_contract,
    harness, target_grounder_sha256: str, target_executor_sha256: str,
) -> dict[str, Any]:
    """Freeze all predictions without opening the current row's outcome."""

    arguments = {
        "task_id": str(row["task_id"]),
        "target_state_sha256": str(row["runtime_receipt_sha256"]),
        "target_grounder_sha256": target_grounder_sha256,
        "source_program_sha256": source_contract.program_sha256,
        "obligation_kind": str(row["query_plan"]["obligation_kind"]),
        "operand_runs": row["operand_runs"],
        "grounder_qualified": _runtime_integrity(row),
        "formal_outcome_read": False,
    }
    binding = bind_robust_temporal_pair_program(**arguments)
    target = unified_temporal_grounding(
        contract=source_contract, binding=binding,
    )
    decision = decide_temporal_relation(
        harness=harness, target=target, binding=binding,
        target_executor_sha256=target_executor_sha256,
    )
    direct = str(row["direct_response"])
    source_prediction = decision.source_relation or direct

    shuffled_binding = bind_robust_temporal_pair_program(
        **arguments, effect_binding_authenticated=False,
    )
    shuffled_target = unified_temporal_grounding(
        contract=source_contract, binding=shuffled_binding,
    )
    shuffled_decision = decide_temporal_relation(
        harness=harness, target=shuffled_target,
        binding=shuffled_binding,
        target_executor_sha256=target_executor_sha256,
    )
    wrong_selection = select_source_contract(
        (wrong_contract,), target.requirement,
    )
    body = {
        "task_id": str(row["task_id"]),
        "video_id": str(row["video_id"]),
        "target_native_prediction": direct,
        "source_induced_prediction": source_prediction,
        "effect_shuffled_prediction": direct,
        "wrong_source_prediction": direct,
        "generic_scaffold_prediction": source_prediction,
        "target_written_equivalent_prediction": source_prediction,
        "source_executor_authorized": decision.source_relation is not None,
        "source_relation": decision.source_relation,
        "source_phase7_reason": decision.phase7.reason,
        "source_utility_reason": decision.utility.reason,
        "source_utility_lower_bound": decision.utility.utility_lower_bound,
        "source_authenticity_lower_bound": (
            decision.utility.authenticity_lower_bound
        ),
        "source_binding_receipt_sha256": binding.receipt_sha256,
        "effect_shuffled_binding_receipt_sha256": (
            shuffled_binding.receipt_sha256
        ),
        "effect_shuffled_executor_authorized": (
            shuffled_decision.source_relation is not None
        ),
        "wrong_source_abstained": (
            wrong_selection["status"] != "UNIQUE_SOURCE_CONTRACT_SELECTED"
        ),
        "operand_a_hypothesis_count": len(binding.operand_a_hypotheses),
        "operand_b_hypothesis_count": len(binding.operand_b_hypotheses),
        "cross_view_relation_count": len(binding.cross_view_relations),
        "all_pairs_strictly_separated": (
            binding.all_pairs_strictly_separated
        ),
        "all_pairs_relation_consistent": (
            binding.all_pairs_relation_consistent
        ),
        "runtime_integrity_qualified": _runtime_integrity(row),
        "formal_outcome_used_for_current_authorization": False,
    }
    return body | {"prediction_receipt_sha256": stable_hash(body)}


def evaluate(
    *, config_path: Path, base_report_path: Path, output_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    prereg_path = REPO_ROOT / config["preregistration"]
    if _sha256(prereg_path) != config["preregistration_file_sha256"]:
        raise ValueError("V34 preregistration file hash mismatch")
    prereg = json.loads(prereg_path.read_text())
    if prereg["status"] != (
        "FROZEN_BEFORE_ANY_V34_FORMAL_PROVIDER_OR_OUTCOME_CALL"
    ):
        raise ValueError("V34 preregistration status is not frozen")
    postground = config["postground"]
    for field in ("adapter_module", "evaluator_module"):
        path = REPO_ROOT / postground[field]
        if _sha256(path) != postground[f"{field}_sha256"]:
            raise ValueError(f"V34 {field} changed after freeze")
    if (
        prereg["postground_evaluation_protocol_sha256"]
        != postground["evaluation_protocol_sha256"]
    ):
        raise ValueError("V34 post-ground protocol hash mismatch")
    base = json.loads(base_report_path.read_text())
    base_body = dict(base)
    claimed_base_hash = base_body.pop("report_sha256")
    if stable_hash(base_body) != claimed_base_hash:
        raise ValueError("V34 base report hash mismatch")
    if len(base["rows"]) != int(
        postground["formal_gates"]["required_valid_rows"]
    ):
        raise ValueError("V34 base report has wrong row count")

    sources, _ = _load_sources(config)
    source_contract, wrong_contract = sources[1], sources[2]
    if source_contract.program_sha256 != postground["source_program_sha256"]:
        raise ValueError("V34 source temporal program drifted")
    source_artifact = json.loads((REPO_ROOT / SOURCE_ARTIFACT).read_text())
    calibration = postground["development_calibration"]
    route = build_temporal_route(
        source_program_sha256=source_contract.program_sha256,
        target_grounder_sha256=postground["target_grounder_sha256"],
        target_executor_sha256=postground["target_executor_sha256"],
        evidence_report_sha256=prereg[
            "qualified_v33_development_report_sha256"
        ],
        utility_vs_target_native=PairedCalibration(
            int(calibration["wins"]), int(calibration["losses"]),
            int(calibration["ties"]),
        ),
        authenticity_vs_effect_shuffled=PairedCalibration(
            int(calibration["wins"]), int(calibration["losses"]),
            int(calibration["ties"]),
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
    # Outcome blindness is enforced by freezing every prediction first.
    frozen = [
        _freeze_row(
            row=row, source_contract=source_contract,
            wrong_contract=wrong_contract, harness=harness,
            target_grounder_sha256=postground["target_grounder_sha256"],
            target_executor_sha256=postground["target_executor_sha256"],
        )
        for row in base["rows"]
    ]
    outcomes = [
        _answer_class(row["gold_answer_evaluator_only"])
        for row in base["rows"]
    ]
    if any(value is None for value in outcomes):
        raise ValueError("V34 formal row has non-temporal gold")
    arm_keys = (
        "target_native_prediction", "source_induced_prediction",
        "effect_shuffled_prediction", "wrong_source_prediction",
        "generic_scaffold_prediction",
        "target_written_equivalent_prediction",
    )
    correctness = {
        key: [
            _answer_class(row[key]) == gold
            for row, gold in zip(frozen, outcomes, strict=True)
        ]
        for key in arm_keys
    }
    primary = _paired(
        correctness["source_induced_prediction"],
        correctness["target_native_prediction"],
    )
    shuffled = _paired(
        correctness["source_induced_prediction"],
        correctness["effect_shuffled_prediction"],
    )
    generic = _paired(
        correctness["source_induced_prediction"],
        correctness["generic_scaffold_prediction"],
    )
    target_written = _paired(
        correctness["source_induced_prediction"],
        correctness["target_written_equivalent_prediction"],
    )
    gates = postground["formal_gates"]
    authorizations = sum(row["source_executor_authorized"] for row in frozen)
    unique_videos = len({row["video_id"] for row in frozen})
    qualification = {
        "required_valid_rows": len(frozen) >= gates["required_valid_rows"],
        "required_unique_videos": (
            unique_videos >= gates["required_unique_videos"]
        ),
        "minimum_source_authorizations": (
            authorizations >= gates["minimum_source_authorizations"]
        ),
        "minimum_source_wins": primary["wins"] >= gates["minimum_source_wins"],
        "maximum_source_losses": (
            primary["losses"] <= gates["maximum_source_losses"]
        ),
        "minimum_source_minus_target_correct": (
            primary["left_minus_right_correct"]
            >= gates["minimum_source_minus_target_correct"]
        ),
        "maximum_exact_one_sided_pvalue": (
            primary["exact_one_sided_pvalue"]
            <= gates["maximum_exact_one_sided_pvalue"]
        ),
        "required_effect_shuffled_abstentions": sum(
            not row["effect_shuffled_executor_authorized"] for row in frozen
        ) >= gates["required_effect_shuffled_abstentions"],
        "required_wrong_source_abstentions": sum(
            row["wrong_source_abstained"] for row in frozen
        ) >= gates["required_wrong_source_abstentions"],
        "required_generic_scaffold_matches": sum(
            a == b for a, b in zip(
                correctness["source_induced_prediction"],
                correctness["generic_scaffold_prediction"], strict=True,
            )
        ) >= gates["required_generic_scaffold_matches"],
        "required_target_written_equivalent_matches": sum(
            a == b for a, b in zip(
                correctness["source_induced_prediction"],
                correctness["target_written_equivalent_prediction"],
                strict=True,
            )
        ) >= gates["required_target_written_equivalent_matches"],
        "maximum_reported_provider_cost_usd": (
            float(base["reported_provider_cost_usd"])
            <= gates["maximum_reported_provider_cost_usd"]
        ),
        "runtime_integrity_qualified": all(
            row["runtime_integrity_qualified"] for row in frozen
        ),
        "current_outcome_never_used_for_authorization": all(
            not row["formal_outcome_used_for_current_authorization"]
            for row in frozen
        ),
    }
    status = (
        "AGQA2_ROBUST_TEMPORAL_V34_FORMAL_QUALIFIED"
        if all(qualification.values())
        else "AGQA2_ROBUST_TEMPORAL_V34_FORMAL_NOT_QUALIFIED"
    )
    rows_detail = []
    for prediction, gold in zip(frozen, outcomes, strict=True):
        rows_detail.append(prediction | {
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
    result = {
        "schema_version": "agqa2-robust-temporal-v34-formal-report-v1",
        "status": status,
        "split": "fresh_formal",
        "confirmatory_claim": all(qualification.values()),
        "claim_boundary": prereg["claim_boundary"],
        "rows": len(frozen),
        "unique_video_count": unique_videos,
        "source_program_sha256": source_contract.program_sha256,
        "source_artifact_sha256": source_artifact["artifact_sha256"],
        "target_grounder_sha256": postground["target_grounder_sha256"],
        "target_executor_sha256": postground["target_executor_sha256"],
        "postground_evaluation_protocol_sha256": postground[
            "evaluation_protocol_sha256"
        ],
        "base_report_sha256": base["report_sha256"],
        "source_executor_authorizations": authorizations,
        "source_vs_target_native": primary,
        "source_vs_effect_shuffled": shuffled,
        "source_vs_generic_scaffold": generic,
        "source_vs_target_written_equivalent": target_written,
        "effect_shuffled_abstentions": sum(
            not row["effect_shuffled_executor_authorized"] for row in frozen
        ),
        "wrong_source_abstentions": sum(
            row["wrong_source_abstained"] for row in frozen
        ),
        "qualification_gates": qualification,
        "provider_calls": base["provider_calls"],
        "reported_provider_cost_usd": base["reported_provider_cost_usd"],
        "current_outcome_used_for_authorization": False,
        "interpretation": {
            "second_source_program_family_success_validated": all(
                qualification.values()
            ),
            "source_beats_handwritten_generic": False,
            "source_provenance_is_necessary": False,
            "full_agqa_solved": False,
        },
        "rows_detail": rows_detail,
    }
    result["report_sha256"] = stable_hash(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_robust_temporal_v34_formal.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--base-report", type=Path,
        default=REPO_ROOT / "runs/agqa2_robust_temporal_v34_formal/base_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_robust_temporal_v34_formal/report.json",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    collect_base(
        config_path=args.config.resolve(), keys_path=args.keys.resolve(),
        output_path=args.base_report.resolve(), workers=args.workers,
        limit=None,
    )
    result = evaluate(
        config_path=args.config.resolve(),
        base_report_path=args.base_report.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps({
        key: result[key]
        for key in (
            "status", "rows", "source_executor_authorizations",
            "source_vs_target_native", "qualification_gates",
            "provider_calls", "reported_provider_cost_usd", "report_sha256",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
