#!/usr/bin/env python3
"""Evaluate the preregistered, previously outcome-unread V27 receipt pool."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_goal_relation_evaluation import (  # noqa: E402
    freeze_transfer_predictions,
)
from motif_transfer.agqa_goal_relation_transfer import (  # noqa: E402
    build_harness, build_route,
)
from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    exact_one_sided_pvalue,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.unified_transfer_runtime import PairedCalibration  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _answer_matches,
)
from scripts.collect_agqa2_query_object_v20 import (  # noqa: E402
    _load_selected_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, hash_field: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    body = dict(value)
    claimed = str(body.pop(hash_field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return value


def evaluation_protocol_core(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "manifest": config["manifest"],
        "manifest_file_sha256": config["manifest_file_sha256"],
        "source": config["source"],
        "development": config["development"],
        "adapter": config["adapter"],
        "calibration": config["calibration"],
        "controls": config["controls"],
        "qualification_gates": config["qualification_gates"],
        "failure_policy": config["failure_policy"],
    }


def _paired(rows: Sequence[Mapping[str, Any]], left: str, right: str):
    wins = sum(row[left] and not row[right] for row in rows)
    losses = sum(row[right] and not row[left] for row in rows)
    return {
        "left_correct": sum(bool(row[left]) for row in rows),
        "right_correct": sum(bool(row[right]) for row in rows),
        "left_minus_right_correct": (
            sum(bool(row[left]) for row in rows)
            - sum(bool(row[right]) for row in rows)
        ),
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "exact_one_sided_pvalue": exact_one_sided_pvalue(
            source_wins=wins, source_losses=losses,
        ),
    }


def evaluate(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    prereg_path = REPO_ROOT / config["preregistration"]
    manifest_path = REPO_ROOT / config["manifest"]
    if _sha256(prereg_path) != config["preregistration_file_sha256"]:
        raise ValueError("V30 preregistration file hash mismatch")
    if _sha256(manifest_path) != config["manifest_file_sha256"]:
        raise ValueError("V30 manifest file hash mismatch")
    prereg = _verified(prereg_path, "preregistration_sha256")
    manifest = _verified(manifest_path, "manifest_sha256")
    if prereg["status"] != "FROZEN_BEFORE_V30_FORMAL_GOLD_READ":
        raise ValueError("V30 preregistration status mismatch")
    if manifest["status"] != "FROZEN_V30_OUTCOME_UNREAD_RUNTIME_RECEIPTS":
        raise ValueError("V30 manifest status mismatch")
    protocol_sha256 = stable_hash(evaluation_protocol_core(config))
    if protocol_sha256 != config["expected_evaluation_protocol_sha256"]:
        raise ValueError("V30 evaluation protocol changed after freeze")
    for section, labels in (
        ("source", ("artifact", "confirmation", "inducer")),
        ("adapter", ("module", "prediction_module", "evaluator")),
        ("development", ("report",)),
    ):
        spec = config[section]
        for label in labels:
            path = REPO_ROOT / spec[label]
            if _sha256(path) != spec[f"{label}_file_sha256"]:
                raise ValueError(f"V30 {section}.{label} file hash mismatch")
    development = _verified(
        REPO_ROOT / config["development"]["report"], "report_sha256",
    )
    if not development.get("qualified_for_future_disjoint_reserve"):
        raise ValueError("V30 development route is not qualified")
    artifact = json.loads((REPO_ROOT / config["source"]["artifact"]).read_text())
    confirmation = json.loads((
        REPO_ROOT / config["source"]["confirmation"]
    ).read_text())
    calibration = config["calibration"]
    route = build_route(
        source_program_sha256=str(artifact["artifact_sha256"]),
        target_grounder_sha256=str(config["adapter"]["target_grounder_sha256"]),
        target_executor_sha256=str(config["adapter"]["target_executor_sha256"]),
        evidence_report_sha256=str(development["report_sha256"]),
        utility_vs_target_native=PairedCalibration(**(
            calibration["utility_vs_target_native"]
        )),
        authenticity_vs_effect_shuffled=PairedCalibration(**(
            calibration["authenticity_vs_effect_shuffled"]
        )),
    )
    harness = build_harness(
        artifact=artifact,
        confirmation=confirmation,
        inducer_artifact_sha256=str(config["source"]["inducer_artifact_sha256"]),
        route=route,
    )

    metadata = _load_selected_rows(manifest)
    frozen_rows = []
    for sample in manifest["samples"]:
        task_id = str(sample["task_id"])
        receipt_path = REPO_ROOT / sample["runtime_receipt_path"]
        if _sha256(receipt_path) != sample["runtime_receipt_file_sha256"]:
            raise ValueError(f"V30 runtime receipt file changed: {task_id}")
        runtime = json.loads(receipt_path.read_text())
        body = dict(runtime)
        claimed = str(body.pop("runtime_receipt_sha256", ""))
        if not claimed or stable_hash(body) != claimed:
            raise ValueError(f"V30 runtime receipt hash mismatch: {task_id}")
        if (
            runtime["task_id"] != task_id
            or runtime["video_id"] != sample["video_id"]
            or runtime["question_sha256"] != sample["question_sha256"]
            or runtime["video_sha256"] != sample["video_sha256"]
        ):
            raise ValueError(f"V30 runtime/manifest mismatch: {task_id}")
        # No evaluator-only field is supplied to the prediction API.
        frozen = freeze_transfer_predictions(
            row=runtime,
            artifact=artifact,
            confirmation=confirmation,
            harness=harness,
            target_grounder_sha256=str(
                config["adapter"]["target_grounder_sha256"]
            ),
            target_executor_sha256=str(
                config["adapter"]["target_executor_sha256"]
            ),
            minimum_ontology_confidences=tuple(
                config["adapter"]["minimum_ontology_confidences"]
            ),
        )
        frozen_rows.append((sample, runtime, frozen))

    # Gold/program access starts only after every prediction receipt freezes.
    evaluated = []
    for sample, runtime, frozen in frozen_rows:
        task_id = str(sample["task_id"])
        target = metadata[task_id]
        gold = str(target["answer"])
        row = asdict(frozen)
        row.update({
            "video_id": str(sample["video_id"]),
            "relation_group": str(sample["relation_group"]),
            "gold_answer_evaluator_only": gold,
            "source_correct": _answer_matches(
                frozen.source_harness_prediction, gold,
            ),
            "target_correct": _answer_matches(
                frozen.target_native_prediction, gold,
            ),
            "effect_shuffled_correct": _answer_matches(
                frozen.effect_shuffled_prediction, gold,
            ),
            "generic_scaffold_correct": _answer_matches(
                frozen.generic_scaffold_prediction, gold,
            ),
            "target_written_equivalent_correct": _answer_matches(
                frozen.target_written_equivalent_prediction, gold,
            ),
            "official_answer_first_read_after_all_predictions_froze": True,
        })
        evaluated.append(row)

    source_target = _paired(evaluated, "source_correct", "target_correct")
    source_shuffled = _paired(
        evaluated, "source_correct", "effect_shuffled_correct",
    )
    source_generic = _paired(
        evaluated, "source_correct", "generic_scaffold_correct",
    )
    source_target_written = _paired(
        evaluated, "source_correct", "target_written_equivalent_correct",
    )
    gate = config["qualification_gates"]
    gates = {
        "required_valid_rows": len(evaluated) == gate["required_valid_rows"],
        "minimum_source_authorizations": sum(
            row["source_executor_authorized"] for row in evaluated
        ) >= gate["minimum_source_authorizations"],
        "runtime_integrity_qualified": all(
            row["runtime_integrity_qualified"] for row in evaluated
        ),
        "minimum_source_vs_target_wins": (
            source_target["wins"] >= gate["minimum_source_vs_target_wins"]
        ),
        "maximum_source_vs_target_losses": (
            source_target["losses"] <= gate["maximum_source_vs_target_losses"]
        ),
        "minimum_source_minus_target_correct": (
            source_target["left_minus_right_correct"]
            >= gate["minimum_source_minus_target_correct"]
        ),
        "maximum_exact_one_sided_pvalue": (
            source_target["exact_one_sided_pvalue"]
            <= gate["maximum_exact_one_sided_pvalue"]
        ),
        "authenticity_matches_primary_endpoint": (
            source_shuffled == source_target
        ),
        "effect_shuffled_source_never_executes": all(
            not row["effect_shuffled_executor_authorized"]
            for row in evaluated
        ),
        "target_written_equivalent_is_ceiling": (
            source_target_written["wins"]
            == source_target_written["losses"] == 0
        ),
        "current_outcome_never_used_for_authorization": all(
            not row["current_outcome_read"] for row in evaluated
        ),
        "generic_scaffold_control_reported": len(evaluated) > 0,
    }
    qualified = all(gates.values())
    core = {
        "schema_version": "agqa2-goal-relation-formal-report-v30",
        "status": (
            "AGQA2_GOAL_RELATION_V30_FORMAL_QUALIFIED"
            if qualified else
            "AGQA2_GOAL_RELATION_V30_FORMAL_NOT_QUALIFIED"
        ),
        "claim_boundary": config["claim_boundary"],
        "evaluation_protocol_sha256": protocol_sha256,
        "config_file_sha256": _sha256(config_path),
        "preregistration_sha256": prereg["preregistration_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "development_report_sha256": development["report_sha256"],
        "source_program_sha256": artifact["artifact_sha256"],
        "target_grounder_sha256": config["adapter"]["target_grounder_sha256"],
        "target_executor_sha256": config["adapter"]["target_executor_sha256"],
        "rows": len(evaluated),
        "source_executor_authorizations": sum(
            row["source_executor_authorized"] for row in evaluated
        ),
        "source_vs_target_native": source_target,
        "source_vs_effect_shuffled": source_shuffled,
        "source_vs_generic_scaffold": source_generic,
        "source_vs_target_written_equivalent": source_target_written,
        "qualification_gates": gates,
        "transfer_qualified": qualified,
        "provider_calls": 0,
        "formal_gold_evaluation_started_after_freeze": True,
        "source_provenance_claim": qualified,
        "rows_detail": evaluated,
    }
    result = core | {"report_sha256": stable_hash(core)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/agqa2_goal_relation_v30_legacy.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_goal_relation_v30_legacy/report.json",
    )
    args = parser.parse_args()
    result = evaluate(args.config.resolve(), args.output.resolve())
    print(json.dumps({key: result[key] for key in (
        "status", "rows", "source_executor_authorizations",
        "source_vs_target_native", "source_vs_effect_shuffled",
        "source_vs_generic_scaffold", "qualification_gates",
        "provider_calls", "report_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
