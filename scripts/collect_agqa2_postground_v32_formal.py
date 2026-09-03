#!/usr/bin/env python3
"""Collect and evaluate a fresh V32 post-grounding AGQA confirmation."""

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

from motif_transfer.agqa_goal_relation_transfer import (  # noqa: E402
    build_harness, build_route,
)
from motif_transfer.agqa_postground_relation_evaluation import (  # noqa: E402
    freeze_postground_predictions,
)
from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    exact_one_sided_pvalue,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.unified_transfer_runtime import PairedCalibration  # noqa: E402
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _answer_matches,
)
import scripts.collect_agqa2_query_object_v28 as v28  # noqa: E402


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
    spec = config["postground_formal"]
    return {
        "schema_version": spec["schema_version"],
        "source": spec["source"],
        "development": spec["development"],
        "adapter": spec["adapter"],
        "calibration": spec["calibration"],
        "controls": spec["controls"],
        "qualification_gates": spec["qualification_gates"],
        "failure_policy": spec["failure_policy"],
        "manifest_sha256": spec["manifest_sha256"],
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


def collect(
    *, config_path: Path, keys_path: Path, base_output_path: Path,
    output_path: Path, workers: int,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    spec = config["postground_formal"]
    protocol_sha256 = stable_hash(evaluation_protocol_core(config))
    if protocol_sha256 != spec["expected_evaluation_protocol_sha256"]:
        raise ValueError("V32 postground evaluation protocol changed")
    prereg_path = REPO_ROOT / spec["preregistration"]
    if _sha256(prereg_path) != spec["preregistration_file_sha256"]:
        raise ValueError("V32 postground preregistration file hash mismatch")
    prereg = _verified(prereg_path, "preregistration_sha256")
    if prereg["status"] != "FROZEN_BEFORE_ANY_V32_PROVIDER_OR_GOLD_CALL":
        raise ValueError("V32 postground preregistration status mismatch")
    for section, labels in (
        ("source", ("artifact", "confirmation", "inducer")),
        ("development", ("report",)),
        ("adapter", ("module", "prediction_module", "collector")),
    ):
        values = spec[section]
        for label in labels:
            path = REPO_ROOT / values[label]
            if _sha256(path) != values[f"{label}_file_sha256"]:
                raise ValueError(f"V32 {section}.{label} file hash mismatch")
    development = _verified(
        REPO_ROOT / spec["development"]["report"], "report_sha256",
    )
    if not development.get("qualified_for_future_disjoint_reserve"):
        raise ValueError("V32 development route is not qualified")
    artifact = json.loads((REPO_ROOT / spec["source"]["artifact"]).read_text())
    confirmation = json.loads((
        REPO_ROOT / spec["source"]["confirmation"]
    ).read_text())
    route = build_route(
        source_program_sha256=str(artifact["artifact_sha256"]),
        target_grounder_sha256=str(spec["adapter"]["target_grounder_sha256"]),
        target_executor_sha256=str(spec["adapter"]["target_executor_sha256"]),
        evidence_report_sha256=str(development["report_sha256"]),
        utility_vs_target_native=PairedCalibration(**(
            spec["calibration"]["utility_vs_target_native"]
        )),
        authenticity_vs_effect_shuffled=PairedCalibration(**(
            spec["calibration"]["authenticity_vs_effect_shuffled"]
        )),
    )
    harness = build_harness(
        artifact=artifact,
        confirmation=confirmation,
        inducer_artifact_sha256=str(spec["source"]["inducer_artifact_sha256"]),
        route=route,
    )

    base = v28.collect(
        config_path=config_path,
        keys_path=keys_path,
        output_path=base_output_path,
        workers=workers,
    )
    frozen = []
    for row in base["rows"]:
        prediction = freeze_postground_predictions(
            row=row,
            artifact=artifact,
            confirmation=confirmation,
            harness=harness,
            target_grounder_sha256=str(
                spec["adapter"]["target_grounder_sha256"]
            ),
            target_executor_sha256=str(
                spec["adapter"]["target_executor_sha256"]
            ),
            minimum_ontology_confidences=tuple(
                spec["adapter"]["minimum_ontology_confidences"]
            ),
        )
        frozen.append((row, prediction))

    evaluated = []
    for row, prediction in frozen:
        gold = str(row["gold_answer_evaluator_only"])
        detail = asdict(prediction)
        detail.update({
            "video_id": str(row["video_id"]),
            "relation_group": next(
                sample["relation_group"] for sample in (
                    json.loads((REPO_ROOT / config["manifest"]).read_text())[
                        "samples"
                    ]
                ) if sample["task_id"] == row["task_id"]
            ),
            "gold_answer_evaluator_only": gold,
            "source_correct": _answer_matches(
                prediction.source_harness_prediction, gold,
            ),
            "target_correct": _answer_matches(
                prediction.target_native_prediction, gold,
            ),
            "effect_shuffled_correct": _answer_matches(
                prediction.effect_shuffled_prediction, gold,
            ),
            "generic_scaffold_correct": _answer_matches(
                prediction.generic_scaffold_prediction, gold,
            ),
            "target_written_equivalent_correct": _answer_matches(
                prediction.target_written_equivalent_prediction, gold,
            ),
            "prediction_api_has_no_gold_or_outcome_argument": True,
        })
        evaluated.append(detail)
    source_target = _paired(evaluated, "source_correct", "target_correct")
    source_shuffled = _paired(
        evaluated, "source_correct", "effect_shuffled_correct",
    )
    source_generic = _paired(
        evaluated, "source_correct", "generic_scaffold_correct",
    )
    target_written = _paired(
        evaluated, "source_correct", "target_written_equivalent_correct",
    )
    gate = spec["qualification_gates"]
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
        "authenticity_matches_primary_endpoint": source_shuffled == source_target,
        "effect_shuffled_source_never_executes": all(
            not row["effect_shuffled_executor_authorized"]
            for row in evaluated
        ),
        "source_matches_generic_ceiling": (
            source_generic["wins"] == source_generic["losses"] == 0
        ),
        "source_matches_target_written_ceiling": (
            target_written["wins"] == target_written["losses"] == 0
        ),
        "raw_votes_never_used_as_symbolic_bindings": all(
            not row["raw_neural_votes_used_as_symbolic_bindings"]
            for row in evaluated
        ),
        "prediction_api_has_no_gold_or_outcome_argument": all(
            row["prediction_api_has_no_gold_or_outcome_argument"]
            for row in evaluated
        ),
        "provider_cost_within_cap": (
            float(base["reported_provider_cost_usd"])
            <= gate["maximum_reported_provider_cost_usd"]
        ),
    }
    qualified = all(gates.values())
    core = {
        "schema_version": "agqa2-postground-formal-report-v32",
        "status": (
            "AGQA2_POSTGROUND_V32_FORMAL_QUALIFIED"
            if qualified else "AGQA2_POSTGROUND_V32_FORMAL_NOT_QUALIFIED"
        ),
        "claim_boundary": spec["claim_boundary"],
        "evaluation_protocol_sha256": protocol_sha256,
        "config_file_sha256": _sha256(config_path),
        "preregistration_sha256": prereg["preregistration_sha256"],
        "manifest_sha256": spec["manifest_sha256"],
        "development_report_sha256": development["report_sha256"],
        "source_program_sha256": artifact["artifact_sha256"],
        "target_grounder_sha256": spec["adapter"]["target_grounder_sha256"],
        "target_executor_sha256": spec["adapter"]["target_executor_sha256"],
        "rows": len(evaluated),
        "source_executor_authorizations": sum(
            row["source_executor_authorized"] for row in evaluated
        ),
        "source_vs_target_native": source_target,
        "source_vs_effect_shuffled": source_shuffled,
        "source_vs_generic_scaffold": source_generic,
        "source_vs_target_written_equivalent": target_written,
        "qualification_gates": gates,
        "transfer_qualified": qualified,
        "base_grounder_status_diagnostic_only": base["status"],
        "base_grounder_qualified_diagnostic_only": base["grounder_qualified"],
        "provider_calls": base["provider_calls"],
        "reported_provider_cost_usd": base["reported_provider_cost_usd"],
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
        default=REPO_ROOT / "configs/agqa2_postground_v32_formal.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--base-output", type=Path,
        default=REPO_ROOT / "runs/agqa2_postground_v32_formal/base_report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/agqa2_postground_v32_formal/report.json",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    result = collect(
        config_path=args.config.resolve(),
        keys_path=args.keys.resolve(),
        base_output_path=args.base_output.resolve(),
        output_path=args.output.resolve(),
        workers=args.workers,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "rows", "source_executor_authorizations",
        "source_vs_target_native", "source_vs_effect_shuffled",
        "source_vs_generic_scaffold", "qualification_gates",
        "reported_provider_cost_usd", "report_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
