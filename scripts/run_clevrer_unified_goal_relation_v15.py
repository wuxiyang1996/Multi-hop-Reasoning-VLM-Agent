#!/usr/bin/env python3
"""Prospective zero-provider-cost CLEVRER unified V15 reserve runner."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.run_clevrer_sokoban_proof_v14 import (  # noqa: E402
    _binding_controls,
    _content_hash,
    _paired,
    _sample_parts,
)
from motif_transfer.clevrer_proof_receipts import (  # noqa: E402
    paired_proof_features,
)
from motif_transfer.clevrer_query_compiler import (  # noqa: E402
    compile_choice,
    compile_question,
    normalize_official_program,
)
from motif_transfer.clevrer_unified_goal_relation import (  # noqa: E402
    build_harness,
    build_route,
    decide_recovery,
    source_goal_relation_envelope,
    target_grounding,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.sokoban_video_recovery import (  # noqa: E402
    exact_binomial_two_sided,
)
from motif_transfer.unified_transfer_runtime import (  # noqa: E402
    PairedCalibration,
    TransferVerdict,
)
from motif_transfer.video_proof_grounder import (  # noqa: E402
    V14_FEATURE_NAMES,
    validate_v14_artifact,
)
from motif_transfer.video_recovery_cate import (  # noqa: E402
    FEATURE_NAMES,
    build_features,
)


AUTHENTIC = "authentic_source_induced_goal_relation"
NEURAL = "neural_only_explicit_relation"
TARGET_BASE = "target_base_receipt_recovery"
GENERIC = "generic_error_scaffold"
PERMUTED = "source_permuted_uplift"
SHUFFLED = "shuffled_proof_binding"
INVERTED = "source_inverted_effect"
TRAJECTORY = "target_trajectory_only"
CEILING = "target_native_representation_ceiling"
CONDITIONS = (
    NEURAL, AUTHENTIC, TARGET_BASE, GENERIC, PERMUTED, SHUFFLED,
    INVERTED, TRAJECTORY, CEILING,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value: str) -> Path:
    result = Path(value)
    return result if result.is_absolute() else REPO / result


def _verify_lineage(config: Mapping[str, Any]) -> dict[str, str]:
    if config.get("status") != "FROZEN_BEFORE_CLEVRER_V15_RESERVE_OUTCOMES":
        raise ValueError("CLEVRER V15 config is not a prospective freeze")
    if config["target"].get("role") != "reserve":
        raise ValueError("CLEVRER V15 runner only opens the reserve role")
    observed = {
        key: _sha(_path(str(path)))
        for key, path in config["lineage_paths"].items()
    }
    if observed != config.get("frozen_lineage"):
        mismatches = {
            key: {"expected": config.get("frozen_lineage", {}).get(key),
                  "observed": value}
            for key, value in observed.items()
            if value != config.get("frozen_lineage", {}).get(key)
        }
        raise ValueError(f"CLEVRER V15 frozen lineage mismatch: {mismatches}")
    return observed


def _calibration(value: Mapping[str, Any]) -> PairedCalibration:
    return PairedCalibration(
        wins=int(value["wins"]), losses=int(value["losses"]),
        ties=int(value["ties"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _read(args.config)
    observed_lineage = _verify_lineage(config)

    source = _read(_path(config["source"]["artifact"]))
    confirmation = _read(_path(config["source"]["confirmation"]))
    envelope = source_goal_relation_envelope(
        source, confirmation,
        inducer_artifact_sha256=observed_lineage["source_inducer"],
    )
    if envelope.envelope_sha256 != config["source"]["envelope_sha256"]:
        raise ValueError("frozen source envelope mismatch")
    if envelope.contract.contract_sha256 != config["source"]["contract_sha256"]:
        raise ValueError("frozen source contract mismatch")

    development = _read(_path(config["development"]["report"]))
    body = dict(development)
    claimed = body.pop("report_sha256", None)
    if claimed != stable_hash(body) or development.get("status") != (
        "CLEVRER_UNIFIED_V15_DEVELOPMENT_GATE_PASSED"
    ):
        raise ValueError("V15 development report is invalid or failed")
    calibration = config["development"]["calibration"]

    grounder_artifact = _read(_path(config["grounder"]["artifact"]))
    proof_model, base_model, permuted_model, threshold = validate_v14_artifact(
        grounder_artifact
    )
    if float(threshold) != float(config["grounder"]["decision_threshold"]):
        raise ValueError("frozen grounder threshold mismatch")
    grounder_sha256 = str(grounder_artifact["artifact_sha256"])
    executor_sha256 = observed_lineage["adapter"]
    route = build_route(
        source_program_sha256=envelope.contract.program_sha256,
        target_grounder_sha256=grounder_sha256,
        target_executor_sha256=executor_sha256,
        evidence_report_sha256=observed_lineage["development_report"],
        utility_vs_neural=_calibration(calibration["utility_vs_neural"]),
        authenticity_vs_source_permuted=_calibration(
            calibration["authenticity_vs_source_permuted"]
        ),
    )
    harness = build_harness(envelope, route)

    manifest = _read(_path(config["target"]["split_manifest"]))
    sample_ids = list(manifest["benchmarks"]["clevrer"]["splits"]["reserve"])
    expected = int(config["gates"]["expected_samples"])
    if len(sample_ids) != expected or len(set(sample_ids)) != expected:
        raise ValueError("reserve sample count/uniqueness mismatch")

    official_root = _path(config["target"]["official_root"])
    executor_root = official_root / "executor"
    sys.path.insert(0, str(executor_root))
    from executor import Executor  # type: ignore  # noqa: E402
    from simulation import Simulation  # type: ignore  # noqa: E402

    annotations_path = executor_root / "data/validation.json"
    annotations = {
        int(row["scene_index"]): row
        for row in json.loads(annotations_path.read_text(encoding="utf-8"))
    }
    prediction_hashes: dict[str, str] = {}
    base_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        scene_id, question_id = _sample_parts(sample_id)
        question = next(
            row for row in annotations[scene_id]["questions"]
            if int(row["question_id"]) == question_id
        )
        family = str(question["question_type"])
        question_program = compile_question(str(question["question"]), family)
        choice_programs = [
            compile_choice(str(choice["choice"]), family)
            for choice in question["choices"]
        ]
        paths = {
            "explicit": executor_root / "data/propnet_preds"
            / config["target"]["explicit_relation_prediction_directory"]
            / f"sim_{scene_id:05d}.json",
            "trajectory": executor_root / "data/propnet_preds"
            / config["target"]["trajectory_prediction_directory"]
            / f"sim_{scene_id:05d}.json",
        }
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(paths)
        for path in paths.values():
            prediction_hashes.setdefault(str(path.resolve()), _sha(path))
        explicit_executor = Executor(Simulation(str(paths["explicit"]), use_event_ann=True))
        trajectory_executor = Executor(Simulation(str(paths["trajectory"]), use_event_ann=False))
        proof_features, proof_receipts = paired_proof_features(
            explicit_executor, trajectory_executor, question_program, choice_programs,
        )
        explicit_raw = [str(row["explicit_answer"]) for row in proof_receipts]
        trajectory_raw = [str(row["trajectory_answer"]) for row in proof_receipts]
        explicit_answer = "".join("1" if value == "yes" else "0" for value in explicit_raw)
        trajectory_answer = "".join("1" if value == "yes" else "0" for value in trajectory_raw)
        base_features = build_features(
            family=family, question_program=question_program,
            choice_programs=choice_programs,
            explicit_answer=explicit_answer, trajectory_answer=trajectory_answer,
            explicit_error_count=explicit_raw.count("error"),
        )
        features = tuple(base_features + proof_features)
        if len(features) != len(V14_FEATURE_NAMES):
            raise AssertionError("CLEVRER V15 feature contract drift")
        proof_sha = _content_hash(proof_receipts)
        proof_score = float(proof_model.predict([features])[0])
        target = target_grounding(
            task_id=sample_id, contract=envelope.contract,
            target_grounder_sha256=grounder_sha256,
            proof_receipt_sha256=proof_sha,
            proof_predicted_uplift=proof_score,
            decision_threshold=threshold,
        )
        decision = decide_recovery(
            harness=harness, target=target,
            target_executor_sha256=executor_sha256,
        )
        # Every decision and control below is fixed before evaluator-only gold
        # answers or official functional programs are accessed.
        base_rows.append({
            "sample_id": sample_id,
            "family": family,
            "question": question,
            "question_program": question_program,
            "choice_programs": choice_programs,
            "explicit_answer": explicit_answer,
            "trajectory_answer": trajectory_answer,
            "explicit_raw": explicit_raw,
            "trajectory_raw": trajectory_raw,
            "features": features,
            "proof_receipts_sha256": proof_sha,
            "proof_score": proof_score,
            "base_score": float(base_model.predict([base_features])[0]),
            "permuted_score": float(permuted_model.predict([features])[0]),
            "authentic_recover": decision.selected_native_representation == "trajectory",
            "phase7": decision.phase7,
            "utility": decision.utility,
            "executor_calls": decision.executor_calls,
        })

    binding = _binding_controls(base_rows, proof_model)
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        sample_id = str(base["sample_id"])
        shuffled = binding["shuffled_proof"][sample_id]
        decisions = {
            NEURAL: False,
            AUTHENTIC: base["authentic_recover"],
            TARGET_BASE: base["base_score"] > threshold,
            GENERIC: "error" in base["explicit_raw"],
            PERMUTED: base["permuted_score"] > threshold,
            SHUFFLED: shuffled["score"] > threshold,
            INVERTED: base["proof_score"] < -threshold,
            TRAJECTORY: True,
        }
        # Evaluator-only boundary starts here.
        question = base["question"]
        gold = "".join(
            "1" if choice["answer"] == "correct" else "0"
            for choice in question["choices"]
        )
        condition_rows = {}
        for name, recover in decisions.items():
            answer = base["trajectory_answer"] if recover else base["explicit_answer"]
            condition_rows[name] = {
                "recover": bool(recover), "answer": answer,
                "correct": answer == gold,
                "selected_native_representation": (
                    "trajectory" if recover else "explicit_relation"
                ),
            }
        explicit_correct = base["explicit_answer"] == gold
        trajectory_correct = base["trajectory_answer"] == gold
        ceiling_recover = trajectory_correct and not explicit_correct
        ceiling_answer = base["trajectory_answer"] if ceiling_recover else base["explicit_answer"]
        condition_rows[CEILING] = {
            "recover": ceiling_recover, "answer": ceiling_answer,
            "correct": explicit_correct or trajectory_correct,
            "selected_native_representation": (
                "trajectory" if ceiling_recover else "explicit_relation"
            ),
            "evaluator_only": True,
        }
        rows.append({
            "sample_id": sample_id,
            "family": base["family"],
            "proof_receipts_sha256": base["proof_receipts_sha256"],
            "compiler_question_exact": base["question_program"]
            == normalize_official_program(question["program"]),
            "compiler_choices_exact": all(
                compiled == normalize_official_program(choice["program"])
                for compiled, choice in zip(base["choice_programs"], question["choices"])
            ),
            "target_receipt": {
                "explicit_error_count": base["explicit_raw"].count("error"),
                "answer_disagreement": base["explicit_answer"] != base["trajectory_answer"],
                "proof_predicted_uplift": base["proof_score"],
                "base_predicted_uplift": base["base_score"],
                "permuted_predicted_uplift": base["permuted_score"],
                "shuffled_predicted_uplift": shuffled["score"],
                "decision_threshold": threshold,
            },
            "unified_authority": {
                "phase7": {**asdict(base["phase7"]),
                           "verdict": base["phase7"].verdict.value},
                "utility": {**asdict(base["utility"]),
                            "verdict": base["utility"].verdict.value},
                "executor_calls": base["executor_calls"],
                "current_target_outcome_read": False,
            },
            "conditions": condition_rows,
            "gold_answer_evaluator_only": gold,
        })

    count = len(rows)
    metrics = {
        name: {
            "correct": sum(row["conditions"][name]["correct"] for row in rows),
            "accuracy": sum(row["conditions"][name]["correct"] for row in rows) / count,
            "recoveries": sum(row["conditions"][name]["recover"] for row in rows),
        }
        for name in CONDITIONS
    }
    paired = {
        name: _paired(rows, AUTHENTIC, name)
        for name in CONDITIONS if name != AUTHENTIC
    }
    gates_cfg = config["gates"]
    causal_controls = tuple(gates_cfg["causal_control_conditions"])
    utility = paired[NEURAL]
    authenticity = paired[PERMUTED]
    gates = {
        "expected_sample_count": count == expected,
        "compiler_exact_on_all_rows": all(
            row["compiler_question_exact"] and row["compiler_choices_exact"]
            for row in rows
        ),
        "source_envelope_frozen_and_admitted": envelope.admitted,
        "unified_authority_matches_authentic_decision": all(
            (row["unified_authority"]["phase7"]["verdict"] == "SELECT_SKILL")
            == row["conditions"][AUTHENTIC]["recover"]
            for row in rows
        ),
        "only_selected_rows_reach_target_executor": all(
            row["unified_authority"]["executor_calls"]
            == int(row["conditions"][AUTHENTIC]["recover"])
            for row in rows
        ),
        "zero_runtime_target_outcome_exposure": all(
            row["unified_authority"]["current_target_outcome_read"] is False
            and row["unified_authority"]["phase7"]["current_target_outcome_read"] is False
            and row["unified_authority"]["utility"]["current_outcome_read"] is False
            for row in rows
        ),
        "minimum_authentic_recoveries": metrics[AUTHENTIC]["recoveries"]
        >= int(gates_cfg["minimum_authentic_recoveries"]),
        "authentic_strictly_above_all_causal_controls": all(
            metrics[AUTHENTIC]["correct"] > metrics[name]["correct"]
            for name in causal_controls
        ),
        "authentic_positive_paired_vs_all_causal_controls": all(
            paired[name]["net_wins"] > 0 for name in causal_controls
        ),
        "minimum_utility_net_wins": utility["net_wins"]
        >= int(gates_cfg["minimum_utility_net_wins"]),
        "utility_exact_p_value": utility["exact_two_sided_p"]
        <= float(gates_cfg["maximum_utility_exact_p"]),
        "minimum_authenticity_net_wins": authenticity["net_wins"]
        >= int(gates_cfg["minimum_authenticity_net_wins"]),
        "authenticity_exact_p_value": authenticity["exact_two_sided_p"]
        <= float(gates_cfg["maximum_authenticity_exact_p"]),
        "no_negative_transfer_vs_neural": utility["wins"] >= utility["losses"],
        "zero_external_provider_calls": True,
    }
    passed = all(gates.values())
    report_body = {
        "schema_version": "clevrer-unified-goal-relation-v15-reserve-report",
        "status": (
            "CLEVRER_UNIFIED_GOAL_RELATION_V15_FORMAL_VALIDATED"
            if passed else "CLEVRER_UNIFIED_GOAL_RELATION_V15_FORMAL_FAILED"
        ),
        "benchmark": "clevrer",
        "role": "prospective_reserve",
        "claim_boundary": config["claim_boundary"],
        "samples": count,
        "conditions": metrics,
        "paired_authentic": paired,
        "gates": gates,
        "source_program": config["source"],
        "target_interface": config["target"]["interface"],
        "rows": rows,
        "lineage": {
            "config_file_sha256": _sha(args.config),
            "verified_frozen_lineage": observed_lineage,
            "annotation_file_sha256": _sha(annotations_path),
            "prediction_file_sha256": prediction_hashes,
        },
        "cost": {
            "external_provider_calls": 0,
            "external_provider_cost_usd": 0.0,
            "local_official_prediction_files": len(prediction_hashes),
        },
    }
    report = report_body | {"report_sha256": stable_hash(report_body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "samples": count,
        "conditions": metrics, "paired_authentic": paired,
        "gates": gates, "cost": report["cost"], "output": str(args.output),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
