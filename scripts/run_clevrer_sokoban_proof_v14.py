#!/usr/bin/env python3
"""Prospective V14 Sokoban-to-CLEVRER proof-grounded transfer evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motif_transfer.clevrer_proof_receipts import (  # noqa: E402
    PROOF_FEATURE_NAMES,
    paired_proof_features,
)
from motif_transfer.clevrer_query_compiler import (  # noqa: E402
    compile_choice,
    compile_question,
    normalize_official_program,
)
from motif_transfer.sokoban_video_recovery import (  # noqa: E402
    exact_binomial_two_sided,
    validate_source_receipt,
)
from motif_transfer.video_proof_grounder import (  # noqa: E402
    V14_FEATURE_NAMES,
    validate_v14_artifact,
)
from motif_transfer.video_recovery_cate import FEATURE_NAMES, build_features  # noqa: E402


CONDITIONS = (
    "target_explicit_no_recovery",
    "target_trajectory_only",
    "authentic_sokoban_proof_cate_recover",
    "target_base_receipt_cate_recover",
    "permuted_uplift_cate_recover",
    "shuffled_proof_binding_recover",
    "source_inverted_effect_recover",
    "source_shuffled_action_binding_recover",
    "target_family_matched_marginal_recover",
    "target_error_only_recover",
    "target_disagreement_recover",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_parts(sample_id: str) -> tuple[int, int]:
    video, question = sample_id.split(".mp4.Q", 1)
    return int(video.rsplit("_", 1)[1]), int(question)


def _verify_lineage(config: Mapping[str, Any]) -> dict[str, str]:
    if config.get("status") != "FROZEN_BEFORE_V14_FORMAL_OUTCOMES":
        raise ValueError("V14 formal config was not frozen")
    if config["target"].get("split") != "formal":
        raise ValueError("V14 runner can only evaluate the formal role")
    paths = {
        "source_receipt_sha256": Path(config["source"]["receipt"]),
        "split_manifest_sha256": Path(config["target"]["split_manifest"]),
        "proof_artifact_file_sha256": Path(config["grounder"]["artifact"]),
        "training_report_sha256": Path(config["grounder"]["training_report"]),
        "training_config_sha256": Path(config["grounder"]["training_config"]),
        "runner_sha256": Path(__file__).resolve(),
        "proof_grounder_module_sha256": REPO / "src/motif_transfer/video_proof_grounder.py",
        "proof_receipt_module_sha256": REPO / "src/motif_transfer/clevrer_proof_receipts.py",
        "compiler_module_sha256": REPO / "src/motif_transfer/clevrer_query_compiler.py",
        "base_feature_module_sha256": REPO / "src/motif_transfer/video_recovery_cate.py",
        "recovery_module_sha256": REPO / "src/motif_transfer/sokoban_video_recovery.py",
    }
    observed = {key: _sha256(path) for key, path in paths.items()}
    for key, value in observed.items():
        if value != config["frozen_lineage"].get(key):
            raise ValueError(f"V14 frozen lineage mismatch for {key}: {paths[key]}")
    return observed


def _binding_controls(rows: list[dict[str, Any]], proof_model: Any) -> dict[str, dict[str, Any]]:
    shuffled_proof: dict[str, Any] = {}
    shuffled_actions: dict[str, bool] = {}
    marginal: dict[str, bool] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        ordered = sorted(
            family_rows,
            key=lambda row: hashlib.sha256(
                f"v14-proof-binding|{row['sample_id']}".encode("utf-8")
            ).hexdigest(),
        )
        for index, row in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)]
            features = (
                list(row["features"][: len(FEATURE_NAMES)])
                + list(donor["features"][len(FEATURE_NAMES) :])
            )
            score = float(proof_model.predict([features])[0])
            shuffled_proof[str(row["sample_id"])] = {
                "score": score,
                "donor_sample_id": str(donor["sample_id"]),
            }
        flags = [bool(row["authentic_recover"]) for row in ordered]
        for index, row in enumerate(ordered):
            shuffled_actions[str(row["sample_id"])] = flags[(index + 1) % len(flags)]
        selected_count = sum(flags)
        marginal_order = sorted(
            (str(row["sample_id"]) for row in family_rows),
            key=lambda sample_id: hashlib.sha256(
                f"v14-marginal|{sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        selected = set(marginal_order[:selected_count])
        for row in family_rows:
            sample_id = str(row["sample_id"])
            marginal[sample_id] = sample_id in selected
    return {
        "shuffled_proof": shuffled_proof,
        "shuffled_actions": shuffled_actions,
        "marginal": marginal,
    }


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    wins = sum(
        row["conditions"][left]["correct"] and not row["conditions"][right]["correct"]
        for row in rows
    )
    losses = sum(
        row["conditions"][right]["correct"] and not row["conditions"][left]["correct"]
        for row in rows
    )
    return {
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "ties": len(rows) - wins - losses,
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    observed_lineage = _verify_lineage(config)

    source_path = Path(config["source"]["receipt"])
    source_receipt = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_receipt(source_receipt)
    artifact_path = Path(config["grounder"]["artifact"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    proof_model, base_model, permuted_model, threshold = validate_v14_artifact(artifact)
    training_report = json.loads(
        Path(config["grounder"]["training_report"]).read_text(encoding="utf-8")
    )
    if training_report.get("status") != "V14_PROOF_GROUNDER_DEVELOPMENT_GATE_PASSED":
        raise ValueError("V14 proof grounder development gate failed")
    if training_report.get("artifact_sha256") != artifact.get("artifact_sha256"):
        raise ValueError("V14 artifact/training report content hash mismatch")
    if artifact.get("config_sha256") != observed_lineage["training_config_sha256"]:
        raise ValueError("V14 artifact training-config lineage mismatch")

    target = config["target"]
    manifest = json.loads(Path(target["split_manifest"]).read_text(encoding="utf-8"))
    sample_ids = list(manifest["benchmarks"]["clevrer"]["splits"]["formal"])
    expected_count = int(config["gates"]["expected_formal_samples"])
    if len(sample_ids) != expected_count or len(set(sample_ids)) != expected_count:
        raise ValueError("V14 formal count/uniqueness preflight failed")

    executor_root = Path(target["official_root"]) / "executor"
    sys.path.insert(0, str(executor_root))
    from executor import Executor  # type: ignore
    from simulation import Simulation  # type: ignore

    annotations_path = executor_root / "data/validation.json"
    annotations = {
        int(row["scene_index"]): row
        for row in json.loads(annotations_path.read_text(encoding="utf-8"))
    }
    prediction_hash_cache: dict[str, str] = {}
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
            / str(target["explicit_relation_prediction_directory"])
            / f"sim_{scene_id:05d}.json",
            "trajectory": executor_root / "data/propnet_preds"
            / str(target["trajectory_prediction_directory"])
            / f"sim_{scene_id:05d}.json",
        }
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(paths)
        for path in paths.values():
            prediction_hash_cache.setdefault(str(path.resolve()), _sha256(path))
        explicit_executor = Executor(Simulation(str(paths["explicit"]), use_event_ann=True))
        trajectory_executor = Executor(Simulation(str(paths["trajectory"]), use_event_ann=False))
        proof_features, proof_receipts = paired_proof_features(
            explicit_executor, trajectory_executor, question_program, choice_programs,
        )
        explicit_raw = [str(value["explicit_answer"]) for value in proof_receipts]
        trajectory_raw = [str(value["trajectory_answer"]) for value in proof_receipts]
        explicit_answer = "".join("1" if value == "yes" else "0" for value in explicit_raw)
        trajectory_answer = "".join("1" if value == "yes" else "0" for value in trajectory_raw)
        base_features = build_features(
            family=family,
            question_program=question_program,
            choice_programs=choice_programs,
            explicit_answer=explicit_answer,
            trajectory_answer=trajectory_answer,
            explicit_error_count=explicit_raw.count("error"),
        )
        features = tuple(base_features + proof_features)
        if len(features) != len(V14_FEATURE_NAMES):
            raise AssertionError("V14 runtime feature contract drift")
        proof_score = float(proof_model.predict([features])[0])
        base_score = float(base_model.predict([base_features])[0])
        permuted_score = float(permuted_model.predict([features])[0])
        # Official answers/programs become evaluator-only data after all learned
        # runtime scores above have been computed from target-native receipts.
        gold_answer = "".join(
            "1" if choice["answer"] == "correct" else "0"
            for choice in question["choices"]
        )
        base_rows.append({
            "sample_id": sample_id,
            "family": family,
            "gold_answer": gold_answer,
            "explicit_answer": explicit_answer,
            "trajectory_answer": trajectory_answer,
            "explicit_raw": explicit_raw,
            "trajectory_raw": trajectory_raw,
            "features": features,
            "proof_score": proof_score,
            "base_score": base_score,
            "permuted_score": permuted_score,
            "authentic_recover": proof_score > threshold,
            "compiler_question_exact": question_program
            == normalize_official_program(question["program"]),
            "compiler_choices_exact": all(
                compiled == normalize_official_program(choice["program"])
                for compiled, choice in zip(choice_programs, question["choices"])
            ),
            "proof_receipts_sha256": _content_hash(proof_receipts),
        })

    bindings = _binding_controls(base_rows, proof_model)
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        sample_id = str(base["sample_id"])
        shuffled_proof = bindings["shuffled_proof"][sample_id]
        decisions = {
            "target_explicit_no_recovery": False,
            "target_trajectory_only": True,
            "authentic_sokoban_proof_cate_recover": base["authentic_recover"],
            "target_base_receipt_cate_recover": base["base_score"] > threshold,
            "permuted_uplift_cate_recover": base["permuted_score"] > threshold,
            "shuffled_proof_binding_recover": shuffled_proof["score"] > threshold,
            "source_inverted_effect_recover": base["proof_score"] < -threshold,
            "source_shuffled_action_binding_recover": bindings["shuffled_actions"][sample_id],
            "target_family_matched_marginal_recover": bindings["marginal"][sample_id],
            "target_error_only_recover": "error" in base["explicit_raw"],
            "target_disagreement_recover": base["explicit_answer"] != base["trajectory_answer"],
        }
        conditions = {}
        for condition in CONDITIONS:
            recover = bool(decisions[condition])
            answer = base["trajectory_answer"] if recover else base["explicit_answer"]
            conditions[condition] = {
                "recover": recover,
                "selected_native_representation": "trajectory" if recover else "explicit_relation",
                "answer": answer,
                "correct": answer == base["gold_answer"],
            }
        rows.append({
            "sample_id": sample_id,
            "family": base["family"],
            "gold_answer": base["gold_answer"],
            "compiler_question_exact": base["compiler_question_exact"],
            "compiler_choices_exact": base["compiler_choices_exact"],
            "proof_receipts_sha256": base["proof_receipts_sha256"],
            "typed_target_receipt": {
                "explicit_error_count": base["explicit_raw"].count("error"),
                "explicit_answer": base["explicit_answer"],
                "trajectory_answer": base["trajectory_answer"],
                "answer_disagreement": base["explicit_answer"] != base["trajectory_answer"],
            },
            "grounder": {
                "feature_names": list(V14_FEATURE_NAMES),
                "features": list(map(float, base["features"])),
                "proof_predicted_uplift": base["proof_score"],
                "base_only_predicted_uplift": base["base_score"],
                "permuted_predicted_uplift": base["permuted_score"],
                "shuffled_proof_predicted_uplift": shuffled_proof["score"],
                "shuffled_proof_donor_sample_id": shuffled_proof["donor_sample_id"],
                "decision_threshold": threshold,
            },
            "conditions": conditions,
        })

    count = len(rows)
    metrics = {
        condition: {
            "correct": sum(row["conditions"][condition]["correct"] for row in rows),
            "accuracy": sum(row["conditions"][condition]["correct"] for row in rows) / count,
            "recoveries": sum(row["conditions"][condition]["recover"] for row in rows),
        }
        for condition in CONDITIONS
    }
    authentic = "authentic_sokoban_proof_cate_recover"
    paired = {
        condition: _paired(rows, authentic, condition)
        for condition in CONDITIONS if condition != authentic
    }
    primary = paired["target_explicit_no_recovery"]
    gate_config = config["gates"]
    binding_controls = tuple(gate_config["binding_control_conditions"])
    gates = {
        "confirmed_source_receipt": True,
        "development_gate_passed": True,
        "frozen_lineage_verified": True,
        "formal_sample_count": count == expected_count,
        "compiler_exact_on_all_rows": all(
            row["compiler_question_exact"] and row["compiler_choices_exact"] for row in rows
        ),
        "minimum_authentic_recoveries": metrics[authentic]["recoveries"]
        >= int(gate_config["minimum_authentic_recoveries"]),
        "authentic_above_primary": metrics[authentic]["correct"]
        > metrics["target_explicit_no_recovery"]["correct"],
        "authentic_primary_net_wins": primary["net_wins"]
        >= int(gate_config["minimum_primary_net_wins"]),
        "authentic_primary_p_value": primary["exact_two_sided_p"]
        <= float(gate_config["maximum_primary_exact_p"]),
        "proof_receipts_above_base_only": metrics[authentic]["correct"]
        > metrics["target_base_receipt_cate_recover"]["correct"],
        "proof_receipts_positive_paired_vs_base_only": paired[
            "target_base_receipt_cate_recover"
        ]["net_wins"] > 0,
        "authentic_strictly_above_binding_controls": all(
            metrics[authentic]["correct"] > metrics[condition]["correct"]
            for condition in binding_controls
        ),
        "authentic_positive_paired_vs_binding_controls": all(
            paired[condition]["net_wins"] > 0 for condition in binding_controls
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": 14,
        "status": (
            "SOKOBAN_TO_CLEVRER_PROOF_NEUROSYMBOLIC_TRANSFER_FORMAL_VALIDATED"
            if passed else "SOKOBAN_TO_CLEVRER_PROOF_NEUROSYMBOLIC_TRANSFER_FORMAL_FAILED"
        ),
        "benchmark": "clevrer",
        "split": "formal",
        "samples": count,
        "source_symbolic_contract": config["source"]["transferred_contract"],
        "target_neural_grounder_estimand": artifact["estimand"],
        "conditions": metrics,
        "paired_authentic": paired,
        "gates": gates,
        "rows": rows,
        "lineage": {
            "config_sha256": _sha256(args.config),
            "verified_file_sha256": observed_lineage,
            "artifact_content_sha256": artifact["artifact_sha256"],
            "annotations_sha256": _sha256(annotations_path),
            "prediction_file_sha256": prediction_hash_cache,
        },
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "samples": count,
        "conditions": metrics,
        "paired_primary": primary,
        "paired_vs_base_only": paired["target_base_receipt_cate_recover"],
        "gates": gates,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
