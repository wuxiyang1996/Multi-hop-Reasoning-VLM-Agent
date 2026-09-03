#!/usr/bin/env python3
"""Prospective CLEVRER evaluation of Sokoban recovery with a frozen CATE grounder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motif_transfer.clevrer_query_compiler import (  # noqa: E402
    compile_choice,
    compile_question,
    normalize_official_program,
)
from motif_transfer.sokoban_video_recovery import (  # noqa: E402
    exact_binomial_two_sided,
    parse_executor_effect,
    validate_source_receipt,
)
from motif_transfer.video_recovery_cate import (  # noqa: E402
    FEATURE_NAMES,
    build_features,
    validate_frozen_artifact,
)


CONDITIONS = (
    "target_explicit_no_recovery",
    "target_trajectory_only",
    "authentic_sokoban_cate_recover",
    "permuted_uplift_cate_recover",
    "source_inverted_effect_recover",
    "source_shuffled_grounding_recover",
    "target_family_matched_marginal_recover",
    "target_error_only_recover",
    "target_disagreement_recover",
)
CONTROL_CONDITIONS = tuple(
    value for value in CONDITIONS if value != "authentic_sokoban_cate_recover"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_parts(sample_id: str) -> tuple[int, int]:
    video, question = sample_id.split(".mp4.Q", 1)
    return int(video.rsplit("_", 1)[1]), int(question)


def _verify_lineage(config: Mapping[str, Any]) -> dict[str, str]:
    if config.get("status") != "FROZEN_BEFORE_V13_FORMAL_OUTCOMES":
        raise ValueError("formal config was not frozen")
    if config["target"].get("split") != "formal":
        raise ValueError("V13 runner may only open the frozen formal role")
    paths = {
        "source_receipt_sha256": Path(config["source"]["receipt"]),
        "split_manifest_sha256": Path(config["target"]["split_manifest"]),
        "cate_artifact_file_sha256": Path(config["grounder"]["artifact"]),
        "training_report_sha256": Path(config["grounder"]["training_report"]),
        "training_config_sha256": Path(config["grounder"]["training_config"]),
        "runner_sha256": Path(__file__).resolve(),
        "cate_module_sha256": REPO / "src/motif_transfer/video_recovery_cate.py",
        "recovery_module_sha256": REPO / "src/motif_transfer/sokoban_video_recovery.py",
        "target_compiler_sha256": REPO / "src/motif_transfer/clevrer_query_compiler.py",
    }
    observed = {key: _sha256(path) for key, path in paths.items()}
    for key, value in observed.items():
        expected = str(config.get("frozen_lineage", {}).get(key) or "")
        if not expected or value != expected:
            raise ValueError(f"frozen lineage mismatch for {key}: {paths[key]}")
    return observed


def _execute_pair(
    *,
    executor_root: Path,
    scene_id: int,
    question_program: Sequence[str],
    choice_programs: Sequence[Sequence[str]],
    target: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    from executor import Executor  # type: ignore
    from simulation import Simulation  # type: ignore

    specs = {
        "explicit": (str(target["explicit_relation_prediction_directory"]), True),
        "trajectory": (str(target["trajectory_prediction_directory"]), False),
    }
    output: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name, (directory, use_event_ann) in specs.items():
        path = executor_root / "data/propnet_preds" / directory / f"sim_{scene_id:05d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        executor = Executor(Simulation(str(path), use_event_ann=use_event_ann))
        raw = [
            executor.run(list(choice) + list(question_program), debug=False)
            for choice in choice_programs
        ]
        if any(value not in {"yes", "no", "error"} for value in raw):
            raise ValueError("unexpected CLEVRER executor result")
        output[name] = {
            "answer": "".join("1" if value == "yes" else "0" for value in raw),
            "raw_executor_results": raw,
        }
        hashes[name] = _sha256(path)
    return output, hashes


def _matched_controls(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    shuffled: dict[str, bool] = {}
    marginal: dict[str, bool] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if str(row["family"]) == family]
        ordered = sorted(
            family_rows,
            key=lambda row: hashlib.sha256(
                f"v13-shuffle|{row['sample_id']}".encode("utf-8")
            ).hexdigest(),
        )
        flags = [bool(row["authentic_recover"]) for row in ordered]
        for index, row in enumerate(ordered):
            shuffled[str(row["sample_id"])] = flags[(index + 1) % len(flags)]
        selected_count = sum(flags)
        marginal_order = sorted(
            (str(row["sample_id"]) for row in family_rows),
            key=lambda sample_id: hashlib.sha256(
                f"v13-marginal|{sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        selected = set(marginal_order[:selected_count])
        for row in family_rows:
            sample_id = str(row["sample_id"])
            marginal[sample_id] = sample_id in selected
    return {"shuffled": shuffled, "marginal": marginal}


def _paired(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    wins = sum(
        bool(row["conditions"][left]["correct"])
        and not bool(row["conditions"][right]["correct"])
        for row in rows
    )
    losses = sum(
        bool(row["conditions"][right]["correct"])
        and not bool(row["conditions"][left]["correct"])
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
    authentic_model, permuted_model, threshold = validate_frozen_artifact(artifact)
    training_report = json.loads(
        Path(config["grounder"]["training_report"]).read_text(encoding="utf-8")
    )
    if training_report.get("status") != "CATE_DEVELOPMENT_GATE_PASSED":
        raise ValueError("CATE development gate did not pass")
    if training_report.get("artifact_sha256") != artifact.get("artifact_sha256"):
        raise ValueError("training report and artifact content hashes disagree")
    if artifact.get("config_sha256") != observed_lineage["training_config_sha256"]:
        raise ValueError("artifact was not produced by the frozen training config")

    target = config["target"]
    manifest = json.loads(Path(target["split_manifest"]).read_text(encoding="utf-8"))
    sample_ids = list(manifest["benchmarks"]["clevrer"]["splits"]["formal"])
    if len(sample_ids) != int(config["gates"]["expected_formal_samples"]):
        raise ValueError("unexpected formal sample count")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("formal split contains duplicate sample IDs")

    executor_root = Path(target["official_root"]) / "executor"
    sys.path.insert(0, str(executor_root))
    annotations_path = executor_root / "data/validation.json"
    annotations = {
        int(row["scene_index"]): row
        for row in json.loads(annotations_path.read_text(encoding="utf-8"))
    }
    base_rows: list[dict[str, Any]] = []
    prediction_hashes: dict[str, str] = {}
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
        native, hashes = _execute_pair(
            executor_root=executor_root,
            scene_id=scene_id,
            question_program=question_program,
            choice_programs=choice_programs,
            target=target,
        )
        for name, value in hashes.items():
            prediction_hashes[f"{sample_id}:{name}"] = value
        effect = parse_executor_effect(native["explicit"]["raw_executor_results"])
        features = build_features(
            family=family,
            question_program=question_program,
            choice_programs=choice_programs,
            explicit_answer=native["explicit"]["answer"],
            trajectory_answer=native["trajectory"]["answer"],
            explicit_error_count=effect.error_count,
        )
        authentic_score = float(authentic_model.predict([features])[0])
        permuted_score = float(permuted_model.predict([features])[0])
        # Gold labels and official programs are attached only after every policy
        # decision above has been made from target-native runtime receipts.
        gold_answer = "".join(
            "1" if choice["answer"] == "correct" else "0"
            for choice in question["choices"]
        )
        base_rows.append({
            "sample_id": sample_id,
            "family": family,
            "gold_answer": gold_answer,
            "compiled_question_program": question_program,
            "compiled_choice_programs": choice_programs,
            "compiler_question_exact": question_program
            == normalize_official_program(question["program"]),
            "compiler_choices_exact": all(
                compiled == normalize_official_program(choice["program"])
                for compiled, choice in zip(choice_programs, question["choices"])
            ),
            "explicit": native["explicit"],
            "trajectory": native["trajectory"],
            "effect": effect,
            "features": features,
            "authentic_score": authentic_score,
            "permuted_score": permuted_score,
            "authentic_recover": authentic_score > threshold,
        })

    matched = _matched_controls(base_rows)
    rows: list[dict[str, Any]] = []
    for base in base_rows:
        sample_id = str(base["sample_id"])
        explicit = base["explicit"]
        trajectory = base["trajectory"]
        decisions = {
            "target_explicit_no_recovery": False,
            "target_trajectory_only": True,
            "authentic_sokoban_cate_recover": bool(base["authentic_recover"]),
            "permuted_uplift_cate_recover": base["permuted_score"] > threshold,
            "source_inverted_effect_recover": base["authentic_score"] < -threshold,
            "source_shuffled_grounding_recover": matched["shuffled"][sample_id],
            "target_family_matched_marginal_recover": matched["marginal"][sample_id],
            "target_error_only_recover": base["effect"].error_count > 0,
            "target_disagreement_recover": explicit["answer"] != trajectory["answer"],
        }
        conditions: dict[str, Any] = {}
        for condition in CONDITIONS:
            recover = bool(decisions[condition])
            selected = trajectory if recover else explicit
            conditions[condition] = {
                "recover": recover,
                "selected_native_representation": (
                    "trajectory" if recover else "explicit_relation"
                ),
                "answer": selected["answer"],
                "correct": selected["answer"] == base["gold_answer"],
            }
        effect = base["effect"]
        rows.append({
            "sample_id": sample_id,
            "family": base["family"],
            "gold_answer": base["gold_answer"],
            "compiled_question_program": base["compiled_question_program"],
            "compiled_choice_programs": base["compiled_choice_programs"],
            "compiler_question_exact": base["compiler_question_exact"],
            "compiler_choices_exact": base["compiler_choices_exact"],
            "typed_target_receipt": {
                "explicit_error_count": effect.error_count,
                "explicit_answer": explicit["answer"],
                "trajectory_answer": trajectory["answer"],
                "answer_disagreement": explicit["answer"] != trajectory["answer"],
            },
            "grounder": {
                "feature_names": list(FEATURE_NAMES),
                "features": list(map(float, base["features"])),
                "authentic_predicted_uplift": base["authentic_score"],
                "permuted_predicted_uplift": base["permuted_score"],
                "decision_threshold": threshold,
            },
            "conditions": conditions,
        })

    count = len(rows)
    metrics = {
        condition: {
            "correct": sum(bool(row["conditions"][condition]["correct"]) for row in rows),
            "accuracy": sum(bool(row["conditions"][condition]["correct"]) for row in rows) / count,
            "recoveries": sum(bool(row["conditions"][condition]["recover"]) for row in rows),
        }
        for condition in CONDITIONS
    }
    authentic = "authentic_sokoban_cate_recover"
    paired = {
        condition: _paired(rows, authentic, condition)
        for condition in CONTROL_CONDITIONS
    }
    primary = paired["target_explicit_no_recovery"]
    gate_config = config["gates"]
    contrast_controls = tuple(gate_config["strict_control_conditions"])
    gates = {
        "confirmed_source_receipt": True,
        "development_gate_passed": True,
        "frozen_lineage_verified": True,
        "formal_sample_count": count == int(gate_config["expected_formal_samples"]),
        "compiler_exact_on_all_rows": all(
            row["compiler_question_exact"] and row["compiler_choices_exact"]
            for row in rows
        ),
        "minimum_authentic_recoveries": metrics[authentic]["recoveries"]
        >= int(gate_config["minimum_authentic_recoveries"]),
        "authentic_above_primary": metrics[authentic]["correct"]
        > metrics["target_explicit_no_recovery"]["correct"],
        "authentic_primary_net_wins": primary["net_wins"]
        >= int(gate_config["minimum_primary_net_wins"]),
        "authentic_primary_p_value": primary["exact_two_sided_p"]
        <= float(gate_config["maximum_primary_exact_p"]),
        "authentic_strictly_above_frozen_controls": all(
            metrics[authentic]["correct"] > metrics[condition]["correct"]
            for condition in contrast_controls
        ),
        "authentic_positive_paired_vs_binding_controls": all(
            paired[condition]["net_wins"] > 0
            for condition in contrast_controls
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": 13,
        "status": (
            "SOKOBAN_TO_CLEVRER_NEUROSYMBOLIC_TRANSFER_FORMAL_VALIDATED"
            if passed else "SOKOBAN_TO_CLEVRER_NEUROSYMBOLIC_TRANSFER_FORMAL_FAILED"
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
            "prediction_sha256": prediction_hashes,
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
        "gates": gates,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
