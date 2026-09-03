#!/usr/bin/env python3
"""Evaluate Sokoban VERIFY/REFUTED/REPLAN transfer on CLEVRER."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


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
    authentic_recovery_decision,
    exact_binomial_two_sided,
    parse_executor_effect,
    validate_source_receipt,
)


CONDITIONS = (
    "target_explicit_no_recovery",
    "target_trajectory_only",
    "authentic_sokoban_verify_recover",
    "source_availability_only",
    "source_inverted_effect",
    "source_position_prior",
    "shuffled_refutation_binding",
    "source_marginal_recovery",
    "target_disagreement_recovery",
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


def _verify_lineage(config: Mapping[str, Any]) -> None:
    paths = {
        "source_receipt_sha256": Path(config["source"]["receipt"]),
        "split_manifest_sha256": Path(config["target"]["split_manifest"]),
        "runner_sha256": Path(__file__).resolve(),
        "recovery_module_sha256": REPO / "src/motif_transfer/sokoban_video_recovery.py",
        "target_compiler_sha256": REPO / "src/motif_transfer/clevrer_query_compiler.py",
    }
    base_report = config["target"].get("base_report")
    if base_report:
        paths["base_report_sha256"] = Path(base_report)
    development_report = config.get("development_report")
    if development_report:
        paths["development_report_sha256"] = Path(development_report)
    for key, path in paths.items():
        expected = str(config.get("frozen_lineage", {}).get(key) or "")
        if not expected or _sha256(path) != expected:
            raise ValueError(f"frozen lineage mismatch for {key}: {path}")


def _execute_native_pair(
    *,
    executor_root: Path,
    scene_id: int,
    question_program: Sequence[str],
    choice_programs: Sequence[Sequence[str]],
    target: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    from simulation import Simulation  # type: ignore
    from executor import Executor  # type: ignore

    specs = {
        "explicit": (
            str(target["explicit_relation_prediction_directory"]), True,
        ),
        "trajectory": (
            str(target["trajectory_prediction_directory"]), False,
        ),
    }
    output = {}
    hashes = {}
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


def _base_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    target = config["target"]
    if target.get("base_report"):
        report = json.loads(Path(target["base_report"]).read_text(encoding="utf-8"))
        rows = []
        for source in report["rows"]:
            rows.append({
                "sample_id": source["sample_id"],
                "family": source["family"],
                "gold_answer": source["gold_answer"],
                "compiled_question_program": source["compiled_question_program"],
                "compiled_choice_programs": source["compiled_choice_programs"],
                "compiler_question_exact": source["compiler_question_exact"],
                "compiler_choices_exact": source["compiler_choices_exact"],
                "explicit": {
                    key: source["conditions"]["target_always_explicit_relation"][key]
                    for key in ("answer", "raw_executor_results")
                },
                "trajectory": {
                    key: source["conditions"]["target_always_trajectory"][key]
                    for key in ("answer", "raw_executor_results")
                },
            })
        return rows, {}

    manifest_path = Path(target["split_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(target["split"])
    sample_ids = manifest["benchmarks"]["clevrer"]["splits"][split]
    executor_root = Path(target["official_root"]) / "executor"
    sys.path.insert(0, str(executor_root))
    annotations_path = executor_root / "data/validation.json"
    annotations = {
        int(row["scene_index"]): row
        for row in json.loads(annotations_path.read_text(encoding="utf-8"))
    }
    rows = []
    prediction_hashes = {}
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
        native, hashes = _execute_native_pair(
            executor_root=executor_root,
            scene_id=scene_id,
            question_program=question_program,
            choice_programs=choice_programs,
            target=target,
        )
        for name, value in hashes.items():
            prediction_hashes[f"{sample_id}:{name}"] = value
        rows.append({
            "sample_id": sample_id,
            "family": family,
            "gold_answer": "".join(
                "1" if choice["answer"] == "correct" else "0"
                for choice in question["choices"]
            ),
            "compiled_question_program": question_program,
            "compiled_choice_programs": choice_programs,
            "compiler_question_exact": question_program
            == normalize_official_program(question["program"]),
            "compiler_choices_exact": all(
                compiled == normalize_official_program(choice["program"])
                for compiled, choice in zip(choice_programs, question["choices"])
            ),
            **native,
        })
    return rows, prediction_hashes


def _control_recovery_flags(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, bool]]:
    authentic = {
        str(row["sample_id"]): authentic_recovery_decision(
            parse_executor_effect(row["explicit"]["raw_executor_results"])
        )
        for row in rows
    }
    shuffled: dict[str, bool] = {}
    marginal: dict[str, bool] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        ids = [str(row["sample_id"]) for row in rows if str(row["family"]) == family]
        ordered = sorted(
            ids,
            key=lambda value: hashlib.sha256(f"shuffle|{value}".encode()).hexdigest(),
        )
        values = [authentic[value] for value in ordered]
        for index, sample_id in enumerate(ordered):
            shuffled[sample_id] = values[(index + 1) % len(values)]
        count = sum(values)
        marginal_order = sorted(
            ids,
            key=lambda value: hashlib.sha256(f"marginal|{value}".encode()).hexdigest(),
        )
        selected = set(marginal_order[:count])
        for sample_id in ids:
            marginal[sample_id] = sample_id in selected
    return {"authentic": authentic, "shuffled": shuffled, "marginal": marginal}


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
        "ties": len(rows) - wins - losses,
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _verify_lineage(config)
    source_path = Path(config["source"]["receipt"])
    source_receipt = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_receipt(source_receipt)
    base_rows, prediction_hashes = _base_rows(config)
    flags = _control_recovery_flags(base_rows)

    rows = []
    for base in base_rows:
        sample_id = str(base["sample_id"])
        explicit = dict(base["explicit"])
        trajectory = dict(base["trajectory"])
        effect = parse_executor_effect(explicit["raw_executor_results"])
        decisions = {
            "target_explicit_no_recovery": False,
            "target_trajectory_only": True,
            "authentic_sokoban_verify_recover": flags["authentic"][sample_id],
            "source_availability_only": False,
            "source_inverted_effect": not flags["authentic"][sample_id],
            "source_position_prior": True,
            "shuffled_refutation_binding": flags["shuffled"][sample_id],
            "source_marginal_recovery": flags["marginal"][sample_id],
            "target_disagreement_recovery": explicit["answer"] != trajectory["answer"],
        }
        condition_rows = {}
        for condition in CONDITIONS:
            recover = decisions[condition]
            selected = trajectory if recover else explicit
            condition_rows[condition] = {
                "recover": recover,
                "selected_native_representation": (
                    "trajectory" if recover else "explicit_relation"
                ),
                "answer": selected["answer"],
                "correct": selected["answer"] == base["gold_answer"],
            }
        rows.append({
            **{key: base[key] for key in (
                "sample_id", "family", "gold_answer", "compiled_question_program",
                "compiled_choice_programs", "compiler_question_exact",
                "compiler_choices_exact",
            )},
            "typed_effect_receipt": {
                "expected_effect_observed": effect.expected_effect_observed,
                "expected_effect_refuted": effect.expected_effect_refuted,
                "raw_result_count": effect.raw_result_count,
                "error_count": effect.error_count,
            },
            "conditions": condition_rows,
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
    authentic = "authentic_sokoban_verify_recover"
    paired = {
        condition: _paired(rows, authentic, condition)
        for condition in CONDITIONS if condition != authentic
    }
    primary = paired["target_explicit_no_recovery"]
    controls = [condition for condition in CONDITIONS if condition != authentic]
    gate_config = config["gates"]
    gates = {
        "confirmed_source_receipt": True,
        "compiler_exact_on_all_rows": all(
            row["compiler_question_exact"] and row["compiler_choices_exact"]
            for row in rows
        ),
        "authentic_action_contrast": metrics[authentic]["recoveries"]
        >= int(gate_config["minimum_authentic_recoveries"]),
        "authentic_strictly_above_all_controls": all(
            metrics[authentic]["correct"] > metrics[condition]["correct"]
            for condition in controls
        ),
        "authentic_primary_net_wins": (
            primary["wins"] - primary["losses"]
            >= int(gate_config["minimum_primary_net_wins"])
        ),
        "authentic_primary_p_value": (
            primary["exact_two_sided_p"]
            <= float(gate_config["maximum_primary_exact_p"])
        ),
        "authentic_positive_vs_shuffled_and_marginal": all(
            paired[condition]["wins"] > paired[condition]["losses"]
            for condition in (
                "shuffled_refutation_binding", "source_marginal_recovery",
            )
        ),
    }
    passed = all(gates.values())
    split = str(config["target"]["split"])
    report = {
        "schema_version": 1,
        "status": (
            "SOKOBAN_TO_CLEVRER_RECOVERY_DEVELOPMENT_PASS"
            if passed and split == "development"
            else "SOKOBAN_TO_CLEVRER_RECOVERY_FORMAL_VALIDATED"
            if passed
            else "SOKOBAN_TO_CLEVRER_RECOVERY_TRANSFER_FAIL"
        ),
        "benchmark": "clevrer",
        "split": split,
        "samples": count,
        "source_artifact": {
            "domain": source_receipt["source_domain"],
            "artifact_version": source_receipt["artifact_version"],
            "artifact_sha256": source_receipt["artifact"]["artifact_sha256"],
            "fresh_confirmation_report_sha256": source_receipt["fresh_confirmation"]["report_sha256"],
        },
        "conditions": metrics,
        "paired_authentic": paired,
        "gates": gates,
        "rows": rows,
        "lineage": {
            "config_sha256": _sha256(args.config),
            "source_receipt_file_sha256": _sha256(source_path),
            "split_manifest_sha256": _sha256(Path(config["target"]["split_manifest"])),
            "prediction_sha256": prediction_hashes,
        },
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "split": split, "samples": count,
        "conditions": metrics, "paired_primary": primary,
        "gates": gates, "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
