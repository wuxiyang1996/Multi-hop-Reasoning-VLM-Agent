#!/usr/bin/env python3
"""Run CLEVRER neural dynamics + symbolic executor as a disclosed oracle diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_parts(sample_id: str) -> tuple[int, int]:
    video, question = sample_id.split(".mp4.Q", 1)
    return int(video.rsplit("_", 1)[1]), int(question)


def _program(tokens: Sequence[str]) -> list[str]:
    # Public annotations use the older name; the official executor registers the
    # semantically identical current name.
    return ["filter_counterfact" if token == "get_counterfact" else token for token in tokens]


def _baseline(source: Mapping[str, Any]) -> str:
    return str(max(
        source["world_model"]["particles"],
        key=lambda row: (float(row["prior_weight"]), str(row["native_answer"])),
    )["native_answer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = Path(config["source"]["typed_summary"])
    if _sha256(source_path) != config["source"]["typed_summary_sha256"]:
        raise ValueError("source receipt hash mismatch")
    source_gate = json.loads(source_path.read_text(encoding="utf-8"))
    if source_gate.get("status") != "SOURCE_TYPED_GATE_PASSED":
        raise ValueError("source typed gate did not pass")
    manifest_path = Path(config["split_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(config["split"])
    if split != "adaptation":
        raise ValueError("this diagnostic is adaptation-only")
    sample_ids = manifest["benchmarks"]["clevrer"]["splits"][split]
    sources = {
        str(row["sample_id"]): row
        for row in json.loads(Path(config["structured_receipts"]).read_text(encoding="utf-8"))
    }
    if set(sample_ids) - set(sources):
        raise ValueError("structured receipts do not cover frozen adaptation IDs")

    official_root = Path(config["official_root"])
    executor_root = official_root / "executor"
    sys.path.insert(0, str(executor_root))
    from simulation import Simulation  # type: ignore  # noqa: E402
    from executor import Executor  # type: ignore  # noqa: E402

    annotations_path = executor_root / "data/validation.json"
    annotations = {
        int(row["scene_index"]): row
        for row in json.loads(annotations_path.read_text(encoding="utf-8"))
    }
    rows = []
    prediction_hashes: dict[str, dict[str, str]] = {
        condition: {} for condition in config["conditions"]
    }
    for sample_id in sample_ids:
        scene_id, question_id = _sample_parts(sample_id)
        question = next(
            row for row in annotations[scene_id]["questions"]
            if int(row["question_id"]) == question_id
        )
        # Runtime payload deliberately excludes every choice's `answer` field.
        runtime_question_program = _program(question["program"])
        runtime_choice_programs = [_program(choice["program"]) for choice in question["choices"]]
        gold = "".join("1" if choice["answer"] == "correct" else "0" for choice in question["choices"])
        condition_rows = {}
        for condition, spec in config["conditions"].items():
            prediction_path = (
                executor_root / "data/propnet_preds"
                / spec["prediction_directory"] / f"sim_{scene_id:05d}.json"
            )
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction_hashes[condition][sample_id] = _sha256(prediction_path)
            simulation = Simulation(
                str(prediction_path),
                use_event_ann=bool(spec["use_predicted_collision_edges"]),
            )
            executor = Executor(simulation)
            raw = [
                executor.run(choice_program + runtime_question_program, debug=False)
                for choice_program in runtime_choice_programs
            ]
            if any(value not in {"yes", "no", "error"} for value in raw):
                raise ValueError("unexpected official executor result")
            answer = "".join("1" if value == "yes" else "0" for value in raw)
            condition_rows[condition] = {
                "answer": answer, "raw_executor_results": raw,
                "correct": answer == gold,
            }
        baseline = _baseline(sources[sample_id])
        rows.append({
            "sample_id": sample_id, "family": question["question_type"],
            "gold_answer": gold, "baseline_answer": baseline,
            "baseline_correct": baseline == gold, "conditions": condition_rows,
            "runtime_program_oracle": True,
            "answer_labels_seen_by_dynamics_or_executor": False,
        })

    count = len(rows)
    conditions = {
        name: {
            "correct": sum(bool(row["conditions"][name]["correct"]) for row in rows),
            "accuracy": sum(bool(row["conditions"][name]["correct"]) for row in rows) / count,
        }
        for name in config["conditions"]
    }
    baseline = sum(bool(row["baseline_correct"]) for row in rows)
    without = conditions["target_native_without_edge"]["correct"]
    with_edge = conditions["target_native_with_edge"]["correct"]
    gates = {
        "frozen_adaptation_complete": count == len(sample_ids),
        "source_gate_valid": source_gate["edge_replication_gate"]["status"] == "EDGE_REPLICATION_GATE_PASSED",
        "target_dynamics_executor_headroom": max(without, with_edge) > baseline,
        "edge_aware_dynamics_above_no_edge": with_edge > without,
        # The authentic compilation and a complete target-native controller both
        # choose the same edge-aware target architecture. A tie is evidence that
        # source-specific incremental transfer is not identified.
        "authentic_above_same_architecture_target_control": with_edge > with_edge,
    }
    family_metrics = {}
    for family in sorted({row["family"] for row in rows}):
        subset = [row for row in rows if row["family"] == family]
        family_metrics[family] = {
            name: sum(bool(row["conditions"][name]["correct"]) for row in subset)
            for name in config["conditions"]
        } | {"samples": len(subset)}
    report = {
        "schema_version": 1,
        "status": (
            "TARGET_DYNAMICS_HEADROOM_PASS_TRANSFER_NOT_IDENTIFIED"
            if gates["target_dynamics_executor_headroom"]
            and not gates["authentic_above_same_architecture_target_control"]
            else "DIAGNOSTIC_FAIL"
        ),
        "benchmark": "clevrer", "split": split, "samples": count,
        "baseline": {"correct": baseline, "accuracy": baseline / count},
        "conditions": conditions,
        "authentic_source_bind_relate_compile": conditions["target_native_with_edge"],
        "same_architecture_target_control": conditions["target_native_with_edge"],
        "source_edge_ablation": conditions["target_native_without_edge"],
        "family_metrics": family_metrics, "gates": gates, "rows": rows,
        "lineage": {
            "config_sha256": _sha256(args.config),
            "source_gate_sha256": _sha256(source_path),
            "split_manifest_sha256": _sha256(manifest_path),
            "official_annotations_sha256": _sha256(annotations_path),
            "prediction_sha256": prediction_hashes,
        },
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "baseline": report["baseline"],
        "conditions": conditions, "family_metrics": family_metrics,
        "gates": gates, "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
