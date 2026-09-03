#!/usr/bin/env python3
"""Run controlled causal-query routing on CLEVRER neural dynamics outputs."""

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

from motif_transfer.causal_query_routing import (  # noqa: E402
    CausalQueryState,
    build_source_models,
    select_action,
    source_gate_report,
)
from motif_transfer.clevrer_query_compiler import (  # noqa: E402
    compile_choice,
    compile_question,
    normalize_official_program,
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


def _verify_frozen_lineage(config: Mapping[str, Any]) -> None:
    lineage = config.get("frozen_lineage")
    if not lineage:
        return
    paths = {
        "split_manifest_sha256": Path(config["target"]["split_manifest"]),
        "runner_sha256": Path(__file__).resolve(),
        "source_controller_sha256": REPO / "src/motif_transfer/causal_query_routing.py",
        "target_compiler_sha256": REPO / "src/motif_transfer/clevrer_query_compiler.py",
    }
    for key, path in paths.items():
        expected = str(lineage.get(key) or "")
        if not expected or _sha256(path) != expected:
            raise ValueError(f"frozen lineage mismatch for {key}: {path}")


def _execute(
    *,
    executor_root: Path,
    scene_id: int,
    question_program: Sequence[str],
    choice_programs: Sequence[Sequence[str]],
    action: str,
    target: Mapping[str, Any],
) -> tuple[str, list[str], Path]:
    from simulation import Simulation  # type: ignore
    from executor import Executor  # type: ignore

    if action == "USE_EXPLICIT_RELATION":
        directory = str(target["explicit_relation_prediction_directory"])
        use_event_ann = True
    elif action == "DERIVE_FROM_TRAJECTORY":
        directory = str(target["trajectory_prediction_directory"])
        use_event_ann = False
    else:
        raise ValueError(f"unknown target causal representation action: {action}")
    prediction_path = (
        executor_root / "data/propnet_preds" / directory / f"sim_{scene_id:05d}.json"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    executor = Executor(Simulation(str(prediction_path), use_event_ann=use_event_ann))
    raw = [
        executor.run(list(choice) + list(question_program), debug=False)
        for choice in choice_programs
    ]
    if any(value not in {"yes", "no", "error"} for value in raw):
        raise ValueError("unexpected official CLEVRER executor output")
    return "".join("1" if value == "yes" else "0" for value in raw), raw, prediction_path


def _condition_action(
    condition: str,
    *,
    state: CausalQueryState,
    source_models: Mapping[str, Any],
) -> str:
    if condition == "target_always_explicit_relation":
        return "USE_EXPLICIT_RELATION"
    if condition == "target_always_trajectory":
        return "DERIVE_FROM_TRAJECTORY"
    if condition in source_models:
        return select_action(state, source_models[condition])
    if condition == "intervention_semantics_inverted":
        inverted = CausalQueryState(
            state_id=f"{state.state_id}:inverted",
            intervention_active=not state.intervention_active,
            future_query=state.future_query,
            explicit_relation_reliability=state.explicit_relation_reliability,
            trajectory_reliability=state.trajectory_reliability,
            predicted_intervention_shift=state.predicted_intervention_shift,
            remaining_compute_fraction=state.remaining_compute_fraction,
        )
        return select_action(inverted, source_models["authentic_source_router"])
    raise ValueError(f"unknown experiment condition: {condition}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _verify_frozen_lineage(config)
    source_config = config["source"]
    source_gate = source_gate_report(source_config)
    if source_gate["status"] != "SOURCE_CAUSAL_ROUTING_GATE_PASSED":
        raise ValueError("source causal routing gate failed")
    source_models = build_source_models(source_config)

    target = config["target"]
    manifest_path = Path(target["split_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(target["split"])
    sample_ids = manifest["benchmarks"]["clevrer"]["splits"][split]
    official_root = Path(target["official_root"])
    executor_root = official_root / "executor"
    sys.path.insert(0, str(executor_root))
    annotations_path = executor_root / "data/validation.json"
    annotations = {
        int(row["scene_index"]): row
        for row in json.loads(annotations_path.read_text(encoding="utf-8"))
    }

    conditions = tuple(map(str, config["conditions"]))
    rows = []
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
        question_exact = question_program == normalize_official_program(question["program"])
        choice_exact = [
            compiled == normalize_official_program(choice["program"])
            for compiled, choice in zip(choice_programs, question["choices"])
        ]
        gold = "".join(
            "1" if choice["answer"] == "correct" else "0"
            for choice in question["choices"]
        )
        state = CausalQueryState(
            state_id=sample_id,
            intervention_active="filter_counterfact" in question_program,
            future_query=family in {"predictive", "counterfactual"},
            explicit_relation_reliability=float(target["explicit_relation_reliability"]),
            trajectory_reliability=float(target["trajectory_reliability"]),
            predicted_intervention_shift=float(target["predicted_intervention_shift"]),
        )
        native_outputs: dict[str, dict[str, Any]] = {}
        for action in ("USE_EXPLICIT_RELATION", "DERIVE_FROM_TRAJECTORY"):
            answer, raw, prediction_path = _execute(
                executor_root=executor_root,
                scene_id=scene_id,
                question_program=question_program,
                choice_programs=choice_programs,
                action=action,
                target=target,
            )
            prediction_hashes[f"{sample_id}:{action}"] = _sha256(prediction_path)
            native_outputs[action] = {
                "answer": answer,
                "raw_executor_results": raw,
                "correct": answer == gold,
            }
        condition_rows = {}
        for condition in conditions:
            action = _condition_action(
                condition, state=state, source_models=source_models,
            )
            condition_rows[condition] = {
                "selected_abstract_action": action,
                **native_outputs[action],
            }
        rows.append({
            "sample_id": sample_id,
            "family": family,
            "gold_answer": gold,
            "compiled_question_program": question_program,
            "compiled_choice_programs": choice_programs,
            "compiler_question_exact": question_exact,
            "compiler_choices_exact": all(choice_exact),
            "official_program_runtime_input": False,
            "official_answer_runtime_input": False,
            "intervention_active": state.intervention_active,
            "conditions": condition_rows,
        })

    count = len(rows)
    metrics = {
        condition: {
            "correct": sum(bool(row["conditions"][condition]["correct"]) for row in rows),
            "accuracy": sum(bool(row["conditions"][condition]["correct"]) for row in rows) / count,
            "explicit_relation_actions": sum(
                row["conditions"][condition]["selected_abstract_action"]
                == "USE_EXPLICIT_RELATION"
                for row in rows
            ),
            "trajectory_actions": sum(
                row["conditions"][condition]["selected_abstract_action"]
                == "DERIVE_FROM_TRAJECTORY"
                for row in rows
            ),
        }
        for condition in conditions
    }
    authentic = metrics["authentic_source_router"]["correct"]
    controls = [condition for condition in conditions if condition != "authentic_source_router"]
    gates = {
        "source_gate_passed": source_gate["status"] == "SOURCE_CAUSAL_ROUTING_GATE_PASSED",
        "compiler_exact_on_all_rows": all(
            row["compiler_question_exact"] and row["compiler_choices_exact"]
            for row in rows
        ),
        "authentic_action_contrast": any(
            row["conditions"]["authentic_source_router"]["selected_abstract_action"]
            != row["conditions"]["target_always_explicit_relation"]["selected_abstract_action"]
            for row in rows
        ),
        "authentic_strictly_above_all_controls": all(
            authentic > metrics[condition]["correct"] for condition in controls
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "status": (
            "CLEVRER_CAUSAL_QUERY_ADAPTATION_PASS"
            if passed and split == "adaptation"
            else "CLEVRER_CAUSAL_QUERY_FORMAL_PASS"
            if passed
            else "CLEVRER_CAUSAL_QUERY_TRANSFER_FAIL"
        ),
        "benchmark": "clevrer",
        "split": split,
        "samples": count,
        "source_gate": source_gate,
        "conditions": metrics,
        "gates": gates,
        "rows": rows,
        "lineage": {
            "config_sha256": _sha256(args.config),
            "manifest_sha256": _sha256(manifest_path),
            "official_annotations_sha256": _sha256(annotations_path),
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
        "status": report["status"],
        "split": split,
        "samples": count,
        "conditions": metrics,
        "gates": gates,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
