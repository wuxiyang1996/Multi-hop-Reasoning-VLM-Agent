#!/usr/bin/env python3
"""Freeze outcome-blind predictions for the CLEVRER full five-arm protocol.

All arms consume one cached NS-DR prediction per raw video.  The two native
views below are computations over that same prediction payload: ``explicit``
uses its predicted event edges, while ``trajectory`` reconstructs events from
the predicted trajectories.  No annotation, official program, or answer is
opened in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.clevrer_descriptive_compiler import compile_descriptive_question  # noqa: E402
from motif_transfer.clevrer_query_compiler import compile_choice, compile_question  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


ARMS = (
    "neural_only", "generic_symbolic", "source_permuted",
    "source_induced", "target_written_isomorphic",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction(executor: Any, task: dict[str, Any]) -> tuple[str, str]:
    family = str(task["question_family"])
    if family == "descriptive":
        program = compile_descriptive_question(
            str(task["question"]), str(task["public_subtype"]),
        )
        return str(executor.run(program)), stable_hash(program)
    question_program = compile_question(str(task["question"]), family)
    outputs = []
    program_hashes = []
    for choice in task["choices"]:
        choice_program = compile_choice(str(choice["choice"]), family)
        program = choice_program + question_program
        raw = str(executor.run(program))
        outputs.append("1" if raw == "yes" else "0")
        program_hashes.append(stable_hash(program))
    return "".join(outputs), stable_hash(program_hashes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--grounder-config", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--neural-actor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = _read(args.runtime)
    config = _read(args.grounder_config)
    runtime_body = dict(runtime); claimed_runtime = runtime_body.pop("runtime_sha256")
    if stable_hash(runtime_body) != claimed_runtime:
        raise ValueError("shared runtime hash mismatch")
    if runtime.get("status") != "CLEVRER_SHARED_RUNTIME_FROZEN":
        raise ValueError("shared runtime did not pass")
    if tuple(runtime.get("arms", ())) != ARMS:
        raise ValueError("five-arm contract drift")
    if runtime.get("answers_read") or runtime.get("reserve_programs_read"):
        raise ValueError("oracle boundary already crossed")
    actor = _read(args.neural_actor) if args.neural_actor else None
    actor_predictions: dict[str, str] = {}
    if actor is not None:
        actor_body = dict(actor); actor_claimed = actor_body.pop("runtime_sha256", None)
        if stable_hash(actor_body) != actor_claimed:
            raise ValueError("neural actor artifact hash mismatch")
        if actor.get("status") != "NEURAL_ACTOR_PREDICTIONS_FROZEN_BEFORE_OUTCOMES":
            raise ValueError("neural actor was not frozen")
        if actor.get("shared_runtime_sha256") != claimed_runtime:
            raise ValueError("neural actor/shared runtime mismatch")
        if actor.get("answers_read") or actor.get("official_programs_read"):
            raise ValueError("neural actor crossed oracle boundary")
        actor_predictions = {str(row["task_id"]): str(row["prediction"]) for row in actor["rows"]}
        if len(actor_predictions) != len(runtime["tasks"]):
            raise ValueError("neural actor task coverage mismatch")

    executor_root = args.official_root / "executor"
    if not (executor_root / "executor.py").is_file() or not (executor_root / "simulation.py").is_file():
        raise FileNotFoundError("official CLEVRER executor root is incomplete")
    sys.path.insert(0, str(executor_root))
    from executor import Executor  # type: ignore  # noqa: E402
    from simulation import Simulation  # type: ignore  # noqa: E402

    prediction_root = Path(config["prediction_root"])
    video_receipts = {
        int(row["video_id"]): row["grounder_receipt"] for row in runtime["videos"]
    }
    grouped: dict[int, list[dict[str, Any]]] = {}
    for task in runtime["tasks"]:
        grouped.setdefault(int(task["video_id"]), []).append(task)

    rows = []
    for video_id, tasks in sorted(grouped.items()):
        path = prediction_root / f"sim_{video_id:05d}.json"
        receipt = video_receipts[video_id]
        if _sha(path) != receipt["prediction_sha256"]:
            raise ValueError(f"prediction drift for video {video_id}")
        # Same frozen neural payload and executor code; only event composition differs.
        explicit = Executor(Simulation(str(path), use_event_ann=True))
        trajectory = Executor(Simulation(str(path), use_event_ann=False))
        executor_receipt = stable_hash({
            "prediction_receipt_sha256": receipt["receipt_sha256"],
            "executor_module_sha256": _sha(executor_root / "executor.py"),
            "simulation_module_sha256": _sha(executor_root / "simulation.py"),
            "views": ["explicit_predicted_edges", "trajectory_composed_edges"],
        })
        for task in tasks:
            explicit_answer, explicit_program_sha = _prediction(explicit, task)
            trajectory_answer, trajectory_program_sha = _prediction(trajectory, task)
            source_commit = bool(task.get(
                "source_execution_authorized",
                task["source_applicability"]["status"] == "AUTHORIZED",
            ))
            permuted_commit = task["permuted_applicability"]["status"] == "AUTHORIZED"
            if actor is None:
                # Legacy V1 diagnostic: retained for immutable failed-artifact audit.
                predictions = {
                    "neural_only": explicit_answer,
                    "generic_symbolic": trajectory_answer,
                    "source_permuted": trajectory_answer if permuted_commit else explicit_answer,
                    "source_induced": trajectory_answer if source_commit else explicit_answer,
                    "target_written_isomorphic": trajectory_answer if source_commit else explicit_answer,
                }
            else:
                fallback = actor_predictions[str(task["task_id"])]
                # V2: all arms share this frozen neural fallback.  The generic
                # ceiling eagerly executes; only source/iso consult source scope.
                predictions = {
                    "neural_only": fallback,
                    "generic_symbolic": explicit_answer,
                    "source_permuted": explicit_answer if permuted_commit else fallback,
                    "source_induced": explicit_answer if source_commit else fallback,
                    "target_written_isomorphic": explicit_answer if source_commit else fallback,
                }
            row = {
                "task_id": task["task_id"], "video_id": video_id,
                "question_id": int(task["question_id"]),
                "question_family": task["question_family"],
                "task_state_sha256": task["task_state_sha256"],
                "semantic_receipt_sha256": task["semantic_receipt"]["receipt_sha256"],
                "grounder_receipt_sha256": receipt["receipt_sha256"],
                "executor_receipt_sha256": executor_receipt,
                "explicit_program_sha256": explicit_program_sha,
                "trajectory_program_sha256": trajectory_program_sha,
                "source_authorization_sha256": task["source_applicability"]["receipt_sha256"],
                "permuted_authorization_sha256": task["permuted_applicability"]["receipt_sha256"],
                "source_commit": source_commit, "permuted_commit": permuted_commit,
                "explicit_prediction": explicit_answer,
                "trajectory_prediction": trajectory_answer,
                "predictions": predictions,
                "gold_read": False, "official_program_read": False,
            }
            row["prediction_receipt_sha256"] = stable_hash(row)
            rows.append(row)

    gates = {
        "expected_1600_tasks": len(rows) == 1600,
        "five_arms_present": all(set(row["predictions"]) == set(ARMS) for row in rows),
        "one_grounder_and_executor_per_task": all(
            row["grounder_receipt_sha256"] and row["executor_receipt_sha256"] for row in rows
        ),
        "source_permuted_fail_closed": all(not row["permuted_commit"] for row in rows),
        "target_written_isomorphic_exact": all(
            row["predictions"]["source_induced"]
            == row["predictions"]["target_written_isomorphic"] for row in rows
        ),
        "no_oracle_read": all(
            not row["gold_read"] and not row["official_program_read"] for row in rows
        ),
        "neural_actor_complete_when_required": actor is None or len(actor_predictions) == len(rows),
    }
    body = {
        "schema_version": (
            "clevrer-five-arm-outcome-blind-predictions-v2"
            if actor is not None else "clevrer-five-arm-outcome-blind-predictions-v1"
        ),
        "status": "CLEVRER_FIVE_ARM_PREDICTIONS_FROZEN" if all(gates.values()) else "FAILED",
        "shared_runtime_sha256": claimed_runtime,
        "grounder_config_sha256": stable_hash(config),
        "official_executor_code": {
            "executor_sha256": _sha(executor_root / "executor.py"),
            "simulation_sha256": _sha(executor_root / "simulation.py"),
        },
        "arms": ARMS, "rows": rows, "gates": gates,
        "answers_read": False, "official_programs_read": False,
        "external_provider_calls": 0,
        "generic_symbolic_role": "TARGET_NATIVE_EAGER_COMPOSITION_CEILING",
        "neural_actor_runtime_sha256": actor.get("runtime_sha256") if actor is not None else None,
        "neural_actor_model": actor.get("model") if actor is not None else None,
    }
    body["predictions_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "rows": len(rows), "gates": gates,
                      "predictions_sha256": body["predictions_sha256"]}, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
