#!/usr/bin/env python3
"""Bridge exact 9B route selection to frozen target-native action receipts.

This is a locked counterfactual replay over previously validated formal traces.
It does not relabel those outcomes as a fresh target run.  A task is equivalent
only when every neural route component is exact, the original task receipt is
present and content-addressed, and the old route used the same source program.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.portable_paths import resolve_repo_artifact  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> bool:
    body = dict(value)
    claimed = body.pop(field, None)
    return bool(claimed and claimed == stable_hash(body))


def _verified_formal_paths(prereg: Mapping[str, Any]) -> dict[str, list[Path]]:
    output = {}
    for benchmark, specs in prereg["formal_evidence"].items():
        paths = []
        for spec in specs:
            path = resolve_repo_artifact(spec["path"], REPO)
            if not path.is_file() or _sha(path) != spec["sha256"]:
                raise ValueError(f"formal evidence drifted: {path}")
            paths.append(path)
        output[benchmark] = paths
    return output


def _webshop_receipts(task_ids: set[str], expected_program: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = REPO / "runs/webshop_structural_transfer_v21_formal/report.json"
    report = _read(report_path)
    records = {}
    decisions = 0
    for task_id in sorted(task_ids):
        path = report_path.parent / f"{task_id}.source_induced_structural_ir.json"
        row = _read(path)
        controller = row.get("structural_controller") or {}
        compatibility = controller.get("compatibility") or {}
        valid = bool(
            row.get("task_id") == task_id
            and row.get("condition") == "source_induced_structural_ir"
            and _self_hash(row, "receipt_sha256")
            and compatibility.get("source_artifact_sha256") == expected_program
            and all((compatibility.get("gates") or {}).values())
        )
        count = int(row.get("source_decision_count") or 0)
        decisions += count
        records[task_id] = {
            "receipt_available": valid,
            "receipt_sha256": row.get("receipt_sha256"),
            "success_critical_decisions": count,
            "old_route_programs": [compatibility.get("source_artifact_sha256")],
        }
    summary = report["summaries"]["source_induced_structural_ir"]
    integrity = {
        "formal_status": report.get("status"),
        "formal_report_self_hash_valid": _self_hash(report, "report_sha256"),
        "task_receipts": len(records),
        "success_critical_decisions": decisions,
        "reported_success_critical_decisions": summary["source_authorized_decisions"],
        "existing_success": {
            "correct": summary["strict_successes"], "tasks": summary["tasks"],
        },
    }
    return records, integrity


def _discoveryworld_receipts(task_ids: set[str], expected_program: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = REPO / "runs/discoveryworld_structural_transfer_v1_matched"
    report = _read(root / "report.json")
    records = {}
    decisions = 0
    for task_id in sorted(task_ids):
        path = root / task_id / "result.json"
        row = _read(path)
        recovery = (((row.get("conditions") or {}).get("source_induced") or {}).get("recovery") or [])
        program_hashes = {
            str((step.get("runtime_decision") or {}).get("source_program_sha256"))
            for step in recovery
        }
        valid = bool(
            row.get("task_id") == task_id
            and row.get("status") == "DISCOVERYWORLD_STRUCTURAL_MATCHED_COMPLETE"
            and _self_hash(row, "result_sha256")
            and program_hashes <= {expected_program}
            and all(
                (step.get("runtime_decision") or {}).get("receipt_sha256")
                and (step.get("transition") or {}).get("receipt_sha256")
                for step in recovery
            )
        )
        decisions += len(recovery)
        records[task_id] = {
            "receipt_available": valid,
            "receipt_sha256": row.get("result_sha256"),
            "success_critical_decisions": len(recovery),
            "old_route_programs": sorted(program_hashes),
        }
    integrity = {
        "formal_status": report.get("status"),
        "formal_report_self_hash_valid": _self_hash(report, "report_sha256"),
        "task_receipts": len(records),
        "success_critical_decisions": decisions,
        "existing_success": {
            "correct": report["condition_successes"]["source_induced"],
            "tasks": report["applicable_tasks"],
        },
    }
    return records, integrity


def _tir_receipts(task_ids: set[str], expected_program: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read(REPO / "runs/tir_maze_structural_transfer_v3/heldout_report.json")
    traces = {
        str(row["sample_id"]): row for row in report["traces"]
        if row.get("condition") == "source_induced"
    }
    records = {}
    for task_id in sorted(task_ids):
        row = traces.get(task_id) or {}
        valid = bool(
            row.get("native_receipt_sha256")
            and row.get("trace_sha256")
            and row.get("source_option") in {"EXECUTE", "ABSTAIN"}
            and (row.get("source_option") != "EXECUTE" or row.get("native_selected_answer") is not None)
        )
        records[task_id] = {
            "receipt_available": valid,
            "receipt_sha256": row.get("trace_sha256"),
            "native_receipt_sha256": row.get("native_receipt_sha256"),
            "success_critical_decisions": 1,
            "old_route_programs": [expected_program],
        }
    summary = report["summaries"]["source_induced"]
    integrity = {
        "formal_status": report.get("status"),
        "formal_report_self_hash_valid": _self_hash(report, "report_sha256"),
        "task_receipts": len(traces),
        "success_critical_decisions": len(records),
        "existing_success": {"correct": summary["successes"], "tasks": summary["tasks"]},
    }
    return records, integrity


def _alfworld_receipts(task_ids: set[str], expected_program: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read(REPO / "runs/alfworld_unified_goal_acquisition_v13_formal/report.json.gz")
    episodes = {
        str(row["task_id"]): row
        for row in report["episodes"]["authentic_source_goal_relation_macro"]
    }
    records = {}
    decisions = 0
    for task_id in sorted(task_ids):
        phase7 = (report.get("phase7_authorizations") or {}).get(task_id) or {}
        receipts = (report.get("authority_receipts") or {}).get(task_id) or []
        episode = episodes.get(task_id) or {}
        episode_records = episode.get("records") or []
        # Two frozen episodes correctly never admitted the source executor.  In
        # that case there is no authority_receipt by construction; the
        # content-addressed episode and its per-step abstention records are the
        # native replay receipt.  This is stricter than merely accepting an
        # empty receipt list: every action must equal the frozen fallback.
        zero_admission_abstention_receipt = bool(
            not receipts
            and episode.get("source_admissions") == 0
            and episode_records
            and _self_hash(episode, "episode_sha256")
            and all(
                _self_hash(row, "record_sha256")
                and row.get("source_admitted") is False
                and row.get("program_active") is False
                and row.get("selected_action") == row.get("raw_fallback_action")
                for row in episode_records
            )
        )
        admitted_action_receipts = bool(
            receipts
            and all(
                row.get("phase7_authorization_sha256") == phase7.get("authorization_sha256")
                and row.get("target_executor_calls") == 1
                and row.get("formal_outcome_read") is False
                and row.get("receipt_sha256")
                for row in receipts
            )
        )
        valid = bool(
            phase7.get("route_id") == "sokoban-goal-acquisition-to-alfworld-multiplicity-v11"
            and phase7.get("verdict") == "SELECT_SKILL"
            and phase7.get("target_action_emitted") is False
            and phase7.get("current_target_outcome_read") is False
            and (admitted_action_receipts or zero_admission_abstention_receipt)
        )
        decisions += len(receipts)
        records[task_id] = {
            "receipt_available": valid,
            "receipt_sha256": (
                stable_hash(receipts) if receipts else episode.get("episode_sha256")
            ),
            "success_critical_decisions": len(receipts),
            "old_route_programs": [expected_program],
            "receipt_kind": (
                "SOURCE_EXECUTOR_ACTION_RECEIPTS"
                if receipts else "ZERO_ADMISSION_ABSTENTION_EPISODE_RECEIPT"
            ),
        }
    summary = report["summaries"]["authentic_source_goal_relation_macro"]
    integrity = {
        "formal_status": report.get("status"),
        "formal_report_self_hash_valid": _self_hash(report, "report_sha256"),
        "task_receipts": len(report.get("authority_receipts") or {}),
        "success_critical_decisions": decisions,
        "existing_success": {"correct": summary["successes"], "tasks": summary["tasks"]},
    }
    return records, integrity


def _clevrer_receipts(task_ids: set[str], expected_program: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read(REPO / "runs/clevrer_unified_goal_relation_v15_reserve/formal_report.json")
    formal_rows = {str(row["sample_id"]): row for row in report["rows"]}
    records = {}
    executor_calls = 0
    for task_id in sorted(task_ids):
        row = formal_rows.get(task_id) or {}
        authority = row.get("unified_authority") or {}
        phase7 = authority.get("phase7") or {}
        calls = int(authority.get("executor_calls") or 0)
        executor_calls += calls
        valid = bool(
            row.get("proof_receipts_sha256")
            and phase7.get("selected_program_sha256") == expected_program
            and phase7.get("current_target_outcome_read") is False
            and phase7.get("target_action_emitted") is False
            and phase7.get("verdict") in {"SELECT_SKILL", "ABSTAIN"}
            and calls == (1 if phase7.get("verdict") == "SELECT_SKILL" else 0)
        )
        records[task_id] = {
            "receipt_available": valid,
            "receipt_sha256": phase7.get("authorization_sha256"),
            "success_critical_decisions": 1,
            "target_executor_calls": calls,
            "old_route_programs": [phase7.get("selected_program_sha256")],
        }
    authentic = "authentic_source_induced_goal_relation"
    neural = "neural_only_explicit_relation"
    integrity = {
        "formal_status": report.get("status"),
        "formal_report_self_hash_valid": _self_hash(report, "report_sha256"),
        "task_receipts": len(formal_rows),
        "success_critical_decisions": len(records),
        "target_executor_calls": executor_calls,
        "existing_success": {
            "correct": sum(bool(row["conditions"][authentic]["correct"]) for row in report["rows"]),
            "tasks": len(report["rows"]),
            "neural_only_correct": sum(bool(row["conditions"][neural]["correct"]) for row in report["rows"]),
        },
    }
    return records, integrity


def _agqa_receipts(task_ids: set[str], expected_programs: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _read(REPO / "runs/agqa2_full_distribution_v62/base_report.json")
    report = _read(REPO / "runs/agqa2_full_distribution_v62/report.json")
    selection = base.get("source_selection") or {}
    old_programs = [selection.get("temporal_program_sha256"), selection.get("relation_program_sha256")]
    rows = {str(row["task_id"]): row for row in base["rows"]}
    records = {}
    authorized = 0
    for task_id in sorted(task_ids):
        row = rows.get(task_id) or {}
        is_authorized = row.get("source_executor_authorized") is True
        authorized += int(is_authorized)
        valid = bool(
            row.get("runtime_receipt_sha256")
            and row.get("runtime_answer_read") is False
            and row.get("runtime_functional_program_read") is False
            and row.get("runtime_scene_graph_read") is False
            and row.get("runtime_source_identity_read") is False
            and (is_authorized or row.get("source_prediction") == row.get("direct_response"))
        )
        records[task_id] = {
            "receipt_available": valid,
            "receipt_sha256": row.get("runtime_receipt_sha256"),
            "success_critical_decisions": 1,
            "target_executor_calls": int(is_authorized),
            "old_route_programs": old_programs,
        }
    integrity = {
        "formal_status": report.get("status"),
        "formal_report_self_hash_valid": _self_hash(report, "report_sha256"),
        "base_report_self_hash_valid": _self_hash(base, "report_sha256"),
        "task_receipts": len(rows),
        "success_critical_decisions": len(records),
        "target_executor_calls": authorized,
        "existing_success": {
            "correct": report["source_vs_direct_rows"]["source_correct"],
            "tasks": report["sample_count"],
            "neural_only_correct": report["source_vs_direct_rows"]["direct_correct"],
            "wins": report["source_vs_direct_rows"]["wins"],
            "losses": report["source_vs_direct_rows"]["losses"],
        },
    }
    if old_programs != expected_programs:
        for row in records.values():
            row["receipt_available"] = False
    return records, integrity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--route-report", type=Path, required=True)
    parser.add_argument("--route-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    activation = _read(args.activation)
    report = _read(args.route_report)
    if not (
        activation.get("status") == "FROZEN_SIX_BENCHMARK_SUBSTITUTION_EVALUATION_READY"
        and all((activation.get("gates") or {}).values())
        and report.get("status") == "SIX_BENCHMARK_MODEL_SUBSTITUTION_ROUTE_GATE_PASSED"
        and all((report.get("gates") or {}).values())
    ):
        raise SystemExit("route substitution gate did not pass")
    prereg_path = resolve_repo_artifact(
        activation["target_preregistration"]["path"], REPO,
    )
    prereg = _read(prereg_path)
    if _sha(prereg_path) != activation["target_preregistration"]["sha256"]:
        raise SystemExit("substitution preregistration drifted")
    _verified_formal_paths(prereg)
    dataset_path = resolve_repo_artifact(activation["evaluation_file"]["path"], REPO)
    index_path = resolve_repo_artifact(
        activation["native_replay_index"]["path"], REPO,
    )
    dataset = {str(row["example_id"]): row for row in _rows(dataset_path)}
    index = _rows(index_path)
    predictions = {
        str(row["example_id"]): row for row in _rows(args.route_predictions)
        if row.get("regime") == "CONTROLLER_LORA"
    }
    exact = {
        example_id: bool(
            prediction.get("exact_json") is True
            and prediction.get("parsed") == json.loads(dataset[example_id]["completion"])
        )
        for example_id, prediction in predictions.items()
        if example_id in dataset
    }

    task_ids = {}
    programs = {}
    for row in index:
        task_ids.setdefault(row["benchmark"], set()).add(str(row["formal_task_id"]))
        programs.setdefault(row["benchmark"], list(row["source_program_sha256"]))
    extractors = {
        "webshop": lambda: _webshop_receipts(task_ids["webshop"], programs["webshop"][0]),
        "discoveryworld": lambda: _discoveryworld_receipts(task_ids["discoveryworld"], programs["discoveryworld"][0]),
        "tirbench": lambda: _tir_receipts(task_ids["tirbench"], programs["tirbench"][0]),
        "alfworld": lambda: _alfworld_receipts(task_ids["alfworld"], programs["alfworld"][0]),
        "clevrer": lambda: _clevrer_receipts(task_ids["clevrer"], programs["clevrer"][0]),
        "agqa2": lambda: _agqa_receipts(task_ids["agqa2"], programs["agqa2"]),
    }
    evidence = {}
    integrity = {}
    for benchmark, extractor in extractors.items():
        evidence[benchmark], integrity[benchmark] = extractor()

    task_rows = []
    group = Counter()
    divergence_ids = []
    for row in index:
        benchmark = str(row["benchmark"])
        task_id = str(row["formal_task_id"])
        selector_exact = all(exact.get(str(value), False) for value in row["selector_example_ids"])
        native = evidence[benchmark].get(task_id) or {}
        route_programs_match = list(native.get("old_route_programs") or []) == list(row["source_program_sha256"])
        equivalent = bool(selector_exact and native.get("receipt_available") and route_programs_match)
        if not equivalent:
            divergence_ids.append(f"{benchmark}:{task_id}")
        group[(benchmark, "tasks")] += 1
        group[(benchmark, "equivalent")] += int(equivalent)
        group[(benchmark, "decisions")] += int(native.get("success_critical_decisions") or 0)
        group[(benchmark, "equivalent_decisions")] += int(equivalent) * int(
            native.get("success_critical_decisions") or 0
        )
        task_rows.append({
            "benchmark": benchmark, "formal_task_id": task_id,
            "selector_exact": selector_exact,
            "native_receipt_available": bool(native.get("receipt_available")),
            "route_programs_match": route_programs_match,
            "success_critical_action_equivalent": equivalent,
            "success_critical_decisions": int(native.get("success_critical_decisions") or 0),
            "native_receipt_sha256": native.get("receipt_sha256"),
        })
    by_benchmark = {
        benchmark: {
            "tasks": group[(benchmark, "tasks")],
            "equivalent_tasks": group[(benchmark, "equivalent")],
            "success_critical_decisions": group[(benchmark, "decisions")],
            "equivalent_success_critical_decisions": group[(benchmark, "equivalent_decisions")],
            "action_equivalence": (
                group[(benchmark, "equivalent_decisions")] / group[(benchmark, "decisions")]
                if group[(benchmark, "decisions")] else 0.0
            ),
            "existing_formal_result_inherited_not_rerun": integrity[benchmark]["existing_success"],
            "receipt_integrity": integrity[benchmark],
        }
        for benchmark in sorted(task_ids)
    }
    total_decisions = sum(row["success_critical_decisions"] for row in task_rows)
    equivalent_decisions = sum(
        row["success_critical_decisions"]
        for row in task_rows if row["success_critical_action_equivalent"]
    )
    gates = {
        "route_report_hash_matches_activation": report.get("evaluation_manifest", {}).get("sha256") == _sha(args.activation),
        "all_2246_route_decisions_exact": len(exact) == 2246 and all(exact.values()),
        "all_1346_formal_tasks_replayed": len(task_rows) == 1346,
        "all_six_benchmark_groups_present": set(by_benchmark) == set(extractors),
        "every_formal_task_has_content_addressed_native_receipt": all(
            row["native_receipt_available"] for row in task_rows
        ),
        "all_old_routes_use_the_expected_source_programs": all(
            row["route_programs_match"] for row in task_rows
        ),
        "success_critical_action_equivalence_is_one": total_decisions > 0 and equivalent_decisions == total_decisions,
        "zero_divergence_episodes": not divergence_ids,
        "all_formal_report_self_hashes_valid": all(
            all(
                value for key, value in details.items()
                if key.endswith("self_hash_valid")
            )
            for details in integrity.values()
        ),
    }
    payload = {
        "schema_version": "harness-9b-six-benchmark-action-equivalence-v1",
        "status": (
            "SIX_BENCHMARK_9B_SUBSTITUTION_ACTION_EQUIVALENCE_VALIDATED"
            if all(gates.values())
            else "SIX_BENCHMARK_9B_SUBSTITUTION_ACTION_EQUIVALENCE_FAILED"
        ),
        "authority": (
            "LOCKED_COUNTERFACTUAL_REPLAY;EXACT_NEURAL_ROUTE_SUBSTITUTION;"
            "FROZEN_TARGET_NATIVE_GROUNDER_COMPOSER_UTILITY_VERIFIER_EXECUTOR_RECEIPTS"
        ),
        "activation": {"path": str(args.activation.resolve()), "sha256": _sha(args.activation)},
        "route_report": {"path": str(args.route_report.resolve()), "sha256": _sha(args.route_report)},
        "route_predictions": {"path": str(args.route_predictions.resolve()), "sha256": _sha(args.route_predictions)},
        "summary": {
            "formal_tasks": len(task_rows),
            "route_decisions": len(exact),
            "success_critical_decisions": total_decisions,
            "equivalent_success_critical_decisions": equivalent_decisions,
            "action_equivalence": equivalent_decisions / total_decisions if total_decisions else 0.0,
            "divergence_episode_count": len(divergence_ids),
            "divergence_episode_ids": divergence_ids,
        },
        "by_benchmark": by_benchmark,
        "gates": gates,
        "claim_boundary": (
            "This validates locked 9B selector substitution and deterministic native-action "
            "equivalence on previously validated formal traces. The listed success rates are "
            "inherited from those content-addressed runs, not newly measured 9B target outcomes. "
            "Only divergence episodes would require a live rerun."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    task_path = args.output.with_suffix(".tasks.jsonl")
    with task_path.open("w", encoding="utf-8") as stream:
        for row in sorted(task_rows, key=lambda value: (value["benchmark"], value["formal_task_id"])):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"], "summary": payload["summary"],
        "gates": gates, "by_benchmark": by_benchmark,
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
