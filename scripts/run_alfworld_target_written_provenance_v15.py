#!/usr/bin/env python3
"""Diagnose whether ALFWorld policy efficacy depends on source provenance.

This consumed-population diagnostic reruns a source-blind, target-written
controller on the complete V13+V14 ALFWorld population.  It compares full
action/state traces against the previously frozen authentic source-induced
arm.  It is not a new prospective success-rate experiment.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
for _site in sorted((REPO.parent / "conda/envs/alfworld/lib").glob(
    "python*/site-packages"
)):
    if (_site / "alfworld").is_dir():
        sys.path.append(str(_site))
        break

import run_alfworld_goal_relation_macro_v3 as base  # noqa: E402
from run_alfworld_unified_goal_acquisition_v13 import (  # noqa: E402
    _ValidTrainBatchEnvironment,
)
from motif_transfer.alfworld_goal_relation_macro import (  # noqa: E402
    AUTHENTIC,
    RAW,
)
from motif_transfer.alfworld_target_written_equivalent import (  # noqa: E402
    TARGET_WRITTEN_EQUIVALENT,
    TargetWrittenExecutionState,
    choose_target_written_action,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_POPULATIONS = (
    (
        "v13_formal_reserve",
        REPO / "configs/alfworld_unified_goal_acquisition_v13_formal.json",
        REPO / "runs/alfworld_unified_goal_acquisition_v13_formal/report.json",
    ),
    (
        "v14_remaining_population",
        REPO / "configs/alfworld_program_driven_policy_v14_formal.json",
        REPO / "runs/alfworld_program_driven_policy_v14_formal/report.json",
    ),
)
DEFAULT_OUTPUT = REPO / "runs/alfworld_target_written_provenance_v15/report.json"


class _ForbiddenSourceArtifact(Mapping[str, Any]):
    """A capability tripwire: every possible source read aborts the run."""

    def __init__(self) -> None:
        self.read_attempts = 0

    def _deny(self, operation: str) -> None:
        self.read_attempts += 1
        raise RuntimeError(f"target-written controller attempted {operation}")

    def __getitem__(self, key: str) -> Any:
        self._deny(f"source lookup: {key}")

    def __iter__(self) -> Iterator[str]:
        self._deny("source iteration")

    def __len__(self) -> int:
        self._deny("source length inspection")


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        raw = gzip.open(path, "rt", encoding="utf-8").read()
    else:
        raw = path.read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _normalize_task_id(task_id: str, config: Mapping[str, Any]) -> str:
    virtual_root = (
        Path(str(config["alfworld_data"])) / "json_2.1.1" / "train"
    ).resolve()
    return str(Path(task_id).resolve().relative_to(virtual_root))


def _trace_projection(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "selected_action": str(row["selected_action"]),
            "before_state_sha256": str(row["before_state_sha256"]),
            "after_state_sha256": str(row["after_state_sha256"]),
            "target_effect_receipt": str(row["target_effect_receipt"]),
            "official_success_after": bool(row["official_success_after"]),
        }
        for row in episode["records"]
    ]


def _verify_population_inputs(
    config_path: Path, report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _read(config_path)
    report = _read(report_path)
    _verify_self_hash(config, "config_sha256")
    _verify_self_hash(report, "report_sha256")
    if report.get("config_sha256") != config.get("config_sha256"):
        raise ValueError(f"config/report lineage mismatch: {report_path}")
    if list(map(str, report["task_ids"])) != list(map(str, config["task_ids"])):
        raise ValueError(f"config/report task mismatch: {report_path}")
    target_path = REPO / str(config["target_grounder"])
    causal_path = REPO / str(config["target_causal_effect_artifact"])
    if _file_sha256(target_path) != config["target_grounder_file_sha256"]:
        raise ValueError(f"target grounder changed: {target_path}")
    if _file_sha256(causal_path) != config["target_causal_effect_file_sha256"]:
        raise ValueError(f"target causal head changed: {causal_path}")
    target = _read(target_path)
    causal = _read(causal_path)
    if not target.get("target_grounder_gate", {}).get("passed"):
        raise ValueError("target-native grounder is not qualified")
    if not causal.get("gates", {}).get(
        "effect_balanced_accuracy_at_least_0p80"
    ):
        raise ValueError("target-native causal head is not qualified")
    return config, report, target, causal


def _run_population(
    label: str, config_path: Path, report_path: Path,
) -> dict[str, Any]:
    config, reference, target, causal = _verify_population_inputs(
        config_path, report_path,
    )
    source_tripwire = _ForbiddenSourceArtifact()
    reference_authentic = {
        str(row["task_id"]): row for row in reference["episodes"][AUTHENTIC]
    }
    task_ids = tuple(map(str, config["task_ids"]))
    environment = _ValidTrainBatchEnvironment(
        config_path=str(config["alfworld_config"]),
        data_path=str(config["alfworld_data"]),
        split="train",
        seed=int(config["seed"]),
        game_ids=task_ids,
        max_steps=int(config["max_steps"]),
    )
    rows: list[dict[str, Any]] = []
    try:
        for index in range(len(task_ids)):
            episode = base._run_episode(
                environment=environment,
                condition=TARGET_WRITTEN_EQUIVALENT,
                source_artifact=source_tripwire,
                target_grounder=target["target_grounder"],
                target_causal_effect_head=causal["target_causal_effect_head"],
                max_steps=int(config["max_steps"]),
                thresholds=config["thresholds"],
            )
            task_id = _normalize_task_id(str(episode["task_id"]), config)
            authentic = reference_authentic[task_id]
            actions_equal = list(episode["actions"]) == list(authentic["actions"])
            trace_equal = _trace_projection(episode) == _trace_projection(authentic)
            outcome_equal = (
                bool(episode["official_success"])
                == bool(authentic["official_success"])
                and int(episode["steps"]) == int(authentic["steps"])
            )
            row_body = {
                "population": label,
                "task_id": task_id,
                "target_written_success": bool(episode["official_success"]),
                "authentic_source_induced_success": bool(
                    authentic["official_success"]
                ),
                "steps": int(episode["steps"]),
                "source_artifact_read_attempts": source_tripwire.read_attempts,
                "target_written_source_admissions": int(
                    episode["source_admissions"]
                ),
                "action_trace_sha256": stable_hash(list(episode["actions"])),
                "actions_exactly_match_authentic": actions_equal,
                "state_effect_trace_exactly_matches_authentic": trace_equal,
                "outcome_and_steps_exactly_match_authentic": outcome_equal,
                "reference_episode_sha256": str(authentic["episode_sha256"]),
            }
            rows.append(row_body | {"row_sha256": stable_hash(row_body)})
            print(json.dumps({
                "population": label,
                "task_index": index,
                "task_id": task_id,
                "success": episode["official_success"],
                "steps": episode["steps"],
                "exact_action_trace": actions_equal,
                "source_reads": source_tripwire.read_attempts,
            }), flush=True)
    finally:
        environment.close()

    raw_successes = int(reference["summaries"][RAW]["successes"])
    authentic_successes = int(reference["summaries"][AUTHENTIC]["successes"])
    body = {
        "population": label,
        "config_path": str(config_path.relative_to(REPO)),
        "config_file_sha256": _file_sha256(config_path),
        "config_sha256": str(config["config_sha256"]),
        "reference_report_path": str(report_path.relative_to(REPO)),
        "reference_report_file_sha256": _file_sha256(report_path),
        "reference_report_sha256": str(reference["report_sha256"]),
        "target_grounder_file_sha256": str(
            config["target_grounder_file_sha256"]
        ),
        "target_causal_effect_file_sha256": str(
            config["target_causal_effect_file_sha256"]
        ),
        "tasks": len(rows),
        "target_written_successes": sum(
            row["target_written_success"] for row in rows
        ),
        "authentic_source_induced_successes": authentic_successes,
        "raw_target_only_successes": raw_successes,
        "source_induced_gain_over_raw": authentic_successes - raw_successes,
        "source_artifact_read_attempts": source_tripwire.read_attempts,
        "exact_action_trace_matches": sum(
            row["actions_exactly_match_authentic"] for row in rows
        ),
        "exact_state_effect_trace_matches": sum(
            row["state_effect_trace_exactly_matches_authentic"] for row in rows
        ),
        "exact_outcome_and_step_matches": sum(
            row["outcome_and_steps_exactly_match_authentic"] for row in rows
        ),
        "tasks_with_target_written_source_admission": sum(
            row["target_written_source_admissions"] > 0 for row in rows
        ),
        "rows": rows,
    }
    return body | {"population_sha256": stable_hash(body)}


def run() -> dict[str, Any]:
    base.choose_goal_relation_action = choose_target_written_action
    base.TargetRelationExecutionState = TargetWrittenExecutionState
    populations = [
        _run_population(label, config, report)
        for label, config, report in DEFAULT_POPULATIONS
    ]
    task_count = sum(int(row["tasks"]) for row in populations)
    source_successes = sum(
        int(row["authentic_source_induced_successes"]) for row in populations
    )
    raw_successes = sum(
        int(row["raw_target_only_successes"]) for row in populations
    )
    target_written_successes = sum(
        int(row["target_written_successes"]) for row in populations
    )
    gates = {
        "complete_45_task_consumed_population": task_count == 45,
        "zero_source_artifact_reads": sum(
            int(row["source_artifact_read_attempts"]) for row in populations
        ) == 0,
        "zero_source_admissions": sum(
            int(row["tasks_with_target_written_source_admission"])
            for row in populations
        ) == 0,
        "all_action_traces_exactly_match": sum(
            int(row["exact_action_trace_matches"]) for row in populations
        ) == task_count,
        "all_state_effect_traces_exactly_match": sum(
            int(row["exact_state_effect_trace_matches"]) for row in populations
        ) == task_count,
        "all_outcomes_and_steps_exactly_match": sum(
            int(row["exact_outcome_and_step_matches"]) for row in populations
        ) == task_count,
        "target_written_recovers_all_source_induced_successes": (
            target_written_successes == source_successes
        ),
        "program_structure_has_nonzero_policy_utility": (
            source_successes > raw_successes
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "alfworld-target-written-provenance-v15",
        "status": (
            "ALFWORLD_SOURCE_PROVENANCE_NOT_BEHAVIORALLY_IDENTIFIABLE"
            if passed else "ALFWORLD_TARGET_WRITTEN_EQUIVALENCE_FAILED"
        ),
        "role": "posthoc_consumed_population_identifiability_diagnostic",
        "claim_boundary": (
            "This diagnostic tests whether already-specified program behavior "
            "requires source provenance. It does not add prospective success "
            "evidence and does not test the acquisition cost of writing the "
            "target controller."
        ),
        "controller": TARGET_WRITTEN_EQUIVALENT,
        "controller_file_sha256": _file_sha256(
            REPO / "src/motif_transfer/alfworld_target_written_equivalent.py"
        ),
        "source_artifact_paths_loaded": [],
        "source_identifiers_consumed_by_controller": [],
        "target_native_components_reused": [
            "frozen_neural_action_grounder",
            "frozen_target_causal_effect_head",
            "native_action_interface",
            "target_slot_ledger",
        ],
        "populations": populations,
        "combined": {
            "tasks": task_count,
            "target_written_successes": target_written_successes,
            "authentic_source_induced_successes": source_successes,
            "raw_target_only_successes": raw_successes,
            "source_induced_and_target_written_gain_over_raw": (
                source_successes - raw_successes
            ),
            "exact_action_trace_matches": sum(
                int(row["exact_action_trace_matches"])
                for row in populations
            ),
            "exact_state_effect_trace_matches": sum(
                int(row["exact_state_effect_trace_matches"])
                for row in populations
            ),
        },
        "identifiability_result": {
            "program_structure_causally_sufficient_for_observed_policy_effect": (
                passed
            ),
            "source_provenance_necessary_after_program_is_specified": False,
            "source_provenance_recoverable_from_extensional_behavior": False,
            "remaining_source_value_estimand": (
                "acquisition information/cost under matched target evidence"
            ),
        },
        "gates": gates,
    }
    return body | {"report_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run()
    output = args.output if args.output.is_absolute() else REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "combined": report["combined"],
        "identifiability_result": report["identifiability_result"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
