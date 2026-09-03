#!/usr/bin/env python3
"""Execute the K=1 target-induced recurrence on consumed ALFWorld V14."""

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
from motif_transfer.alfworld_goal_relation_macro import AUTHENTIC  # noqa: E402
from motif_transfer.alfworld_target_recurrent_induction import (  # noqa: E402
    TARGET_INDUCED,
    choose_target_induced_action,
    execution_normal_form,
    validate_target_recurrent_program,
)
from motif_transfer.alfworld_target_written_equivalent import (  # noqa: E402
    TargetWrittenExecutionState,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_CONFIG = REPO / "configs/alfworld_target_induced_policy_v17_consumed.json"


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        raw = gzip.open(path, "rt", encoding="utf-8").read()
    else:
        raw = path.read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _choose(**kwargs: Any) -> dict[str, Any]:
    program = kwargs.pop("source_artifact")
    return choose_target_induced_action(
        **kwargs, program_artifact=program,
    )


def _task_id(value: str, v14_config: Mapping[str, Any]) -> str:
    virtual_root = (
        Path(str(v14_config["alfworld_data"])) / "json_2.1.1" / "train"
    ).resolve()
    return str(Path(value).resolve().relative_to(virtual_root))


def _trace(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
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


def run(config_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    if config.get("status") != "CONSUMED_V14_POLICY_DIAGNOSTIC":
        raise ValueError("V17 is restricted to a consumed-data diagnostic")
    dependencies = {
        "runner_file_sha256": Path(__file__).resolve(),
        "v14_config_file_sha256": REPO / str(config["v14_config"]),
        "v14_reference_report_file_sha256": REPO / str(
            config["v14_reference_report"]
        ),
        "v16_qualification_file_sha256": REPO / str(
            config["v16_qualification"]
        ),
        "target_program_file_sha256": REPO / str(config["target_program"]),
        "target_grounder_file_sha256": REPO / str(config["target_grounder"]),
        "target_causal_effect_file_sha256": REPO / str(
            config["target_causal_effect_artifact"]
        ),
        "target_inducer_file_sha256": REPO / (
            "src/motif_transfer/alfworld_target_recurrent_induction.py"
        ),
    }
    for field, path in dependencies.items():
        if _sha(path) != config[field]:
            raise ValueError(f"V17 dependency changed: {path}")
    v14_config = _read(dependencies["v14_config_file_sha256"])
    reference = _read(dependencies["v14_reference_report_file_sha256"])
    qualification = _read(dependencies["v16_qualification_file_sha256"])
    program = _read(dependencies["target_program_file_sha256"])
    target = _read(dependencies["target_grounder_file_sha256"])
    causal = _read(dependencies["target_causal_effect_file_sha256"])
    _self_hash(v14_config, "config_sha256")
    _self_hash(reference, "report_sha256")
    _self_hash(qualification, "report_sha256")
    validate_target_recurrent_program(program)
    if not all(qualification["gates"].values()):
        raise ValueError("target K=1 program did not pass V16 qualification")
    if qualification["lineage"]["target_k1_program_sha256"] != program[
        "program_sha256"
    ]:
        raise ValueError("V16 qualification/target program mismatch")
    task_ids = tuple(map(str, v14_config["task_ids"]))
    if stable_hash(list(task_ids)) != config["task_ids_sha256"]:
        raise ValueError("V17 task identities differ from consumed V14")
    if not target.get("target_grounder_gate", {}).get("passed"):
        raise ValueError("target-native grounder is not qualified")
    if not causal.get("gates", {}).get(
        "effect_balanced_accuracy_at_least_0p80"
    ):
        raise ValueError("target causal-effect head is not qualified")

    base.choose_goal_relation_action = _choose
    base.TargetRelationExecutionState = TargetWrittenExecutionState
    environment = _ValidTrainBatchEnvironment(
        config_path=str(v14_config["alfworld_config"]),
        data_path=str(v14_config["alfworld_data"]), split="train",
        seed=int(v14_config["seed"]), game_ids=task_ids,
        max_steps=int(v14_config["max_steps"]),
    )
    episodes = []
    try:
        for index in range(len(task_ids)):
            episode = base._run_episode(
                environment=environment, condition=TARGET_INDUCED,
                source_artifact=program,
                target_grounder=target["target_grounder"],
                target_causal_effect_head=causal["target_causal_effect_head"],
                max_steps=int(v14_config["max_steps"]),
                thresholds=v14_config["thresholds"],
            )
            body = dict(episode)
            body.pop("episode_sha256", None)
            body["task_id"] = _task_id(str(body["task_id"]), v14_config)
            episode = body | {"episode_sha256": stable_hash(body)}
            episodes.append(episode)
            print(json.dumps({
                "task_index": index, "task_id": episode["task_id"],
                "success": episode["official_success"],
                "steps": episode["steps"],
                "program_active_steps": sum(
                    row["program_active"] for row in episode["records"]
                ),
            }), flush=True)
    finally:
        environment.close()

    authentic = {
        str(row["task_id"]): row
        for row in reference["episodes"][AUTHENTIC]
    }
    comparisons = []
    for episode in episodes:
        expected = authentic[str(episode["task_id"])]
        body = {
            "task_id": str(episode["task_id"]),
            "target_induced_success": bool(episode["official_success"]),
            "source_induced_success": bool(expected["official_success"]),
            "actions_exactly_match_source": (
                list(episode["actions"]) == list(expected["actions"])
            ),
            "state_effect_trace_exactly_matches_source": (
                _trace(episode) == _trace(expected)
            ),
            "steps_exactly_match_source": int(episode["steps"]) == int(
                expected["steps"]
            ),
            "source_admissions": int(episode["source_admissions"]),
            "program_active_steps": sum(
                row["program_active"] for row in episode["records"]
            ),
            "program_origin_values": sorted({
                str(row.get("program_origin"))
                for row in episode["records"] if row.get("program_origin")
            }),
        }
        comparisons.append(body | {"comparison_sha256": stable_hash(body)})
    successes = sum(bool(row["official_success"]) for row in episodes)
    source_successes = int(reference["summaries"][AUTHENTIC]["successes"])
    raw_successes = int(reference["summaries"]["raw_target_only"]["successes"])
    diagnostics = dict(Counter(
        str(record["diagnostic"])
        for episode in episodes for record in episode["records"]
    ))
    gates = {
        "complete_consumed_v14_population": len(episodes) == 21,
        "matched_task_identities": {
            str(row["task_id"]) for row in episodes
        } == set(authentic),
        "target_program_reads_no_source_artifact": (
            program["source_artifact_read"] is False
        ),
        "zero_source_admissions": all(
            row["source_admissions"] == 0 for row in comparisons
        ),
        "target_induced_program_changes_policy": any(
            int(row["changed_actions_after_first_relation"]) > 0
            for row in episodes
        ),
        "target_induced_recovers_source_successes": (
            successes == source_successes == 18
        ),
        "target_induced_exceeds_raw_reference": successes > raw_successes,
        "all_action_traces_exactly_match_source": all(
            row["actions_exactly_match_source"] for row in comparisons
        ),
        "all_state_effect_traces_exactly_match_source": all(
            row["state_effect_trace_exactly_matches_source"]
            for row in comparisons
        ),
        "all_step_counts_exactly_match_source": all(
            row["steps_exactly_match_source"] for row in comparisons
        ),
        "no_new_task_or_outcome_claim": True,
    }
    passed = all(gates.values())
    report_body = {
        "schema_version": "alfworld-target-induced-policy-v17-report",
        "status": (
            "ALFWORLD_TARGET_INDUCED_POLICY_EQUIVALENCE_VALIDATED"
            if passed else "ALFWORLD_TARGET_INDUCED_POLICY_EQUIVALENCE_FAILED"
        ),
        "role": "posthoc_consumed_v14_policy_diagnostic",
        "claim_boundary": str(config["claim_boundary"]),
        "config_sha256": str(config["config_sha256"]),
        "target_program_sha256": str(program["program_sha256"]),
        "target_program_normal_form": execution_normal_form(program),
        "source_artifact_paths_loaded": [],
        "tasks": len(episodes),
        "target_induced_successes": successes,
        "source_induced_reference_successes": source_successes,
        "raw_target_only_reference_successes": raw_successes,
        "target_induced_gain_over_raw_reference": successes - raw_successes,
        "diagnostics": diagnostics,
        "comparisons": comparisons,
        "gates": gates,
        "episodes": episodes,
    }
    return report_body | {"report_sha256": stable_hash(report_body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO / args.config
    report = run(config_path)
    config = _read(config_path)
    output = REPO / str(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "tasks": report["tasks"],
        "target_induced_successes": report["target_induced_successes"],
        "source_induced_reference_successes": report[
            "source_induced_reference_successes"
        ],
        "raw_target_only_reference_successes": report[
            "raw_target_only_reference_successes"
        ],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
