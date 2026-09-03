#!/usr/bin/env python3
"""Run the frozen fresh ALFWorld source-vs-target acquisition replication."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


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
import run_alfworld_unified_goal_acquisition_v11 as frozen  # noqa: E402
from motif_transfer.active_video_transfer import (  # noqa: E402
    exact_binomial_two_sided,
)
from motif_transfer.alfworld_env import (  # noqa: E402
    ALFWorldTextBatchEnvironment as _FrozenBatchEnvironment,
)
from motif_transfer.alfworld_goal_relation_macro import (  # noqa: E402
    AUTHENTIC,
    CARDINALITY_CONTROL,
    EFFECT_CONTROL,
    GENERIC,
    RAW,
)
from motif_transfer.alfworld_target_recurrent_induction import (  # noqa: E402
    TARGET_INDUCED,
    choose_target_induced_action,
    execution_normal_form,
    permute_binding_relation,
    validate_target_recurrent_program,
)
from motif_transfer.alfworld_target_written_equivalent import (  # noqa: E402
    TargetWrittenExecutionState,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_CONFIG = (
    REPO / "configs/alfworld_target_acquisition_fresh_v18/formal.json"
)
FORMAL_STATUS = "FROZEN_BEFORE_ANY_V19_FRESH_POLICY_RESET_OR_OUTCOME"
TARGET_K1 = "target_only_k1_induced_program"
TARGET_PERMUTED = "target_only_k1_binding_relation_permuted"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
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


class _FrozenPhysicalSplitBatchEnvironment(_FrozenBatchEnvironment):
    """Expose a frozen physical split through V11's virtual train token."""

    physical_split = "valid_seen"

    def __init__(self, *args, data_path: str, split: str, **kwargs) -> None:
        if split != "train":
            raise ValueError("V19 expects V11's frozen train split token")
        self._v19_data_root = Path(data_path).resolve()
        transport = {
            "train": "train",
            "valid_seen": "eval_in_distribution",
            "valid_unseen": "eval_out_of_distribution",
        }[self.physical_split]
        super().__init__(
            *args, data_path=data_path, split=transport, **kwargs,
        )

    def reset(self):
        observation = super().reset()
        actual = Path(self.resolved_game_file).resolve()
        relative = actual.relative_to(
            self._v19_data_root / "json_2.1.1" / self.physical_split
        )
        self.resolved_game_file = str(
            self._v19_data_root / "json_2.1.1" / "train" / relative
        )
        return observation


def _normalize_episode(
    episode: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    fake_root = (
        Path(str(config["alfworld_data"])) / "json_2.1.1" / "train"
    ).resolve()
    actual_root = (
        Path(str(config["alfworld_data"])) / "json_2.1.1"
        / str(config["alfworld_physical_split"])
    ).resolve()
    body = dict(episode)
    body.pop("episode_sha256", None)
    relative = str(Path(str(body["task_id"])).resolve().relative_to(fake_root))
    actual = actual_root / relative
    if _sha(actual) != config["task_file_sha256"].get(relative):
        raise ValueError(f"V19 physical game hash mismatch: {relative}")
    body["task_id"] = relative
    return body | {"episode_sha256": stable_hash(body)}


def _normalize_source_report(
    report: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(report)
    body.pop("report_sha256", None)
    episodes = {
        condition: [
            _normalize_episode(episode, config) for episode in rows
        ]
        for condition, rows in report["episodes"].items()
    }
    body["episodes"] = episodes
    body["schema_version"] = "alfworld-target-acquisition-fresh-v19-source-report"
    body["role"] = "prospective_execution_untouched_mechanism_replication"
    body["physical_data_split"] = str(config["alfworld_physical_split"])
    body["v11_frozen_action_runtime_reused"] = True
    return body | {"report_sha256": stable_hash(body)}


def _choose_target(**kwargs: Any) -> dict[str, Any]:
    program = kwargs.pop("source_artifact")
    kwargs["condition"] = TARGET_INDUCED
    return choose_target_induced_action(
        **kwargs, program_artifact=program,
    )


def _run_target_condition(
    *, condition: str, program: Mapping[str, Any],
    config: Mapping[str, Any], target: Mapping[str, Any],
    causal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    base.choose_goal_relation_action = _choose_target
    base.TargetRelationExecutionState = TargetWrittenExecutionState
    task_ids = tuple(map(str, config["task_ids"]))
    environment = _FrozenPhysicalSplitBatchEnvironment(
        config_path=str(config["alfworld_config"]),
        data_path=str(config["alfworld_data"]), split="train",
        seed=int(config["seed"]), game_ids=task_ids,
        max_steps=int(config["max_steps"]),
    )
    episodes = []
    try:
        for index in range(len(task_ids)):
            episode = base._run_episode(
                environment=environment, condition=condition,
                source_artifact=program,
                target_grounder=target["target_grounder"],
                target_causal_effect_head=causal["target_causal_effect_head"],
                max_steps=int(config["max_steps"]),
                thresholds=config["thresholds"],
            )
            episode = _normalize_episode(episode, config)
            episodes.append(episode)
            print(json.dumps({
                "condition": condition,
                "task_index": index,
                "task_id": episode["task_id"],
                "success": episode["official_success"],
                "steps": episode["steps"],
            }), flush=True)
    finally:
        environment.close()
    return episodes


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


def _paired(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    right_by_task = {str(row["task_id"]): row for row in right}
    wins = losses = 0
    for row in left:
        other = right_by_task[str(row["task_id"])]
        left_success = bool(row["official_success"])
        right_success = bool(other["official_success"])
        wins += int(left_success and not right_success)
        losses += int(not left_success and right_success)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(left) - wins - losses,
        "net_wins": wins - losses,
        "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
    }


def _changed_vs(
    episodes: Mapping[str, list[dict[str, Any]]], reference: str,
) -> None:
    baseline = {
        str(row["task_id"]): row for row in episodes[reference]
    }
    for condition, rows in episodes.items():
        for row in rows:
            other = baseline[str(row["task_id"])]
            row["changed_actions_vs_raw_trajectory"] = sum(
                left != right for left, right in zip_longest(
                    row["actions"], other["actions"], fillvalue=None,
                )
            )


def run(config_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    if config.get("status") != FORMAL_STATUS:
        raise ValueError("V19 fresh protocol is not frozen")
    dependencies = {
        "v19_runner_file_sha256": Path(__file__).resolve(),
        "preparer_file_sha256": REPO / str(config["preparer"]),
        "selection_file_sha256": REPO / str(config["selection"]),
        "compiler_audit_file_sha256": REPO / str(config["compiler_audit"]),
        "target_k1_program_file_sha256": REPO / str(
            config["target_k1_program"]
        ),
        "target_inducer_file_sha256": REPO / (
            "src/motif_transfer/alfworld_target_recurrent_induction.py"
        ),
        "alfworld_config_file_sha256": Path(str(config["alfworld_config"])),
        "required_python_executable_sha256": Path(str(
            config["required_python_executable"]
        )),
    }
    for field, path in dependencies.items():
        if _sha(path) != config[field]:
            raise ValueError(f"V19 frozen dependency changed: {path}")
    required_python = Path(str(config["required_python_executable"])).resolve()
    if Path(sys.executable).resolve() != required_python:
        raise ValueError(
            f"V19 requires frozen interpreter {required_python}, got "
            f"{Path(sys.executable).resolve()}"
        )
    if f"{sys.version_info.major}.{sys.version_info.minor}" != config[
        "required_python_major_minor"
    ]:
        raise ValueError("V19 Python major/minor contract changed")
    selection = _read(dependencies["selection_file_sha256"])
    compiler = _read(dependencies["compiler_audit_file_sha256"])
    _self_hash(selection, "selection_sha256")
    _self_hash(compiler, "compiler_audit_sha256")
    if selection["selection_sha256"] != config["selection_sha256"]:
        raise ValueError("V19 selection lineage mismatch")
    if compiler["compiler_audit_sha256"] != config["compiler_audit_sha256"]:
        raise ValueError("V19 compiler lineage mismatch")
    if compiler["selection_sha256"] != selection["selection_sha256"]:
        raise ValueError("V19 compiler preceded the frozen selection")

    program = _read(dependencies["target_k1_program_file_sha256"])
    validate_target_recurrent_program(program)
    if int(program["complete_target_trajectory_budget"]) != 1:
        raise ValueError("V19 target program must be the K=1 artifact")
    target = _read(REPO / str(config["target_grounder"]))
    causal = _read(REPO / str(config["target_causal_effect_artifact"]))

    # V11 is reused as a frozen action runtime.  V19 deliberately omits a
    # powered significance gate because the complete frozen reserve has n=8.
    physical_split = str(config["alfworld_physical_split"])
    if physical_split not in {"train", "valid_seen", "valid_unseen"}:
        raise ValueError(f"unsupported V19 physical split: {physical_split}")
    _FrozenPhysicalSplitBatchEnvironment.physical_split = physical_split
    frozen.DEVELOPMENT_STATUS = FORMAL_STATUS
    frozen.ALFWorldTextBatchEnvironment = _FrozenPhysicalSplitBatchEnvironment
    source_report = _normalize_source_report(
        frozen.run(config_path), config,
    )
    episodes: dict[str, list[dict[str, Any]]] = {
        condition: list(rows)
        for condition, rows in source_report["episodes"].items()
    }
    episodes[TARGET_K1] = _run_target_condition(
        condition=TARGET_K1, program=program, config=config,
        target=target, causal=causal,
    )
    permuted = permute_binding_relation(program)
    episodes[TARGET_PERMUTED] = _run_target_condition(
        condition=TARGET_PERMUTED, program=permuted, config=config,
        target=target, causal=causal,
    )
    _changed_vs(episodes, RAW)

    expected = set(map(str, config["task_ids"]))
    selected = {str(row["task_id"]) for row in selection["tasks"]}
    compiler_ids = set(map(str, compiler["compiler_results"]))
    generated_ids = set(map(str, compiler["generated_game_file_sha256"]))
    eligible = {
        task_id for task_id, row in compiler["compiler_results"].items()
        if row["solvable"]
    }
    eligibility_filter = str(config.get("compiler_eligibility_filter", "ITT_ALL"))
    summaries = {
        condition: base._summary(rows) for condition, rows in episodes.items()
    }
    source_by_task = {
        str(row["task_id"]): row for row in episodes[AUTHENTIC]
    }
    target_comparisons = []
    for row in episodes[TARGET_K1]:
        source = source_by_task[str(row["task_id"])]
        body = {
            "task_id": str(row["task_id"]),
            "source_success": bool(source["official_success"]),
            "target_k1_success": bool(row["official_success"]),
            "actions_exactly_match": list(source["actions"]) == list(row["actions"]),
            "state_effect_trace_exactly_matches": _trace(source) == _trace(row),
            "steps_exactly_match": int(source["steps"]) == int(row["steps"]),
            "target_source_admissions": int(row["source_admissions"]),
        }
        target_comparisons.append(
            body | {"comparison_sha256": stable_hash(body)}
        )
    paired = {
        "source_vs_raw": _paired(episodes[AUTHENTIC], episodes[RAW]),
        "source_vs_target_k1": _paired(
            episodes[AUTHENTIC], episodes[TARGET_K1],
        ),
        "target_k1_vs_raw": _paired(episodes[TARGET_K1], episodes[RAW]),
        "target_k1_vs_target_permuted": _paired(
            episodes[TARGET_K1], episodes[TARGET_PERMUTED],
        ),
    }
    base_gates = source_report["gates"]
    source_success = int(summaries[AUTHENTIC]["successes"])
    raw_success = int(summaries[RAW]["successes"])
    target_success = int(summaries[TARGET_K1]["successes"])
    permuted_success = int(summaries[TARGET_PERMUTED]["successes"])
    gates = {
        "all_identities_frozen_before_compilation": (
            len(expected) >= int(config["gates"]["minimum_fresh_tasks"])
            and selection["status"] == config["selection_frozen_status"]
        ),
        "identity_selection_used_no_observation_or_outcome": (
            selection["selection_used_observation_walkthrough_or_policy_outcome"]
            is False
        ),
        "all_frozen_tasks_compile_audited_without_replacement": (
            compiler_ids == selected
            and generated_ids == selected
            and compiler["compiler_solvability_used_for_identity_selection"]
            is False
        ),
        "compiler_eligibility_rule_applied_without_policy_outcome": (
            (eligibility_filter == "ITT_ALL" and expected == selected)
            or (
                eligibility_filter == "SOLVABLE_TRUE_ONLY"
                and expected == eligible
            )
        ),
        "compiler_walkthrough_never_exposed_to_policy": (
            compiler["compiler_walkthrough_exposed_to_transfer_policy"] is False
        ),
        "complete_matched_fresh_task_matrix": all(
            len(rows) == len(expected)
            and {str(row["task_id"]) for row in rows} == expected
            for rows in episodes.values()
        ),
        "unified_source_harness_authorized_every_task": bool(
            base_gates["all_tasks_pre_authorized_by_unified_harness"]
            and base_gates["unified_route_is_exact"]
            and base_gates[
                "selector_emits_no_action_and_reads_no_current_outcome"
            ]
            and base_gates[
                "every_source_active_action_uses_target_native_executor"
            ]
        ),
        "source_and_target_grounding_qualification_reused": bool(
            base_gates["source_acquisition_fresh_confirmation_passed"]
            and base_gates["target_neural_grounder_gate_passed"]
        ),
        "source_uses_zero_complete_target_trajectories": (
            int(config["source_complete_target_trajectory_budget"]) == 0
        ),
        "target_only_requires_one_complete_target_trajectory": (
            int(config["target_complete_trajectory_budget"]) == 1
            and int(program["complete_target_trajectory_budget"]) == 1
            and program["source_artifact_read"] is False
        ),
        "source_and_target_k1_successes_match": source_success == target_success,
        "source_and_target_k1_action_traces_match": all(
            row["actions_exactly_match"] for row in target_comparisons
        ),
        "source_and_target_k1_state_effect_traces_match": all(
            row["state_effect_trace_exactly_matches"]
            for row in target_comparisons
        ),
        "target_k1_has_zero_source_admissions": all(
            row["target_source_admissions"] == 0
            for row in target_comparisons
        ),
        "source_has_fresh_success_gain_over_raw": source_success > raw_success,
        "target_k1_has_fresh_success_gain_over_permuted": (
            target_success > permuted_success
        ),
        "source_strictly_beats_source_program_controls": all(
            source_success > int(summaries[name]["successes"])
            for name in (CARDINALITY_CONTROL, EFFECT_CONTROL, GENERIC)
        ),
        "zero_source_negative_transfer_vs_raw": (
            paired["source_vs_raw"]["losses"] == 0
        ),
        "zero_target_k1_negative_transfer_vs_raw": (
            paired["target_k1_vs_raw"]["losses"] == 0
        ),
    }
    passed = all(gates.values())
    diagnostics = {
        condition: dict(Counter(
            str(record["diagnostic"])
            for episode in rows for record in episode["records"]
        ))
        for condition, rows in episodes.items()
    }
    report_body = {
        "schema_version": "alfworld-target-acquisition-fresh-v19-report",
        "status": (
            "ALFWORLD_SOURCE_ACQUISITION_VALUE_FRESH_VALIDATED"
            if passed else "ALFWORLD_SOURCE_ACQUISITION_VALUE_FRESH_FAILED"
        ),
        "role": "prospective_execution_untouched_mechanism_replication",
        "claim_boundary": str(config["claim_boundary"]),
        "config_sha256": str(config["config_sha256"]),
        "selection_sha256": str(selection["selection_sha256"]),
        "compiler_audit_sha256": str(compiler["compiler_audit_sha256"]),
        "compiler_solvable_tasks": sum(
            bool(row["solvable"])
            for row in compiler["compiler_results"].values()
        ),
        "compiler_unsolvable_tasks_retained_in_itt": sum(
            not bool(row["solvable"])
            for row in compiler["compiler_results"].values()
        ),
        "source_complete_target_trajectory_budget": 0,
        "target_only_complete_target_trajectory_budget": 1,
        "target_k1_program_sha256": str(program["program_sha256"]),
        "target_k1_execution_normal_form": execution_normal_form(program),
        "source_artifact_paths_loaded_by_target_conditions": [],
        "identifiability_estimand": (
            "Execution efficacy of matched program content versus acquisition "
            "cost of obtaining that content from source interventions or "
            "complete target demonstrations."
        ),
        "summaries": summaries,
        "paired": paired,
        "target_k1_source_comparisons": target_comparisons,
        "diagnostics": diagnostics,
        "source_runtime_gates": base_gates,
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
        "summaries": report["summaries"],
        "paired": report["paired"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
