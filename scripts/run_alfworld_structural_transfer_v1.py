#!/usr/bin/env python3
"""Run the frozen matched ALFWorld structural-transfer replication."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_structural_induction import (  # noqa: E402
    ALFWorldStructuralGrounder,
    validate_grounder,
    validate_target_sequence_program,
)
from motif_transfer.alfworld_structural_runtime_v1 import (  # noqa: E402
    ALFWorldStructuralSelector,
    CONDITIONS,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.structural_delta_induction import validate_structural_program  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_config(config: Mapping[str, Any]) -> None:
    body = dict(config)
    claimed = body.pop("config_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("ALFWorld structural frozen config hash mismatch")
    if tuple(config.get("conditions") or ()) != CONDITIONS:
        raise ValueError("ALFWorld structural condition matrix changed")
    for relative, expected in config["integrity"]["file_sha256"].items():
        path = (REPO / str(relative)).resolve()
        if _sha256(path) != str(expected):
            raise ValueError(f"frozen dependency changed: {path}")


def _paired(success: Mapping[str, Mapping[str, bool]], comparator: str) -> dict[str, Any]:
    authentic = success["source_induced"]
    other = success[comparator]
    ids = tuple(authentic)
    wins = sum(authentic[key] and not other[key] for key in ids)
    losses = sum(other[key] and not authentic[key] for key in ids)
    discordant = wins + losses
    exact_p = (
        min(1.0, 2.0 * sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1)) / (2 ** discordant))
        if discordant else 1.0
    )
    return {
        "wins": wins, "losses": losses,
        "ties": len(ids) - discordant,
        "negative_transfer_rate": losses / max(1, len(ids)),
        "two_sided_exact_sign_p": exact_p,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite formal report: {args.output}")
    config = _read(args.config.resolve())
    _validate_config(config)
    grounder = _read((REPO / config["grounder"]["path"]).resolve())
    validate_grounder(grounder)
    scorer = ALFWorldStructuralGrounder(grounder)
    if grounder["grounder_sha256"] != config["grounder"]["grounder_sha256"]:
        raise SystemExit("frozen ALFWorld structural grounder changed")
    target_program = grounder["target_program"]
    validate_target_sequence_program(target_program)
    source_program = _read((REPO / config["source_induced"]["path"]).resolve())
    permuted_program = _read((REPO / config["source_permuted"]["path"]).resolve())
    validate_structural_program(source_program)
    validate_structural_program(permuted_program)
    if source_program["program_sha256"] != config["source_induced"]["program_sha256"]:
        raise SystemExit("frozen authentic source program changed")
    if permuted_program["program_sha256"] != config["source_permuted"]["program_sha256"]:
        raise SystemExit("frozen permuted source program changed")

    task_ids = tuple(map(str, config["target"]["task_ids"]))
    split = str(config["target"]["split"])
    root_name = {
        "train": "train",
        "eval_in_distribution": "valid_seen",
        "eval_out_of_distribution": "valid_unseen",
    }[split]
    data_root = Path(config["target"]["alfworld_data"]) / "json_2.1.1" / root_name
    episodes: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(Path(config["target"]["alfworld_config"]).resolve()),
            data_path=str(Path(config["target"]["alfworld_data"]).resolve()),
            split=split,
            seed=int(config["target"]["seed"]),
            game_ids=task_ids,
            max_steps=int(config["target"]["max_steps"]),
            expose_expert_plan=condition == "target_native_ceiling",
        )
        try:
            for _ in task_ids:
                observation = environment.reset()
                actual_id = Path(environment.resolved_game_file).resolve().relative_to(
                    data_root.resolve()
                ).as_posix()
                source_sequence = (
                    source_program["induced_sequence"]
                    if condition == "source_induced"
                    else permuted_program["induced_sequence"]
                    if condition == "source_permuted" else ()
                )
                selector = ALFWorldStructuralSelector(
                    condition=condition,
                    target_sequence=target_program["induced_sequence"],
                    source_sequence=source_sequence,
                    threshold=float(grounder["threshold"]),
                )
                history: list[str] = []
                records = []
                for step in range(int(config["target"]["max_steps"])):
                    goal = str(observation.state.get("task_goal") or "")
                    current_text = str(observation.state.get("observation") or "")
                    rows = scorer.score_candidates(
                        goal=goal, observation=current_text,
                        actions=observation.native_actions, step=step,
                        action_history=history,
                    )
                    expert_action = (
                        environment.expert_action()
                        if condition == "target_native_ceiling" else None
                    )
                    decision = selector.select(
                        rows=rows, history=history, goal=goal,
                        expert_action=expert_action,
                    )
                    selected = str(decision["action"])
                    after, reward = environment.step(selected)
                    transition_receipt = selector.observe_transition(
                        after_observation=str(after.state.get("observation") or ""),
                    )
                    record_body = {
                        "step": step,
                        "selected_action": selected,
                        "selected_action_sha256": rows[selected]["action_sha256"],
                        "decision": decision,
                        "transition_receipt": transition_receipt,
                        "target_native_observation_sha256": stable_hash({
                            "observation": str(after.state.get("observation") or ""),
                            "native_actions": list(after.native_actions),
                        }),
                        "official_success_evaluator_only": bool(after.official_success),
                        "reward_evaluator_only": float(reward),
                    }
                    records.append(record_body | {"record_sha256": stable_hash(record_body)})
                    history.append(selected)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                episode_body = {
                    "task_id": actual_id,
                    "condition": condition,
                    "official_success": bool(observation.official_success),
                    "steps": len(records),
                    "records": records,
                    "source_admissions": selector.source_admissions,
                    "source_abstentions": selector.source_abstentions,
                    "transition_mismatches": selector.transition_mismatches,
                    "observed_operator_sequence": selector.observed_operator_sequence,
                    "source_cursor_final": selector.cursor,
                    "formal_outcome_available_to_controller": False,
                }
                episodes[condition].append(
                    episode_body | {"episode_sha256": stable_hash(episode_body)}
                )
                print(
                    f"{condition} {len(episodes[condition])}/{len(task_ids)} "
                    f"success={int(observation.official_success)} steps={len(records)}",
                    flush=True,
                )
        finally:
            environment.close()

    success = {
        condition: {row["task_id"]: bool(row["official_success"]) for row in rows}
        for condition, rows in episodes.items()
    }
    summaries = {}
    for condition, rows in episodes.items():
        records = [record for row in rows for record in row["records"]]
        summaries[condition] = {
            "tasks": len(rows),
            "successes": sum(bool(row["official_success"]) for row in rows),
            "success_rate": sum(bool(row["official_success"]) for row in rows) / len(rows),
            "mean_steps": sum(int(row["steps"]) for row in rows) / len(rows),
            "source_admissions": sum(int(row["source_admissions"]) for row in rows),
            "source_admission_rate": sum(int(row["source_admissions"]) for row in rows) / max(1, len(records)),
            "changed_from_neural_actions": sum(
                record["selected_action"] != record["decision"]["neural_action"]
                for record in records
            ),
            "changed_from_neural_rate": sum(
                record["selected_action"] != record["decision"]["neural_action"]
                for record in records
            ) / max(1, len(records)),
        }
    comparisons = {
        comparator: _paired(success, comparator)
        for comparator in ("neural_only", "source_permuted", "generic_scaffold")
    }
    realized_orders = {
        condition: [row["task_id"] for row in episodes[condition]]
        for condition in CONDITIONS
    }
    exact_matrix = all(
        realized_orders[condition] == realized_orders[CONDITIONS[0]]
        for condition in CONDITIONS
    )
    exact_frozen_set = all(
        set(realized_orders[condition]) == set(task_ids)
        and len(realized_orders[condition]) == len(task_ids)
        for condition in CONDITIONS
    )
    preregistered = dict(config.get("preregistered_gates") or {})
    gates = {
        "exact_matched_task_matrix": exact_matrix,
        "exact_frozen_task_set": exact_frozen_set,
        "minimum_twelve_fresh_tasks": len(task_ids) >= int(
            preregistered.get("minimum_tasks", 12)
        ),
        "source_strictly_beats_neural": summaries["source_induced"]["successes"] > summaries["neural_only"]["successes"],
        "source_strictly_beats_permuted": summaries["source_induced"]["successes"] > summaries["source_permuted"]["successes"],
        "source_strictly_beats_generic": summaries["source_induced"]["successes"] > summaries["generic_scaffold"]["successes"],
        "paired_significance_vs_neural": comparisons["neural_only"]["two_sided_exact_sign_p"] <= float(
            preregistered.get("two_sided_exact_sign_p_vs_neural_max", 0.05)
        ),
        "negative_transfer_at_most_ten_percent": comparisons["neural_only"]["negative_transfer_rate"] <= float(
            preregistered.get("negative_transfer_rate_vs_neural_max", 0.10)
        ),
        "target_native_ceiling_capable": summaries["target_native_ceiling"]["success_rate"] >= float(
            preregistered.get("target_native_ceiling_success_rate_min", 0.80)
        ),
        "source_behavior_nontrivial": summaries["source_induced"]["changed_from_neural_actions"] >= int(
            preregistered.get("changed_from_neural_action_count_min", 1)
        ),
        "source_operator_admitted": summaries["source_induced"]["source_admissions"] >= int(
            preregistered.get("source_operator_admissions_min", len(task_ids) * 2)
        ),
        "formal_outcome_never_available_to_controller": all(
            row["formal_outcome_available_to_controller"] is False
            for rows in episodes.values() for row in rows
        ),
        "frozen_unique_source_applicability": config["source_induced"]["source_name_evaluator_label_only"] == "put_near",
    }
    development_mode = config.get("evaluation_mode") == "DEVELOPMENT"
    if development_mode:
        gates = {
            "exact_matched_task_matrix": exact_matrix,
            "exact_frozen_task_set": exact_frozen_set,
            "source_has_positive_success": summaries["source_induced"]["successes"] > 0,
            "source_not_below_neural": summaries["source_induced"]["successes"] >= summaries["neural_only"]["successes"],
            "source_strictly_beats_permuted": summaries["source_induced"]["successes"] > summaries["source_permuted"]["successes"],
            "source_operator_admitted": summaries["source_induced"]["source_admissions"] >= len(task_ids) * 2,
            "formal_outcome_never_available_to_controller": gates["formal_outcome_never_available_to_controller"],
        }
    passed = all(gates.values())
    report_body = {
        "schema_version": "alfworld-structural-transfer-formal-report-v1",
        "role": str(config.get("role") or "FRESH_MATCHED_CAUSAL_SECOND_TARGET_REPLICATION"),
        "status": (
            "ALFWORLD_STRUCTURAL_DEVELOPMENT_PASSED" if development_mode and passed
            else "ALFWORLD_STRUCTURAL_DEVELOPMENT_FAILED" if development_mode
            else "ALFWORLD_STRUCTURAL_TRANSFER_VALIDATED" if passed
            else "ALFWORLD_STRUCTURAL_TRANSFER_FAILED"
        ),
        "config_path": str(args.config.resolve().relative_to(REPO)),
        "config_sha256": config["config_sha256"],
        "grounder_sha256": grounder["grounder_sha256"],
        "target_program_sha256": target_program["program_sha256"],
        "source_program_sha256": source_program["program_sha256"],
        "source_permuted_program_sha256": permuted_program["program_sha256"],
        "conditions": list(CONDITIONS),
        "task_ids": list(task_ids),
        "episodes": episodes,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "preregistered_gates": preregistered,
        "gates": gates,
        "formal_results_used_to_change_protocol": False if not development_mode else None,
        "claim_boundary": (
            "Fresh task replication of a source-induced anonymous structural "
            "subprogram with ALFWorld-native neural grounding; not evidence of "
            "zero-shot target policy transfer."
        ),
    }
    report = report_body | {"report_sha256": stable_hash(report_body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "summaries": summaries,
        "paired_comparisons": comparisons, "gates": gates,
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
