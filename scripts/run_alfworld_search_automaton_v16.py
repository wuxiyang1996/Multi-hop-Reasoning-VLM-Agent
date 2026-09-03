#!/usr/bin/env python3
"""Run V16 source search control on consumed ALFWorld development tasks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.active_video_transfer import (  # noqa: E402
    exact_binomial_two_sided,
)
from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_hierarchical_grounder import score_actions  # noqa: E402
from motif_transfer.alfworld_search_automaton_v16 import (  # noqa: E402
    AUTHENTIC,
    CEILING,
    CONDITIONS,
    LEDGER_BLIND,
    PERMUTED,
    RAW,
    classify_target_outcome,
    summarize_episodes,
    target_policy_rank,
    target_scope_id,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    AttemptLedger,
    OUTCOME_REFUTED,
    OUTCOME_TERMINAL_VERIFIED,
    SourceSearchAutomaton,
    bind_native_action,
    ground_target_event,
)
from motif_transfer.sokoban_search_automaton_v16 import (  # noqa: E402
    BACKTRACK,
    COMMIT,
    EXPLORE,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route(
    source: SourceSearchAutomaton,
    *,
    event_name: str,
    episode_id: str,
    decision_index: int,
    evidence_kind: str,
    evidence_payload: Mapping[str, Any],
    source_action: str,
    native_action_id: str,
    native_action: Any,
    confidence: float,
) -> dict[str, Any]:
    event = ground_target_event(
        domain="alfworld",
        episode_id=episode_id,
        decision_index=decision_index,
        untried_candidate_available=event_name == "UNBOUND",
        active_candidate_refuted=event_name == "REFUTED",
        terminal_commit_verified=event_name == "VERIFIED",
        evidence_kind=evidence_kind,
        evidence_payload=evidence_payload,
        grounding_confidence=confidence,
    )
    if event is None:
        raise RuntimeError("ALFWorld target event unexpectedly abstained")
    binding = bind_native_action(
        event,
        abstract_action=source_action,
        native_action_id=native_action_id,
        native_action=native_action,
        grounding_confidence=confidence,
    )
    return asdict(source.route(event, {source_action: binding}))


def _run_episode(
    *,
    environment: ALFWorldTextBatchEnvironment,
    condition: str,
    source: SourceSearchAutomaton,
    target_grounder: Mapping[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    observation = environment.reset()
    episode_id = str(Path(environment.resolved_game_file).name)
    attempted_history: list[str] = []
    effect_history: list[str] = []
    records: list[dict[str, Any]] = []
    source_decisions: list[dict[str, Any]] = []
    ledger = AttemptLedger()
    for step in range(max_steps):
        native_actions = list(observation.native_actions)
        goal = str(observation.state.get("task_goal", ""))
        before_text = str(observation.state.get("observation", ""))
        grounded = score_actions(
            goal=goal,
            observation=before_text,
            native_actions=native_actions,
            step=step,
            action_history=effect_history,
            artifact=target_grounder,
        )
        if not grounded:
            break
        ranked = target_policy_rank(
            grounded,
            attempted_history,
            discount_repeats=condition != LEDGER_BLIND,
            structured=condition in {AUTHENTIC, LEDGER_BLIND, CEILING},
        )
        fallback = ranked[0]
        selected = fallback
        decision_receipt = None
        ledger_active = False
        if condition in {AUTHENTIC, LEDGER_BLIND, CEILING}:
            scope = target_scope_id(
                goal=goal,
                native_actions=list(grounded),
                history=effect_history,
            )
            ledger.begin_scope(
                f"blind-{step}" if condition == LEDGER_BLIND else scope
            )
            selected_id = next(
                (candidate for candidate in ranked if candidate not in ledger.tried),
                None,
            )
            if selected_id is None:
                # The source contract has no edge for an exhausted target
                # candidate set.  It must abstain to the target policy, not
                # terminate an otherwise viable target episode.
                selected = fallback
            else:
                selected = selected_id
                ledger_active = True
            if condition != CEILING and ledger_active:
                confidence = float(grounded[selected]["applicability"])
                decision_receipt = _route(
                    source,
                    event_name="UNBOUND",
                    episode_id=episode_id,
                    decision_index=len(source_decisions),
                    evidence_kind="target_neural_untried_native_action",
                    evidence_payload={
                        "target_action_id": stable_hash(selected),
                        "target_scope_sha256": scope,
                        "untried_target_action_count": sum(
                            action not in ledger.tried for action in ranked
                        ),
                    },
                    source_action=EXPLORE,
                    native_action_id=stable_hash(selected),
                    native_action=selected,
                    confidence=confidence,
                )
                source_decisions.append(decision_receipt)
                if not decision_receipt["admitted"]:
                    # A low-confidence target action binding revokes source
                    # authority for this step; it must not terminate the
                    # target episode.
                    selected = fallback
                    ledger_active = False
                else:
                    selected = str(decision_receipt["native_action"])
            if ledger_active:
                ledger.next_untried([selected])
        elif condition == PERMUTED:
            # Mislabeled UNBOUND -> REFUTED cannot select a candidate; target
            # fallback remains responsible for the actual action.
            confidence = float(grounded[fallback]["applicability"])
            decision_receipt = _route(
                source,
                event_name="REFUTED",
                episode_id=episode_id,
                decision_index=len(source_decisions),
                evidence_kind="permuted_unbound_as_refuted_control",
                evidence_payload={"target_action_id": stable_hash(fallback)},
                source_action=BACKTRACK,
                native_action_id="abstain_to_target_fallback",
                native_action={"operation": "target_fallback"},
                confidence=confidence,
            )
            source_decisions.append(decision_receipt)

        before_native = tuple(observation.native_actions)
        after, reward = environment.step(selected)
        outcome = classify_target_outcome(
            goal=goal,
            selected_action=selected,
            selected_grounding=grounded[selected],
            effect_history=effect_history,
            before_observation=before_text,
            after_observation=str(after.state.get("observation", "")),
            before_native_actions=before_native,
            after_native_actions=after.native_actions,
            official_success_after=after.official_success,
        )
        attempted_history.append(selected)
        if outcome != OUTCOME_REFUTED:
            effect_history.append(selected)
        if ledger_active:
            event_name = ledger.observe(selected, outcome)
            if condition != CEILING and event_name is not None:
                if outcome == OUTCOME_REFUTED:
                    source_action = BACKTRACK
                    native_id = "reject_target_action_and_replan"
                    native_action = {"target_action_id": stable_hash(selected)}
                    kind = "target_native_action_effect_refuted"
                elif outcome == OUTCOME_TERMINAL_VERIFIED:
                    source_action = COMMIT
                    native_id = "accept_official_terminal_success"
                    native_action = {"operation": "terminate_success"}
                    kind = "target_native_terminal_success_verified"
                else:
                    raise RuntimeError("unexpected routed ALFWorld outcome")
                outcome_decision = _route(
                    source,
                    event_name=(
                        "REFUTED" if outcome == OUTCOME_REFUTED else "VERIFIED"
                    ),
                    episode_id=episode_id,
                    decision_index=len(source_decisions),
                    evidence_kind=kind,
                    evidence_payload={
                        "target_action_id": stable_hash(selected),
                        "before_native_actions_sha256": stable_hash(before_native),
                        "after_native_actions_sha256": stable_hash(after.native_actions),
                        "official_success_after": bool(after.official_success),
                    },
                    source_action=source_action,
                    native_action_id=native_id,
                    native_action=native_action,
                    confidence=1.0,
                )
                source_decisions.append(outcome_decision)
        records.append({
            "step": step,
            "selected_action": selected,
            "fallback_action": fallback,
            "changed_from_target_fallback": selected != fallback,
            "target_option": str(grounded[selected]["option"]),
            "target_policy_probability": float(grounded[selected]["policy"]),
            "target_applicability": float(grounded[selected]["applicability"]),
            "target_outcome": outcome,
            "reward_diagnostic_only": float(reward),
            "official_success_after": bool(after.official_success),
            "before_state_sha256": stable_hash({
                "observation": before_text,
                "native_actions": before_native,
            }),
            "after_state_sha256": stable_hash({
                "observation": str(after.state.get("observation", "")),
                "native_actions": after.native_actions,
            }),
            "source_decision_receipt_sha256": (
                decision_receipt["receipt_sha256"]
                if decision_receipt is not None else None
            ),
        })
        observation = after
        if after.terminal or after.official_success:
            break
    return {
        "task_id": str(environment.resolved_game_file),
        "condition": condition,
        "official_success": bool(records and records[-1]["official_success_after"]),
        "steps": len(records),
        "actions": [row["selected_action"] for row in records],
        "target_outcomes": dict(Counter(row["target_outcome"] for row in records)),
        "source_decisions": len(source_decisions),
        "source_action_counts": dict(Counter(
            row["source_action"] for row in source_decisions if row["admitted"]
        )),
        "source_trace": source_decisions,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO / "configs/alfworld_search_automaton_v16_development.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = REPO / config["source"]["artifact"]
    if _file_sha256(source_path) != config["source"]["artifact_file_sha256"]:
        raise SystemExit("source artifact file changed after config freeze")
    source_artifact = json.loads(source_path.read_text(encoding="utf-8"))
    source = SourceSearchAutomaton(
        source_artifact,
        expected_sha256=str(config["source"]["artifact_sha256"]),
    )
    target_config = config["target"]
    grounder_path = REPO / target_config["target_grounder_artifact"]
    manifest_path = REPO / target_config["manifest"]
    if _file_sha256(grounder_path) != target_config[
        "target_grounder_artifact_file_sha256"
    ]:
        raise SystemExit("target grounder artifact changed after config freeze")
    if _file_sha256(manifest_path) != target_config["manifest_file_sha256"]:
        raise SystemExit("target manifest changed after config freeze")
    target_artifact = json.loads(grounder_path.read_text(encoding="utf-8"))
    if not target_artifact.get("target_grounder_gate", {}).get("passed"):
        raise SystemExit("target-native neural grounder gate did not pass")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_ids = tuple(map(str, manifest["cells"][
        target_config["manifest_cell"]
    ]["splits"][target_config["manifest_split"]]))
    configured_conditions = tuple(config["conditions"])
    if configured_conditions != CONDITIONS:
        raise SystemExit("ALFWorld V16 condition matrix changed")
    episodes: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(target_config["alfworld_config"]),
            data_path=str(target_config["alfworld_data"]),
            split=str(target_config["split"]),
            seed=int(target_config["seed"]),
            game_ids=task_ids,
            max_steps=int(target_config["max_steps"]),
        )
        try:
            for task_index in range(len(task_ids)):
                episode = _run_episode(
                    environment=environment,
                    condition=condition,
                    source=source,
                    target_grounder=target_artifact["target_grounder"],
                    max_steps=int(target_config["max_steps"]),
                )
                episodes[condition].append(episode)
                print(json.dumps({
                    "condition": condition,
                    "task_index": task_index,
                    "success": episode["official_success"],
                    "steps": episode["steps"],
                }), flush=True)
        finally:
            environment.close()

    raw_by_task = {row["task_id"]: row for row in episodes[RAW]}
    for condition, rows in episodes.items():
        for row in rows:
            raw_actions = raw_by_task[row["task_id"]]["actions"]
            row["changed_actions_vs_raw"] = sum(
                left != right
                for left, right in zip_longest(
                    row["actions"], raw_actions, fillvalue=None
                )
            )
    summaries = summarize_episodes(episodes)
    authentic = {row["task_id"]: row for row in episodes[AUTHENTIC]}
    paired = {}
    for comparator in CONDITIONS:
        if comparator == AUTHENTIC:
            continue
        other = {row["task_id"]: row for row in episodes[comparator]}
        wins = losses = 0
        for task_id, row in authentic.items():
            a = bool(row["official_success"])
            b = bool(other[task_id]["official_success"])
            wins += a and not b
            losses += b and not a
        paired[comparator] = {
            "wins": wins,
            "losses": losses,
            "ties": len(task_ids) - wins - losses,
            "net_wins": wins - losses,
            "exact_two_sided_p": exact_binomial_two_sided(wins, losses),
        }
    source_actions = {
        action
        for row in episodes[AUTHENTIC]
        for action in row["source_action_counts"]
    }
    authentic_summary = summaries[AUTHENTIC]
    actual_task_sets = {
        condition: {row["task_id"] for row in rows}
        for condition, rows in episodes.items()
    }
    actual_task_ids = actual_task_sets[RAW]
    gates = {
        "complete_matched_task_matrix": all(
            len(rows) == len(task_ids) for rows in episodes.values()
        ),
        "matched_actual_task_identity": all(
            ids == actual_task_ids for ids in actual_task_sets.values()
        ),
        "matched_initial_state_hashes": all(
            len({
                row["records"][0]["before_state_sha256"]
                for rows in episodes.values()
                for row in rows
                if row["task_id"] == actual_task_id and row["records"]
            }) == 1
            for actual_task_id in actual_task_ids
        ),
        "target_neural_grounder_adaptation_gate_passed": True,
        "all_three_source_actions_exercised": source_actions
        == {BACKTRACK, COMMIT, EXPLORE},
        "nontrivial_action_changes": authentic_summary[
            "changed_actions_vs_raw"
        ] >= int(config["gates"]["minimum_changed_actions"]),
        "success_gain_over_raw_target": authentic_summary["successes"]
        > summaries[RAW]["successes"],
        "zero_negative_transfer_vs_raw": paired[RAW]["losses"] == 0,
        "strictly_beats_destructive_controls": all(
            authentic_summary["successes"] > summaries[name]["successes"]
            for name in (PERMUTED, LEDGER_BLIND)
        ),
        "matches_isomorphic_target_search_ceiling": (
            authentic_summary["successes"] == summaries[CEILING]["successes"]
        ),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "alfworld-search-automaton-transfer-v16",
        "status": (
            "CONSUMED_DEVELOPMENT_TRANSFER_GATE_PASSED"
            if passed else "CONSUMED_DEVELOPMENT_TRANSFER_GATE_FAILED"
        ),
        "claim_boundary": str(config["claim_boundary"]),
        "source_artifact_sha256": source.artifact_sha256,
        "target_grounder_kind": target_artifact["target_grounder"]["kind"],
        "task_ids": list(task_ids),
        "mapping": {
            "EXPLORE_UNTRIED": "execute_highest_ranked_untried_target_neural_action",
            "BACKTRACK_REPLAN": "reject_no-effect_action_within_target_scope",
            "COMMIT_VERIFY": "accept_target_official_terminal_success",
        },
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "episodes": episodes,
    }
    report = body | {"report_sha256": stable_hash(body)}
    output = REPO / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "summaries": summaries,
        "paired": paired,
        "gates": gates,
        "report_sha256": report["report_sha256"],
        "output": str(output.resolve()),
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
