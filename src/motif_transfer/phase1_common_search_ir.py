"""Intervention-grounded common search IR for the six Phase-1 games.

Each source state supplies outcome-blind native-action candidates.  Candidates
are executed from the exact same replayed fork and then followed by one frozen,
state-dependent continuation policy.  A candidate set is eligible only when
duplicate executions are byte-stable and exactly one candidate has the best
multi-step official return.  The resulting candidate ledger contains no native
action token and can be consumed by the existing three-edge search-automaton
source gate.

This module does not read target data and does not itself establish transfer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import stable_hash
from .instrumented_import import import_native_source_batch
from .real_source_interventions import (
    SourceSnapshot,
    content_hash,
    split_seeds,
    validate_plan,
)
from .sokoban_search_automaton_v16 import (
    ACTIONS,
    EVENTS,
    matched_decision_rows,
    summarize_source_gate,
)
from .skill_internal import build_execution_sets


ROW_SCHEMA = "PHASE1_MATCHED_OPTION_FORK_V1"
RECEIPT_SCHEMA = "PHASE1_COMMON_SEARCH_LEDGER_V1"
SPLIT_MAP = {
    "development": "discovery",
    "qualification": "calibration",
    "heldout": "fresh",
}


def canonical_policy_sha256(policy: Mapping[str, str]) -> str:
    """Hash only the alpha-stable routing contract, never source lineage."""

    return stable_hash({
        "events": list(EVENTS),
        "actions": list(ACTIONS),
        "policy": {event: str(policy[event]) for event in EVENTS},
    })


def _repeat_to_horizon(actions: Sequence[str], horizon: int) -> tuple[str, ...]:
    if not actions:
        raise ValueError("option template is empty")
    return tuple(str(actions[index % len(actions)]) for index in range(horizon))


def build_discovery_option_template_artifact(
    evidence_dir: str | Path,
    *,
    game: str,
    horizon: int,
) -> dict[str, Any]:
    """Freeze one rewarded discovery execution plus structural corruptions."""

    evidence_dir = Path(evidence_dir).resolve()
    episodes = import_native_source_batch(evidence_dir)
    execution_sets = build_execution_sets(game, episodes)
    records = {
        row.transition.receipt_id: row
        for episode in episodes for row in episode.records
    }
    candidates: list[dict[str, Any]] = []
    for execution_set in execution_sets:
        for execution in execution_set.executions:
            if execution.split != "discovery":
                continue
            execution_rows = [
                records[receipt_id]
                for receipt_id in execution.transition_receipt_ids
            ]
            official_return = float(sum(row.reward for row in execution_rows))
            if official_return <= 0:
                continue
            actions = tuple(row.action for row in execution_rows)
            candidates.append({
                "skill_id": execution.skill_id,
                "execution_id": execution.execution_id,
                "transition_receipt_ids": list(execution.transition_receipt_ids),
                "official_cumulative_return": official_return,
                "actions": list(actions),
            })
    if not candidates:
        raise ValueError("no positive discovery execution is available")
    candidates.sort(key=lambda row: (
        -float(row["official_cumulative_return"]),
        stable_hash(row),
    ))
    selected = candidates[0]
    authentic = _repeat_to_horizon(selected["actions"], horizon)
    unique_tokens = sorted(set(authentic))
    if len(unique_tokens) < 2:
        raise ValueError("authentic option has fewer than two native tokens")
    permutation = {
        token: unique_tokens[(index + 1) % len(unique_tokens)]
        for index, token in enumerate(unique_tokens)
    }
    rotated = authentic[horizon // 2:] + authentic[:horizon // 2]
    templates = (
        ("AUTHENTIC_DISCOVERY_EXECUTION", authentic),
        ("ORDER_REVERSED_CONTROL", tuple(reversed(authentic))),
        ("PHASE_ROTATED_CONTROL", rotated),
        ("ACTION_TOKEN_PERMUTED_CONTROL", tuple(
            permutation[action] for action in authentic
        )),
    )
    if len({actions for _name, actions in templates}) != len(templates):
        raise ValueError("template corruption families are not distinct")
    body = {
        "schema_version": "phase1-discovery-option-template-artifact-v1",
        "status": "FROZEN_BEFORE_FRESH_OPTION_FORKS",
        "game": game,
        "horizon": horizon,
        "selection": (
            "highest official cumulative return among discovery-only maximal "
            "skill executions; stable-hash tie break"
        ),
        "selected_discovery_execution": selected,
        "templates": [
            {
                "template_id": stable_hash({
                    "game": game,
                    "family": family,
                    "actions": list(actions),
                }),
                "family": family,
                "actions": list(actions),
            }
            for family, actions in templates
        ],
        "claim_boundary": (
            "SOURCE_NATIVE_DISCOVERY_ACTIONS_ONLY;TARGET_UNREAD;TEMPLATES_"
            "NEVER_EXPORTED_IN_COMMON_IR"
        ),
    }
    return body | {"artifact_sha256": stable_hash(body)}


def build_discovery_primitive_template_artifact(
    evidence_dir: str | Path,
    *,
    game: str,
    horizon: int,
) -> dict[str, Any]:
    """Freeze all native action primitives observed in discovery executions.

    No reward or target observation participates in vocabulary selection.  The
    source-specific primitive is repeated to create a matched macro-option;
    only its later intervention verdict enters the common symbolic ledger.
    """

    evidence_dir = Path(evidence_dir).resolve()
    episodes = import_native_source_batch(evidence_dir)
    execution_sets = build_execution_sets(game, episodes)
    records = {
        row.transition.receipt_id: row
        for episode in episodes for row in episode.records
    }
    action_counts: Counter[str] = Counter()
    skill_ids: set[str] = set()
    discovery_execution_ids: set[str] = set()
    discovery_execution_lengths: list[int] = []
    for execution_set in execution_sets:
        for execution in execution_set.executions:
            if execution.split != "discovery":
                continue
            skill_ids.add(str(execution.skill_id))
            discovery_execution_ids.add(str(execution.execution_id))
            discovery_execution_lengths.append(
                len(execution.transition_receipt_ids)
            )
            for receipt_id in execution.transition_receipt_ids:
                action_counts[str(records[receipt_id].action)] += 1
    actions = sorted(
        action_counts,
        key=lambda action: (stable_hash({"game": game, "action": action}), action),
    )
    if len(actions) < 2:
        raise ValueError("fewer than two discovery-native action primitives")
    templates = []
    for action in actions:
        option_actions = [action] * horizon
        family = "DISCOVERY_NATIVE_PRIMITIVE"
        templates.append({
            "template_id": stable_hash({
                "game": game,
                "family": family,
                "actions": option_actions,
            }),
            "family": family,
            "actions": option_actions,
            "discovery_observation_count": action_counts[action],
        })
    body = {
        "schema_version": "phase1-discovery-primitive-template-artifact-v1",
        "status": "FROZEN_BEFORE_FRESH_OPTION_FORKS",
        "game": game,
        "horizon": horizon,
        "selection": (
            "all distinct native action tokens observed in discovery-only "
            "maximal skill executions; reward-blind stable-hash ordering"
        ),
        "discovery_execution_count": len(discovery_execution_ids),
        "maximum_discovery_execution_length": max(
            discovery_execution_lengths
        ),
        "discovery_skill_ids": sorted(skill_ids),
        "templates": templates,
        "claim_boundary": (
            "SOURCE_DISCOVERY_VOCABULARY_ONLY;OUTCOME_BLIND_CANDIDATE_"
            "SELECTION;TARGET_UNREAD;NATIVE_TOKENS_NEVER_EXPORTED_IN_COMMON_IR"
        ),
    }
    return body | {"artifact_sha256": stable_hash(body)}


def validate_option_template_artifact(
    artifact: Mapping[str, Any], *, game: str, horizon: int
) -> tuple[dict[str, Any], ...]:
    body = dict(artifact)
    claimed = body.pop("artifact_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("option-template artifact hash mismatch")
    if artifact.get("status") != "FROZEN_BEFORE_FRESH_OPTION_FORKS":
        raise ValueError("option-template artifact is not frozen")
    if artifact.get("game") != game or int(artifact.get("horizon", 0)) != horizon:
        raise ValueError("option-template artifact game/horizon mismatch")
    templates = tuple(dict(row) for row in artifact.get("templates") or [])
    if len(templates) < 4:
        raise ValueError("option-template artifact has too few candidates")
    for row in templates:
        actions = tuple(map(str, row.get("actions") or []))
        if len(actions) != horizon:
            raise ValueError("option template does not cover the frozen horizon")
        expected = stable_hash({
            "game": game,
            "family": str(row["family"]),
            "actions": list(actions),
        })
        if row.get("template_id") != expected:
            raise ValueError("option-template ID mismatch")
    return templates


def build_observed_prefix_plan(
    adapter_class: type,
    *,
    game: str,
    seeds: Sequence[int],
    namespace: str,
    max_steps: int,
    rollout_steps: int,
    snapshots_per_episode: int,
    actions_per_snapshot: int,
    minimum_step: int,
    runtime_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze hash-ranked snapshots from prefixes that were actually observed.

    Unlike the older live-plan helper, this routine does not assume that every
    environment survives to a preselected late step.  It records all available
    prefix states without reading rewards, then ranks only by seed and step.
    Natural termination can reduce the eligible prefix set but cannot influence
    the ranking among states that exist.
    """

    if not 0 <= minimum_step < rollout_steps <= max_steps:
        raise ValueError("invalid rollout interval")
    if snapshots_per_episode < 1 or actions_per_snapshot < 2:
        raise ValueError("invalid snapshot or candidate count")
    seed_splits = split_seeds(seeds, namespace=namespace)
    snapshots: list[SourceSnapshot] = []
    episode_audit: list[dict[str, Any]] = []
    for seed in sorted({int(value) for value in seeds}):
        adapter = adapter_class(game, max_steps)
        prefix: list[str] = []
        candidates: list[SourceSnapshot] = []
        try:
            adapter.reset(seed=seed)
            for step in range(rollout_steps):
                native = tuple(map(str, adapter.admissible_actions()))
                if len(native) < actions_per_snapshot:
                    raise RuntimeError(
                        f"seed {seed} step {step}: fewer native actions than candidates"
                    )
                rollout_action = min(
                    native,
                    key=lambda action: (
                        content_hash([
                            namespace, seed, step, "rollout", action
                        ]),
                        action,
                    ),
                )
                if step >= minimum_step:
                    alternatives = sorted(
                        (action for action in native if action != rollout_action),
                        key=lambda action: (
                            content_hash([
                                namespace, seed, step, "fork", action
                            ]),
                            action,
                        ),
                    )
                    selected_actions = tuple(
                        [rollout_action, *alternatives[: actions_per_snapshot - 1]]
                    )
                    candidates.append(SourceSnapshot(
                        snapshot_id=f"{game}.observed_hash_policy.{seed}.step_{step}",
                        split=seed_splits[seed],
                        condition="observed_reward_blind_hash_policy",
                        episode_id=f"{game}_observed_seed_{seed}",
                        seed=seed,
                        step=step,
                        max_steps=max_steps,
                        expected_fork_state_sha256=_observable_hash(
                            adapter.state_receipt()
                        ),
                        expected_native_actions_sha256=content_hash(list(native)),
                        prefix_actions=tuple(prefix),
                        selected_actions=selected_actions,
                        logged_action=rollout_action,
                        grounding_state=str(adapter.state_receipt()),
                    ))
                adapter.step(rollout_action)
                prefix.append(rollout_action)
                if adapter.last_terminated or adapter.last_truncated:
                    break
        finally:
            adapter.close()
        ranked = sorted(
            candidates,
            key=lambda snapshot: (
                content_hash([
                    namespace, seed, "observed-snapshot", snapshot.step
                ]),
                snapshot.step,
            ),
        )
        selected = ranked[:snapshots_per_episode]
        if len(selected) < snapshots_per_episode:
            raise RuntimeError(
                f"seed {seed}: only {len(selected)} observed eligible prefixes"
            )
        snapshots.extend(sorted(selected, key=lambda snapshot: snapshot.step))
        episode_audit.append({
            "seed": seed,
            "split": seed_splits[seed],
            "observed_steps": len(prefix),
            "eligible_prefixes": len(candidates),
            "selected_steps": sorted(snapshot.step for snapshot in selected),
            "terminated_or_truncated_before_rollout_limit": len(prefix) < rollout_steps,
        })
    plan_core: dict[str, Any] = {
        "schema_version": "real-source-intervention-plan-v1",
        "selection": {
            "namespace": namespace,
            "rule": (
                "observe reward-blind hash-policy prefixes; rank available "
                "snapshots by sha256(namespace, seed, step)"
            ),
            "trajectory_policy": (
                "lowest sha256(namespace, seed, step, rollout, native_action)"
            ),
            "snapshots_per_episode": snapshots_per_episode,
            "actions_per_snapshot": actions_per_snapshot,
            "minimum_step": minimum_step,
            "rollout_steps": rollout_steps,
            "content_or_outcome_used_for_selection": False,
            "reward_read_during_plan_collection": False,
            "terminal_status_used_only_to_stop_invalid_future_steps": True,
            "seed_splits": {
                str(key): value for key, value in sorted(seed_splits.items())
            },
        },
        "source": {
            "game": game,
            "collection_kind": "fresh_same_runtime_observed_prefix_reward_blind",
            "runtime_receipt": dict(runtime_receipt or {}),
        },
        "episode_audit": episode_audit,
        "snapshots": [
            {
                "snapshot_id": snapshot.snapshot_id,
                "split": snapshot.split,
                "condition": snapshot.condition,
                "episode_id": snapshot.episode_id,
                "seed": snapshot.seed,
                "step": snapshot.step,
                "max_steps": snapshot.max_steps,
                "expected_fork_state_sha256": snapshot.expected_fork_state_sha256,
                "expected_native_actions_sha256": snapshot.expected_native_actions_sha256,
                "prefix_actions": list(snapshot.prefix_actions),
                "selected_actions": list(snapshot.selected_actions),
                "logged_action": snapshot.logged_action,
                "grounding_state": snapshot.grounding_state,
            }
            for snapshot in snapshots
        ],
    }
    return plan_core | {"plan_sha256": content_hash(plan_core)}


def _observable_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _common_continuation_action(
    native_actions: Sequence[str],
    *,
    namespace: str,
    snapshot_id: str,
    decision_index: int,
    candidate_id: str | None = None,
) -> str:
    if not native_actions:
        raise RuntimeError("continuation state has no native action")
    return min(
        map(str, native_actions),
        key=lambda action: (
            content_hash([
                namespace,
                snapshot_id,
                "COMMON_STATE_DEPENDENT_CONTINUATION_V1",
                candidate_id,
                decision_index,
                action,
            ]),
            action,
        ),
    )


def execute_option_fork(
    adapter_class: type,
    *,
    game: str,
    snapshot: Mapping[str, Any],
    candidate_action: str,
    candidate_rank: int,
    repeat_index: int,
    horizon: int,
    namespace: str,
    continuation_mode: str = "common",
    option_actions: Sequence[str] | None = None,
    option_template_id: str | None = None,
) -> dict[str, Any]:
    """Replay one exact fork and evaluate one first-action intervention."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if continuation_mode not in {"common", "candidate_conditioned"}:
        raise ValueError("unsupported continuation mode")
    status = "INTERVENTION_OBSERVED"
    error: str | None = None
    adapter = None
    rewards: list[float] = []
    actions: list[str] = []
    state_hashes: list[str] = []
    transition_effects: list[dict[str, Any]] = []
    candidate_id = stable_hash({
        "snapshot_id": str(snapshot["snapshot_id"]),
        "candidate_rank": candidate_rank,
        "candidate_action": candidate_action,
        "option_template_id": option_template_id,
        "option_actions_sha256": (
            stable_hash(list(map(str, option_actions)))
            if option_actions is not None else None
        ),
    })
    try:
        adapter = adapter_class(game, int(snapshot["max_steps"]))
        adapter.reset(seed=int(snapshot["seed"]))
        for prefix_action in snapshot["prefix_actions"]:
            adapter.step(str(prefix_action))
            if adapter.last_terminated or adapter.last_truncated:
                raise RuntimeError("prefix terminated before frozen fork")
        observed_fork_hash = _observable_hash(adapter.state_receipt())
        observed_native = tuple(map(str, adapter.admissible_actions()))
        observed_native_hash = content_hash(list(observed_native))
        if observed_fork_hash != str(snapshot["expected_fork_state_sha256"]):
            raise RuntimeError("fork observable differs from frozen plan")
        if observed_native_hash != str(snapshot["expected_native_actions_sha256"]):
            raise RuntimeError("fork native action set differs from frozen plan")
        if candidate_action not in observed_native:
            raise RuntimeError("candidate action is not native at frozen fork")

        for decision_index in range(horizon):
            if adapter.last_terminated or adapter.last_truncated:
                break
            if option_actions is not None:
                action = str(option_actions[decision_index])
                if action not in tuple(map(str, adapter.admissible_actions())):
                    # Invalid candidate paths are real refutations, not replay
                    # failures. Preserve the observed prefix and stop.
                    break
            elif decision_index == 0:
                action = candidate_action
            else:
                action = _common_continuation_action(
                    tuple(map(str, adapter.admissible_actions())),
                    namespace=namespace,
                    snapshot_id=str(snapshot["snapshot_id"]),
                    decision_index=decision_index,
                    candidate_id=(
                        candidate_id
                        if continuation_mode == "candidate_conditioned"
                        else None
                    ),
                )
            before_hash = _observable_hash(adapter.state_receipt())
            adapter.step(action)
            after_hash = _observable_hash(adapter.state_receipt())
            actions.append(action)
            rewards.append(float(adapter.last_reward))
            transition_effects.append({
                # This is the explicit (state, effect, next_state) projection
                # consumed by Phase-3 typed-effect induction.  The native
                # action remains only in the source audit row and is never
                # exported in the induced symbolic program.
                "before_observable_sha256": before_hash,
                "effect": {
                    "official_reward": float(adapter.last_reward),
                    "terminated": bool(adapter.last_terminated),
                    "truncated": bool(adapter.last_truncated),
                },
                "after_observable_sha256": after_hash,
                "observable_changed": before_hash != after_hash,
            })
            state_hashes.append(stable_hash({
                "decision_index": decision_index,
                "before_observable_sha256": before_hash,
                "action": action,
                "after_observable_sha256": after_hash,
                "reward": float(adapter.last_reward),
                "terminated": bool(adapter.last_terminated),
                "truncated": bool(adapter.last_truncated),
            }))
        final_observable_sha256 = _observable_hash(adapter.state_receipt())
    except Exception as exc:
        status = "INTERVENTION_FAILED"
        error = f"{type(exc).__name__}: {exc}"
        observed_fork_hash = None
        observed_native_hash = None
        final_observable_sha256 = None
    finally:
        if adapter is not None:
            adapter.close()

    cumulative: dict[str, float] = {}
    for endpoint in sorted({1, 2, 4, 8, 16, 32, horizon}):
        if endpoint <= horizon:
            cumulative[f"h{endpoint}"] = float(sum(rewards[:endpoint]))
    body = {
        "schema_version": ROW_SCHEMA,
        "status": status,
        "game": game,
        "snapshot_id": str(snapshot["snapshot_id"]),
        "source_split": str(snapshot["split"]),
        "candidate_id": candidate_id,
        "candidate_rank": int(candidate_rank),
        "repeat_index": int(repeat_index),
        "candidate_action": candidate_action,
        "option_template_id": option_template_id,
        "horizon": int(horizon),
        "continuation_mode": continuation_mode,
        "observed_actions": len(actions),
        "actions": actions,
        "step_rewards": rewards,
        "cumulative_returns": cumulative,
        "transition_hashes": state_hashes,
        "transition_effects": transition_effects,
        "expected_fork_observable_sha256": str(
            snapshot["expected_fork_state_sha256"]
        ),
        "observed_fork_observable_sha256": observed_fork_hash,
        "observed_native_actions_sha256": observed_native_hash,
        "final_observable_sha256": final_observable_sha256,
        "terminated": bool(getattr(adapter, "last_terminated", False)),
        "truncated": bool(getattr(adapter, "last_truncated", False)),
        "error": error,
        "claim_boundary": (
            "SOURCE_NATIVE_ACTION_TOKEN_RETAINED_FOR_CAUSAL_AUDIT_ONLY_"
            "NEVER_EXPORTED_IN_COMMON_IR"
        ),
    }
    return body | {"row_sha256": stable_hash(body)}


def collect_option_forks(
    plan: Mapping[str, Any],
    *,
    adapter_class: type,
    game: str,
    horizon: int,
    repeats: int,
    namespace: str,
    workers: int,
    continuation_mode: str = "common",
    option_templates: Sequence[Mapping[str, Any]] | None = None,
    snapshot_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect every frozen candidate with duplicate matched executions."""

    validate_plan(plan)
    if str((plan.get("source") or {}).get("game")) != game:
        raise ValueError("plan/source game mismatch")
    if repeats < 2:
        raise ValueError("at least two repeats are required for stability")
    if workers < 1:
        raise ValueError("workers must be positive")
    templates = tuple(dict(row) for row in (option_templates or ()))
    jobs = []
    for snapshot in plan.get("snapshots") or []:
        if snapshot_ids is not None and str(snapshot["snapshot_id"]) not in snapshot_ids:
            continue
        if templates:
            ranked_templates = sorted(
                templates,
                key=lambda row: stable_hash({
                    "snapshot_id": str(snapshot["snapshot_id"]),
                    "template_id": str(row["template_id"]),
                    "rule": "OUTCOME_BLIND_TEMPLATE_ORDER_V1",
                }),
            )
            candidates = [
                (rank, str(row["actions"][0]), tuple(map(str, row["actions"])),
                 str(row["template_id"]))
                for rank, row in enumerate(ranked_templates)
            ]
        else:
            candidates = [
                (rank, str(action), None, None)
                for rank, action in enumerate(snapshot["selected_actions"])
            ]
        for candidate_rank, action, option_actions, template_id in candidates:
            for repeat_index in range(repeats):
                jobs.append((
                    snapshot, action, candidate_rank, repeat_index,
                    option_actions, template_id,
                ))

    def execute(
        job: tuple[
            Mapping[str, Any], str, int, int, tuple[str, ...] | None, str | None
        ]
    ) -> dict[str, Any]:
        snapshot, action, rank, repeat, option_actions, template_id = job
        return execute_option_fork(
            adapter_class,
            game=game,
            snapshot=snapshot,
            candidate_action=action,
            candidate_rank=rank,
            repeat_index=repeat,
            horizon=horizon,
            namespace=namespace,
            continuation_mode=continuation_mode,
            option_actions=option_actions,
            option_template_id=template_id,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(execute, jobs))


def option_rows_to_ledgers(
    rows: Iterable[Mapping[str, Any]],
    *,
    primary_horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove native actions and create only stable, unique-best ledgers."""

    key = f"h{primary_horizon}"
    by_snapshot: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_snapshot[str(row["snapshot_id"])].append(row)

    ledgers: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    split_totals: Counter[str] = Counter()
    split_eligible: Counter[str] = Counter()
    verified_ranks: Counter[int] = Counter()
    for snapshot_id, snapshot_rows in sorted(by_snapshot.items()):
        source_splits = {str(row["source_split"]) for row in snapshot_rows}
        if len(source_splits) != 1:
            raise ValueError("snapshot spans multiple source splits")
        source_split = next(iter(source_splits))
        split_totals[source_split] += 1
        by_candidate: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in snapshot_rows:
            by_candidate[int(row["candidate_rank"])].append(row)
        expected_ranks = list(range(len(by_candidate)))
        if sorted(by_candidate) != expected_ranks:
            exclusions["INCOMPLETE_CANDIDATE_RANKS"] += 1
            continue
        if any(
            any(str(row["status"]) != "INTERVENTION_OBSERVED" for row in repeats)
            for repeats in by_candidate.values()
        ):
            exclusions["INTERVENTION_FAILURE"] += 1
            continue
        if any(len(repeats) < 2 for repeats in by_candidate.values()):
            exclusions["INSUFFICIENT_REPEATS"] += 1
            continue

        candidate_rows: list[Mapping[str, Any]] = []
        unstable = False
        for candidate_rank in expected_ranks:
            repeats = by_candidate[candidate_rank]
            signatures = {
                (
                    float((row.get("cumulative_returns") or {})[key]),
                    str(row.get("final_observable_sha256")),
                    tuple(row.get("transition_hashes") or []),
                )
                for row in repeats
            }
            if len(signatures) != 1:
                unstable = True
                break
            candidate_rows.append(repeats[0])
        if unstable:
            exclusions["NONDETERMINISTIC_OPTION_OUTCOME"] += 1
            continue

        returns = [
            float((row.get("cumulative_returns") or {})[key])
            for row in candidate_rows
        ]
        maximum = max(returns)
        verified = [index for index, value in enumerate(returns) if value == maximum]
        if len(verified) != 1 or maximum <= min(returns):
            exclusions["NO_UNIQUE_VALUED_CANDIDATE"] += 1
            continue
        verified_rank = verified[0]
        verified_ranks[verified_rank] += 1
        split_eligible[source_split] += 1
        ledger_body = {
            "schema_version": RECEIPT_SCHEMA,
            "snapshot_id": snapshot_id,
            "source_split": source_split,
            "automaton_split": SPLIT_MAP[source_split],
            "candidate_count": len(candidate_rows),
            "verified_candidate_rank": verified_rank,
            "verification_authority": (
                f"UNIQUE_BEST_MATCHED_OFFICIAL_CUMULATIVE_RETURN_{key.upper()}"
            ),
            "return_gap": maximum - sorted(returns)[-2],
            "attempts": [
                {
                    "candidate_receipt_id": str(row["candidate_id"]),
                    "verified": index == verified_rank,
                    "refuted": index != verified_rank,
                    "observed_actions": int(row["observed_actions"]),
                    "transition_hashes": list(row["transition_hashes"]),
                    # Official values stay in source audit receipts.  Target
                    # adapters receive only the canonical routing artifact.
                    "official_cumulative_return": returns[index],
                }
                for index, row in enumerate(candidate_rows)
            ],
        }
        ledgers.append(
            ledger_body | {"receipt_sha256": stable_hash(ledger_body)}
        )

    audit = {
        "snapshots": len(by_snapshot),
        "eligible_ledgers": len(ledgers),
        "exclusions": dict(sorted(exclusions.items())),
        "split_totals": dict(sorted(split_totals.items())),
        "split_eligible": dict(sorted(split_eligible.items())),
        "split_eligible_fraction": {
            split: split_eligible[split] / total if total else 0.0
            for split, total in sorted(split_totals.items())
        },
        "verified_candidate_rank_counts": {
            str(rank): count for rank, count in sorted(verified_ranks.items())
        },
        "native_action_tokens_exported_to_ir": False,
    }
    return ledgers, audit


def analyze_common_search_ir(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_horizon: int,
    source_gate_requirements: Mapping[str, Any],
    minimum_eligible_fraction_each_split: float,
    expected_policy_sha256: str | None = None,
    maximum_intervention_failed_rows: int | None = None,
) -> dict[str, Any]:
    failed_rows = sum(
        str(row.get("status")) != "INTERVENTION_OBSERVED" for row in rows
    )
    infrastructure_gate = {
        "intervention_failed_rows": failed_rows,
        "maximum_intervention_failed_rows": maximum_intervention_failed_rows,
        "passed": (
            maximum_intervention_failed_rows is None
            or failed_rows <= maximum_intervention_failed_rows
        ),
    }
    ledgers, ledger_audit = option_rows_to_ledgers(
        rows, primary_horizon=primary_horizon
    )
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledgers:
        by_split[str(row["automaton_split"])].append(row)
    required = {"discovery", "calibration", "fresh"}
    if set(by_split) != required:
        return {
            "status": "SOURCE_GATE_FAILED_INCOMPLETE_ELIGIBLE_SPLITS",
            "source_gate_passed": False,
            "ledger_audit": ledger_audit,
            "infrastructure_gate": infrastructure_gate,
        }
    branch_support = {
        split: dict(Counter(
            str(row["event"])
            for row in matched_decision_rows(split_ledgers)
        ))
        for split, split_ledgers in sorted(by_split.items())
    }
    if any(
        branch_support[split].get(event, 0) == 0
        for split in required for event in EVENTS
    ):
        return {
            "schema_version": "phase1-common-search-ir-source-report-v1",
            "status": "SOURCE_GATE_FAILED_INCOMPLETE_BRANCH_SUPPORT",
            "source_gate_passed": False,
            "primary_horizon": primary_horizon,
            "ledger_audit": ledger_audit,
            "branch_support": branch_support,
            "infrastructure_gate": infrastructure_gate,
            "expected_policy_sha256": expected_policy_sha256,
            "claim_boundary": (
                "SOURCE_ONLY_MATCHED_MULTI_STEP_OFFICIAL_RETURN;NO_TARGET_"
                "DATA_READ;NO_SOURCE_NATIVE_ACTION_TOKEN_EXPORTED"
            ),
        }
    gate = summarize_source_gate(
        discovery_receipts=by_split["discovery"],
        calibration_receipts=by_split["calibration"],
        fresh_receipts=by_split["fresh"],
        requirements=source_gate_requirements,
    )
    coverage_gate = all(
        fraction >= minimum_eligible_fraction_each_split
        for fraction in ledger_audit["split_eligible_fraction"].values()
    )
    policy_hash = canonical_policy_sha256(gate["learned_policy"])
    equivalence_gate = (
        expected_policy_sha256 is None or policy_hash == expected_policy_sha256
    )
    all_gates = bool(
        gate["source_gate_passed"]
        and coverage_gate
        and equivalence_gate
        and infrastructure_gate["passed"]
    )
    return {
        "schema_version": "phase1-common-search-ir-source-report-v1",
        "status": (
            "SOURCE_COMMON_SEARCH_IR_GATE_PASSED"
            if all_gates else "SOURCE_COMMON_SEARCH_IR_GATE_FAILED"
        ),
        "source_gate_passed": all_gates,
        "primary_horizon": primary_horizon,
        "ledger_audit": ledger_audit,
        "automaton_gate": gate,
        "branch_support": branch_support,
        "coverage_gate": {
            "minimum_eligible_fraction_each_split": (
                minimum_eligible_fraction_each_split
            ),
            "passed": coverage_gate,
        },
        "infrastructure_gate": infrastructure_gate,
        "canonical_policy_sha256": policy_hash,
        "expected_policy_sha256": expected_policy_sha256,
        "canonical_policy_equivalence_passed": equivalence_gate,
        "claim_boundary": (
            "SOURCE_ONLY_MATCHED_MULTI_STEP_OFFICIAL_RETURN;NO_TARGET_DATA_READ;"
            "NO_SOURCE_NATIVE_ACTION_TOKEN_EXPORTED"
        ),
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    return rows


__all__ = [
    "analyze_common_search_ir",
    "build_observed_prefix_plan",
    "build_discovery_option_template_artifact",
    "build_discovery_primitive_template_artifact",
    "canonical_policy_sha256",
    "collect_option_forks",
    "execute_option_fork",
    "option_rows_to_ledgers",
    "read_jsonl",
    "validate_option_template_artifact",
    "write_jsonl",
]
