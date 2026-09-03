"""Receipt-grounded discovery and evaluation for event-level source controllers.

The module deliberately ignores historical skill names and descriptions.  It
uses only native source transitions to (1) derive pre-action event roles, (2)
induce a PERSIST/SWITCH branch map from discovery episodes, and (3) freeze
outcome-blind qualification/held-out replay snapshots.

The observational induction is candidate generation, not causal evidence.
Only matched multi-horizon replay can upgrade the resulting controller.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .instrumented_import import ImportedSourceEpisode, import_native_source_batch
from .multihorizon_replay import HORIZONS, cumulative_returns, file_hash, stable_hash
from .multihorizon_runner import ForkState, PolicyHistoryStep
from .phase1_assets import read_jsonl


SPLITS = ("discovery", "qualification", "held_out")
MODES = ("COMMON_HASH_CONTINUATION", "FULL_TREATMENT_REGIME")
TREATMENTS = (
    "EVENT_CONTROLLER",
    "SHUFFLED_EVENT_CONTROLLER",
    "ALWAYS_PERSIST",
    "ALWAYS_SWITCH",
    "HASH_RANDOM",
)


class ReceiptEvent(str, Enum):
    PROGRESS = "PROGRESS"
    FAILURE = "FAILURE"
    AFFORDANCE_CHANGE = "AFFORDANCE_CHANGE"
    STALL = "STALL"
    UNKNOWN = "UNKNOWN"


class ControlBranch(str, Enum):
    PERSIST = "PERSIST"
    SWITCH = "SWITCH"


@dataclass(frozen=True)
class MicroDecisionPoint:
    point_id: str
    game: str
    episode_id: str
    episode_seed: int
    split: str
    step: int
    event: ReceiptEvent
    previous_action: str
    native_actions: tuple[str, ...]
    prefix_actions: tuple[str, ...]
    prefix_rewards: tuple[float, ...]
    expected_fork_observable_hash: str
    source_history_receipt_ids: tuple[str, ...]
    observed_branch: ControlBranch
    future_returns: Mapping[str, float]
    future_positive: Mapping[str, bool]

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError("unsupported source split")
        if self.event == ReceiptEvent.UNKNOWN:
            raise ValueError("UNKNOWN points are not replay candidates")
        if self.step != len(self.prefix_actions):
            raise ValueError("point step and prefix length differ")
        if len(self.prefix_actions) != len(self.prefix_rewards):
            raise ValueError("prefix action/reward lengths differ")
        if not self.prefix_actions or self.previous_action != self.prefix_actions[-1]:
            raise ValueError("point lacks a valid previous action")
        if self.previous_action not in self.native_actions:
            raise ValueError("PERSIST action is not native at the fork")
        if len(set(self.native_actions) - {self.previous_action}) == 0:
            raise ValueError("point has no native SWITCH alternative")
        body = asdict(self)
        expected = body.pop("point_id")
        body["event"] = self.event.value
        body["observed_branch"] = self.observed_branch.value
        if expected != stable_hash(body):
            raise ValueError("micro decision point hash mismatch")


def split_by_episode_id(episode_ids: Sequence[str]) -> dict[str, str]:
    unique = sorted(set(episode_ids))
    if len(unique) < 3:
        raise ValueError("micro-controller discovery requires at least three episodes")
    return {
        episode_id: SPLITS[index % len(SPLITS)]
        for index, episode_id in enumerate(unique)
    }


def classify_receipt_event(
    rewards: Sequence[float],
    affordance_changes: Sequence[bool],
    terminals: Sequence[bool],
    *,
    stall_window: int,
) -> ReceiptEvent:
    """Classify the next decision from receipt history only."""

    if stall_window < 1:
        raise ValueError("stall_window must be positive")
    if not (
        len(rewards) == len(affordance_changes) == len(terminals)
    ):
        raise ValueError("receipt history fields must have equal length")
    if not rewards or terminals[-1]:
        return ReceiptEvent.UNKNOWN
    if rewards[-1] < 0:
        return ReceiptEvent.FAILURE
    if rewards[-1] > 0:
        return ReceiptEvent.PROGRESS
    if affordance_changes[-1]:
        return ReceiptEvent.AFFORDANCE_CHANGE
    if len(rewards) >= stall_window and all(
        reward == 0 and not terminal and not affordance_change
        for reward, terminal, affordance_change in zip(
            rewards[-stall_window:],
            terminals[-stall_window:],
            affordance_changes[-stall_window:],
        )
    ):
        return ReceiptEvent.STALL
    return ReceiptEvent.UNKNOWN


def _episode_seeds(evidence_dir: Path) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for row in read_jsonl(evidence_dir / "events.jsonl"):
        if row.get("kind") != "RESET":
            continue
        seed = (row.get("payload") or {}).get("requested_seed")
        if not isinstance(seed, int):
            raise ValueError("source RESET lacks an integer requested_seed")
        episode_id = str(row.get("episode_id"))
        if episode_id in seeds:
            raise ValueError("source episode has multiple RESET events")
        seeds[episode_id] = seed
    return seeds


def _event_history(records, end: int) -> tuple[list[float], list[bool], list[bool]]:
    previous = records[:end]
    return (
        [float(row.reward) for row in previous],
        [set(row.before.native_actions) != set(row.after.native_actions) for row in previous],
        [bool(row.after.terminal) for row in previous],
    )


def _future_outcome(records, start: int) -> tuple[dict[str, float], dict[str, bool]]:
    rewards = [float(row.reward) for row in records[start:start + max(HORIZONS)]]
    returns = cumulative_returns(rewards)
    positives = {
        f"h{horizon}": any(value > 0 for value in rewards[:horizon])
        for horizon in HORIZONS
    }
    return returns, positives


def extract_micro_decision_points(
    episodes: Sequence[ImportedSourceEpisode],
    episode_seeds: Mapping[str, int],
    *,
    stall_window: int = 2,
    maximum_horizon: int = max(HORIZONS),
    maximum_steps: int | None = None,
) -> tuple[MicroDecisionPoint, ...]:
    """Extract pre-action points without reading skill identity or prose."""

    if maximum_horizon != max(HORIZONS):
        raise ValueError("v1 freezes h8 as the maximum horizon")
    split_map = split_by_episode_id([episode.episode_id for episode in episodes])
    points: list[MicroDecisionPoint] = []
    for episode in sorted(episodes, key=lambda item: item.episode_id):
        if episode.gaps:
            continue
        records = tuple(sorted(episode.records, key=lambda item: item.step))
        if not records:
            continue
        steps = [row.step for row in records]
        if steps != list(range(len(records))):
            continue
        seed = episode_seeds.get(episode.episode_id)
        if seed is None:
            continue
        limit = maximum_steps if maximum_steps is not None else len(records)
        for index, record in enumerate(records):
            if index == 0 or index + maximum_horizon > limit:
                continue
            rewards, affordance_changes, terminals = _event_history(records, index)
            event = classify_receipt_event(
                rewards, affordance_changes, terminals,
                stall_window=stall_window,
            )
            if event == ReceiptEvent.UNKNOWN:
                continue
            previous_action = records[index - 1].action
            native_actions = tuple(record.before.native_actions)
            if previous_action not in native_actions:
                continue
            if not set(native_actions) - {previous_action}:
                continue
            future_returns, future_positive = _future_outcome(records, index)
            observed_branch = (
                ControlBranch.PERSIST
                if record.action == previous_action
                else ControlBranch.SWITCH
            )
            body: dict[str, Any] = {
                "game": episode.game,
                "episode_id": episode.episode_id,
                "episode_seed": seed,
                "split": split_map[episode.episode_id],
                "step": record.step,
                "event": event.value,
                "previous_action": previous_action,
                "native_actions": native_actions,
                "prefix_actions": tuple(row.action for row in records[:index]),
                "prefix_rewards": tuple(float(row.reward) for row in records[:index]),
                "expected_fork_observable_hash": stable_hash(
                    record.before.state.get("observable_state", "")
                ),
                "source_history_receipt_ids": tuple(
                    row.transition.receipt_id
                    for row in records[max(0, index - stall_window):index]
                ),
                "observed_branch": observed_branch.value,
                "future_returns": future_returns,
                "future_positive": future_positive,
            }
            point = MicroDecisionPoint(
                stable_hash(body),
                **{
                    **body,
                    "event": event,
                    "observed_branch": observed_branch,
                },
            )
            point.validate()
            points.append(point)
    return tuple(points)


def summarize_observational_points(
    points: Sequence[MicroDecisionPoint],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        result[split] = {}
        for event in ReceiptEvent:
            if event == ReceiptEvent.UNKNOWN:
                continue
            event_rows = [
                row for row in points
                if row.split == split and row.event == event
            ]
            if not event_rows:
                continue
            by_branch = {}
            for branch in ControlBranch:
                branch_rows = [row for row in event_rows if row.observed_branch == branch]
                by_branch[branch.value] = {
                    "n": len(branch_rows),
                    "games": len({row.game for row in branch_rows}),
                    "future_positive_rate": {
                        f"h{horizon}": (
                            sum(bool(row.future_positive[f"h{horizon}"]) for row in branch_rows)
                            / len(branch_rows)
                            if branch_rows else None
                        )
                        for horizon in HORIZONS
                    },
                    "mean_future_return": {
                        f"h{horizon}": (
                            sum(float(row.future_returns[f"h{horizon}"]) for row in branch_rows)
                            / len(branch_rows)
                            if branch_rows else None
                        )
                        for horizon in HORIZONS
                    },
                }
            result[split][event.value] = by_branch
    return result


def induce_discovery_branch_map(
    points: Sequence[MicroDecisionPoint],
    *,
    min_branch_support: int,
    min_games_per_branch: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Choose branches from discovery-only, game-balanced h8 positive rates."""

    mapping: dict[str, str] = {}
    audit: dict[str, Any] = {}
    for event in ReceiptEvent:
        if event == ReceiptEvent.UNKNOWN:
            continue
        rows = [
            row for row in points
            if row.split == "discovery" and row.event == event
        ]
        branch_scores: dict[str, float | None] = {}
        branch_support: dict[str, Any] = {}
        eligible = True
        for branch in ControlBranch:
            branch_rows = [row for row in rows if row.observed_branch == branch]
            by_game: dict[str, list[MicroDecisionPoint]] = defaultdict(list)
            for row in branch_rows:
                by_game[row.game].append(row)
            game_rates = {
                game: sum(bool(row.future_positive["h8"]) for row in game_rows)
                / len(game_rows)
                for game, game_rows in sorted(by_game.items())
            }
            branch_support[branch.value] = {
                "n": len(branch_rows),
                "games": len(by_game),
                "game_h8_positive_rates": game_rates,
            }
            if (
                len(branch_rows) < min_branch_support
                or len(by_game) < min_games_per_branch
            ):
                branch_scores[branch.value] = None
                eligible = False
            else:
                branch_scores[branch.value] = sum(game_rates.values()) / len(game_rates)
        selected: str | None = None
        if eligible:
            persist = branch_scores[ControlBranch.PERSIST.value]
            switch = branch_scores[ControlBranch.SWITCH.value]
            if persist is not None and switch is not None and persist != switch:
                selected = (
                    ControlBranch.PERSIST.value
                    if persist > switch else ControlBranch.SWITCH.value
                )
                mapping[event.value] = selected
        audit[event.value] = {
            "selection": selected,
            "game_balanced_h8_positive_rate": branch_scores,
            "support": branch_support,
            "selection_authority": "DISCOVERY_OBSERVATIONAL_RECEIPTS_ONLY",
        }
    return mapping, audit


def _switch_action(point: MicroDecisionPoint) -> str:
    alternatives = [
        action for action in point.native_actions
        if action != point.previous_action
    ]
    alternatives.sort(key=lambda action: stable_hash({
        "point_id": point.point_id,
        "action": action,
        "rule": "HASH_NATIVE_ALTERNATIVE_V1",
    }))
    return alternatives[0]


def build_source_microcontroller_plan(
    evidence: Sequence[tuple[str, str | Path]],
    *,
    config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported source micro-controller config")
    stall_window = int(config.get("stall_window", 2))
    maximum_per_cell = int(config.get("maximum_per_game_event_split", 1))
    if maximum_per_cell < 1:
        raise ValueError("maximum_per_game_event_split must be positive")
    all_points: list[MicroDecisionPoint] = []
    inputs = []
    requested_games = set()
    for expected_game, raw_path in evidence:
        evidence_dir = Path(raw_path).resolve()
        manifest_path = evidence_dir / "manifest.json"
        events_path = evidence_dir / "events.jsonl"
        episodes_path = evidence_dir / "episodes.jsonl"
        for path in (manifest_path, events_path, episodes_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.get("metadata") or {}
        game = str(metadata.get("game"))
        if game != expected_game or game in requested_games:
            raise ValueError("evidence game identity is missing, duplicated, or mismatched")
        requested_games.add(game)
        maximum_steps = int(metadata.get("max_steps", 0))
        episodes = import_native_source_batch(evidence_dir, include_base_replays=False)
        points = extract_micro_decision_points(
            episodes,
            _episode_seeds(evidence_dir),
            stall_window=stall_window,
            maximum_steps=maximum_steps,
        )
        all_points.extend(points)
        inputs.append({
            "game": game,
            "evidence_dir": str(evidence_dir),
            "maximum_steps": maximum_steps,
            "manifest_sha256": file_hash(manifest_path),
            "events_sha256": file_hash(events_path),
            "episodes_sha256": file_hash(episodes_path),
            "point_count": len(points),
        })
    branch_map, induction_audit = induce_discovery_branch_map(
        all_points,
        min_branch_support=int(config.get("min_branch_support", 3)),
        min_games_per_branch=int(config.get("min_games_per_branch", 2)),
    )
    requested_events = {
        str(value) for value in config.get(
            "included_events", ["PROGRESS", "STALL"]
        )
    }
    replayable_games = {
        str(value) for value in config.get(
            "replayable_games", sorted(requested_games)
        )
    }
    if not replayable_games or not replayable_games <= requested_games:
        raise ValueError("replayable_games must be a non-empty evidence subset")
    eligible_events = requested_events & set(branch_map)
    snapshots = []
    for game in sorted(replayable_games):
        for split in ("qualification", "held_out"):
            for event in sorted(eligible_events):
                candidates = [
                    point for point in all_points
                    if point.game == game
                    and point.split == split
                    and point.event.value == event
                ]
                # Selection is deliberately independent of current/future outcome.
                candidates.sort(key=lambda point: stable_hash({
                    "point_id": point.point_id,
                    "rule": "OUTCOME_BLIND_POINT_HASH_V1",
                }))
                for point in candidates[:maximum_per_cell]:
                    snapshots.append({
                        "snapshot_id": point.point_id,
                        "game": point.game,
                        "episode_id": point.episode_id,
                        "episode_seed": point.episode_seed,
                        "split": point.split,
                        "fork_step": point.step,
                        "event": point.event.value,
                        "prefix_actions": list(point.prefix_actions),
                        "prefix_rewards": list(point.prefix_rewards),
                        "expected_fork_observable_hash": point.expected_fork_observable_hash,
                        "previous_action": point.previous_action,
                        "switch_action": _switch_action(point),
                        "source_history_receipt_ids": list(
                            point.source_history_receipt_ids
                        ),
                    })
    plan_body = {
        "schema_version": 1,
        "protocol_status": "FROZEN_BEFORE_QUALIFICATION_OR_HELDOUT_REPLAY",
        "claim_boundary": (
            "Discovery observations propose an event-conditioned controller. "
            "Only matched qualification and held-out multi-horizon replay can "
            "support source value; target transfer is not tested here."
        ),
        "selection_authority": (
            "DISCOVERY_OUTCOMES_FOR_BRANCH_INDUCTION;BLIND_RECEIPT_HISTORY_AND_"
            "LINEAGE_ONLY_FOR_QUALIFICATION_HELDOUT_SNAPSHOT_SELECTION"
        ),
        "skill_semantics_used": False,
        "config": str(config_path),
        "config_sha256": file_hash(config_path),
        "inputs": inputs,
        "stall_window": stall_window,
        "horizons": list(HORIZONS),
        "modes": list(MODES),
        "treatments": list(TREATMENTS),
        "branch_map": branch_map,
        "induction_audit": induction_audit,
        "replayable_games": sorted(replayable_games),
        "snapshots": snapshots,
        "selected_counts": {
            split: sum(row["split"] == split for row in snapshots)
            for split in ("qualification", "held_out")
        },
    }
    plan = plan_body | {"plan_sha256": stable_hash(plan_body)}
    observational_report = {
        "schema_version": 1,
        "claim_boundary": "OBSERVATIONAL_CANDIDATE_GENERATION_ONLY_NOT_CAUSAL",
        "skill_semantics_used": False,
        "point_count": len(all_points),
        "game_count": len(requested_games),
        "split_event_branch_stats": summarize_observational_points(all_points),
        "branch_map": branch_map,
        "induction_audit": induction_audit,
        "plan_sha256": plan["plan_sha256"],
    }
    return plan, observational_report


def validate_source_microcontroller_plan(plan: Mapping[str, Any]) -> None:
    body = dict(plan)
    claimed = body.pop("plan_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("source micro-controller plan hash mismatch")
    if plan.get("protocol_status") != "FROZEN_BEFORE_QUALIFICATION_OR_HELDOUT_REPLAY":
        raise ValueError("source micro-controller plan is not frozen")
    if plan.get("skill_semantics_used") is not False:
        raise ValueError("source micro-controller plan used skill semantics")
    if tuple(plan.get("treatments", ())) != TREATMENTS:
        raise ValueError("source micro-controller treatments changed")
    if tuple(plan.get("modes", ())) != MODES:
        raise ValueError("source micro-controller modes changed")
    branch_map = plan.get("branch_map") or {}
    if not branch_map:
        raise ValueError("no discovery-supported event branch was induced")
    for event, branch in branch_map.items():
        ReceiptEvent(event)
        ControlBranch(branch)
    seen = set()
    for snapshot in plan.get("snapshots", ()):
        snapshot_id = str(snapshot.get("snapshot_id"))
        if snapshot_id in seen:
            raise ValueError("duplicate source micro-controller snapshot")
        seen.add(snapshot_id)
        if snapshot.get("split") not in {"qualification", "held_out"}:
            raise ValueError("replay plan contains a discovery snapshot")
        if snapshot.get("event") not in branch_map:
            raise ValueError("snapshot event lacks a frozen controller branch")
        if snapshot.get("game") not in set(plan.get("replayable_games", ())):
            raise ValueError("snapshot game is not replayable under the frozen plan")
        prefix = snapshot.get("prefix_actions") or []
        rewards = snapshot.get("prefix_rewards") or []
        if len(prefix) != len(rewards) or len(prefix) != int(snapshot["fork_step"]):
            raise ValueError("snapshot prefix is inconsistent")
        if not snapshot.get("source_history_receipt_ids"):
            raise ValueError("snapshot lacks source receipt lineage")


def _live_event(
    prefix_rewards: Sequence[float],
    history: Sequence[PolicyHistoryStep],
    *,
    stall_window: int,
) -> ReceiptEvent:
    rewards = [float(value) for value in prefix_rewards] + [
        float(row.reward) for row in history
    ]
    # Live fork receipts currently expose reward/terminal but not the next
    # native set in PolicyHistoryStep, so affordance changes are UNKNOWN here.
    changes = [False] * len(rewards)
    terminals = [False] * len(prefix_rewards) + [
        bool(row.official_success) for row in history
    ]
    return classify_receipt_event(
        rewards, changes, terminals, stall_window=stall_window,
    )


def _hash_action(
    state: ForkState,
    *,
    snapshot_id: str,
    decision_index: int,
    exclude: str | None = None,
) -> str:
    choices = [action for action in state.admissible_actions if action != exclude]
    if not choices:
        choices = list(state.admissible_actions)
    if not choices:
        return "__NO_NATIVE_ACTION__"
    choices.sort(key=lambda action: stable_hash({
        "snapshot_id": snapshot_id,
        "decision_index": decision_index,
        "state_hash": state.receipt_hash,
        "action": action,
        "rule": "HASH_NATIVE_ACTION_V1",
    }))
    return choices[0]


def choose_microcontroller_action(
    state: ForkState,
    *,
    snapshot: Mapping[str, Any],
    branch_map: Mapping[str, str],
    treatment: str,
    decision_index: int,
    history: Sequence[PolicyHistoryStep],
    stall_window: int,
) -> tuple[str, dict[str, Any]]:
    if treatment not in TREATMENTS:
        raise ValueError("unknown source micro-controller treatment")
    prior_actions = [str(value) for value in snapshot["prefix_actions"]] + [
        row.action for row in history
    ]
    previous_action = prior_actions[-1] if prior_actions else None
    event = _live_event(
        snapshot["prefix_rewards"], history, stall_window=stall_window,
    )
    branch: str | None = None
    if treatment == "EVENT_CONTROLLER":
        branch = branch_map.get(event.value)
    elif treatment == "SHUFFLED_EVENT_CONTROLLER":
        authentic = branch_map.get(event.value)
        if authentic is not None:
            branch = (
                ControlBranch.SWITCH.value
                if authentic == ControlBranch.PERSIST.value
                else ControlBranch.PERSIST.value
            )
    elif treatment == "ALWAYS_PERSIST":
        branch = ControlBranch.PERSIST.value
    elif treatment == "ALWAYS_SWITCH":
        branch = ControlBranch.SWITCH.value

    fallback = False
    if branch == ControlBranch.PERSIST.value and previous_action in state.admissible_actions:
        action = str(previous_action)
    elif branch == ControlBranch.SWITCH.value:
        frozen_switch = str(snapshot.get("switch_action", ""))
        if (
            decision_index == 0
            and frozen_switch in state.admissible_actions
            and frozen_switch != previous_action
        ):
            action = frozen_switch
        else:
            action = _hash_action(
                state,
                snapshot_id=str(snapshot["snapshot_id"]),
                decision_index=decision_index,
                exclude=previous_action,
            )
    else:
        fallback = treatment != "HASH_RANDOM"
        action = _hash_action(
            state,
            snapshot_id=str(snapshot["snapshot_id"]),
            decision_index=decision_index,
        )
    return action, {
        "event": event.value,
        "branch": branch,
        "fallback_to_hash": fallback,
        "authority": "SYMBOLIC_ROUTE_OVER_RECEIPT_EVENT",
    }


def run_microcontroller_snapshot(
    environment_factory,
    *,
    snapshot: Mapping[str, Any],
    branch_map: Mapping[str, str],
    stall_window: int,
) -> tuple[dict[str, Any], ...]:
    """Execute matched deterministic treatments from one verified fork."""

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for treatment in TREATMENTS:
            env = environment_factory()
            try:
                state = env.reset(seed=int(snapshot["episode_seed"]))
                prefix_state_hashes = [state.receipt_hash]
                replay_failed = False
                for action in snapshot["prefix_actions"]:
                    if state.terminal or action not in state.admissible_actions:
                        replay_failed = True
                        break
                    result = env.step(str(action))
                    state = result.state
                    prefix_state_hashes.append(state.receipt_hash)
                base = {
                    "snapshot_id": snapshot["snapshot_id"],
                    "game": snapshot["game"],
                    "episode_id": snapshot["episode_id"],
                    "episode_seed": snapshot["episode_seed"],
                    "fork_step": snapshot["fork_step"],
                    "split": snapshot["split"],
                    "event": snapshot["event"],
                    "mode": mode,
                    "treatment": treatment,
                    "prefix_actions": list(snapshot["prefix_actions"]),
                    "prefix_state_hashes": prefix_state_hashes,
                    "expected_fork_observable_hash": snapshot[
                        "expected_fork_observable_hash"
                    ],
                    "observed_fork_observable_hash": state.observable_hash,
                }
                if (
                    replay_failed
                    or state.observable_hash
                    != snapshot["expected_fork_observable_hash"]
                ):
                    rows.append(base | {
                        "status": "REPLAY_MISMATCH",
                        "actions": [],
                        "step_rewards": [],
                    })
                    continue
                actions: list[str] = []
                rewards: list[float] = []
                history: list[PolicyHistoryStep] = []
                step_receipts = []
                official_success: bool | None = None
                initial_event = _live_event(
                    snapshot["prefix_rewards"], (), stall_window=stall_window,
                )
                if initial_event.value != snapshot["event"]:
                    rows.append(base | {
                        "status": "EVENT_RECOMPUTE_MISMATCH",
                        "actions": [],
                        "step_rewards": [],
                        "recomputed_event": initial_event.value,
                    })
                    continue
                for decision_index in range(max(HORIZONS)):
                    if state.terminal or state.truncated:
                        break
                    active_treatment = (
                        treatment
                        if mode == "FULL_TREATMENT_REGIME" or decision_index == 0
                        else "HASH_RANDOM"
                    )
                    action, decision_metadata = choose_microcontroller_action(
                        state,
                        snapshot=snapshot,
                        branch_map=branch_map,
                        treatment=active_treatment,
                        decision_index=decision_index,
                        history=tuple(history),
                        stall_window=stall_window,
                    )
                    if action not in state.admissible_actions:
                        rows.append(base | {
                            "status": "POLICY_ACTION_INADMISSIBLE",
                            "actions": actions + [action],
                            "step_rewards": rewards,
                            "failed_decision_index": decision_index,
                        })
                        break
                    before_hash = state.receipt_hash
                    result = env.step(action)
                    state = result.state
                    actions.append(action)
                    rewards.append(float(result.reward))
                    official_success = (
                        bool(official_success) or bool(result.official_success)
                        if result.official_success is not None
                        else official_success
                    )
                    history_step = PolicyHistoryStep(
                        decision_index=decision_index,
                        treatment=active_treatment,
                        action=action,
                        reward=float(result.reward),
                        before_state_hash=before_hash,
                        after_state_hash=state.receipt_hash,
                        official_value=result.official_value,
                        official_success=result.official_success,
                    )
                    history.append(history_step)
                    receipt_body = {
                        "decision_index": decision_index,
                        "active_treatment": active_treatment,
                        "action": action,
                        "decision_metadata": decision_metadata,
                        "transition": asdict(history_step),
                        "terminal": state.terminal,
                        "truncated": state.truncated,
                        "environment_metadata": dict(result.metadata),
                    }
                    step_receipts.append(
                        receipt_body | {"receipt_sha256": stable_hash(receipt_body)}
                    )
                if rows and all(
                    rows[-1].get(key) == base[key]
                    for key in ("snapshot_id", "mode", "treatment")
                ) and rows[-1]["status"] == "POLICY_ACTION_INADMISSIBLE":
                    continue
                if not rewards:
                    # A terminal fork is not a valid pre-action decision point.
                    rows.append(base | {
                        "status": "NO_POST_FORK_ACTION",
                        "actions": actions,
                        "step_rewards": rewards,
                    })
                    continue
                rows.append(base | {
                    "status": "INTERVENTION_OBSERVED",
                    "actions": actions,
                    "step_rewards": rewards,
                    "cumulative_returns": cumulative_returns(rewards),
                    "first_positive_reward_step": next(
                        (index + 1 for index, value in enumerate(rewards) if value > 0),
                        None,
                    ),
                    "official_success": official_success,
                    "observed_horizon": len(rewards),
                    "final_state_hash": state.receipt_hash,
                    "step_receipts": step_receipts,
                })
            finally:
                env.close()
    return tuple(rows)


def analyze_microcontroller_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = [row for row in rows if row.get("status") == "INTERVENTION_OBSERVED"]
    snapshot_keys = {
        (str(row["snapshot_id"]), str(row["mode"])) for row in rows
    }
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observed:
        grouped[(str(row["snapshot_id"]), str(row["mode"]))].append(row)
    complete = {}
    invalid_cells = []
    for key in sorted(snapshot_keys):
        by_treatment: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in grouped.get(key, ()):
            by_treatment[str(row["treatment"])].append(row)
        if set(by_treatment) == set(TREATMENTS) and all(
            len(by_treatment[treatment]) == 1 for treatment in TREATMENTS
        ):
            complete[key] = {
                treatment: by_treatment[treatment][0]
                for treatment in TREATMENTS
            }
        else:
            invalid_cells.append({
                "snapshot_id": key[0],
                "mode": key[1],
                "treatment_counts": {
                    treatment: len(by_treatment.get(treatment, ()))
                    for treatment in TREATMENTS
                },
            })

    stats: dict[str, Any] = {}
    for split in ("qualification", "held_out"):
        stats[split] = {}
        for mode in MODES:
            cells = [
                cell for cell in complete.values()
                if next(iter(cell.values()))["split"] == split
                and next(iter(cell.values()))["mode"] == mode
            ]
            horizons = {}
            for horizon in HORIZONS:
                field = f"h{horizon}"
                means = {
                    treatment: (
                        sum(float(cell[treatment]["cumulative_returns"][field]) for cell in cells)
                        / len(cells)
                        if cells else None
                    )
                    for treatment in TREATMENTS
                }
                progress_rates = {
                    treatment: (
                        sum(
                            any(
                                float(value) > 0
                                for value in cell[treatment]["step_rewards"][:horizon]
                            )
                            for cell in cells
                        ) / len(cells)
                        if cells else None
                    )
                    for treatment in TREATMENTS
                }
                controller = means["EVENT_CONTROLLER"]
                deltas = {
                    control: (
                        sum(
                            float(cell["EVENT_CONTROLLER"]["cumulative_returns"][field])
                            - float(cell[control]["cumulative_returns"][field])
                            for cell in cells
                        ) / len(cells)
                        if cells else None
                    )
                    for control in TREATMENTS if control != "EVENT_CONTROLLER"
                }
                progress_deltas = {
                    control: (
                        progress_rates["EVENT_CONTROLLER"] - progress_rates[control]
                        if progress_rates["EVENT_CONTROLLER"] is not None
                        else None
                    )
                    for control in TREATMENTS if control != "EVENT_CONTROLLER"
                }
                horizons[field] = {
                    "mean_return": means,
                    "controller_paired_mean_delta": deltas,
                    "progress_rate": progress_rates,
                    "controller_progress_rate_delta": progress_deltas,
                    "best_static_mean": (
                        max(means["ALWAYS_PERSIST"], means["ALWAYS_SWITCH"])
                        if controller is not None else None
                    ),
                }
            stats[split][mode] = {
                "complete_snapshots": len(cells),
                "horizons": horizons,
            }

    game_event_h8 = []
    for split in ("qualification", "held_out"):
        for mode in MODES:
            for game in sorted({
                str(next(iter(cell.values()))["game"])
                for cell in complete.values()
            }):
                for event in sorted({
                    str(next(iter(cell.values()))["event"])
                    for cell in complete.values()
                }):
                    cells = [
                        cell for cell in complete.values()
                        if next(iter(cell.values()))["split"] == split
                        and next(iter(cell.values()))["mode"] == mode
                        and next(iter(cell.values()))["game"] == game
                        and next(iter(cell.values()))["event"] == event
                    ]
                    if not cells:
                        continue
                    means = {
                        treatment: sum(
                            float(cell[treatment]["cumulative_returns"]["h8"])
                            for cell in cells
                        ) / len(cells)
                        for treatment in TREATMENTS
                    }
                    progress = {
                        treatment: sum(
                            any(float(value) > 0 for value in cell[treatment]["step_rewards"][:8])
                            for cell in cells
                        ) / len(cells)
                        for treatment in TREATMENTS
                    }
                    game_event_h8.append({
                        "split": split,
                        "mode": mode,
                        "game": game,
                        "event": event,
                        "complete_snapshots": len(cells),
                        "mean_return": means,
                        "progress_rate": progress,
                    })

    def positive(split: str, mode: str, control: str) -> bool:
        cell = stats[split][mode]
        if cell["complete_snapshots"] == 0:
            return False
        h8 = cell["horizons"]["h8"]
        controller = h8["mean_return"]["EVENT_CONTROLLER"]
        if control == "BEST_STATIC":
            baseline = h8["best_static_mean"]
            return controller is not None and baseline is not None and controller > baseline
        delta = h8["controller_paired_mean_delta"][control]
        return delta is not None and delta > 0

    gates = {
        "BLIND_CELLS_COMPLETE": not invalid_cells,
    }
    for split in ("qualification", "held_out"):
        for mode in MODES:
            prefix = f"{split}_{mode}_H8".upper()
            gates[f"{prefix}_GT_SHUFFLED"] = positive(
                split, mode, "SHUFFLED_EVENT_CONTROLLER"
            )
            gates[f"{prefix}_GT_BEST_STATIC"] = positive(
                split, mode, "BEST_STATIC"
            )
            gates[f"{prefix}_GT_HASH_RANDOM"] = positive(
                split, mode, "HASH_RANDOM"
            )
    gates["SOURCE_MICROCONTROLLER_SUPPORTED"] = all(gates.values())
    return {
        "schema_version": 1,
        "claim_boundary": (
            "The h8 source gate requires the discovery-induced event controller "
            "to beat shuffled topology, the best static PERSIST/SWITCH policy, "
            "and hash-random under both estimands in qualification and held-out."
        ),
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "selected_snapshots": len({str(row["snapshot_id"]) for row in rows}),
        "complete_cells": len(complete),
        "invalid_cells": invalid_cells,
        "split_stats": stats,
        "posthoc_scale_robustness_diagnostic": {
            "authority": "DIAGNOSTIC_ONLY_PRIMARY_GATE_REMAINS_RAW_PAIRED_H8",
            "game_event_h8": game_event_h8,
        },
        "gates": gates,
    }


__all__ = [
    "ControlBranch", "MicroDecisionPoint", "MODES", "ReceiptEvent", "SPLITS",
    "TREATMENTS", "analyze_microcontroller_rows", "build_source_microcontroller_plan",
    "choose_microcontroller_action", "classify_receipt_event",
    "extract_micro_decision_points", "induce_discovery_branch_map",
    "run_microcontroller_snapshot", "summarize_observational_points",
    "validate_source_microcontroller_plan",
]
