from __future__ import annotations

from collections import defaultdict
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .multihorizon_replay import (
    HORIZONS,
    SPLITS,
    TREATMENTS,
    choose_lineage_snapshots,
    file_hash,
    snapshot_id,
    stable_hash,
)


_GAME_STATE_MARKER = "Game state:"
_AVAILABLE_ACTIONS_MARKER = "Available actions (pick ONE by number):"
_RECENT_HEADER = "Recent actions and rewards:"
_SUBGOAL_RE = re.compile(r"^Subgoal:\s*(.*)$", re.MULTILINE)
_RECENT_RE = re.compile(r"^\s{2}(.+?)\s+->\s+reward\s+(-?\d+(?:\.\d+)?)\s*$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def episode_splits(episodes: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    episode_ids = sorted({str(row["episode_id"]) for row in episodes})
    if len(episode_ids) < 3:
        raise ValueError("at least three episodes are required for frozen splitting")
    return {
        episode_id: SPLITS[index % len(SPLITS)]
        for index, episode_id in enumerate(episode_ids)
    }


def _snapshot_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["episode_id"]),
        int(row["episode_seed"]),
        int(row["step"]),
    )


def _shared_field(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = {stable_hash(row.get(field)) for row in rows}
    if len(values) != 1:
        raise ValueError(f"snapshot field differs across treatments: {field}")
    return rows[0].get(field)


def validate_matched_snapshot(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_steps: int,
    maximum_horizon: int = max(HORIZONS),
) -> None:
    if len(rows) != len(TREATMENTS):
        raise ValueError("snapshot does not contain exactly four treatment rows")
    by_treatment = {str(row.get("treatment")): row for row in rows}
    if set(by_treatment) != set(TREATMENTS):
        raise ValueError("snapshot treatment identities are incomplete")
    for field in (
        "episode_id",
        "episode_seed",
        "step",
        "source_skill_id",
        "prefix_actions",
        "before_observable_sha256",
        "native_actions",
        "native_actions_sha256",
    ):
        _shared_field(rows, field)
    step = int(rows[0]["step"])
    if step + maximum_horizon > maximum_steps:
        raise ValueError("snapshot cannot expose the frozen h8 endpoint")
    for treatment, row in by_treatment.items():
        if row.get("replay_status") != "INTERVENTION_OBSERVED":
            raise ValueError(f"{treatment} one-step replay is not observed")
        if bool(row.get("parser_fallback")):
            raise ValueError(f"{treatment} first action used parser fallback")
        if str(row.get("parsed_action")) not in {
            str(item) for item in row.get("native_actions", ())
        }:
            raise ValueError(f"{treatment} first action is not native")
        if stable_hash(str(row.get("prompt", ""))) != row.get("prompt_sha256"):
            raise ValueError(f"{treatment} prompt hash mismatch")
        if stable_hash(str(row.get("raw_response", ""))) != row.get(
            "raw_response_sha256"
        ):
            raise ValueError(f"{treatment} raw response hash mismatch")
        expected_adapter = None if treatment == "B" else "action_taking"
        if row.get("requested_adapter") != expected_adapter:
            raise ValueError(f"{treatment} requested adapter mismatch")
        if row.get("used_adapter") != expected_adapter:
            raise ValueError(f"{treatment} used adapter mismatch")


def prompt_parts(prompt: str) -> dict[str, Any]:
    position = prompt.find(_GAME_STATE_MARKER)
    if position < 0:
        raise ValueError("matched prompt lacks Game state marker")
    prefix = prompt[:position].rstrip()
    suffix = prompt[position + len(_GAME_STATE_MARKER):]
    subgoal_match = _SUBGOAL_RE.search(suffix)
    if subgoal_match is None:
        raise ValueError("matched prompt lacks Subgoal line")
    subgoal = subgoal_match.group(1).strip()
    after_subgoal = suffix[subgoal_match.end():]
    available_position = after_subgoal.find(_AVAILABLE_ACTIONS_MARKER)
    if available_position < 0:
        raise ValueError("matched prompt lacks Available actions marker")
    between = after_subgoal[:available_position]
    recent_position = between.find(_RECENT_HEADER)
    static_context = (
        between[:recent_position] if recent_position >= 0 else between
    ).strip()
    recent: list[tuple[str, float]] = []
    if recent_position >= 0:
        recent_block = between[recent_position:].splitlines()[1:]
        for line in recent_block:
            match = _RECENT_RE.match(line)
            if match is not None:
                recent.append((match.group(1), float(match.group(2))))
    return {
        "prefix": prefix,
        "subgoal": subgoal,
        "static_context": static_context,
        "initial_recent": recent[-5:],
    }


def render_continuation_prompt(
    source_prompt: str,
    *,
    state_markup: str,
    native_actions: Sequence[str],
    branch_history: Sequence[tuple[str, float]],
) -> str:
    parts = prompt_parts(source_prompt)
    recent = [
        (str(action), float(reward))
        for action, reward in parts["initial_recent"] + list(branch_history)
    ][-5:]
    recent_text = ""
    if recent:
        lines = [_RECENT_HEADER]
        for action, reward in recent:
            lines.append(f"  {action} -> reward {reward:.1f}")
        if len(recent) >= 3 and sum(reward for _, reward in recent) == 0:
            lines.append("WARNING: Recent actions got 0 reward. Try a DIFFERENT action!")
        recent_text = "\n".join(lines) + "\n\n"
    static_context = (
        str(parts["static_context"]).rstrip() + "\n"
        if parts["static_context"] else ""
    )
    numbered_actions = "\n".join(
        f"  {index + 1}. {action}"
        for index, action in enumerate(native_actions)
    )
    return (
        f"{parts['prefix']}\n\n{_GAME_STATE_MARKER}\n\n{state_markup}\n\n"
        f"Subgoal: {parts['subgoal']}\n{static_context}{recent_text}"
        f"{_AVAILABLE_ACTIONS_MARKER}\n{numbered_actions}\n\n"
        "Brief REASONING (1 sentence max) then ACTION: <number>."
    )


def build_multihorizon_plan(
    evidence_dir: Path,
    *,
    config_path: Path,
    maximum_per_split: int,
    included_splits: Sequence[str] = ("qualification", "held_out"),
) -> dict[str, Any]:
    if maximum_per_split <= 0:
        raise ValueError("maximum_per_split must be positive")
    if not included_splits or any(split not in SPLITS for split in included_splits):
        raise ValueError("included_splits must use frozen split names")
    manifest_path = evidence_dir / "manifest.json"
    episodes_path = evidence_dir / "episodes.jsonl"
    records_path = evidence_dir / "matched_policy_records.jsonl"
    for path in (manifest_path, episodes_path, records_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata") or {}
    maximum_steps = int(metadata["max_steps"])
    episodes = read_jsonl(episodes_path)
    records = read_jsonl(records_path)
    split_by_episode = episode_splits(episodes)
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_snapshot_key(row)].append(row)
    valid_records: list[dict[str, Any]] = []
    for snapshot_rows in grouped.values():
        try:
            validate_matched_snapshot(snapshot_rows, maximum_steps=maximum_steps)
        except ValueError:
            continue
        split = split_by_episode[str(snapshot_rows[0]["episode_id"])]
        valid_records.extend(dict(row) | {"split": split} for row in snapshot_rows)
    split_by_seed = {
        int(row["episode_seed"]): str(row["split"])
        for row in valid_records
    }
    chosen = choose_lineage_snapshots(
        valid_records,
        split_by_seed,
        maximum_per_split=maximum_per_split,
    )
    selected_ids = {
        snapshot for split in included_splits for snapshot in chosen[split]
    }
    snapshots = []
    for key, snapshot_rows in sorted(grouped.items()):
        if snapshot_id(snapshot_rows[0]) not in selected_ids:
            continue
        validate_matched_snapshot(snapshot_rows, maximum_steps=maximum_steps)
        split = split_by_episode[key[0]]
        by_treatment = {
            str(row["treatment"]): row for row in snapshot_rows
        }
        snapshots.append({
            "snapshot_id": snapshot_id(snapshot_rows[0]),
            "episode_id": key[0],
            "episode_seed": key[1],
            "fork_step": key[2],
            "split": split,
            "source_skill_id": _shared_field(snapshot_rows, "source_skill_id"),
            "prefix_actions": _shared_field(snapshot_rows, "prefix_actions"),
            "expected_fork_observable_hash": _shared_field(
                snapshot_rows, "before_observable_sha256"
            ),
            "treatments": {
                treatment: {
                    "prompt": str(row["prompt"]),
                    "prompt_sha256": str(row["prompt_sha256"]),
                    "raw_response_sha256": str(row["raw_response_sha256"]),
                    "parsed_action": str(row["parsed_action"]),
                    "requested_adapter": row.get("requested_adapter"),
                    "used_adapter": row.get("used_adapter"),
                    "context_skill_id": row.get("context_skill_id"),
                }
                for treatment, row in sorted(by_treatment.items())
            },
        })
    selected_counts = {
        split: sum(row["split"] == split for row in snapshots)
        for split in included_splits
    }
    if any(selected_counts[split] == 0 for split in included_splits):
        raise ValueError(f"no eligible snapshot in a requested split: {selected_counts}")
    body = {
        "schema_version": 1,
        "protocol_status": "FROZEN_BEFORE_MULTIHORIZON_OUTCOMES",
        "claim_boundary": "MECHANISM_SMOKE_NOT_CONFIRMATORY",
        "selection_authority": (
            "LINEAGE_ONLY_NO_ACTION_REWARD_SKILL_TEXT_OR_AFTER_STATE"
        ),
        "game": str(metadata.get("game")),
        "maximum_steps": maximum_steps,
        "maximum_horizon": max(HORIZONS),
        "included_splits": list(included_splits),
        "maximum_per_split": maximum_per_split,
        "selected_counts": selected_counts,
        "input_receipts": {
            "config": str(config_path.resolve()),
            "config_sha256": file_hash(config_path),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": file_hash(manifest_path),
            "episodes": str(episodes_path.resolve()),
            "episodes_sha256": file_hash(episodes_path),
            "matched_policy_records": str(records_path.resolve()),
            "matched_policy_records_sha256": file_hash(records_path),
        },
        "snapshots": snapshots,
    }
    return body | {"plan_sha256": stable_hash(body)}


def validate_plan(plan: Mapping[str, Any]) -> None:
    body = dict(plan)
    claimed = body.pop("plan_sha256", None)
    if claimed != stable_hash(body):
        raise ValueError("multihorizon plan hash mismatch")
    if plan.get("protocol_status") != "FROZEN_BEFORE_MULTIHORIZON_OUTCOMES":
        raise ValueError("multihorizon plan is not frozen")
    for snapshot in plan.get("snapshots", ()):
        treatments = snapshot.get("treatments") or {}
        if set(treatments) != set(TREATMENTS):
            raise ValueError("plan snapshot treatment identities are incomplete")
        for treatment, row in treatments.items():
            if stable_hash(str(row.get("prompt", ""))) != row.get("prompt_sha256"):
                raise ValueError(f"plan prompt hash mismatch for {treatment}")


__all__ = [
    "read_jsonl",
    "episode_splits",
    "validate_matched_snapshot",
    "prompt_parts",
    "render_continuation_prompt",
    "build_multihorizon_plan",
    "validate_plan",
]
