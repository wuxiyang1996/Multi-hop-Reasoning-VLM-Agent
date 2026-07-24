#!/usr/bin/env python3
"""Compare skill-on/off source-policy runs under exactly matched initial states.

This report intentionally does not infer game semantics from action strings.  Action
validity is established only against the native action list recorded by the wrapper;
the wrapper identity is copied from the collection manifest for interpretation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
from typing import Any

from motif_transfer.instrumented_import import ImportedSourceEpisode, import_native_source_batch
from motif_transfer.phase1_assets import read_jsonl


CONDITIONS = ("authentic_skill_loaded", "skill_disabled")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _episode_summary(
    episode: ImportedSourceEpisode,
    *,
    requested_seed: int | None,
) -> dict[str, Any]:
    records = episode.records
    skill_ids = tuple(record.selected_skill_id for record in records)
    switches = sum(left != right for left, right in zip(skill_ids, skill_ids[1:]))
    native_membership = tuple(record.action in record.before.native_actions for record in records)
    return {
        "episode_id": episode.episode_id,
        "requested_seed": requested_seed,
        "initial_state_hash": records[0].transition.before_hash if records else None,
        "steps": len(records),
        "total_reward": episode.total_reward,
        "official_success": episode.official_success,
        "import_gaps": list(episode.gaps),
        "selected_skill_id_counts": dict(sorted(Counter(
            skill_id or "NONE" for skill_id in skill_ids
        ).items())),
        "selected_skill_switches": switches,
        "action_origin_counts": dict(sorted(Counter(
            record.action_origin for record in records
        ).items())),
        "all_actions_in_recorded_native_interface": all(native_membership),
        "action_samples": [record.action for record in records[:5]],
    }


def _load_game(
    root: Path,
    condition: str,
    game: str,
) -> tuple[dict[str, Any], list[ImportedSourceEpisode], dict[str, int]]:
    evidence = root / condition / game / "evidence"
    if not evidence.is_dir():
        raise FileNotFoundError(f"missing evidence directory: {evidence}")
    manifest = _read_json(evidence / "manifest.json")
    reset_events = [
        event for event in read_jsonl(evidence / "events.jsonl")
        if event.get("kind") == "RESET"
    ]
    reset_fingerprints = {
        json.dumps(
            (event.get("payload") or {}).get("environment_fingerprint") or {},
            sort_keys=True,
        )
        for event in reset_events
    }
    requested_seed_by_episode: dict[str, int] = {}
    for event in reset_events:
        episode_id = str(event.get("episode_id", ""))
        requested_seed = (event.get("payload") or {}).get("requested_seed")
        if episode_id and isinstance(requested_seed, int):
            requested_seed_by_episode[episode_id] = requested_seed
    manifest["_observed_reset_fingerprints"] = sorted(reset_fingerprints)
    episodes = list(import_native_source_batch(evidence))
    return manifest, episodes, requested_seed_by_episode


def _wrapper_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    serialized = manifest.get("_observed_reset_fingerprints") or []
    fingerprint = json.loads(serialized[0]) if len(serialized) == 1 else {}
    return {
        "game": fingerprint.get("game", manifest.get("game")),
        "wrapper_class": fingerprint.get("wrapper_class"),
        "wrapper_module": fingerprint.get("wrapper_module"),
        "one_consistent_reset_fingerprint": len(serialized) == 1,
    }


def _checkpoint_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest.get("metadata") or {}
    receipt = metadata.get("checkpoint_receipt") or {}
    checkpoint_metadata = receipt.get("metadata") or {}
    return {
        "path": receipt.get("path"),
        "game": checkpoint_metadata.get("game"),
        "best": checkpoint_metadata.get("best"),
        "historical_mean_reward": checkpoint_metadata.get("mean_reward"),
        "original_step": checkpoint_metadata.get("original_step"),
        "files_sha256": receipt.get("files_sha256"),
    }


def _pair_by_initial_state(
    skill_on: list[ImportedSourceEpisode],
    skill_off: list[ImportedSourceEpisode],
    skill_on_seeds: dict[str, int],
    skill_off_seeds: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    off_by_identity: dict[tuple[int, str], deque[ImportedSourceEpisode]] = defaultdict(deque)
    for episode in skill_off:
        seed = skill_off_seeds.get(episode.episode_id)
        if episode.records and seed is not None:
            identity = (seed, episode.records[0].transition.before_hash)
            off_by_identity[identity].append(episode)

    pairs: list[dict[str, Any]] = []
    unmatched_on: list[str] = []
    for on_episode in skill_on:
        if not on_episode.records:
            unmatched_on.append(on_episode.episode_id)
            continue
        seed = skill_on_seeds.get(on_episode.episode_id)
        if seed is None:
            unmatched_on.append(on_episode.episode_id)
            continue
        initial_hash = on_episode.records[0].transition.before_hash
        identity = (seed, initial_hash)
        if not off_by_identity[identity]:
            unmatched_on.append(on_episode.episode_id)
            continue
        off_episode = off_by_identity[identity].popleft()
        on_summary = _episode_summary(on_episode, requested_seed=seed)
        off_summary = _episode_summary(off_episode, requested_seed=seed)
        pairs.append({
            "requested_seed": seed,
            "initial_state_hash": initial_hash,
            "skill_on": on_summary,
            "skill_off": off_summary,
            "reward_delta_skill_on_minus_off": (
                on_episode.total_reward - off_episode.total_reward
            ),
        })

    unmatched_off = [
        episode.episode_id
        for queue in off_by_identity.values()
        for episode in queue
    ]
    return pairs, {
        "skill_on_episode_ids": sorted(unmatched_on),
        "skill_off_episode_ids": sorted(unmatched_off),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--games", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    games = args.games or sorted(
        path.name for path in (root / CONDITIONS[0]).iterdir() if path.is_dir()
    )

    game_reports: dict[str, Any] = {}
    for game in games:
        on_manifest, on_episodes, on_seeds = _load_game(root, CONDITIONS[0], game)
        off_manifest, off_episodes, off_seeds = _load_game(root, CONDITIONS[1], game)
        pairs, unmatched = _pair_by_initial_state(
            on_episodes, off_episodes, on_seeds, off_seeds,
        )
        deltas = [pair["reward_delta_skill_on_minus_off"] for pair in pairs]
        all_episodes = on_episodes + off_episodes
        game_reports[game] = {
            "skill_on_wrapper": _wrapper_identity(on_manifest),
            "skill_off_wrapper": _wrapper_identity(off_manifest),
            "wrapper_identity_matches": _wrapper_identity(on_manifest) == _wrapper_identity(off_manifest),
            "skill_on_checkpoint": _checkpoint_identity(on_manifest),
            "skill_off_checkpoint": _checkpoint_identity(off_manifest),
            "checkpoint_identity_matches": (
                _checkpoint_identity(on_manifest) == _checkpoint_identity(off_manifest)
            ),
            "skill_on_episode_count": len(on_episodes),
            "skill_off_episode_count": len(off_episodes),
            "matched_pair_count": len(pairs),
            "unmatched": unmatched,
            "all_import_gaps_empty": all(not episode.gaps for episode in all_episodes),
            "all_actions_in_recorded_native_interface": all(
                record.action in record.before.native_actions
                for episode in all_episodes
                for record in episode.records
            ),
            "mean_reward_skill_on": (
                sum(episode.total_reward for episode in on_episodes) / len(on_episodes)
                if on_episodes else None
            ),
            "mean_reward_skill_off": (
                sum(episode.total_reward for episode in off_episodes) / len(off_episodes)
                if off_episodes else None
            ),
            "mean_paired_reward_delta_skill_on_minus_off": (
                sum(deltas) / len(deltas) if deltas else None
            ),
            "positive_zero_negative_delta_counts": dict(sorted(Counter(
                "POSITIVE" if delta > 0 else "NEGATIVE" if delta < 0 else "ZERO"
                for delta in deltas
            ).items())),
            "pairs": pairs,
        }

    report = {
        "schema_version": 1,
        "root": str(root),
        "pairing_rule": "EXACT_REQUESTED_SEED_AND_INITIAL_SOURCE_TRANSITION_BEFORE_HASH_V2",
        "interpretation_limit": (
            "This paired pilot separates skill-context treatment from the loaded action-policy LoRA. "
            "It does not by itself establish cross-domain motif transfer."
        ),
        "games": game_reports,
        "gates": {
            "all_requested_games_present": set(game_reports) == set(games),
            "all_initial_states_paired": all(
                not report["unmatched"]["skill_on_episode_ids"]
                and not report["unmatched"]["skill_off_episode_ids"]
                for report in game_reports.values()
            ),
            "all_import_gaps_empty": all(
                report["all_import_gaps_empty"] for report in game_reports.values()
            ),
            "all_actions_native": all(
                report["all_actions_in_recorded_native_interface"]
                for report in game_reports.values()
            ),
            "wrapper_identity_matches": all(
                report["wrapper_identity_matches"] for report in game_reports.values()
            ),
            "checkpoint_identity_matches": all(
                report["checkpoint_identity_matches"] for report in game_reports.values()
            ),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
