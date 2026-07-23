from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLITS = ("discovery", "qualification", "heldout")
HORIZONS = (1, 2, 4, 8)


def _stable_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_discovery_candidate(
    events: Sequence[Mapping[str, Any]],
    *,
    events_sha256: str,
) -> dict[str, Any]:
    seeds_by_episode = {
        str(row["episode_id"]): int(row["payload"]["requested_seed"])
        for row in events if row.get("kind") == "RESET"
    }
    seeds = sorted(set(seeds_by_episode.values()))
    split_by_seed = {seed: SPLITS[index % 3] for index, seed in enumerate(seeds)}
    discovery_episodes = {
        episode_id for episode_id, seed in seeds_by_episode.items()
        if split_by_seed[seed] == "discovery"
    }
    selected: dict[str, dict[int, str]] = defaultdict(dict)
    rewards: dict[str, dict[int, float]] = defaultdict(dict)
    discovery_event_hashes = []
    for row in events:
        episode_id = str(row["episode_id"])
        if episode_id not in discovery_episodes:
            continue
        if row.get("event_sha256"):
            discovery_event_hashes.append(str(row["event_sha256"]))
        payload = row.get("payload") or {}
        if row.get("kind") == "AGENT_PROPOSAL_SET" and payload.get("selected_skill_id"):
            selected[episode_id][int(payload["step"])] = str(payload["selected_skill_id"])
        elif row.get("kind") == "ENVIRONMENT_STEP":
            rewards[episode_id][int(payload["step"])] = float(payload["reward"])

    skill_ids = sorted(
        {skill_id for episode in selected.values() for skill_id in episode.values()},
        key=_stable_hash,
    )
    candidates = []
    for skill_id in skill_ids:
        episode_coverage = 0
        selected_steps = 0
        continuous_edges = 0
        support = {horizon: 0 for horizon in HORIZONS}
        reward_episode_coverage_h8 = 0
        for episode_id in sorted(discovery_episodes):
            episode_steps = selected.get(episode_id, {})
            occurrences = sorted(
                step for step, value in episode_steps.items() if value == skill_id
            )
            if not occurrences:
                continue
            episode_coverage += 1
            selected_steps += len(occurrences)
            continuous_edges += sum(
                right == left + 1 for left, right in zip(occurrences, occurrences[1:])
            )
            positive_steps = {
                step for step, reward in rewards.get(episode_id, {}).items() if reward > 0
            }
            per_episode_h8 = False
            for step in occurrences:
                for horizon in HORIZONS:
                    observed = any(
                        step <= reward_step < step + horizon
                        for reward_step in positive_steps
                    )
                    support[horizon] += observed
                    if horizon == 8:
                        per_episode_h8 = per_episode_h8 or observed
            reward_episode_coverage_h8 += per_episode_h8
        eligible = (
            episode_coverage >= 2
            and continuous_edges >= episode_coverage
            and reward_episode_coverage_h8 >= 1
        )
        score = (
            reward_episode_coverage_h8,
            support[8],
            support[1],
            continuous_edges,
            episode_coverage,
            selected_steps,
        )
        candidates.append({
            "skill_id": skill_id,
            "skill_id_tiebreak_sha256": _stable_hash(skill_id),
            "eligible": eligible,
            "score": list(score),
            "discovery_episode_coverage": episode_coverage,
            "selected_steps": selected_steps,
            "continuous_edges": continuous_edges,
            "positive_reward_support": {
                f"h{horizon}": support[horizon] for horizon in HORIZONS
            },
            "positive_reward_episode_coverage_h8": reward_episode_coverage_h8,
        })
    candidates.sort(
        key=lambda row: tuple(-int(value) for value in row["score"])
        + (row["skill_id_tiebreak_sha256"],)
    )
    eligible = [row for row in candidates if row["eligible"]]
    body = {
        "schema_version": "DISCOVERY_ONLY_VALUE_AWARE_SELECTION_V1_FROZEN",
        "events_sha256": events_sha256,
        "split_rule": "ascending_reset_seed_round_robin_v1",
        "split_by_seed": {str(seed): split_by_seed[seed] for seed in seeds},
        "content_scope": "DISCOVERY_EVENTS_ONLY",
        "discovery_episode_ids": sorted(discovery_episodes),
        "discovery_event_hashes": discovery_event_hashes,
        "horizons": list(HORIZONS),
        "eligibility_rule": (
            "discovery_episode_coverage>=2 AND continuous_edges>=episode_coverage "
            "AND positive_reward_episode_coverage_h8>=1"
        ),
        "ranking_rule": (
            "descending reward_episode_coverage_h8, h8 support, h1 support, "
            "continuous edges, episode coverage, selected steps; SHA256 ID tie-break"
        ),
        "candidates": candidates,
        "selected_skill_id": eligible[0]["skill_id"] if eligible else None,
        "status": "FROZEN_CANDIDATE" if eligible else "NO_ELIGIBLE_CANDIDATE",
    }
    return body | {"selection_receipt_sha256": _stable_hash(body)}


def select_from_evidence(evidence_dir: str | Path) -> dict[str, Any]:
    root = Path(evidence_dir)
    events_path = root / "events.jsonl"
    events = [
        json.loads(line) for line in events_path.read_text().splitlines() if line.strip()
    ]
    report = select_discovery_candidate(events, events_sha256=_file_hash(events_path))
    manifest = json.loads((root / "manifest.json").read_text())
    expected = (manifest.get("files") or {}).get("events.jsonl", {}).get("sha256")
    if expected != report["events_sha256"]:
        raise ValueError("events hash does not match source evidence manifest")
    return report


__all__ = ["SPLITS", "HORIZONS", "select_discovery_candidate", "select_from_evidence"]
