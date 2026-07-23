"""Compact batch persistence for instrumented source rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness.reasoning_event_log import (
    reasoning_event_log_from_dict,
    validate_reasoning_protocol,
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_evidence_batch(
    output_dir: Path,
    results: Sequence[Any],
    *,
    manifest_metadata: Mapping[str, Any],
    protocol_profile: str = "source_agent",
) -> Mapping[str, Any]:
    """Write one manifest and two JSONL files, never one file per frame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    events_path = output_dir / "events.jsonl"
    episode_lines = []
    event_lines = []
    protocol_failures = {}
    for result in sorted(results, key=lambda item: item.episode_id):
        log = dict(result.reasoning_event_log or {})
        events = reasoning_event_log_from_dict(log)
        failures = list(validate_reasoning_protocol(events, profile=protocol_profile))
        if failures:
            protocol_failures[result.episode_id] = failures
        episode_lines.append(_json({
            "episode_id": result.episode_id,
            "game": result.game,
            "steps": int(result.steps),
            "total_reward": float(result.total_reward),
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "reasoning_log_sha256": log["log_sha256"],
            "protocol_failures": failures,
        }))
        event_lines.extend(_json(row) for row in log["events"])
    episodes_path.write_text("\n".join(episode_lines) + "\n", encoding="utf-8")
    events_path.write_text("\n".join(event_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "report_role": "source_observation_and_agent_decision_receipts",
        "n_episodes": len(results),
        "n_events": len(event_lines),
        "protocol_failures": protocol_failures,
        "files": {
            "episodes.jsonl": {"sha256": _sha256(episodes_path)},
            "events.jsonl": {"sha256": _sha256(events_path)},
        },
        "metadata": dict(manifest_metadata),
        "protocol_profile": protocol_profile,
        "claim_limit": (
            "Agent-origin decisions are observational evidence. Policy transforms, "
            "fallbacks, and replay forks are separately classified."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n", encoding="utf-8")
    return manifest


__all__ = ["write_source_evidence_batch"]
