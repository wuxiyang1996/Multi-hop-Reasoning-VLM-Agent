"""`RewardLogger` — central sink for SkillEpisode + reward events.

PLAN-HARNESS §5.6: a single sink so the orchestrator, gate, and
training cadence all read from the same source of truth.

This implementation is a thin file-backed JSONL log; the orchestrator's
`ArtifactStore` provides longer-lived storage. We intentionally keep
this module dependency-free so tests can use it standalone.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from data_structure.extensions.skill_episode import SkillEpisode


@dataclass
class RewardLogEntry:
    episode_id: str
    skill_id: str
    skill_version: str
    domain: str
    success: bool
    score: Optional[float]
    cost: Dict[str, float] = field(default_factory=dict)
    parent_run_id: Optional[str] = None
    transfer_label: Optional[str] = None
    timestamp: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "domain": self.domain,
            "success": self.success,
            "score": self.score,
            "cost": dict(self.cost),
            "parent_run_id": self.parent_run_id,
            "transfer_label": self.transfer_label,
            "timestamp": self.timestamp,
        }


class RewardLogger:
    """Append-only JSONL sink. Thread-safe via a single mutex."""

    def __init__(
        self,
        log_path: Optional[str] = None,
        *,
        episode_dir: Optional[str] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._log_path = log_path
        self._episode_dir = episode_dir
        self._clock = clock
        self._lock = threading.Lock()
        self._memory: List[RewardLogEntry] = []
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if episode_dir:
            os.makedirs(episode_dir, exist_ok=True)

    # -- writes ------------------------------------------------------------

    def log_episode(self, episode: SkillEpisode) -> RewardLogEntry:
        outcome = episode.outcome
        entry = RewardLogEntry(
            episode_id=episode.episode_id,
            skill_id=episode.skill_id,
            skill_version=episode.skill_version,
            domain=episode.domain,
            success=bool(outcome and outcome.success),
            score=(outcome.score if outcome else None),
            cost=dict(episode.cost),
            parent_run_id=episode.parent_run_id,
            transfer_label=episode.transfer_label,
            timestamp=self._clock(),
        )
        with self._lock:
            self._memory.append(entry)
            if self._log_path:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry.to_json()) + "\n")
            if self._episode_dir:
                ep_path = os.path.join(self._episode_dir, f"{episode.episode_id}.json")
                with open(ep_path, "w", encoding="utf-8") as fh:
                    json.dump(episode.to_json(), fh, indent=2)
        return entry

    # -- reads -------------------------------------------------------------

    def entries(self) -> List[RewardLogEntry]:
        with self._lock:
            return list(self._memory)

    def filter(
        self,
        *,
        skill_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Iterable[RewardLogEntry]:
        for e in self.entries():
            if skill_id and e.skill_id != skill_id:
                continue
            if domain and e.domain != domain:
                continue
            yield e


__all__ = ["RewardLogEntry", "RewardLogger"]
