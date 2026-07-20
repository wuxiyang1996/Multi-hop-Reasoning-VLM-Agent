"""Record a real successful ALFWorld episode as one immutable demo receipt."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from harness.alfworld_grammar import parse_alfworld_action
from harness.skill_admission import TargetActionEvidence, TargetDemoReceipt


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _won(info: Dict[str, Any], reward: float) -> bool:
    won = info.get("won", False)
    if isinstance(won, (list, tuple)):
        won = won[0] if won else False
    return bool(won) or float(reward) >= 1.0


class AlfworldDemoRecorder:
    """Thin wrapper that records exact admissibility and state hashes."""

    def __init__(
        self,
        env: Any,
        *,
        demo_id: str,
        task_family: str,
        split: str = "train",
        episode_id: Optional[str] = None,
    ) -> None:
        self.env = env
        self.demo_id = demo_id
        self.task_family = task_family
        self.split = split
        self.episode_id = episode_id or demo_id
        self._observation = ""
        self._info: Dict[str, Any] = {}
        self._actions: List[TargetActionEvidence] = []
        self._best_score = 0.0
        self._success = False
        self._started = False

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        observation, info = self.env.reset()
        self._observation = str(observation)
        self._info = dict(info or {})
        self._actions.clear()
        self._best_score = 0.0
        self._success = _won(self._info, 0.0)
        self._started = True
        return observation, info

    def step(self, action: str):
        if not self._started:
            raise RuntimeError("reset must be called before recording steps")
        admissible = list(self._info.get("action_names") or [])
        parsed = parse_alfworld_action(action, admissible=admissible)
        before = {
            "observation": self._observation,
            "structured_state": self._info.get("structured_state"),
        }
        result = self.env.step(action)
        observation, reward, terminated, truncated, info = result
        after_info = dict(info or {})
        after = {
            "observation": str(observation),
            "structured_state": after_info.get("structured_state"),
        }
        self._actions.append(
            TargetActionEvidence(
                transition_index=len(self._actions),
                action=parsed.raw,
                operator=parsed.operator,
                arguments=dict(parsed.arguments),
                argument_types=dict(parsed.argument_types),
                admissible_actions_sha256=_hash(admissible),
                state_sha256=_hash(before),
                next_state_sha256=_hash(after),
            )
        )
        self._observation = str(observation)
        self._info = after_info
        self._best_score = max(self._best_score, float(reward))
        self._success = self._success or _won(after_info, float(reward))
        return result

    def receipt(self) -> TargetDemoReceipt:
        trace = {
            "demo_id": self.demo_id,
            "episode_id": self.episode_id,
            "split": self.split,
            "task_family": self.task_family,
            "actions": [asdict(item) for item in self._actions],
            "official_success": self._success,
            "official_score": self._best_score,
        }
        return TargetDemoReceipt(
            demo_id=self.demo_id,
            target_domain="alfworld",
            task_family=self.task_family,
            split=self.split,
            episode_id=self.episode_id,
            source_file_sha256=_hash(trace),
            executor_kind="real",
            evaluator="alfworld_official",
            official_success=self._success,
            official_score=self._best_score,
            actions=list(self._actions),
            held_out=False,
        )


__all__ = ["AlfworldDemoRecorder"]
