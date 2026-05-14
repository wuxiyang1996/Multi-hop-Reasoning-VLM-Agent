"""Gym-like environment wrapper for Visual/Video Reasoning QA tasks.

Converts the ``VisualReasoningExecutor`` / ``VideoReasoningExecutor``
tool-loop interface into a standard ``reset()`` / ``step()`` interface
so that QA multi-hop reasoning tasks can use the SAME decision-agent
pipeline as games and web tasks.

Interface contract (matches ``_GymLikeWrapper`` in ``gym_like.py``)::

    env = VRReasoningEnv(task_config)
    obs_nl, info = env.reset()
    obs_nl, reward, terminated, truncated, info = env.step(action)
    env.close()

Actions are strings from ``InnerAction`` vocabulary:
    ``"GROUND"``, ``"RETRIEVE"``, ``"CHECK"``, ``"VERIFY"``, ``"COMMIT"``

Each ``step()`` executes one reasoning hop via the underlying
``HopExecutor``, accumulates evidence, and transitions the observation.
The episode terminates when a ``COMMIT`` action is taken (the executor
produces a final answer) or when ``max_steps`` is reached.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

INNER_ACTIONS = ["GROUND", "RETRIEVE", "CHECK", "VERIFY", "COMMIT"]


@dataclass
class VRReasoningEnv:
    """Gym-like wrapper around a visual reasoning HopExecutor.

    Parameters
    ----------
    question : str
        The QA question text.
    image_source : Any
        The image (PIL.Image, np.ndarray, or path) for visual QA, or
        a list of frames for video QA.
    ground_truth : str
        The ground-truth answer (for reward computation at COMMIT).
    executor_factory : callable
        A callable that creates a HopExecutor from the image_source.
        Signature: ``(image_source) -> HopExecutor``.
    max_steps : int
        Maximum number of reasoning hops before truncation.
    skill_bank_context : dict | None
        Optional context from the skill bank (protocol steps, etc.).
    answer_checker : callable | None
        ``(predicted, ground_truth) -> float`` reward function.
        Defaults to exact string match (1.0 / 0.0).
    """

    question: str
    image_source: Any
    ground_truth: str
    executor_factory: Callable
    max_steps: int = 15
    skill_bank_context: Optional[Dict[str, Any]] = None
    answer_checker: Optional[Callable] = None

    _executor: Any = field(default=None, init=False, repr=False)
    _step_count: int = field(default=0, init=False)
    _evidence_chain: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _hop_history: List[str] = field(default_factory=list, init=False)
    _done: bool = field(default=False, init=False)
    _episode_id: str = field(default="", init=False)
    _final_answer: Optional[str] = field(default=None, init=False)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment for a new episode.

        Returns ``(obs_nl, info)`` where ``obs_nl`` is a textual
        description of the initial state.
        """
        self._step_count = 0
        self._evidence_chain = []
        self._hop_history = []
        self._done = False
        self._final_answer = None
        self._episode_id = f"vr_{uuid.uuid4().hex[:8]}"

        self._executor = self.executor_factory(self.image_source)

        obs_nl = self._build_observation()
        info: Dict[str, Any] = {
            "action_names": INNER_ACTIONS,
            "question": self.question,
            "episode_id": self._episode_id,
            "domain": "qa",
        }
        return obs_nl, info

    def step(
        self, action: Union[str, Dict[str, Any]],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Execute one reasoning hop.

        Parameters
        ----------
        action : str or dict
            If str, just the action type (e.g. ``"GROUND"``).
            If dict, ``{"action_type": "GROUND", "payload": {...}}``.

        Returns
        -------
        obs_nl, reward, terminated, truncated, info
        """
        if self._done:
            return self._build_observation(), 0.0, True, False, self._build_info()

        if isinstance(action, dict):
            action_type = action.get("action_type", action.get("action", ""))
            payload = action.get("payload", {})
        else:
            action_type = str(action).strip().upper()
            payload = self._default_payload(action_type)

        self._step_count += 1
        self._hop_history.append(action_type)

        terminated = False
        truncated = False
        reward = 0.0

        try:
            from harness.skill_adapter import AdapterRunContext
            ctx = AdapterRunContext(
                question=self.question,
                evidence=list(self._evidence_chain),
            )
            result = self._executor(action_type, payload, ctx)
        except Exception as e:
            logger.warning("HopExecutor failed for %s: %s", action_type, e)
            result = {"error": str(e), "role": "GATHER"}

        evidence_entry = {
            "hop": self._step_count,
            "action": action_type,
            "result": result,
            "role": result.get("role", "GATHER"),
        }
        self._evidence_chain.append(evidence_entry)

        if action_type == "COMMIT":
            terminated = True
            self._done = True
            self._final_answer = result.get("answer", result.get("value", ""))
            reward = self._compute_reward()

        if self._step_count >= self.max_steps and not terminated:
            truncated = True
            self._done = True

        obs_nl = self._build_observation()
        info = self._build_info()
        info["hop_type"] = action_type
        info["raw_env_reward"] = reward

        return obs_nl, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up executor resources."""
        if self._executor is not None and hasattr(self._executor, "close"):
            try:
                self._executor.close()
            except Exception:
                pass

    # ── Internal helpers ────────────────────────────────────────────

    def _build_observation(self) -> str:
        """Build a textual observation from the current evidence chain."""
        parts = [f"Question: {self.question}"]

        if self._evidence_chain:
            parts.append(f"\nEvidence gathered ({len(self._evidence_chain)} hops):")
            for ev in self._evidence_chain[-5:]:
                role = ev.get("role", "?")
                action = ev.get("action", "?")
                result = ev.get("result", {})
                summary = str(result.get("summary", result.get("value", "")))[:200]
                parts.append(f"  [{role}] {action}: {summary}")

        parts.append(f"\nStep {self._step_count}/{self.max_steps}")
        parts.append(f"Available actions: {', '.join(INNER_ACTIONS)}")

        return "\n".join(parts)

    def _build_info(self) -> Dict[str, Any]:
        return {
            "action_names": INNER_ACTIONS,
            "question": self.question,
            "episode_id": self._episode_id,
            "domain": "qa",
            "hop_history": list(self._hop_history),
            "evidence_count": len(self._evidence_chain),
            "final_answer": self._final_answer,
        }

    def _default_payload(self, action_type: str) -> Dict[str, Any]:
        """Generate a default payload for simple string actions."""
        if action_type == "GROUND":
            return {"query": self.question}
        if action_type == "RETRIEVE":
            return {}
        if action_type == "CHECK":
            return {"kind": "COMPARE"}
        if action_type == "VERIFY":
            return {}
        if action_type == "COMMIT":
            evidence_text = " ".join(
                str(ev.get("result", {}).get("value", ""))
                for ev in self._evidence_chain
            )
            return {"value": evidence_text}
        return {}

    def _compute_reward(self) -> float:
        """Compute episode-end reward by comparing answer to ground truth."""
        if self._final_answer is None:
            return 0.0

        if self.answer_checker is not None:
            return float(self.answer_checker(self._final_answer, self.ground_truth))

        pred = str(self._final_answer).strip().lower()
        gt = str(self.ground_truth).strip().lower()
        if pred == gt:
            return 1.0
        if gt in pred or pred in gt:
            return 0.5
        return 0.0


def make_vr_env(
    question: str,
    image_source: Any,
    ground_truth: str,
    max_steps: int = 15,
    answer_checker: Optional[Callable] = None,
    prefer_gdino: bool = True,
    confidence: float = 0.8,
) -> VRReasoningEnv:
    """Factory function to create a VRReasoningEnv with the standard executor.

    Uses ``VisualReasoningExecutor.from_image()`` under the hood.
    """
    def _factory(img):
        from visual_reasoning_wrapper.skill_executor import VisualReasoningExecutor
        return VisualReasoningExecutor.from_image(
            img, prefer_gdino=prefer_gdino, confidence=confidence,
        )

    return VRReasoningEnv(
        question=question,
        image_source=image_source,
        ground_truth=ground_truth,
        executor_factory=_factory,
        max_steps=max_steps,
        answer_checker=answer_checker,
    )
