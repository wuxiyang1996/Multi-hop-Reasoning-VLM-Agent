"""
Policy interface for the VLM Decision Agent under GRPO training.

Abstracts logprob extraction and action sampling from the underlying LLM,
so the GRPO trainer can compute the ranking objective without knowing model
internals.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PolicyOutput:
    """Output of a single policy forward pass."""

    action: str
    logprob: float
    entropy: float = 0.0
    value: Optional[float] = None
    raw_response: str = ""
    metadata: Optional[Dict[str, Any]] = None


class PolicyInterface(ABC):
    """Abstract interface for a trainable policy.

    Concrete implementations wrap a specific LLM/VLM backend and provide:
      - sample(): stochastic action sampling with logprobs
      - logprob(): compute logprob of a given action under current policy
      - update(): apply a gradient step given GRPO loss
    """

    @abstractmethod
    def sample(
        self,
        observation: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> PolicyOutput:
        """Sample an action from the policy.

        Args:
            observation: current observation text/state
            context: additional context (skill cards, memory, internal state)
            temperature: sampling temperature

        Returns:
            PolicyOutput with action string, logprob, and optional value estimate
        """

    @abstractmethod
    def logprob(
        self,
        observation: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute log-probability of a given action under current policy."""

    @abstractmethod
    def batch_logprobs(
        self,
        observations: List[str],
        actions: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """Compute log-probabilities for a batch of (obs, action) pairs."""

    @abstractmethod
    def update(self, loss: float, grads: Optional[Any] = None) -> Dict[str, float]:
        """Apply one gradient step.

        Returns training stats (loss, grad_norm, etc.)
        """

    @abstractmethod
    def get_parameters(self) -> Any:
        """Return current model parameters (for checkpointing)."""

    @abstractmethod
    def load_parameters(self, params: Any) -> None:
        """Load model parameters (for checkpoint restoration)."""


class LLMPolicy(PolicyInterface):
    """Concrete policy implementation wrapping an LLM API.

    For GRPO, we need:
      1. Multiple rollouts per prompt (group sampling)
      2. Log-probability of each chosen action
      3. Parameter updates via the GRPO objective

    This implementation uses the model's logprobs API when available,
    and falls back to approximation otherwise.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        prompt_builder: Any = None,
        lr: float = 1e-5,
    ):
        self.model_name = model_name
        self.prompt_builder = prompt_builder
        self.lr = lr
        self._parameters: Dict[str, Any] = {}
        self._step_count = 0

    def sample(
        self,
        observation: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
    ) -> PolicyOutput:
        """Sample an action using the LLM."""
        try:
            from API_func import ask_model
        except ImportError:
            return PolicyOutput(action="no-op", logprob=-1.0, raw_response="")

        prompt = self._build_prompt(observation, context)
        response = ask_model(
            prompt,
            model=self.model_name,
            temperature=temperature,
            max_tokens=400,
        )
        action = self._extract_action(response or "")
        lp = self._estimate_logprob(prompt, action, temperature)

        return PolicyOutput(
            action=action,
            logprob=lp,
            raw_response=response or "",
            metadata={"prompt_len": len(prompt)},
        )

    def logprob(
        self,
        observation: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute (approximate) logprob for a given action."""
        prompt = self._build_prompt(observation, context)
        return self._estimate_logprob(prompt, action, temperature=0.0)

    def batch_logprobs(
        self,
        observations: List[str],
        actions: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """Batch logprob computation."""
        contexts = contexts or [None] * len(observations)
        return [
            self.logprob(obs, act, ctx)
            for obs, act, ctx in zip(observations, actions, contexts)
        ]

    def update(self, loss: float, grads: Optional[Any] = None) -> Dict[str, float]:
        """Record training step (actual gradient update depends on backend)."""
        self._step_count += 1
        return {"loss": loss, "step": self._step_count, "lr": self.lr}

    def get_parameters(self) -> Any:
        return dict(self._parameters)

    def load_parameters(self, params: Any) -> None:
        if isinstance(params, dict):
            self._parameters = params

    def _build_prompt(self, observation: str, context: Optional[Dict[str, Any]]) -> str:
        if self.prompt_builder is not None:
            return self.prompt_builder(observation, context)
        parts = [f"Observation: {observation[:2000]}"]
        if context:
            if context.get("skill_cards"):
                parts.append(f"Available skills: {context['skill_cards']}")
            if context.get("active_skill"):
                parts.append(f"Active skill: {context['active_skill']}")
        parts.append("Choose ONE action:")
        return "\n".join(parts)

    @staticmethod
    def _extract_action(response: str) -> str:
        import re
        m = re.search(r'"action"\s*:\s*"([^"]+)"', response)
        if m:
            return m.group(1)
        m = re.search(r"TOOL:\s*take_action.*?\"action\":\s*\"([^\"]+)\"", response, re.DOTALL)
        if m:
            return m.group(1)
        words = response.strip().split()
        return words[-1] if words else "no-op"

    @staticmethod
    def _estimate_logprob(prompt: str, action: str, temperature: float) -> float:
        """Approximate logprob via heuristic (placeholder for real API logprobs)."""
        action_len = max(len(action.split()), 1)
        return -0.5 * action_len - 0.1 * math.log(max(temperature, 0.01))
