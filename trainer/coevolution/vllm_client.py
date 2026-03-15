"""Async vLLM client wrapper for co-evolution inference.

Wraps ``openai.AsyncOpenAI`` to provide adapter-aware completions via the
vLLM multi-LoRA server.  Both decision agent and skill bank agent call the
same vLLM instance; the ``adapter`` parameter selects the active LoRA.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

ADAPTER_MAP = {
    "skill_selection": "skill_selection",
    "action_taking": "action_taking",
    "segment": "segment",
    "contract": "contract",
    "curator": "curator",
    "base": None,
}

# HTTP 400/404 status codes that indicate a missing LoRA adapter
_ADAPTER_NOT_FOUND_CODES = {400, 404}


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    adapter: Optional[str] = None


def _is_adapter_missing(exc: Exception) -> bool:
    """Check if an OpenAI API error indicates the LoRA adapter is not loaded."""
    status = getattr(exc, "status_code", None)
    if status in _ADAPTER_NOT_FOUND_CODES:
        return True
    msg = str(exc).lower()
    return "not found" in msg or "does not exist" in msg or "unknown model" in msg


class AsyncVLLMClient:
    """Thin async wrapper over vLLM's OpenAI-compatible API with multi-LoRA."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-14B",
        default_temperature: float = 0.3,
        default_max_tokens: int = 512,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        self._call_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    async def generate(
        self,
        prompt: str,
        *,
        adapter: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> GenerateResult:
        """Generate a completion via vLLM.

        Parameters
        ----------
        prompt : str
            Full prompt text (system + user rolled into one string).
        adapter : str | None
            LoRA adapter name (e.g. ``"action_taking"``).  ``None`` = base model.
        """
        t0 = time.monotonic()
        temp = temperature if temperature is not None else self.default_temperature
        mtok = max_tokens if max_tokens is not None else self.default_max_tokens

        model_id = self.model
        used_adapter = adapter
        if adapter and adapter in ADAPTER_MAP and ADAPTER_MAP[adapter] is not None:
            model_id = ADAPTER_MAP[adapter]

        try:
            resp = await self._client.completions.create(
                model=model_id,
                prompt=prompt,
                temperature=temp,
                max_tokens=mtok,
                stop=stop,
            )
        except Exception as exc:
            # If the adapter isn't loaded yet (cold start), fall back to base model
            if adapter and model_id != self.model and _is_adapter_missing(exc):
                logger.info(
                    "Adapter '%s' not loaded in vLLM, falling back to base model",
                    adapter,
                )
                used_adapter = None
                try:
                    resp = await self._client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        temperature=temp,
                        max_tokens=mtok,
                        stop=stop,
                    )
                except Exception as exc2:
                    logger.warning("vLLM base-model fallback failed: %s", exc2)
                    return GenerateResult(text="", adapter=used_adapter)
            else:
                logger.warning("vLLM call failed (adapter=%s): %s", adapter, exc)
                return GenerateResult(text="", adapter=used_adapter)

        text = resp.choices[0].text if resp.choices else ""
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        elapsed = (time.monotonic() - t0) * 1000
        self._call_count += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        return GenerateResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed,
            adapter=used_adapter,
        )

    async def generate_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        adapter: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerateResult:
        """Chat-completion variant for skill bank stages that use chat format."""
        t0 = time.monotonic()
        temp = temperature if temperature is not None else self.default_temperature
        mtok = max_tokens if max_tokens is not None else self.default_max_tokens

        model_id = self.model
        used_adapter = adapter
        if adapter and adapter in ADAPTER_MAP and ADAPTER_MAP[adapter] is not None:
            model_id = ADAPTER_MAP[adapter]

        try:
            resp = await self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temp,
                max_tokens=mtok,
            )
        except Exception as exc:
            if adapter and model_id != self.model and _is_adapter_missing(exc):
                logger.info(
                    "Adapter '%s' not loaded in vLLM (chat), falling back to base model",
                    adapter,
                )
                used_adapter = None
                try:
                    resp = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=mtok,
                    )
                except Exception as exc2:
                    logger.warning("vLLM chat base-model fallback failed: %s", exc2)
                    return GenerateResult(text="", adapter=used_adapter)
            else:
                logger.warning("vLLM chat call failed (adapter=%s): %s", adapter, exc)
                return GenerateResult(text="", adapter=used_adapter)

        text = resp.choices[0].message.content if resp.choices else ""
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        elapsed = (time.monotonic() - t0) * 1000
        self._call_count += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        return GenerateResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed,
            adapter=used_adapter,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "call_count": self._call_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
        }

    def reset_stats(self) -> None:
        self._call_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    async def health_check(self) -> bool:
        try:
            resp = await self._client.models.list()
            return len(resp.data) > 0
        except Exception:
            return False
