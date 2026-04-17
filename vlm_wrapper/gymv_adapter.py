"""Gym-V adapter: screenshot → GPT-4o → structured schema.

Sends the game frame to a vision LLM (GPT-4o by default) and receives
the canonical <state>…</state> schema.  The environment's native text
(obs.text, env.description) is optionally attached as grounding context
but the **image is the primary input**.

Two modes:
  1. ``generate_label``  — offline batch label generation for training data.
  2. ``GymVSchemaWrapper`` — online Gym-V ObservationWrapper that replaces
     obs.text with the GPT-produced schema on every step (expensive; for
     data collection, not real-time play).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import openai
from PIL import Image

from .schema import (
    build_system_prompt,
    build_user_message,
    parse_schema_output,
    validate_schema,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("VLM_LABEL_MODEL", "gpt-4o")
_DEFAULT_MAX_TOKENS = int(os.environ.get("VLM_LABEL_MAX_TOKENS", "1200"))
_DEFAULT_TEMPERATURE = float(os.environ.get("VLM_LABEL_TEMPERATURE", "0.2"))


# ======================================================================
# Core function: image → GPT-4o → schema
# ======================================================================

def generate_label(
    image: Image.Image,
    *,
    goal: str = "",
    task_id: str = "",
    step: int = 0,
    game_rules: str = "",
    obs_text: str = "",
    max_entities: int = 20,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    """Send a game screenshot to a vision LLM and return the structured schema.

    Parameters
    ----------
    image : PIL.Image
        The rendered game frame (visual input).
    goal : str
        Short task/goal description (from ``env.description``).
    task_id : str
        Environment ID (e.g. ``"Game2048-v0"``).
    step : int
        Current episode step.
    game_rules : str
        Full game rules from ``env.description`` — passed as text context
        so GPT can use correct action names and entity types.
    obs_text : str
        The native ``obs.text`` — passed as additional context, NOT as
        the primary input.  Used to help GPT ground ambiguous elements.
    max_entities : int
        Entity cap for the schema output.
    model : str
        Vision model to call (default: ``$VLM_LABEL_MODEL`` or ``gpt-4o``).
    api_key / base_url : str
        Override the OpenAI client config (useful for OpenRouter routing).
    temperature / max_tokens : float / int
        Generation parameters.
    retries : int
        Number of retries on parse failure.

    Returns
    -------
    dict with keys:
        ``"schema"``   – the ``<state>…</state>`` string (or None on failure)
        ``"raw"``      – raw LLM output before parsing
        ``"warnings"`` – list of validation warnings
        ``"model"``    – model used
    """
    model = model or _DEFAULT_MODEL
    temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
    max_tokens = max_tokens or _DEFAULT_MAX_TOKENS

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    system = build_system_prompt("gymv", max_entities=max_entities)

    extra_ctx_parts: list[str] = []
    if game_rules:
        extra_ctx_parts.append(f"Game rules:\n{game_rules}")
    if obs_text:
        extra_ctx_parts.append(f"Environment text state (for reference):\n{obs_text}")
    extra_context = "\n\n".join(extra_ctx_parts)

    user_content = build_user_message(
        image,
        domain="gymv",
        task_id=task_id,
        goal=goal,
        step=step,
        extra_context=extra_context,
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    raw = ""
    schema = None
    warnings: list[str] = []

    for attempt in range(1, retries + 2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content or ""
            schema = parse_schema_output(raw)
            if schema:
                warnings = validate_schema(schema)
                break
            else:
                logger.warning("Attempt %d: no <state> block in GPT output", attempt)
        except Exception as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)
            raw = f"Error: {exc}"

    return {"schema": schema, "raw": raw, "warnings": warnings, "model": model}


# ======================================================================
# Gym-V ObservationWrapper (online, expensive)
# ======================================================================

try:
    from gym_v.core import Env, Observation, ObservationWrapper

    class GymVSchemaWrapper(ObservationWrapper):
        """Gym-V wrapper that replaces ``obs.text`` with GPT-produced schema.

        **This is for offline data collection, not real-time play.**
        Each step makes a vision-LLM API call (~1-3 s latency).

        The original ``obs.text`` is preserved in ``obs.metadata["raw_text"]``.
        The raw GPT output is in ``obs.metadata["raw_gpt"]``.
        """

        def __init__(
            self,
            env: Env,
            *,
            max_entities: int = 20,
            model: str | None = None,
            api_key: str | None = None,
            base_url: str | None = None,
        ):
            super().__init__(env)
            self.max_entities = max_entities
            self.model = model
            self.api_key = api_key
            self.base_url = base_url
            self._step_count = 0

        def reset(self, *, seed=None, options=None):
            self._step_count = 0
            return super().reset(seed=seed, options=options)

        def step(self, action):
            self._step_count += 1
            return super().step(action)

        def observation(self, obs: Observation) -> Observation:
            image = obs.image
            if image is None:
                return obs

            description = ""
            if hasattr(self.env, "description"):
                description = self.env.description

            task_id = ""
            if self.env.spec is not None:
                task_id = self.env.spec.id or ""
            if not task_id:
                task_id = type(self.env.unwrapped).__name__

            goal_line = description.strip().split("\n")[0] if description else ""

            result = generate_label(
                image,
                goal=goal_line,
                task_id=task_id,
                step=self._step_count,
                game_rules=description,
                obs_text=obs.text or "",
                max_entities=self.max_entities,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
            )

            new_meta = dict(obs.metadata) if obs.metadata else {}
            new_meta["raw_text"] = obs.text
            new_meta["raw_gpt"] = result["raw"]
            new_meta["schema_warnings"] = result["warnings"]

            return Observation(
                image=obs.image,
                text=result["schema"] or obs.text,
                metadata=new_meta,
            )

except ImportError:
    GymVSchemaWrapper = None  # gym_v not installed
