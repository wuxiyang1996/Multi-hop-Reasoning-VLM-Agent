"""Small provider clients used by evidence-collection runners.

Provider replies remain untrusted Agent output.  These clients deliberately do
not request server-side structured output: every model is evaluated by the same
local fail-closed parser in :mod:`harness.agent_reasoning_cycle`.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any, Dict

import httpx


def load_literal_secret(path: str | Path, variable: str) -> str:
    """Read one literal string assignment without importing/executing a key file."""
    source_path = Path(path)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    values: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
            try:
                value = ast.literal_eval(value_node)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{variable} must be a literal string") from exc
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{variable} must be a non-empty literal string")
            values.append(value.strip())
    if len(values) != 1:
        raise ValueError(f"expected exactly one assignment for {variable}, found {len(values)}")
    return values[0]


class StrictOpenAIResponsesClient:
    """Minimal synchronous OpenAI Responses API client with exact text extraction."""

    def __init__(self, base_url: str, *, timeout_s: float, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        reasoning_effort: str = "low",
    ) -> tuple[str, Dict[str, Any]]:
        if reasoning_effort not in {"minimal", "low", "medium", "high"}:
            raise ValueError("unsupported OpenAI reasoning effort")
        started = time.monotonic()
        response = self._client.post(
            f"{self.base_url}/responses",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "input": prompt,
                "max_output_tokens": int(max_tokens),
                "reasoning": {"effort": reasoning_effort},
                "text": {"verbosity": "low"},
                "store": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        texts: list[str] = []
        for output in payload.get("output") or []:
            if not isinstance(output, dict) or output.get("type") != "message":
                continue
            for content in output.get("content") or []:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        texts.append(text)
        usage = dict(payload.get("usage") or {})
        usage["prompt_tokens"] = int(usage.get("input_tokens", 0) or 0)
        usage["completion_tokens"] = int(usage.get("output_tokens", 0) or 0)
        usage["generation_id"] = str(payload.get("id") or "")
        usage["response_status"] = str(payload.get("status") or "")
        usage["incomplete_details"] = payload.get("incomplete_details")
        usage["latency_s"] = time.monotonic() - started
        usage["model_requested"] = model
        usage["reasoning_effort"] = reasoning_effort
        return "".join(texts), usage


__all__ = ["StrictOpenAIResponsesClient", "load_literal_secret"]
