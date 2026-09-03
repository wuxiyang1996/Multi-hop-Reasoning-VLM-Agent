#!/usr/bin/env python3
"""V20 wrapper for models that require the default sampling temperature."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_natural_video_cross_model_proof_v17 as base  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_temperature_json_call(
    client: OpenAI,
    *,
    model: str,
    system: str,
    content: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )
    raw = response.choices[0].message.content or ""
    if not raw:
        raise ValueError("model returned no JSON content")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("model JSON root must be an object")
    usage = response.usage
    return payload, {
        "model": str(response.model),
        "finish_reason": str(response.choices[0].finish_reason),
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "completion_tokens": int(usage.completion_tokens if usage else 0),
        "cost": float(getattr(usage, "cost", 0.0) or 0.0),
        "response_sha256": base._content_hash(payload),
        "temperature": "provider_default_required_by_pinned_model",
    }


def main() -> None:
    # Read the same required --config argument without consuming argv; base.main
    # remains the sole CLI/parser and collection implementation.
    try:
        config_index = sys.argv.index("--config") + 1
        config_path = Path(sys.argv[config_index])
    except (ValueError, IndexError) as exc:
        raise SystemExit("--config is required") from exc
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = str(config["frozen_lineage"].get("transport_wrapper_sha256") or "")
    actual = _sha256(Path(__file__).resolve())
    if expected != actual:
        raise ValueError("V20 default-temperature transport wrapper lineage mismatch")
    if str(config["model"]["id"]) != "gpt-5.5-2026-04-23":
        raise ValueError("V20 transport wrapper is pinned to GPT-5.5-2026-04-23")
    base.paired.media_helpers._json_call = _default_temperature_json_call
    base.main()


if __name__ == "__main__":
    main()
