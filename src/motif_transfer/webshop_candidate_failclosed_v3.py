"""Task-agnostic fail-closed handling for exhausted WebShop candidate generation."""

from __future__ import annotations

import json
from typing import Any, Callable

from .contracts import stable_hash


SAFE_FALLBACK_ACTION = "noop()"
FALLBACK_SCHEMA = "webshop-target-native-candidate-fallback-v3"


def failclosed_decision_candidates(
    base: Callable[..., tuple[tuple[str, ...], str, list[dict[str, Any]]]],
    **kwargs: Any,
) -> tuple[tuple[str, ...], str, list[dict[str, Any]]]:
    """Use one explicit safe no-op only after every schema retry is invalid.

    Transport, authentication, and other runtime errors are not swallowed.  The
    fallback is deliberately independent of the task, goal, state, source game,
    and experimental condition, so it cannot manufacture a transfer advantage.
    """

    attempts = kwargs.pop("attempts_out", None)
    if attempts is None:
        attempts = []
    try:
        return base(**kwargs, attempts_out=attempts)
    except (json.JSONDecodeError, ValueError) as exc:
        body = {
            "candidates": [{"action": SAFE_FALLBACK_ACTION}],
            "deterministic_fallback": {
                "schema_version": FALLBACK_SCHEMA,
                "reason": "all_model_candidates_invalid_after_frozen_schema_retries",
                "validation_error_type": type(exc).__name__,
                "task_or_goal_information_used": False,
                "source_information_used": False,
                "condition_information_used": False,
                "provider_call": False,
            },
        }
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True)
        attempts.append({
            "attempt": len(attempts),
            "response_sha256": stable_hash(raw),
            "completion_diagnostic": {
                "character_count": len(raw),
                "json_type": "dict",
                "empty_or_none": False,
            },
            "cache_usage": {"provider_call": False, "deterministic": True},
            "valid_candidates": 1,
            "deterministic_fallback": body["deterministic_fallback"],
        })
        return (SAFE_FALLBACK_ACTION,), raw, attempts
