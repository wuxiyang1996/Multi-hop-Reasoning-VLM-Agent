#!/usr/bin/env python3
"""Import-only and synthetic-failure dry-check for WebShop Phase-2 V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.webshop_candidate_failclosed_v3 import (  # noqa: E402
    FALLBACK_SCHEMA,
    failclosed_decision_candidates,
)


EXPECTED_PYTHON = Path(
    "/fs/gamma-projects/vlm-robot/conda/envs/vlm_benchmarks/bin/python"
).resolve()
WRAPPER_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent"
)


def _synthetic_invalid_base(**kwargs):
    attempts = kwargs["attempts_out"]
    attempts.append({"attempt": 0, "validation_error": "synthetic-invalid"})
    raise ValueError("Decision response contains no valid target-native action")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "docs/results/phase2_webshop_runtime_v3_drycheck.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    args = parser.parse_args()

    import browsergym  # noqa: F401
    import gymnasium
    import scripts.run_phase2_webshop_utility_v3 as phase2_runner  # noqa: F401

    sys.path.insert(0, str(WRAPPER_ROOT))
    from webshop_wrapper import register_webshop_tasks

    key_values = runpy.run_path(str(args.keys))
    key_present = bool(
        key_values.get("OPENROUTER_API_KEY")
        or key_values.get("openrouter_api_key")
    )
    candidates, raw, attempts = failclosed_decision_candidates(
        _synthetic_invalid_base,
        payload={"synthetic": True},
        axtree="synthetic tree with no target data",
        maximum=5,
        schema_retries=3,
    )
    fallback = attempts[-1].get("deterministic_fallback") or {}
    gates = {
        "benchmark_python_selected": Path(sys.executable).resolve() == EXPECTED_PYTHON,
        "python_3_11_selected": sys.version_info[:2] == (3, 11),
        "gymnasium_imported": bool(gymnasium.__version__),
        "browsergym_imported": True,
        "v3_runner_imported": True,
        "webshop_wrapper_imported": callable(register_webshop_tasks),
        "openrouter_key_present": key_present,
        "synthetic_invalid_response_falls_back_to_noop": candidates == ("noop()",),
        "fallback_raw_is_valid_json": json.loads(raw)["candidates"][0]["action"] == "noop()",
        "fallback_schema_bound": fallback.get("schema_version") == FALLBACK_SCHEMA,
        "fallback_used_no_target_information": fallback.get("task_or_goal_information_used") is False,
        "fallback_used_no_source_information": fallback.get("source_information_used") is False,
        "fallback_made_no_provider_call": fallback.get("provider_call") is False,
        "zero_target_registrations": True,
        "zero_environment_constructions": True,
        "zero_target_resets": True,
        "zero_target_actions": True,
        "zero_provider_calls": True,
        "zero_target_outcomes_read": True,
    }
    body = {
        "schema_version": "phase2-webshop-runtime-drycheck-v3",
        "status": (
            "PHASE2_WEBSHOP_RUNTIME_V3_DRYCHECK_PASSED"
            if all(gates.values())
            else "PHASE2_WEBSHOP_RUNTIME_V3_DRYCHECK_FAILED"
        ),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "gymnasium_version": gymnasium.__version__,
        "wrapper_root": str(WRAPPER_ROOT),
        "fallback_schema": FALLBACK_SCHEMA,
        "gates": gates,
    }
    result = body | {"drycheck_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite runtime dry-check: {args.output}")
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
