#!/usr/bin/env python3
"""Fail-closed import-only runtime check before freezing Phase-2 WebShop V2."""

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


EXPECTED_PYTHON = Path(
    "/fs/gamma-projects/vlm-robot/conda/envs/vlm_benchmarks/bin/python"
).resolve()
WRAPPER_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs/results/phase2_webshop_runtime_v2_drycheck.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    args = parser.parse_args()

    import browsergym  # noqa: F401
    import gymnasium
    import scripts.run_phase2_webshop_utility_v1 as phase2_runner  # noqa: F401
    import scripts.run_webshop_neural_symbolic_v9 as v9_runner  # noqa: F401

    sys.path.insert(0, str(WRAPPER_ROOT))
    from webshop_wrapper import register_webshop_tasks

    key_values = runpy.run_path(str(args.keys))
    key_present = bool(
        key_values.get("OPENROUTER_API_KEY")
        or key_values.get("openrouter_api_key")
    )
    gates = {
        "benchmark_python_selected": Path(sys.executable).resolve() == EXPECTED_PYTHON,
        "python_3_11_selected": sys.version_info[:2] == (3, 11),
        "gymnasium_imported": bool(gymnasium.__version__),
        "browsergym_imported": True,
        "phase2_runner_imported": True,
        "v9_runner_imported": True,
        "webshop_wrapper_imported": callable(register_webshop_tasks),
        "openrouter_key_present": key_present,
        "zero_target_registrations": True,
        "zero_environment_constructions": True,
        "zero_target_resets": True,
        "zero_target_actions": True,
        "zero_provider_calls": True,
        "zero_target_outcomes_read": True,
    }
    body = {
        "schema_version": "phase2-webshop-runtime-drycheck-v2",
        "status": (
            "PHASE2_WEBSHOP_RUNTIME_DRYCHECK_PASSED"
            if all(gates.values())
            else "PHASE2_WEBSHOP_RUNTIME_DRYCHECK_FAILED"
        ),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "gymnasium_version": gymnasium.__version__,
        "wrapper_root": str(WRAPPER_ROOT),
        "gates": gates,
    }
    result = body | {"drycheck_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite runtime dry-check: {args.output}")
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
