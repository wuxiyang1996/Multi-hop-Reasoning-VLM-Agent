#!/usr/bin/env python3
"""Bind one frozen source memory artifact to a target adaptation split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.cross_domain_memory_baselines import (  # noqa: E402
    TargetDomain,
    bind_memory_artifact_to_target,
    gate_memory_artifact_to_target,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected one JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--adaptation", type=Path, required=True)
    parser.add_argument("--target-domain", choices=[row.value for row in TargetDomain], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--maximum-items-per-call", type=int, default=8)
    parser.add_argument("--maximum-output-tokens", type=int, default=5000)
    parser.add_argument(
        "--mode", choices=("gate-only", "rewrite"), default="gate-only",
        help="Main comparison uses gate-only; rewrite is a legacy diagnostic ablation.",
    )
    args = parser.parse_args()
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    os.environ["XD_MEMORY_BINDER_OPENROUTER_KEY"] = str(key)
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url, {
                "memory_binder": args.model,
                "memory_binding_verifier": args.model,
                "memory_gate_verifier": args.model,
            },
            api_key_env="XD_MEMORY_BINDER_OPENROUTER_KEY",
            json_mode=True, temperature=0, timeout_seconds=240,
            request_overrides={
                "max_tokens": args.maximum_output_tokens,
                "reasoning": {"effort": "none", "exclude": True},
            },
        ),
        cache_path=args.cache,
    )
    source = _object(args.artifact)
    bind = (
        gate_memory_artifact_to_target
        if args.mode == "gate-only"
        else bind_memory_artifact_to_target
    )
    artifact = bind(
        source, args.target_domain, _object(args.adaptation), backend,
        maximum_items_per_call=args.maximum_items_per_call,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "method": artifact["method"], "target_domain": args.target_domain,
        "source_items": len(source["items"]), "bound_items": len(artifact["items"]),
        "artifact_sha256": artifact["artifact_sha256"], "output": str(args.output),
        "mode": args.mode,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
