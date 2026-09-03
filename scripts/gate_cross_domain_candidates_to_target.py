#!/usr/bin/env python3
"""Apply the shared no-rewrite target gate to a method-neutral candidate bank."""

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
    gate_candidates_to_target,
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
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--adaptation", type=Path, required=True)
    parser.add_argument("--target-domain", choices=[row.value for row in TargetDomain], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=Path("/fs/gamma-projects/vlm-robot/keys.py"))
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--maximum-items-per-call", type=int, default=8)
    args = parser.parse_args()
    payload = _object(args.candidates)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit("candidate input must contain a candidates list")
    key = runpy.run_path(str(args.keys)).get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    os.environ["XD_SHARED_MEMORY_GATE_KEY"] = str(key)
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            args.base_url, {"memory_gate_verifier": args.model},
            api_key_env="XD_SHARED_MEMORY_GATE_KEY", json_mode=True,
            temperature=0, timeout_seconds=240,
        ),
        cache_path=args.cache,
    )
    receipt = gate_candidates_to_target(
        candidates, args.target_domain, _object(args.adaptation), backend,
        maximum_items_per_call=args.maximum_items_per_call,
    )
    result = {
        "method": str(payload.get("method") or "unspecified"),
        "source_candidates_sha256": receipt["candidate_payload_sha256"],
        "admitted_candidates": [
            candidate for candidate in candidates
            if str(candidate.get("candidate_id")) in set(receipt["admitted_candidate_ids"])
        ],
        "gate_receipt": receipt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "method": result["method"], "candidates": len(candidates),
        "admitted": len(result["admitted_candidates"]), "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
