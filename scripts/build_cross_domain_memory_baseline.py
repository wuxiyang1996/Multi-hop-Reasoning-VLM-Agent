#!/usr/bin/env python3
"""Build a frozen ExpeL, AWM, or ReasoningBank artifact from source receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.cross_domain_memory_baselines import (  # noqa: E402
    MemoryBaseline,
    induce_memory_artifact,
)
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=[row.value for row in MemoryBaseline])
    parser.add_argument("--source", type=Path, required=True, help="Canonical source episode JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--maximum-items-per-call", type=int, default=8)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise SystemExit("source input must be one JSON object")
    raw_backend = OpenAICompatibleBackend(
        args.base_url,
        {"memory_inducer": args.model},
        api_key_env=args.api_key_env,
        json_mode=True,
        temperature=0,
    )
    backend = MemoizedCompletionBackend(raw_backend, cache_path=args.cache)
    artifact = induce_memory_artifact(
        args.method,
        source,
        backend,
        maximum_items_per_call=args.maximum_items_per_call,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "method": artifact["method"],
        "source_domains": artifact["source_domains"],
        "episodes": len(artifact["source_episode_ids"]),
        "items": len(artifact["items"]),
        "artifact_sha256": artifact["artifact_sha256"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
