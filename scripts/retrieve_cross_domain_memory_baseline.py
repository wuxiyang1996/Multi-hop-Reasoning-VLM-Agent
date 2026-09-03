#!/usr/bin/env python3
"""Retrieve action-free cross-domain memory for one target-domain context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.cross_domain_memory_baselines import (  # noqa: E402
    CrossDomainMemoryAdvisor,
    LocalSentenceTransformerEmbeddingBackend,
    TargetDomain,
    adapt_target_context,
    retrieve_memory_items,
)


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected one JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--domain", required=True, choices=[row.value for row in TargetDomain])
    parser.add_argument("--task", required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    artifact = _object(args.artifact)
    raw = _object(args.context)
    target = adapt_target_context(
        args.domain,
        task=args.task,
        observation=raw.get("observation") or raw,
        native_actions=raw.get("native_actions") or (),
        history=raw.get("history") or (),
        proposal=raw.get("proposal"),
    )
    embedding = LocalSentenceTransformerEmbeddingBackend(args.embedding_model)
    retrieval = retrieve_memory_items(
        artifact, target, embedding, top_k=args.top_k,
    )
    advisory = CrossDomainMemoryAdvisor(retrieval).advisory()
    output = {
        "schema_version": 1,
        "target_context": target,
        "retrieval": retrieval,
        "advisory": {
            "verdict": advisory.verdict.value,
            "reason": advisory.reason,
            "evidence_receipt_ids": list(advisory.evidence_receipt_ids),
            "current_role": advisory.current_role,
            "information_need": advisory.information_need,
            "expected_transition": advisory.expected_transition,
            "failure_route": advisory.failure_route,
            "termination_test": advisory.termination_test,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "method": retrieval["method"], "domain": args.domain,
        "retrieved": len(retrieval["retrieved"]), "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
