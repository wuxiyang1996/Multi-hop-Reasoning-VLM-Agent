#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import runpy

from motif_transfer.frozen_motif_agent import OpenAICompatibleBackend
from motif_transfer.instrumented_import import import_native_source_batch
from motif_transfer.source_execution_motifs import (
    SourceExecutionMotifAgent,
    audit_execution_graph,
    build_execution_traces,
    execution_affordance_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose a receipt-grounded motif over old skill sub-episodes"
    )
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--receipt-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    api_env = args.api_key_env
    if args.key_file:
        value = runpy.run_path(str(args.key_file)).get(args.api_key_env)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"missing {args.api_key_env} in key file")
        api_env = "SOURCE_EXECUTION_MOTIF_API_KEY"
        os.environ[api_env] = value
    episodes = import_native_source_batch(args.evidence_dir)
    traces = build_execution_traces(episodes)
    games = {trace.game for trace in traces}
    if len(games) != 1:
        raise ValueError(f"expected one game, found {sorted(games)}")
    backend = OpenAICompatibleBackend(
        args.endpoint,
        {"source_execution_motif": args.model},
        api_key_env=api_env,
        json_mode=True,
        temperature=None,
    )
    agent = SourceExecutionMotifAgent(backend)
    graph = agent.propose(
        next(iter(games)), traces, include_reasoning=not args.receipt_only,
    )
    audit = audit_execution_graph(graph, traces)
    payload = {
        "schema_version": 1,
        "authority": "UNTRUSTED_AGENT_GRAPH_OVER_RECEIPT_BOUND_SKILL_EXECUTIONS",
        "condition": "RECEIPT_ONLY" if args.receipt_only else "AUTHENTIC_REASONING",
        "evidence_dir": str(args.evidence_dir.resolve()),
        "model_identity": dict(backend.identity),
        "affordance": execution_affordance_report(traces),
        "graph": asdict(graph),
        "audit": asdict(audit),
        "agent_call": agent.last_call,
        "source_supported": False,
        "source_supported_reason": (
            "Graph audit alone is insufficient; blind recurrence and matched "
            "h1/h2/h4/h8 value remain required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "affordance": payload["affordance"]["split_stats"],
        "audit": payload["audit"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
