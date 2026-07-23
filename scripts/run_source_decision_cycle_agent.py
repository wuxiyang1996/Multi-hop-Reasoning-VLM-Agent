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
from motif_transfer.source_decision_cycles import (
    SourceDecisionCycleAgent,
    audit_graph,
    build_decision_traces,
    score_blind_predictions,
    shuffled_graph_edges,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose and blindly qualify a whole-decision-cycle source graph"
    )
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-predictions-per-trace", type=int, default=8)
    args = parser.parse_args()

    api_env = args.api_key_env
    if args.key_file:
        value = runpy.run_path(str(args.key_file)).get(args.api_key_env)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"missing {args.api_key_env} in key file")
        api_env = "SOURCE_DECISION_CYCLE_API_KEY"
        os.environ[api_env] = value
    episodes = import_native_source_batch(args.evidence_dir)
    traces = build_decision_traces(episodes)
    games = {trace.game for trace in traces}
    if len(games) != 1:
        raise ValueError(f"expected one game, found {sorted(games)}")
    backend = OpenAICompatibleBackend(
        args.endpoint,
        {
            "source_decision_graph": args.model,
            "source_decision_prediction": args.model,
        },
        api_key_env=api_env,
        json_mode=True,
        temperature=None,
    )
    agent = SourceDecisionCycleAgent(backend)
    graph = agent.propose(next(iter(games)), traces)
    audit = audit_graph(graph, traces)
    predictions = {}
    scores = {}
    if audit.accepted:
        controls = {
            "AUTHENTIC_GRAPH": None,
            "SHUFFLED_TOPOLOGY": shuffled_graph_edges(graph),
        }
        for condition, edge_override in controls.items():
            predictions[condition] = {}
            scores[condition] = {}
            for split in ("qualification", "held_out"):
                rows = agent.predict(
                    graph, traces, split=split,
                    maximum_queries_per_trace=args.max_predictions_per_trace,
                    condition=condition,
                    edge_override=edge_override,
                )
                predictions[condition][split] = [asdict(row) for row in rows]
                scores[condition][split] = score_blind_predictions(
                    graph, traces, rows, split=split,
                    edge_override=edge_override,
                )
    payload = {
        "schema_version": 2,
        "authority": "UNTRUSTED_AGENT_GRAPH_WITH_BLIND_RECEIPT_SCORING",
        "evidence_dir": str(args.evidence_dir.resolve()),
        "model_identity": dict(backend.identity),
        "graph": asdict(graph),
        "audit": asdict(audit),
        "predictions": predictions,
        "scores": scores,
        "agent_calls": agent.last_calls,
        "source_supported": False,
        "source_supported_reason": (
            "Matched h1/h2/h4/h8 authentic value is a separate required gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "audit_accepted": audit.accepted,
        "nontrivial": audit.nontrivial,
        "crosses_recorded_skill_boundary": audit.crosses_recorded_skill_boundary,
        "scores": {
            condition: {
                split: {
                    key: value for key, value in report.items()
                    if key in {"queries_sampled", "coverage", "exact_accuracy",
                               "null_exact_accuracy", "field_accuracy",
                               "null_field_accuracy", "graph_edge_validity"}
                }
                for split, report in condition_scores.items()
            }
            for condition, condition_scores in scores.items()
        },
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
