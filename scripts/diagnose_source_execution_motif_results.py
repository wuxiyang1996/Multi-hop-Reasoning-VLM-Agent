#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from motif_transfer.instrumented_import import import_native_source_batch
from motif_transfer.execution_graph_scoring import (
    fit_execution_graph_model,
    score_execution_graph_blind,
)
from motif_transfer.source_execution_motifs import (
    audit_execution_graph,
    build_execution_traces,
    graph_from_response,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly re-audit saved untrusted execution-motif proposals"
    )
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.results:
        saved = json.loads(path.read_text(encoding="utf-8"))
        traces = build_execution_traces(
            import_native_source_batch(Path(saved["evidence_dir"]))
        )
        raw_graph = saved["graph"]
        graph = graph_from_response(raw_graph["game"], traces, {
            "description": raw_graph.get("untrusted_description", ""),
            "nodes": [{
                "node_id": node["node_id"],
                "execution_ids": node["execution_ids"],
                "role": node.get("untrusted_role", ""),
            } for node in raw_graph["nodes"]],
            "edges": [{
                "source": edge["source"],
                "target": edge["target"],
            } for edge in raw_graph["edges"]],
        })
        strict = audit_execution_graph(graph, traces)
        exploratory_blind = {}
        if len(graph.nodes) >= 2:
            model = fit_execution_graph_model(graph, traces)
            exploratory_blind = {
                split: score_execution_graph_blind(model, traces, split=split)
                for split in ("qualification", "held_out")
            }
        rows.append({
            "result_file": str(path),
            "condition": saved["condition"],
            "game": graph.game,
            "proposal_description_untrusted": graph.untrusted_description,
            "strict_audit": asdict(strict),
            "exploratory_blind_scoring": exploratory_blind,
            "exploratory_blind_is_confirmatory": False,
            "status": (
                "DISCOVERY_CANDIDATE"
                if strict.accepted else "REJECTED_OR_NEEDS_FRESH_RECURRENCE"
            ),
        })
    payload = {
        "schema_version": 1,
        "authority": "MECHANICAL_STRICT_REAUDIT",
        "gates": {
            "node_recurrence": "each node supported in >=2 discovery episodes",
            "edge_recurrence": "each edge supported in >=2 discovery episodes",
            "shortcut_exclusion": (
                "node labels not exactly recoverable from one recorded "
                "skill/action/length/reward field"
            ),
            "still_required_after_acceptance": [
                "blind qualification recurrence",
                "blind held-out recurrence",
                "matched h1/h2/h4/h8 official return",
                "renamed/shuffled/random controls",
            ],
        },
        "rows": rows,
        "source_supported": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "rows": len(rows),
        "strictly_accepted": sum(
            row["strict_audit"]["accepted"] for row in rows
        ),
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
