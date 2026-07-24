#!/usr/bin/env python3
"""Re-run deterministic Harness audit on saved untrusted model candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import (
    DecisionStepSignature,
    Lifecycle,
    MotifCandidate,
    MotifEdge,
    MotifNode,
    SourceStepSignature,
)
from motif_transfer.harness import DeterministicHarness
from motif_transfer.instrumented_import import import_native_source_batch


def _candidate(raw):
    nodes = []
    for node in raw["nodes"]:
        signatures = []
        for signature in node.get("decision_signatures", []):
            if "proposal_count" in signature:
                signatures.append(DecisionStepSignature(**signature))
            else:
                signatures.append(SourceStepSignature(**signature))
        nodes.append(MotifNode(
            str(node["node_id"]),
            tuple(node.get("transition_receipt_ids", [])),
            tuple(signatures),
        ))
    edges = tuple(MotifEdge(
        str(edge["source"]),
        str(edge["target"]),
        tuple(edge.get("replay_receipt_ids", [])),
        str(edge.get("untrusted_claim", "")),
    ) for edge in raw["edges"])
    return MotifCandidate(
        str(raw["motif_id"]),
        tuple(raw.get("source_lineage", [])),
        tuple(nodes),
        edges,
        Lifecycle(str(raw.get("status", "CANDIDATE"))),
        str(raw.get("untrusted_description", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("evidence_dir")
    parser.add_argument("--supplemental-replays")
    parser.add_argument("--supplemental-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = import_native_source_batch(
        args.evidence_dir,
        args.supplemental_replays,
        include_base_replays=not args.supplemental_only,
    )
    imported = {episode.episode_id: episode for episode in episodes}
    harness = DeterministicHarness()
    rows = []
    for source_row in report["episodes"]:
        episode = imported[str(source_row["episode_id"])]
        receipt_map = {
            row.transition.receipt_id: row.transition for row in episode.records
        } | {row.receipt_id: row for row in episode.replay_forks}
        audits = [
            harness.audit_motif(_candidate(raw), receipt_map)
            for raw in source_row.get("candidates", [])
        ]
        rows.append({
            "episode_id": episode.episode_id,
            "audits": [asdict(row) for row in audits],
        })
    result = {
        "schema_version": 1,
        "authority": "DETERMINISTIC_REAUDIT_ONLY",
        "input_report": str(report_path.resolve()),
        "input_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "prompt_condition": report.get("prompt_condition"),
        "episodes": rows,
        "totals": {
            "candidates": sum(len(row["audits"]) for row in rows),
            "accepted": sum(
                audit["accepted"] for row in rows for audit in row["audits"]
            ),
        },
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
