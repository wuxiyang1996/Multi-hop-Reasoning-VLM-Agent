#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.instrumented_import import import_native_source_batch
from motif_transfer.source_decision_cycles import (
    build_decision_traces,
    structural_affordance_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whole skill-conditioned source decision cycles"
    )
    parser.add_argument("evidence_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for evidence_dir in args.evidence_dirs:
        episodes = import_native_source_batch(evidence_dir)
        traces = build_decision_traces(episodes)
        reports.append({
            "evidence_dir": str(evidence_dir.resolve()),
            "import_gaps": sum(len(episode.gaps) for episode in episodes),
            "episodes": len(episodes),
            "report": structural_affordance_report(traces),
        })
    payload = {
        "schema_version": 1,
        "authority": "MECHANICAL_ANONYMOUS_EVENT_AUDIT_ONLY",
        "datasets": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "datasets": len(reports),
        "episodes": sum(row["episodes"] for row in reports),
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
