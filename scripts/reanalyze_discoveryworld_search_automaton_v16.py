#!/usr/bin/env python3
"""Hash-locked V16 equivalence replay of DiscoveryWorld replication V1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.discoveryworld_search_automaton_v16 import (  # noqa: E402
    evaluate_discovery_relineage,
)
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: dict, field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-artifact", type=Path,
        default=REPO / "runs/sokoban_search_automaton_v16/artifact.json",
    )
    parser.add_argument(
        "--historical-summary", type=Path,
        default=REPO / "docs/results/discoveryworld_replication_v1_summary.json",
    )
    parser.add_argument(
        "--historical-root", type=Path,
        default=REPO / "runs/discoveryworld_replication_v1_matched",
    )
    parser.add_argument(
        "--output", type=Path,
        default=(
            REPO
            / "runs/discoveryworld_search_automaton_v16/equivalence_report.json"
        ),
    )
    args = parser.parse_args()
    source = SourceSearchAutomaton(json.loads(
        args.source_artifact.read_text(encoding="utf-8")
    ))
    summary = json.loads(args.historical_summary.read_text(encoding="utf-8"))
    _self_hash(summary, "report_sha256")
    results = []
    for task_id in map(str, summary["eligible_task_ids"]):
        expected = str(summary["integrity"]["result_file_sha256"][task_id])
        matches = [
            path for path in args.historical_root.rglob(f"{task_id}.json")
            if _file_sha256(path) == expected
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"expected one hash-matched result for {task_id}; got {matches}"
            )
        result = json.loads(matches[0].read_text(encoding="utf-8"))
        _self_hash(result, "result_sha256")
        results.append(result)
    report = evaluate_discovery_relineage(
        source=source, summary=summary, results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "historical_success_counts": report["historical_success_counts"],
        "paired": report["paired"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
