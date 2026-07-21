#!/usr/bin/env python3
"""Merge complete enumeration with endpoint-only retries without outcome selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != _hash(unsigned):
        raise ValueError(f"enumeration artifact hash mismatch: {path}")
    payload.setdefault(
        "total_eligible_graph_count", payload["registered_graph_count"],
    )
    payload.setdefault(
        "selected_graph_indices", list(range(payload["registered_graph_count"])),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [_load(path) for path in args.artifact]
    identity_keys = (
        "condition", "source_treatment", "model", "role", "demo_ids", "demo_hashes",
        "total_eligible_graph_count",
    )
    baseline = payloads[0]
    if any(any(payload.get(key) != baseline.get(key) for key in identity_keys)
           for payload in payloads[1:]):
        raise SystemExit("enumeration retry protocol identity mismatch")
    attempts = {}
    graph_registry = {}
    candidates_by_index = {}
    for payload in payloads:
        indices = payload.get("selected_graph_indices")
        if indices is None:
            indices = list(range(len(payload["rows"])))
        if len(indices) != len(payload["rows"]) or len(indices) != len(payload["source_graphs"]):
            raise SystemExit("enumeration artifact index cardinality mismatch")
        candidate_by_index = {
            int(candidate["proposal_source"].rsplit("graph", 1)[1]): candidate
            for candidate in payload["candidates"]
        }
        for index, row, graph in zip(indices, payload["rows"], payload["source_graphs"]):
            attempts.setdefault(index, []).append(row)
            if index in graph_registry and graph_registry[index] != graph:
                raise SystemExit("graph identity changed across endpoint retry")
            graph_registry[index] = graph
            if row["error"] is None:
                candidate = candidate_by_index.get(index)
                if candidate is None:
                    raise SystemExit("successful enumeration row lacks parsed candidate")
                if index in candidates_by_index:
                    raise SystemExit("multiple successful attempts for one slot are not selectable")
                candidates_by_index[index] = candidate
    expected = set(range(int(baseline["total_eligible_graph_count"])))
    if set(attempts) != expected or set(graph_registry) != expected:
        raise SystemExit("merged enumeration does not cover every registered slot")
    chosen_rows = []
    for index in sorted(expected):
        successful = [row for row in attempts[index] if row["error"] is None]
        if len(successful) == 1:
            chosen_rows.append(successful[0])
            continue
        errors = [str(row["error"] or "") for row in attempts[index]]
        if successful or not all("429 Too Many Requests" in error for error in errors):
            raise SystemExit("retry merge is allowed only for unresolved endpoint 429 rows")
        chosen_rows.append(attempts[index][-1])
    output = {
        key: baseline[key] for key in (
            "schema_version", "candidate_source", "condition", "source_treatment",
            "source_control_receipt", "source_graph_edge_evidence_gate", "demo_ids",
            "demo_hashes", "model", "role", "semantic_alignment_claimed",
        )
    }
    output.update({
        "source_graphs": [graph_registry[index] for index in sorted(expected)],
        "enumeration_complete": True,
        "total_eligible_graph_count": len(expected),
        "selected_graph_indices": sorted(expected),
        "registered_graph_count": len(expected), "rows": chosen_rows,
        "attempt_rows": [row for index in sorted(expected) for row in attempts[index]],
        "candidates": [candidates_by_index[index] for index in sorted(candidates_by_index)],
    })
    output["n_candidates"] = len(output["candidates"])
    output["n_abstain"] = sum(bool(row["abstained"]) for row in chosen_rows)
    output["n_invalid"] = sum(row["error"] is not None for row in chosen_rows)
    output["artifact_sha256"] = _hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "condition": output["condition"], "n_candidates": output["n_candidates"],
        "n_invalid": output["n_invalid"], "n_attempts": len(output["attempt_rows"]),
        "artifact_sha256": output["artifact_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
