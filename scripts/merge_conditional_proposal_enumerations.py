#!/usr/bin/env python3
"""Merge endpoint-only retries into a frozen conditional enumeration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def _hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if not claimed or _hash(unsigned) != claimed:
        raise ValueError(f"artifact hash mismatch: {path}")
    return payload


def _index(row: dict) -> int:
    return int(row["receipt_payload"]["graph_index"])


def _is_endpoint_failure(row: dict) -> bool:
    error = str(row.get("error") or "")
    return error.startswith("HTTPStatusError:") and bool(
        re.search(r"\b(?:429|5\d\d)\b", error)
    )


def _candidate_index(candidate: dict) -> int:
    return int(str(candidate["proposal_source"]).rsplit("graph", 1)[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retry", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = _load(args.base)
    rows = {_index(row): row for row in base["rows"]}
    if len(rows) != int(base["total_eligible_graph_count"]):
        raise SystemExit("base is not a complete registered enumeration")
    candidates = {_candidate_index(row): row for row in base.get("candidates") or ()}
    retry_hashes = []
    for path in args.retry:
        retry = _load(path)
        retry_hashes.append(retry["artifact_sha256"])
        for key in ("condition", "model", "role", "demo_ids", "demo_hashes"):
            if retry.get(key) != base.get(key):
                raise SystemExit(f"retry metadata mismatch: {key}")
        retry_candidates = {
            _candidate_index(row): row for row in retry.get("candidates") or ()
        }
        for replacement in retry["rows"]:
            index = _index(replacement)
            original = rows.get(index)
            if original is None or not _is_endpoint_failure(original):
                raise SystemExit(f"retry attempted non-endpoint slot: {index}")
            if replacement.get("error") is not None or index not in retry_candidates:
                raise SystemExit(f"retry did not produce one successful candidate: {index}")
            for key in ("condition", "graph_index", "proposal_seed"):
                if replacement["receipt_payload"].get(key) != original["receipt_payload"].get(key):
                    raise SystemExit(f"retry receipt mismatch at slot {index}: {key}")
            rows[index] = replacement
            candidates[index] = retry_candidates[index]
    merged = dict(base)
    merged["rows"] = [rows[index] for index in sorted(rows)]
    merged["candidates"] = [candidates[index] for index in sorted(candidates)]
    merged["n_candidates"] = len(candidates)
    merged["n_invalid"] = sum(row.get("error") is not None for row in merged["rows"])
    merged["n_abstain"] = sum(bool(row.get("abstained")) for row in merged["rows"])
    merged["enumeration_complete"] = True
    merged["endpoint_retry_merge"] = {
        "base_artifact_sha256": base["artifact_sha256"],
        "retry_artifact_sha256s": retry_hashes,
        "rule": "replace_only_endpoint_failure_with_same_slot_seed_success",
    }
    merged.pop("artifact_sha256", None)
    merged["artifact_sha256"] = _hash(merged)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "condition": merged["condition"], "n_candidates": merged["n_candidates"],
        "n_invalid": merged["n_invalid"], "artifact_sha256": merged["artifact_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
