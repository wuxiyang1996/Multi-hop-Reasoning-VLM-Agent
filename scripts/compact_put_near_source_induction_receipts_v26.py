#!/usr/bin/env python3
"""Create a portable receipt for the PutNear source induction inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


DEFAULT_OUTPUT = (
    REPO / "docs/results/put_near_source_induction_receipts_v26.json"
)


INPUTS = {
    "discovery": (
        "runs/source_structural_v5b_development/put_near/seed_21.json",
        "runs/source_structural_v5b_development/put_near/seed_22.json",
    ),
    "source_qualification": (
        "runs/source_structural_v5b_development/put_near/seed_23.json",
        "runs/source_structural_v5b_development/put_near/seed_24.json",
    ),
    "fresh_confirmation": (
        "runs/source_structural_v5c_fresh/put_near/seed_31.json",
        "runs/source_structural_v5c_fresh/put_near/seed_32.json",
        "runs/source_structural_v5c_fresh/put_near/seed_33.json",
        "runs/source_structural_v5c_fresh/put_near/seed_34.json",
    ),
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError(f"invalid {field}")


def _validate_collection(collection: Mapping[str, Any]) -> None:
    _self_hash(collection, "collection_sha256")
    if collection.get("task_id") != "put_near":
        raise ValueError("compact input is not PutNear")
    for path in collection.get("paths") or ():
        _self_hash(path, "path_sha256")
        for step in path.get("steps") or ():
            _self_hash(step, "transition_sha256")


def compact() -> dict[str, Any]:
    roles = {}
    input_hashes = {}
    for role, relatives in INPUTS.items():
        rows = []
        for relative in relatives:
            path = REPO / relative
            collection = _read(path)
            _validate_collection(collection)
            input_hashes[relative] = _sha(path)
            rows.append({
                "seed": int(collection["seed"]),
                "split": str(collection["split"]),
                "collection_sha256": str(collection["collection_sha256"]),
                "paths": [dict(row) for row in collection["paths"]],
            })
        roles[role] = rows
    body = {
        "schema_version": "put-near-source-induction-portable-receipt-v1",
        "status": "PORTABLE_SOURCE_ONLY_INDUCTION_RECEIPT",
        "source_task": "put_near",
        "authority": "SOURCE_STATE_ACTION_NEXT_STATE_DELTAS_ONLY",
        "roles": roles,
        "input_file_sha256": input_hashes,
        "raw_source_action_tokens_exported": False,
        "target_data_read": False,
    }
    return body | {"receipt_sha256": stable_hash(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite compact receipt: {output}")
    receipt = compact()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": receipt["status"],
        "receipt_sha256": receipt["receipt_sha256"],
        "role_collections": {
            key: len(value) for key, value in receipt["roles"].items()
        },
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
