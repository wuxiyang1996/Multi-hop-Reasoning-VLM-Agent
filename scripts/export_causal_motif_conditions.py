#!/usr/bin/env python3
"""Export a frozen five-condition target pilot from a causal-motif audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    payload["artifact_hash"] = _hash(payload)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--motif-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    claimed = str(audit.get("artifact_sha256") or "")
    unsigned = dict(audit)
    unsigned.pop("artifact_sha256", None)
    if not claimed or _hash(unsigned) != claimed:
        raise SystemExit("causal motif audit hash mismatch")
    motifs = list(audit.get("causal_motifs") or ())
    if not 0 <= args.motif_index < len(motifs):
        raise SystemExit("motif index out of range")
    selected = motifs[args.motif_index]
    if selected["anti_collapse_audit"]["status"] != "STRUCTURALLY_SPECIFIC_CANDIDATE":
        raise SystemExit("selected motif did not pass structural anti-collapse gates")
    controls = selected["registered_conditioning_controls"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for treatment in (
        "authentic", "generic_protocol", "shuffled_topology", "receipt_null",
    ):
        control = controls[treatment]
        payload = {
            "schema_version": 1,
            "protocol": "causal_motif_conditioning_v1",
            "treatment": treatment,
            "status": "READY",
            "source_audit_sha256": claimed,
            "source_refs": [selected["motif"]["motif_id"]],
            "source_contexts": [control["conditioning"]],
            "conditioning_sha256": control["conditioning_sha256"],
            "source_attribution_verified": False,
            "target_incremental_value_verified": False,
        }
        _write(args.output_dir / f"{treatment}.json", payload)
    other = next((
        row for index, row in enumerate(motifs)
        if index != args.motif_index
        and row.get("causal_fingerprint_sha256")
        != selected.get("causal_fingerprint_sha256")
        and row.get("anti_collapse_audit", {}).get("status")
        == "STRUCTURALLY_SPECIFIC_CANDIDATE"
    ), None)
    if other is None:
        raise SystemExit("no structurally distinct other-source motif is available")
    other_control = other["registered_conditioning_controls"]["authentic"]
    _write(args.output_dir / "other_source.json", {
        "schema_version": 1,
        "protocol": "causal_motif_conditioning_v1",
        "treatment": "other_source",
        "status": "READY",
        "source_audit_sha256": claimed,
        "source_refs": [other["motif"]["motif_id"]],
        "source_contexts": [other_control["conditioning"]],
        "conditioning_sha256": other_control["conditioning_sha256"],
        "source_attribution_verified": False,
        "target_incremental_value_verified": False,
    })
    target_only = {
        "schema_version": 1,
        "protocol": "causal_motif_conditioning_v1",
        "treatment": "target_only",
        "status": "READY",
        "source_audit_sha256": claimed,
        "source_refs": [],
        "source_contexts": [],
        "conditioning_sha256": _hash([]),
        "source_attribution_verified": False,
        "target_incremental_value_verified": False,
    }
    _write(args.output_dir / "target_only.json", target_only)
    manifest = {
        "schema_version": 1,
        "source_audit_sha256": claimed,
        "selected_motif_index": args.motif_index,
        "selected_motif_hash": selected["motif_hash"],
        "conditions": [
            "authentic", "generic_protocol", "shuffled_topology", "receipt_null",
            "other_source", "target_only",
        ],
        "claim_limit": (
            "Frozen conditioning artifacts only; no source attribution or target value "
            "is established by export."
        ),
    }
    _write(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "output_dir": str(args.output_dir), "conditions": manifest["conditions"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
