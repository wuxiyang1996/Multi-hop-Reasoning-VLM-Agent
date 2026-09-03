#!/usr/bin/env python3
"""Audit AGQA/CLEVRER family coverage before any new target reserve is opened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from motif_transfer.video_target_signature_binding import (
    authorize_target_signature,
    permuted_algebra,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algebra", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    algebra = json.loads(args.algebra.read_text(encoding="utf-8"))
    bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
    permuted = permuted_algebra(algebra)
    receipts = []
    for domain in ("agqa", "clevrer"):
        for family in sorted(bindings[domain]):
            authentic = authorize_target_signature(
                algebra=algebra, binding_spec=bindings,
                target_domain=domain, question_family=family,
            )
            # Applicability is intentionally inventory-based.  The permuted
            # control remains applicable but its type-safe composition graph
            # differs and is evaluated only after grounding is frozen.
            permuted_auth = authorize_target_signature(
                algebra=permuted, binding_spec=bindings,
                target_domain=domain, question_family=family,
            )
            receipts.append({
                "domain": domain,
                "family": family,
                "authentic": asdict_safe(authentic),
                "permuted": asdict_safe(permuted_auth),
            })
    all_authorized = all(row["authentic"]["status"] == "AUTHORIZED" for row in receipts)
    body = {
        "schema_version": "video-target-signature-coverage-v1",
        "status": "TARGET_SIGNATURE_COVERAGE_PASSED" if all_authorized else "TARGET_SIGNATURE_COVERAGE_FAILED",
        "authority": "SOURCE_ALGEBRA_PLUS_PUBLIC_TARGET_SCHEMA;NO_TARGET_OUTCOME",
        "source_algebra_sha256": algebra["artifact_sha256"],
        "binding_spec_sha256": stable_hash(bindings),
        "authentic_composition_edges": algebra["composition_edge_count"],
        "permuted_composition_edges": permuted["composition_edge_count"],
        "receipts": receipts,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: body[k] for k in (
        "status", "source_algebra_sha256", "authentic_composition_edges",
        "permuted_composition_edges", "report_sha256"
    )}, indent=2))
    return 0 if all_authorized else 1


def asdict_safe(value: object) -> dict:
    from dataclasses import asdict
    return asdict(value)


if __name__ == "__main__":
    raise SystemExit(main())
