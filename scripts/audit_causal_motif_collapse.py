#!/usr/bin/env python3
"""Audit legacy mega-skill collapse and compile receipt-grounded motif candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.causal_reasoning_motif import (  # noqa: E402
    LegacyMegaSkillLineage,
    audit_motif_anti_collapse,
    compile_causal_motif,
    motif_conditioning_view,
)
from scripts.propose_alfworld_multistep_bindings_35b import _source_graphs  # noqa: E402


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _legacy_report(path: Path) -> tuple[dict[str, Any], list[LegacyMegaSkillLineage]]:
    rows = _load_jsonl(path)
    source_hash = _file_hash(path)
    lineages = [
        LegacyMegaSkillLineage.from_record(row, source_artifact_sha256=source_hash)
        for row in rows
    ]
    signatures = Counter(item.legacy_template_signature for item in lineages)
    event_vocabulary = {
        token.strip()
        for signature in signatures
        for token in signature.split("→")
        if token.strip()
    }
    members = sum(len(item.member_refs) for item in lineages)
    largest = max((len(item.member_refs) for item in lineages), default=0)
    return {
        "path": str(path),
        "file_sha256": source_hash,
        "families": len(rows),
        "members": members,
        "unique_template_signatures": len(signatures),
        "event_vocabulary": sorted(event_vocabulary),
        "largest_family_members": largest,
        "largest_family_fraction": largest / members if members else 0.0,
        "families_with_executable_control_flow_receipts": sum(
            any(key in row for key in ("edges", "branches", "control_flow", "replan"))
            for row in rows
        ),
        "authority": "LINEAGE_RETRIEVAL_ONLY",
        "collapse_warning": (
            "Legacy semantic clustering is not receipt-grounded executable transfer evidence."
        ),
    }, lineages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-megaskills", type=Path, required=True)
    parser.add_argument("--source-hypotheses", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.legacy_megaskills.is_file():
        parser.error(f"missing legacy mega-skills: {args.legacy_megaskills}")
    legacy, lineages = _legacy_report(args.legacy_megaskills)
    motifs = []
    if args.source_hypotheses is not None:
        payload = json.loads(args.source_hypotheses.read_text(encoding="utf-8"))
        # Lineage remains an index only. Do not infer which old family is the
        # correct semantic parent of a new causal motif.
        graphs = _source_graphs(payload, require_agent_reasoning_receipts=True)
        for graph in graphs:
            motif = compile_causal_motif(graph)
            audit = audit_motif_anti_collapse(motif)
            motifs.append({
                "motif": motif.transferable_view(),
                "motif_hash": motif.content_hash(),
                "causal_fingerprint_sha256": motif.causal_fingerprint(),
                "anti_collapse_audit": asdict(audit),
                "registered_conditioning_controls": {
                    treatment: motif_conditioning_view(
                        motif, treatment=treatment, seed=1729,
                    )
                    for treatment in (
                        "authentic", "generic_protocol", "shuffled_topology",
                        "receipt_null",
                    )
                },
            })
    output = {
        "schema_version": 1,
        "legacy": legacy,
        "legacy_lineage_count": len(lineages),
        "causal_motifs": motifs,
        "summary": {
            "n_causal_motifs": len(motifs),
            "n_structurally_specific_candidates": sum(
                row["anti_collapse_audit"]["status"]
                == "STRUCTURALLY_SPECIFIC_CANDIDATE"
                for row in motifs
            ),
            "source_attribution_verified": False,
            "target_incremental_value_verified": False,
        },
        "claim_limit": (
            "This audit detects representational collapse. It does not use legacy labels, "
            "Agent prose, or a GPT verdict as semantic/causal truth."
        ),
    }
    output["artifact_sha256"] = _hash(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
