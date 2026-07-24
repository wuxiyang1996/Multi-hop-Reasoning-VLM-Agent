#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.phase1_assets import (
    audit_batches,
    audit_checkpoint_manifest,
    discover_evidence_batches,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase-1 evidence without trusting skill text")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--checkpoint-manifest", action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args()

    batches = discover_evidence_batches(args.evidence_root)
    report = audit_batches(batches)
    report["checkpoint_manifests"] = [
        audit_checkpoint_manifest(path) for path in args.checkpoint_manifest
    ]
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
