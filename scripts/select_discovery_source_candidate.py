#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.discovery_selection import select_from_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze one source candidate using discovery content only."
    )
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select_from_evidence(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_skill_id": report["selected_skill_id"],
        "status": report["status"],
        "selection_receipt_sha256": report["selection_receipt_sha256"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
