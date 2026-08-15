#!/usr/bin/env python3
"""Write the fail-closed Phase-1 six-game by four-target transfer audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.phase1_six_game_transfer_audit import (
    build_phase1_six_game_four_target_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = build_phase1_six_game_four_target_audit(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "audit_sha256": report["audit_sha256"],
        "output": str(args.output.resolve()),
        "status": report["status"],
        "validated_cells": report["aggregate"][
            "validated_phase1_transfer_cells"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

