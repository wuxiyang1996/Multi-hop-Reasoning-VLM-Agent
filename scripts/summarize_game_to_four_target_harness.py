#!/usr/bin/env python3
"""Create the auditable game-to-four-target transfer matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.cross_benchmark_transfer_audit import (  # noqa: E402
    build_cross_benchmark_audit,
)


def _read(relative: str) -> dict:
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alfworld-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_cross_benchmark_audit(
        source_receipt=_read(
            "docs/results/sokoban_effect_program_v2_compact_receipt.json"
        ),
        webshop=_read(
            "docs/results/webshop_sokoban_effect_transfer_v13_summary.json"
        ),
        discovery_adaptation=_read(
            "docs/results/discoveryworld_v18_v21_adaptation_summary.json"
        ),
        discovery_formal=_read(
            "docs/results/discoveryworld_v22_normal_formal_early_stop.json"
        ),
        alfworld=json.loads(args.alfworld_report.read_text(encoding="utf-8")),
        tir=_read(
            "docs/results/sokoban_tir_effect_v5_consumed_summary.json"
        ),
        tir_other_source_formal=_read(
            "runs/active_tir_wrapper_neurosymbolic_v3_formal/qualification_report.json"
        ),
        tir_target_diagnosis=_read(
            "runs/active_tir_wrapper_gpt41mini_v4_development/adaptation_report.json"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "validated_mechanism_cells": report["validated_mechanism_cells"],
        "validated_heldout_cells": report["validated_heldout_cells"],
        "all_four_share_one_source_artifact": report[
            "all_four_share_one_source_artifact"
        ],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0 if report["status"] == "ALL_FOUR_TARGETS_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
