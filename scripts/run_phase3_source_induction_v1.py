#!/usr/bin/env python3
"""Run source-only anonymous symbolic-program induction on frozen receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_source_induction import (  # noqa: E402
    build_lineage_report,
    file_sha256,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _read(args.config)
    if config.get("schema_version") != "phase3-source-induction-config-v1":
        raise SystemExit("unsupported Phase-3 source induction config")
    lineages = list(config.get("source_lineages") or ())
    if len(lineages) != 6 or len({row["source_game"] for row in lineages}) != 6:
        raise SystemExit("Phase-3 V1 requires exactly six source lineages")

    reports = []
    for lineage in lineages:
        rows_path = REPO / str(lineage["rows_path"])
        actual_hash = file_sha256(rows_path)
        if actual_hash != str(lineage["rows_file_sha256"]):
            raise SystemExit(f"source rows changed: {rows_path}")
        report = build_lineage_report(
            source_game=str(lineage["source_game"]),
            rows_path=str(lineage["rows_path"]),
            primary_horizon=int(lineage["primary_horizon"]),
            thresholds=config["thresholds"],
        )
        reports.append(report)
        _write(
            args.output_dir / "lineages" / f"{lineage['source_game']}.json",
            report,
        )

    structures = {
        report["source_game"]: stable_hash([
            {
                "preconditions": row["preconditions"],
                "state_delta": row["state_delta"],
            }
            for row in report["authentic_program"]["operators"]
        ])
        for report in reports
    }
    profiles = {
        report["source_game"]: report["source_profile"]["profile_sha256"]
        for report in reports
    }
    gates = {
        "exact_six_lineages": len(reports) == 6,
        "all_source_only_programs_qualified": all(
            report["authentic_program"]["status"]
            == "SOURCE_INDUCED_PROGRAM_QUALIFIED"
            for report in reports
        ),
        "all_historical_heldout_gates_pass": all(
            report["status"] == "SOURCE_ONLY_INDUCTION_HELDOUT_VALIDATED"
            for report in reports
        ),
        "all_authentic_strictly_beat_shuffled": all(
            report["gates"]["heldout_authentic_strictly_beats_shuffled"]
            for report in reports
        ),
        "all_source_profiles_content_distinct": len(set(profiles.values())) == 6,
        # This is diagnostic, not required for the shared-structure claim.
        "operator_structures_content_distinct": len(set(structures.values())) == 6,
    }
    required = (
        "exact_six_lineages",
        "all_source_only_programs_qualified",
        "all_historical_heldout_gates_pass",
        "all_authentic_strictly_beat_shuffled",
    )
    body = {
        "schema_version": "phase3-source-induction-summary-v1",
        "status": (
            "PHASE3_SOURCE_INDUCTION_HISTORICAL_REPLAY_VALIDATED"
            if all(gates[key] for key in required)
            else "PHASE3_SOURCE_INDUCTION_HISTORICAL_REPLAY_NOT_VALIDATED"
        ),
        "config_file_sha256": file_sha256(args.config),
        "config_status": config["status"],
        "lineages": [
            {
                "source_game": report["source_game"],
                "status": report["status"],
                "report_sha256": report["report_sha256"],
                "program_sha256": report["authentic_program"]["program_sha256"],
                "operator_structure_sha256": structures[report["source_game"]],
                "source_profile_sha256": profiles[report["source_game"]],
                "heldout": report["heldout"],
            }
            for report in reports
        ],
        "gates": gates,
        "required_gates": list(required),
        "claim_boundary": (
            "RETROSPECTIVE_REUSE_OF_PREVIOUSLY_EXECUTED_SOURCE_HELDOUT_ROWS;"
            "VALIDATES_INDUCTION_IMPLEMENTATION_AND_SHUFFLED_CONTROL_ONLY;"
            "DOES_NOT_COUNT_AS_NEW_PROSPECTIVE_SOURCE_CONFIRMATION_OR_TARGET_UTILITY"
        ),
    }
    summary = body | {"summary_sha256": stable_hash(body)}
    _write(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
