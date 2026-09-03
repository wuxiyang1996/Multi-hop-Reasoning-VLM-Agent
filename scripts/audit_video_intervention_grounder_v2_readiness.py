#!/usr/bin/env python3
"""Audit whether existing video evidence can train intervention grounder V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from motif_transfer.video_intervention_grounder_v2 import (
    summarize_ledger_readiness,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/video_intervention_grounder_v2.json"
DEFAULT_OUTPUT = REPO_ROOT / "runs/video_intervention_grounder_v2_readiness/report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def audit(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_INTERVENTION_LEDGER_READINESS_AUDIT":
        raise ValueError("video V2 readiness config is not frozen")

    threshold = config["readiness_gates"]
    by_benchmark = {}
    source_receipts = {}
    for benchmark, source in sorted(config["development_sources"].items()):
        path = _resolve(source["path"])
        actual_hash = _sha256(path)
        if actual_hash != source["sha256"]:
            raise ValueError(f"{benchmark} source hash mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get(source["rows_field"], [])
        if not isinstance(rows, list):
            raise ValueError(f"{benchmark} rows field is not a list")
        summary = summarize_ledger_readiness(rows)
        summary["gates"] = {
            "minimum_eligible_records": (
                summary["eligible_records"]
                >= int(threshold["minimum_eligible_records_per_benchmark"])
            ),
            "minimum_unique_videos": (
                summary["eligible_unique_videos"]
                >= int(threshold["minimum_unique_videos_per_benchmark"])
            ),
        }
        by_benchmark[benchmark] = summary
        source_receipts[benchmark] = {
            "path": str(path.resolve()), "sha256": actual_hash,
        }

    gates = {
        "every_benchmark_has_minimum_eligible_records": all(
            value["gates"]["minimum_eligible_records"]
            for value in by_benchmark.values()
        ),
        "every_benchmark_has_minimum_unique_videos": all(
            value["gates"]["minimum_unique_videos"]
            for value in by_benchmark.values()
        ),
        "real_transition_tuples_exist": sum(
            value["eligible_records"] for value in by_benchmark.values()
        ) > 0,
    }
    ready = all(gates.values())
    return {
        "schema_version": "video-intervention-grounder-v2-readiness-v1",
        "status": (
            "READY_FOR_VIDEO_INTERVENTION_GROUNDER_V2_INDUCTION"
            if ready else "BLOCKED_NEEDS_INTERVENTION_LEDGER_COLLECTION"
        ),
        "authority": (
            "READINESS_ONLY;DEVELOPMENT_RECEIPTS;NO_GROUNDER_TRAINING;"
            "NO_FORMAL_OR_RESERVE_VIDEO_TARGETS"
        ),
        "config": {
            "path": str(config_path.resolve()), "sha256": _sha256(config_path),
        },
        "source_receipts": source_receipts,
        "by_benchmark": by_benchmark,
        "gates": gates,
        "next_legal_step": (
            "Freeze train/held-out video identities and induce V2 with shuffled-effect controls."
            if ready else
            "Collect development-only state/intervention/effect/next-state ledgers under the frozen V2 schema; do not train on legacy static QA receipts."
        ),
        "claim_boundary": config["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "gates": report["gates"],
        "eligible": {
            key: value["eligible_records"]
            for key, value in report["by_benchmark"].items()
        },
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
