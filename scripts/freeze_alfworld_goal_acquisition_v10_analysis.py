#!/usr/bin/env python3
"""Freeze the deterministic V10 gate-scope correction inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    output = REPO / "configs/alfworld_goal_acquisition_v10_analysis.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite V10 analysis config: {output}")
    analyzer = REPO / "scripts/analyze_alfworld_goal_acquisition_v10.py"
    source = REPO / "runs/alfworld_goal_acquisition_v10_development/report.json"
    body = {
        "schema_version": "alfworld-goal-acquisition-v10-analysis-config-v1",
        "status": "FROZEN_BEFORE_DETERMINISTIC_GATE_SCOPE_CORRECTION",
        "source_report": str(source.relative_to(REPO)),
        "source_report_file_sha256": _sha256(source),
        "analyzer_file_sha256": _sha256(analyzer),
        "output": (
            "runs/alfworld_goal_acquisition_v10_development/analysis_report.json"
        ),
    }
    config = body | {"config_sha256": stable_hash(body)}
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
