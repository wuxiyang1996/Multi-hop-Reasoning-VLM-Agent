#!/usr/bin/env python3
"""Run the controlled intervention-grounded game-to-diagnosis pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.controlled_exploration_transfer import run_controlled_transfer  # noqa: E402
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "controlled_exploration_transfer_v1_discovery.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "runs" / "controlled_exploration_transfer_v1_discovery",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "config_sha256" in config:
        body = dict(config)
        claimed = str(body.pop("config_sha256"))
        if stable_hash(body) != claimed:
            raise SystemExit("invalid frozen controlled-transfer config hash")
    for receipt in config.get("implementation", {}).values():
        path = Path(str(receipt["path"]))
        if _sha256(path) != str(receipt["file_sha256"]):
            raise SystemExit(f"frozen implementation changed: {path}")
    if (
        str(config.get("status", "")).startswith("FROZEN_BEFORE_")
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise SystemExit(f"refusing to overwrite frozen run: {args.output_dir}")
    report = run_controlled_transfer(config)
    core_path = REPO / "src/motif_transfer/controlled_exploration_transfer.py"
    report["implementation"] = {
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha256(Path(__file__)),
        "core_path": str(core_path.resolve()),
        "core_sha256": _sha256(core_path),
    }
    report["config"] = {
        "path": str(config_path),
        "sha256": _sha256(config_path),
        "content": config,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = report.pop("episode_rows")
    rows_path = args.output_dir / "episode_rows.json"
    rows_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    report["episode_rows"] = {
        "path": str(rows_path.resolve()),
        "sha256": _sha256(rows_path),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "gate": report["gate"],
        "source_value_mse": report["source_value_mse"],
        "report": str(report_path.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
