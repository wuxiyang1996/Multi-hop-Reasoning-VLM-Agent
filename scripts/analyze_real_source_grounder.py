#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.candy_native_grounder import run_source_grounder_gate
from motif_transfer.real_source_interventions import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_source_grounder_gate(
        read_jsonl(Path(config["receipts_path"])), str(config["namespace"])
    )
    output = Path(config["grounder_report_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "SOURCE_GROUNDER_GATE_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
