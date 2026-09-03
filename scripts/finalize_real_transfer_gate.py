#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.real_transfer_gate import finalize_transfer_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = json.loads(Path(config["gate_report_path"]).read_text(encoding="utf-8"))
    grounder = json.loads(
        Path(config["grounder_report_path"]).read_text(encoding="utf-8")
    )
    report = finalize_transfer_gate(source, grounder)
    output = Path(config["transfer_gate_report_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["target_execution_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
