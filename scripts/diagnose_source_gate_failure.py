#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.source_gate_diagnosis import build_failure_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a failed source Phase-7 gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matched-evidence", type=Path, required=True)
    parser.add_argument("--phase7-report", type=Path, required=True)
    parser.add_argument("--no-hint", action="append", default=[], metavar="GAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    no_hint = {}
    for value in args.no_hint:
        game, separator, path = value.partition("=")
        if not separator or not game or not path:
            parser.error(f"invalid --no-hint value: {value!r}")
        no_hint[game] = Path(path)
    report = build_failure_diagnosis(
        json.loads(args.config.read_text()),
        args.matched_evidence,
        args.phase7_report,
        no_hint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "classifications": report["matched_gate"]["classifications"],
        "failure_tree": report["failure_tree"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
