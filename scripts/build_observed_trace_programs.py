#!/usr/bin/env python3
"""Build full-episode, non-segmented source TracePrograms and replay them."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skill_bank.trace_program_validator import (  # noqa: E402
    TraceProgramValidator,
    compile_observed_episode,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path)
    source.add_argument(
        "--source-file", type=Path, action="append",
        help="Exact preregistered episode path; repeat for multiple source games.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = (
        sorted(args.source_file)
        if args.source_file
        else sorted(args.source_root.glob("*/episode_*.json"))
    )
    if not paths:
        raise SystemExit("no source episodes selected")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"source episodes do not exist: {missing}")
    validator = TraceProgramValidator()
    programs = []
    receipts = []
    for path in paths:
        program = compile_observed_episode(path)
        replay = validator.validate(program, path)
        programs.append(program.to_dict())
        receipts.append({
            "program_id": replay.program_id,
            "passed": replay.passed,
            "verified_transitions": replay.verified_transitions,
            "failures": list(replay.failures),
            "source_path": str(path),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in programs),
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    report = {
        "schema_version": 1,
        "program_kind": "full_episode_observed_trace",
        "segmentation": "none",
        "n_programs": len(programs),
        "n_transitions": sum(len(item["transitions"]) for item in programs),
        "n_replay_pass": sum(bool(item["passed"]) for item in receipts),
        "n_replay_fail": sum(not bool(item["passed"]) for item in receipts),
        "reasoning_claim": "none_observational_trace_only",
        "official_success_claim": False,
        "source_selection": (
            "exact_preregistered_paths" if args.source_file else "sorted_root_glob"
        ),
        "source_paths": [str(path) for path in paths],
        "receipts": receipts,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "n_programs", "n_transitions", "n_replay_pass", "n_replay_fail",
        "reasoning_claim", "official_success_claim",
    )}, indent=2))
    return 0 if report["n_replay_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
