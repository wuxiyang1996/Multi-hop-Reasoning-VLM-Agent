#!/usr/bin/env python3
"""Replay every canonical source program against immutable source episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skill_bank.program_ir import canonical_program_from_dict  # noqa: E402
from skill_bank.source_replay_validator import SourceReplayValidator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--programs",
        type=Path,
        default=REPO_ROOT / "artifacts/source_evidence_index/source_programs.jsonl",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "labeling/gpt54_skill_labeled",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/source_evidence_index/replay_receipts.jsonl",
    )
    args = parser.parse_args()
    validator = SourceReplayValidator(args.source_root)
    receipts = []
    with args.programs.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                receipts.append(validator.validate(canonical_program_from_dict(json.loads(line))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for receipt in receipts:
            handle.write(json.dumps(receipt.to_dict(), sort_keys=True) + "\n")
    tmp.replace(args.output)
    n_pass = sum(item.passed for item in receipts)
    n_evidence = sum(item.n_evidence for item in receipts)
    n_verified = sum(item.n_verified for item in receipts)
    print(json.dumps({
        "n_programs": len(receipts),
        "n_pass": n_pass,
        "n_fail": len(receipts) - n_pass,
        "n_evidence": n_evidence,
        "n_verified": n_verified,
        "output": str(args.output),
    }, indent=2))
    return 0 if n_pass == len(receipts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
