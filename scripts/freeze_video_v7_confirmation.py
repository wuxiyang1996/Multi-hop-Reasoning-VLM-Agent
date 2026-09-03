#!/usr/bin/env python3
"""Split the previously untouched video held-out IDs into confirmation/reserve.

The original manifest stores ten held-out IDs per family in family order.  This
script uses only IDs and manifest structure: six per family become prospective
confirmation and four remain sealed reserve.  No annotation answer or outcome
is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmation-per-family", type=int, default=6)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    heldout_per_family = int(source["counts_per_family"]["held_out"])
    take = args.confirmation_per_family
    if not 0 < take < heldout_per_family:
        raise ValueError("confirmation count must leave a nonempty reserve")
    benchmarks = {}
    for benchmark, spec in source["benchmarks"].items():
        families = list(spec["families"])
        heldout = list(spec["splits"]["held_out"])
        expected = len(families) * heldout_per_family
        if len(heldout) != expected:
            raise ValueError(f"unexpected held-out layout for {benchmark}")
        confirmation: list[str] = []
        reserve: list[str] = []
        family_roles = {}
        for index, family in enumerate(families):
            rows = heldout[
                index * heldout_per_family:(index + 1) * heldout_per_family
            ]
            chosen, remaining = rows[:take], rows[take:]
            confirmation.extend(chosen)
            reserve.extend(remaining)
            family_roles[family] = {
                "confirmation": chosen,
                "reserve": remaining,
            }
        if set(confirmation) & set(reserve):
            raise ValueError("confirmation/reserve overlap")
        benchmarks[benchmark] = {
            "families": families,
            "family_roles": family_roles,
            "splits": {
                "confirmation": confirmation,
                "reserve": reserve,
            },
        }
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_VIDEO_V7_COLLECTION",
        "selection_rule": (
            "For each benchmark family, take the first six IDs from the ten IDs "
            "already frozen as held_out in the V1 manifest; leave four as reserve."
        ),
        "outcomes_or_answers_read": False,
        "source_manifest": str(args.input.resolve()),
        "source_manifest_sha256": _sha256(args.input),
        "confirmation_per_family": take,
        "reserve_per_family": heldout_per_family - take,
        "benchmarks": benchmarks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        key: {split: len(ids) for split, ids in value["splits"].items()}
        for key, value in benchmarks.items()
    }, indent=2))


if __name__ == "__main__":
    main()
