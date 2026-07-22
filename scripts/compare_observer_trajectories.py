#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


KINDS = ("OBSERVATION", "AGENT_PROPOSAL_SET", "ENVIRONMENT_STEP")


def _load(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seeds = {
        str(row["episode_id"]): int(row["payload"]["requested_seed"])
        for row in rows if row.get("kind") == "RESET"
    }
    indexed = defaultdict(dict)
    for row in rows:
        kind = str(row.get("kind"))
        step = (row.get("payload") or {}).get("step")
        if kind in KINDS and isinstance(step, int):
            indexed[(seeds[str(row["episode_id"])], step, kind)] = row["payload"]
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("unobserved", type=Path)
    parser.add_argument("observed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, right = _load(args.unobserved), _load(args.observed)
    all_keys = sorted(set(left) | set(right))
    mismatches = []
    fields = {
        "OBSERVATION": ("observable_state_sha256", "native_actions_sha256"),
        "AGENT_PROPOSAL_SET": (
            "selected_skill_id", "selected_skill_sha256", "selected_skill_context_sha256",
        ),
        "ENVIRONMENT_STEP": ("executed_action", "reward", "terminated", "truncated"),
    }
    for seed, step, kind in all_keys:
        if (seed, step, kind) not in left or (seed, step, kind) not in right:
            mismatches.append({"seed": seed, "step": step, "kind": kind, "field": "EVENT"})
            continue
        for field in fields[kind]:
            if left[(seed, step, kind)].get(field) != right[(seed, step, kind)].get(field):
                mismatches.append({
                    "seed": seed, "step": step, "kind": kind, "field": field,
                    "unobserved": left[(seed, step, kind)].get(field),
                    "observed": right[(seed, step, kind)].get(field),
                })
    report = {
        "schema_version": 1,
        "keys_compared": len(all_keys),
        "mismatch_count": len(mismatches),
        "OBSERVER_INVARIANCE_PASS": not mismatches,
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "keys_compared", "mismatch_count", "OBSERVER_INVARIANCE_PASS"
    )}, indent=2, sort_keys=True))
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
