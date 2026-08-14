#!/usr/bin/env python3
"""Freeze outcome-blind DiscoveryWorld scenario/seed roles before model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


OFFICIAL_COMMIT = "fd591323920be0d3786ef350955de1945aa571e5"
SCENARIOS = (
    "Combinatorial Chemistry",
    "Archaeology Dating",
    "Plant Nutrients",
    "Reactor Lab",
    "Lost in Translation",
    "Space Sick",
    "Proteomics",
    "It's (not) Rocket Science!",
)


def _hash(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roles = {
        "development": [
            {"scenario": scenario, "difficulty": "Easy", "seed": 0}
            for scenario in SCENARIOS
        ],
        "qualification": [
            {"scenario": scenario, "difficulty": "Easy", "seed": 1}
            for scenario in SCENARIOS
        ],
        "formal_reserve": [
            {"scenario": scenario, "difficulty": difficulty, "seed": seed}
            for difficulty in ("Easy", "Normal")
            for seed in (2, 3, 4)
            for scenario in SCENARIOS
        ],
    }
    payload = {
        "schema_version": "discoveryworld-manifest-v1",
        "official_environment_commit": OFFICIAL_COMMIT,
        "assignment_rule": (
            "All eight official scientific themes; seed 0 development, seed 1 "
            "qualification, seeds 2-4 Easy/Normal sealed reserve. No task outcome, "
            "scorecard, world content, or rollout was consulted."
        ),
        "roles": roles,
        "counts": {key: len(value) for key, value in roles.items()},
        "formal_reserve_read": False,
    }
    payload["manifest_sha256"] = _hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": payload["counts"], "manifest_sha256": payload["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
