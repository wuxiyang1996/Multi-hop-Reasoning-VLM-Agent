#!/usr/bin/env python3
"""Freeze zero-trajectory target-schema LLM synthesis calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen config: {output}")
    paths = {
        "synthesis_runtime": "src/motif_transfer/target_schema_synthesis.py",
        "runner": "scripts/run_target_schema_synthesis_baseline_v29.py",
    }
    body = {
        "schema_version": "target-schema-synthesis-v29-protocol",
        "status": "FROZEN_BEFORE_TARGET_SCHEMA_CALLS",
        "claim_boundary": (
            "A target-only pretrained LLM receives the frozen target interface "
            "description and shared IR grammar but zero complete target "
            "trajectories, rewards, outcomes, source receipts, source programs, "
            "or answer keys. Success proves source provenance is not necessary "
            "for that target/program; failure applies only to this frozen model, "
            "prompt, and call budget and does not rule out humans or other LLMs."
        ),
        **paths,
        **{
            f"{field}_file_sha256": _sha(REPO / path)
            for field, path in paths.items()
        },
        "dependency_fields": {
            f"{field}_file_sha256": field for field in paths
        },
        "model": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "id": "openai/gpt-4.1-mini",
            "temperature": 0,
            "maximum_output_tokens": 220,
            "timeout_seconds": 180,
            "max_retries": 2,
        },
        "replicates_per_target": 4,
        "workers": 4,
        "targets": {
            "alfworld": {
                "interface_description": (
                    "The agent observes textual rooms, inventory, objects, "
                    "receptacles, and a natural-language goal. Native actions "
                    "may move, inspect, take, open, modify, and place objects. "
                    "A target-native grounder may label candidate effects with "
                    "the shared typed operator vocabulary. Several requested "
                    "object-goal relations can remain unsatisfied at once."
                ),
            },
            "discoveryworld": {
                "interface_description": (
                    "The agent observes a partially visible world, inventory, "
                    "nearby entities, tools, and dialog. Native actions include "
                    "move, inspect, take, put, use, activate, and talk. A "
                    "target-native grounder can expose anonymous entity-slot "
                    "cardinality deltas and observation-relation deltas."
                ),
            },
            "tir_rotation": {
                "interface_description": (
                    "An image has an unknown cyclic orientation displacement. "
                    "A target-native grounder can compare anonymous candidate "
                    "transformations and identify candidates that restore "
                    "physical uprightness, without exposing numeric angles, "
                    "answer slots, or gold. The executor must bind one action."
                ),
            },
        },
        "forbidden_inputs": {
            "complete_target_trajectories": 0,
            "target_rewards_or_outcomes": 0,
            "source_receipts_or_programs": 0,
            "target_answer_keys": 0,
        },
        "output": "runs/target_schema_synthesis_v29/report.json",
    }
    config = body | {"config_sha256": stable_hash(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/target_schema_synthesis_v29.json",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), indent=2))


if __name__ == "__main__":
    main()
