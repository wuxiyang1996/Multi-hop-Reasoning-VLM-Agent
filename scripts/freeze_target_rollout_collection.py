#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _digest(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()


def _rank(values: list[str], namespace: str, excluded: set[str]) -> list[str]:
    return sorted(
        set(values) - excluded,
        key=lambda value: (_digest(namespace, value), value),
    )


def _alfworld_ids(root: Path, split: str) -> list[str]:
    base = root / split
    return [
        str(path.relative_to(base))
        for path in base.rglob("game.tw-pddl")
    ]


def _partition(ranked: list[str]) -> dict[str, list[str]]:
    if len(ranked) < 40:
        raise ValueError("target pool has fewer than 40 unobserved items")
    return {
        "adaptation": ranked[:8],
        "qualification": ranked[8:16],
        "held_out": ranked[16:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze unobserved 8/8/24 target rollout splits by ID only"
    )
    parser.add_argument(
        "--workspace", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("configs/target_rollout_collection_v1.json"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]

    old_targets = json.loads(
        (repo / "configs/target_manifests_v1.json").read_text()
    )
    old_alf = old_targets["cells"]["alfworld_valid_unseen"]
    alf_excluded = {
        str(old_alf["adaptation_id"]),
        *map(str, old_alf["smoke_test_ids"]),
    }
    # The seen adaptation was also consumed by prior diagnostics.
    alf_excluded.add(str(
        old_targets["cells"]["alfworld_valid_seen"]["adaptation_id"]
    ))
    alf_root = (
        args.workspace
        / "Multi-hop-Reasoning-VLM-Agent-github-main"
        / ".cache/alfworld_data/json_2.1.1"
    )
    alf_namespace = "target-rollout:alfworld-valid-unseen:v1"
    alf_adaptation = _rank(
        _alfworld_ids(alf_root, "train"),
        alf_namespace + ":adaptation",
        alf_excluded,
    )[:8]
    alf_test = _rank(
        _alfworld_ids(alf_root, "valid_unseen"),
        alf_namespace + ":target",
        alf_excluded,
    )
    alf_splits = {
        "adaptation": alf_adaptation,
        "qualification": alf_test[:8],
        "held_out": alf_test[8:32],
    }

    old_vtb = json.loads(
        (repo / "configs/vtb_single_turn_manifest_v2.json").read_text()
    )
    vtb_excluded = {
        str(old_vtb["adaptation_id"]),
        *map(str, old_vtb["smoke_test_ids"]),
        *map(str, old_vtb["observed_before_freeze"]),
    }
    vtb_namespace = "target-rollout:vtb-single-turn:v1"
    vtb_ranked = _rank(
        list(map(str, old_vtb["test_ids"])),
        vtb_namespace,
        vtb_excluded,
    )
    vtb_splits = _partition(vtb_ranked)

    payload = {
        "schema_version": 1,
        "frozen_at": "2026-07-23",
        "selection_used_content_or_outcome": False,
        "protocol": {
            "adaptation": "may train or propose target-native motifs",
            "qualification": "candidate selection only; never update weights",
            "held_out": "read once after all artifacts are frozen",
        },
        "cells": {
            "alfworld_valid_unseen": {
                "namespace": alf_namespace,
                "pool": {
                    "adaptation": "official train",
                    "qualification_and_held_out": "official valid_unseen",
                },
                "excluded_previously_observed": sorted(alf_excluded),
                "splits": alf_splits,
            },
            "visual_toolbench_single_turn": {
                "namespace": vtb_namespace,
                "pool": "pinned public single-turn internal split",
                "excluded_previously_observed": sorted(vtb_excluded),
                "splits": vtb_splits,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        cell: {split: len(ids) for split, ids in row["splits"].items()}
        for cell, row in payload["cells"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
