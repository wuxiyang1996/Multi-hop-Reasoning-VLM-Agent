#!/usr/bin/env python3
"""Freeze an outcome-blind, task-family-balanced ALFWorld train pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_FAMILIES = (
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
)


def _family(relative_path: str) -> str:
    directory = relative_path.split("/", 1)[0]
    matches = [name for name in TASK_FAMILIES if directory.startswith(name + "-")]
    if len(matches) != 1:
        raise ValueError(f"could not identify one task family for {relative_path!r}")
    return matches[0]


def _rank(seed: int, family: str, relative_path: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{family}\0{relative_path}".encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=88401)
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--validation-per-family", type=int, default=2)
    args = parser.parse_args()
    root = args.train_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {output}")
    if not 0 < args.validation_per_family < args.per_family:
        raise SystemExit("validation-per-family must be in (0, per-family)")

    by_family = {family: [] for family in TASK_FAMILIES}
    for path in root.glob("*/trial_*/game.tw-pddl"):
        relative = path.relative_to(root).as_posix()
        by_family[_family(relative)].append(relative)

    adaptation_train: list[str] = []
    adaptation_validation: list[str] = []
    counts = {}
    for family in TASK_FAMILIES:
        available = by_family[family]
        ranked = sorted(available, key=lambda path: (_rank(args.seed, family, path), path))
        if len(ranked) < args.per_family:
            raise RuntimeError(f"only {len(ranked)} games for {family}")
        selected = ranked[: args.per_family]
        validation = selected[: args.validation_per_family]
        training = selected[args.validation_per_family :]
        adaptation_train.extend(training)
        adaptation_validation.extend(validation)
        counts[family] = {
            "available": len(available),
            "selected_train": len(training),
            "selected_validation": len(validation),
        }

    payload = {
        "schema_version": "alfworld-v2-outcome-blind-pool-v1",
        "status": "FROZEN_BEFORE_COLLECTION",
        "selection_authority": "TRAIN_PATH_NAMES_ONLY",
        "selection_used_file_contents": False,
        "selection_used_rollout_outcomes": False,
        "qualification_or_heldout_read": False,
        "train_root": str(root),
        "seed": args.seed,
        "rank_function": "sha256(seed, task_family, relative_game_path)",
        "per_family": args.per_family,
        "validation_per_family": args.validation_per_family,
        "family_counts": counts,
        "splits": {
            "adaptation_train": adaptation_train,
            "adaptation_validation": adaptation_validation,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "train": len(adaptation_train),
        "validation": len(adaptation_validation),
        "family_counts": counts,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
