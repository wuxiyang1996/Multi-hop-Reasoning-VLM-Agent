#!/usr/bin/env python3
"""Freeze an outcome-blind ALFWorld confirmation split from unseen train paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from motif_transfer.contracts import stable_hash


FAMILIES = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)
TASK_PATTERN = re.compile(
    r"(?:"
    + "|".join(map(re.escape, FAMILIES))
    + r")-[^\s\"\\]+/trial_[^\s\"\\]+/game\.tw-pddl"
)
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".txt", ".log"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _task_ids_in_text(value: str) -> set[str]:
    return set(TASK_PATTERN.findall(value))


def _iter_text_files(roots: Iterable[Path], output: Path) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = (root,)
        else:
            candidates = root.rglob("*")
        for path in candidates:
            if (
                path.is_file()
                and path.resolve() != output
                and path.suffix.lower() in TEXT_SUFFIXES
            ):
                yield path


def _family(task_id: str) -> str:
    family = task_id.split("-", 1)[0]
    if family not in FAMILIES:
        raise ValueError(f"unsupported ALFWorld family: {family}")
    return family


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--exclude-root", type=Path, action="append", default=[])
    parser.add_argument("--exclude-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=97101)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument(
        "--split-name",
        choices=("fresh_confirmation", "adaptation_gate"),
        default="fresh_confirmation",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {output}")
    train_root = args.train_root.resolve()
    task_ids = sorted(
        path.resolve().relative_to(train_root).as_posix()
        for path in train_root.rglob("game.tw-pddl")
    )
    if not task_ids:
        raise SystemExit("ALFWorld train root contains no tasks")

    if args.exclude_snapshot:
        snapshot = json.loads(
            args.exclude_snapshot.read_text(encoding="utf-8")
        )
        excluded = set(map(str, snapshot["excluded_task_ids"]))
        excluded_payload = (
            "\n".join(sorted(excluded)) + "\n"
        ).encode("utf-8")
        if _sha256_bytes(excluded_payload) != snapshot["excluded_task_ids_sha256"]:
            raise SystemExit("exclusion snapshot task-ID hash mismatch")
        exclusion_roots = list(map(str, snapshot["exclusion_roots"]))
        scanned_files = int(snapshot["exclusion_files_scanned"])
    else:
        if not args.exclude_root:
            raise SystemExit("provide exclusion roots or an exclusion snapshot")
        excluded = set()
        scanned_files = 0
        for path in _iter_text_files(
            tuple(root.resolve() for root in args.exclude_root), output
        ):
            scanned_files += 1
            try:
                excluded.update(_task_ids_in_text(path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                continue
        exclusion_roots = [str(root.resolve()) for root in args.exclude_root]

    selected: list[str] = []
    eligible_counts: dict[str, int] = {}
    selected_by_family: dict[str, list[str]] = {}
    for family in FAMILIES:
        eligible = [
            task_id for task_id in task_ids
            if _family(task_id) == family and task_id not in excluded
        ]
        eligible_counts[family] = len(eligible)
        ranked = sorted(
            eligible,
            key=lambda task_id: (
                hashlib.sha256(
                    f"{args.seed}:{task_id}".encode("utf-8")
                ).hexdigest(),
                task_id,
            ),
        )
        if len(ranked) < args.per_family:
            raise SystemExit(
                f"family {family} has only {len(ranked)} unconsumed tasks"
            )
        chosen = ranked[: args.per_family]
        selected.extend(chosen)
        selected_by_family[family] = chosen
    if set(selected) & excluded:
        raise RuntimeError("fresh confirmation selection overlaps prior logs")

    excluded_rows = sorted(excluded)
    excluded_payload = ("\n".join(excluded_rows) + "\n").encode("utf-8")
    claim_boundary = (
        "FRESH_UNSEEN_TRAIN_INSTANCES; IN_DISTRIBUTION_CONFIRMATION_ONLY; "
        "EXISTING_VALID_UNSEEN_HELDOUT_REMAINS_UNREAD"
        if args.split_name == "fresh_confirmation"
        else (
            "FRESH_UNSEEN_TRAIN_INSTANCES; TARGET_ADAPTATION_GATE_ONLY; "
            "CONFIRMATION_AND_EXISTING_VALID_UNSEEN_HELDOUT_REMAIN_UNREAD"
        )
    )
    body = {
        "schema_version": (
            "parameterized-alfworld-confirmation-manifest-v7"
            if args.split_name == "fresh_confirmation"
            else "parameterized-alfworld-adaptation-gate-manifest-v7"
        ),
        "status": "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET",
        "claim_boundary": claim_boundary,
        "train_root": str(train_root),
        "train_task_count": len(task_ids),
        "selection_authority": "PATH_NAMES_AND_PRIOR_TASK_IDENTITIES_ONLY",
        "selection_used_task_file_contents": False,
        "selection_used_target_rollout_outcomes": False,
        "selection_used_prior_task_outcomes": False,
        "exclusion_roots": exclusion_roots,
        "exclusion_files_scanned": scanned_files,
        "excluded_task_count": len(excluded_rows),
        "excluded_task_ids_sha256": _sha256_bytes(excluded_payload),
        "excluded_task_ids": excluded_rows,
        "seed": args.seed,
        "rank_function": "SHA256(seed:relative_task_id)",
        "per_family": args.per_family,
        "eligible_counts": eligible_counts,
        "selected_by_family": selected_by_family,
        "splits": {args.split_name: selected},
        "selected_task_count": len(selected),
        "selected_intersection_with_prior_logs": 0,
    }
    payload = body | {"manifest_sha256": stable_hash(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "manifest_sha256": payload["manifest_sha256"],
        "selected_task_count": len(selected),
        "selected_by_family": selected_by_family,
        "excluded_task_count": len(excluded_rows),
        "selection_used_target_rollout_outcomes": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
