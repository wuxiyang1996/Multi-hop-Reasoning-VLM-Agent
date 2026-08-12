#!/usr/bin/env python3
"""Freeze a revised V8 gate while preserving the unread confirmation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash

from freeze_slot_aware_alfworld_v8_manifest import (
    FAMILIES,
    _family,
    _iter_text_files,
    _rank,
    _sha256_bytes,
    _task_ids_in_text,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_manifest(manifest: dict) -> None:
    body = dict(manifest)
    claimed = str(body.pop("manifest_sha256", ""))
    if stable_hash(body) != claimed:
        raise SystemExit("parent V8 manifest hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--exclude-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptation-seed", type=int, default=98401)
    parser.add_argument("--adaptation-per-family", type=int, default=5)
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument(
        "--families",
        default=",".join(FAMILIES),
        help="Comma-separated adaptation families; confirmation is preserved.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite revised manifest: {output}")
    parent = _read(args.parent_manifest)
    _validate_manifest(parent)
    if not str(parent.get("schema_version", "")).startswith(
        "slot-aware-alfworld-manifest-v8"
    ):
        raise SystemExit("wrong parent V8 manifest")
    if parent.get("status") not in {
        "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET",
        "FROZEN_BEFORE_ANY_REVISED_ADAPTATION_RESET",
    }:
        raise SystemExit("parent manifest was not frozen")
    confirmation = tuple(map(str, parent["splits"]["fresh_confirmation"]))
    train_root = args.train_root.resolve()
    task_ids = sorted(
        path.resolve().relative_to(train_root).as_posix()
        for path in train_root.rglob("game.tw-pddl")
    )
    excluded: set[str] = set()
    scanned_files = 0
    for path in _iter_text_files(
        tuple(root.resolve() for root in args.exclude_root), output
    ):
        scanned_files += 1
        try:
            excluded.update(_task_ids_in_text(path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    if not set(confirmation).issubset(excluded):
        raise SystemExit("parent confirmation IDs missing from audit snapshot")
    selected_families = tuple(
        value.strip() for value in args.families.split(",") if value.strip()
    )
    if not selected_families or not set(selected_families).issubset(FAMILIES):
        raise SystemExit("invalid revised adaptation families")
    adaptation: list[str] = []
    by_family: dict[str, list[str]] = {}
    eligible_counts: dict[str, int] = {}
    for family in selected_families:
        eligible = [
            task_id for task_id in task_ids
            if _family(task_id) == family and task_id not in excluded
        ]
        eligible_counts[family] = len(eligible)
        ranked = sorted(
            eligible, key=lambda task_id: _rank(args.adaptation_seed, task_id)
        )
        if len(ranked) < args.adaptation_per_family:
            raise SystemExit(f"family {family} lacks revised adaptation tasks")
        selected = ranked[: args.adaptation_per_family]
        adaptation.extend(selected)
        by_family[family] = selected
    if set(adaptation) & excluded:
        raise RuntimeError("revised adaptation gate overlaps consumed logs")
    if set(adaptation) & set(confirmation):
        raise RuntimeError("revised adaptation overlaps preserved confirmation")
    excluded_rows = sorted(excluded)
    excluded_payload = ("\n".join(excluded_rows) + "\n").encode()
    body = {
        "schema_version": f"slot-aware-alfworld-manifest-v8-revision{args.revision}",
        "status": "FROZEN_BEFORE_ANY_REVISED_ADAPTATION_RESET",
        "claim_boundary": (
            "FIRST_V8_GATE_CONSUMED_FOR_DEVELOPMENT; NEW_ADAPTATION_GATE_FRESH; "
            "PARENT_CONFIRMATION_PRESERVED_AND_UNREAD; EXISTING_VALID_UNSEEN_"
            "HELDOUT_REMAINS_UNREAD"
        ),
        "parent_manifest": {
            "path": str(args.parent_manifest.resolve()),
            "file_sha256": _file_sha256(args.parent_manifest),
            "manifest_sha256": parent["manifest_sha256"],
            "preserved_confirmation_task_count": len(confirmation),
            "preserved_confirmation_reset_before_revision": False,
        },
        "train_root": str(train_root),
        "train_task_count": len(task_ids),
        "selection_authority": "PATH_NAMES_AND_PRIOR_TASK_IDENTITIES_ONLY",
        "selection_used_task_file_contents": False,
        "selection_used_target_rollout_outcomes": False,
        "selection_used_prior_task_outcomes": False,
        "exclusion_roots": [str(root.resolve()) for root in args.exclude_root],
        "exclusion_files_scanned": scanned_files,
        "excluded_task_count": len(excluded_rows),
        "excluded_task_ids_sha256": _sha256_bytes(excluded_payload),
        "excluded_task_ids": excluded_rows,
        "adaptation_seed": args.adaptation_seed,
        "rank_function": "SHA256(seed:relative_task_id)",
        "adaptation_per_family": args.adaptation_per_family,
        "adaptation_families": list(selected_families),
        "eligible_counts": eligible_counts,
        "adaptation_selected_by_family": by_family,
        "splits": {
            "adaptation_gate": adaptation,
            "fresh_confirmation": list(confirmation),
        },
        "adaptation_task_count": len(adaptation),
        "confirmation_task_count": len(confirmation),
        "selected_task_count": len(adaptation) + len(confirmation),
        "adaptation_intersection_with_prior_logs": 0,
        "adaptation_confirmation_intersection": 0,
        "preserved_confirmation_ids_present_in_exclusion_snapshot": len(
            confirmation
        ),
    }
    result = body | {"manifest_sha256": stable_hash(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "manifest_sha256": result["manifest_sha256"],
        "adaptation_task_count": len(adaptation),
        "preserved_confirmation_task_count": len(confirmation),
        "adaptation_intersection_with_prior_logs": 0,
        "adaptation_confirmation_intersection": 0,
        "selection_used_target_rollout_outcomes": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
