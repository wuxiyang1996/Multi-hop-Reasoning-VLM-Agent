#!/usr/bin/env python3
"""Freeze fresh relation-only V9 adaptation and confirmation identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash

from freeze_slot_aware_alfworld_v8_manifest import (
    _family,
    _iter_text_files,
    _rank,
    _sha256_bytes,
    _task_ids_in_text,
)


DEFAULT_FAMILIES = (
    "pick_and_place_simple",
    "pick_two_obj_and_place",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument(
        "--exclude-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--exclude-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptation-seed", type=int, default=99101)
    parser.add_argument("--confirmation-seed", type=int, default=99201)
    parser.add_argument("--adaptation-per-family", type=int, default=8)
    parser.add_argument("--confirmation-per-family", type=int, default=12)
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated structural target families.",
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
    selected_families = tuple(
        value.strip() for value in args.families.split(",")
        if value.strip()
    )
    if not selected_families or not set(selected_families).issubset(
        DEFAULT_FAMILIES
    ):
        raise SystemExit("V9 supports relation-only target families")
    if args.exclude_snapshot:
        snapshot = json.loads(
            args.exclude_snapshot.read_text(encoding="utf-8")
        )
        excluded = set(map(str, snapshot["excluded_task_ids"]))
        excluded_payload = (
            "\n".join(sorted(excluded)) + "\n"
        ).encode()
        if _sha256_bytes(excluded_payload) != snapshot[
            "excluded_task_ids_sha256"
        ]:
            raise SystemExit("exclusion snapshot task-ID hash mismatch")
        exclusion_roots = list(map(str, snapshot["exclusion_roots"]))
        scanned_files = int(snapshot["exclusion_files_scanned"])
    else:
        if not args.exclude_root:
            raise SystemExit("provide exclusion roots or a snapshot")
        excluded: set[str] = set()
        scanned_files = 0
        for path in _iter_text_files(
            tuple(root.resolve() for root in args.exclude_root), output
        ):
            scanned_files += 1
            try:
                excluded.update(_task_ids_in_text(
                    path.read_text(encoding="utf-8")
                ))
            except UnicodeDecodeError:
                continue
        exclusion_roots = [
            str(root.resolve()) for root in args.exclude_root
        ]
    adaptation: list[str] = []
    confirmation: list[str] = []
    by_family: dict[str, dict[str, list[str]]] = {}
    eligible_counts: dict[str, int] = {}
    for family in selected_families:
        eligible = [
            task_id for task_id in task_ids
            if _family(task_id) == family and task_id not in excluded
        ]
        eligible_counts[family] = len(eligible)
        ranked_adaptation = sorted(
            eligible,
            key=lambda task_id: _rank(
                args.adaptation_seed, task_id
            ),
        )
        if len(ranked_adaptation) < args.adaptation_per_family:
            raise SystemExit(f"family {family} lacks fresh V9 tasks")
        family_adaptation = ranked_adaptation[
            :args.adaptation_per_family
        ]
        remaining = [
            task_id for task_id in eligible
            if task_id not in set(family_adaptation)
        ]
        ranked_confirmation = sorted(
            remaining,
            key=lambda task_id: _rank(
                args.confirmation_seed, task_id
            ),
        )
        if len(ranked_confirmation) < args.confirmation_per_family:
            raise SystemExit(
                f"family {family} lacks fresh V9 confirmation tasks"
            )
        family_confirmation = ranked_confirmation[
            :args.confirmation_per_family
        ]
        adaptation.extend(family_adaptation)
        confirmation.extend(family_confirmation)
        by_family[family] = {
            "adaptation_gate": family_adaptation,
            "fresh_confirmation": family_confirmation,
        }
    selected = set(adaptation) | set(confirmation)
    if len(selected) != len(adaptation) + len(confirmation):
        raise RuntimeError("V9 adaptation and confirmation overlap")
    if selected & excluded:
        raise RuntimeError("V9 selected a consumed task identity")
    excluded_rows = sorted(excluded)
    excluded_payload = (
        "\n".join(excluded_rows) + "\n"
    ).encode()
    body = {
        "schema_version": "executable-source-graph-alfworld-manifest-v9",
        "status": "FROZEN_BEFORE_ANY_SELECTED_TASK_RESET",
        "claim_boundary": (
            "FRESH_RELATION_ONLY_TRAIN_INSTANCES; V8_CONFIRMATION_CONSUMED_"
            "FOR_DEVELOPMENT; EXISTING_VALID_UNSEEN_HELDOUT_UNREAD"
        ),
        "train_root": str(train_root),
        "train_task_count": len(task_ids),
        "target_families": list(selected_families),
        "selection_authority": (
            "PATH_IDENTITIES_AND_PRIOR_TASK_IDENTITIES_ONLY"
        ),
        "selection_used_task_file_contents": False,
        "selection_used_target_rollout_outcomes": False,
        "selection_used_prior_task_outcomes": False,
        "exclusion_roots": exclusion_roots,
        "exclusion_files_scanned": scanned_files,
        "excluded_task_count": len(excluded_rows),
        "excluded_task_ids_sha256": _sha256_bytes(excluded_payload),
        "excluded_task_ids": excluded_rows,
        "adaptation_seed": args.adaptation_seed,
        "confirmation_seed": args.confirmation_seed,
        "rank_function": "SHA256(seed:relative_task_id)",
        "adaptation_per_family": args.adaptation_per_family,
        "confirmation_per_family": args.confirmation_per_family,
        "eligible_counts": eligible_counts,
        "selected_by_family": by_family,
        "splits": {
            "adaptation_gate": adaptation,
            "fresh_confirmation": confirmation,
        },
        "adaptation_task_count": len(adaptation),
        "confirmation_task_count": len(confirmation),
        "selected_task_count": len(selected),
        "selected_intersection_with_prior_logs": 0,
        "adaptation_confirmation_intersection": 0,
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
        "excluded_task_count": len(excluded_rows),
        "adaptation_task_count": len(adaptation),
        "confirmation_task_count": len(confirmation),
        "selected_intersection_with_prior_logs": 0,
        "selection_used_target_rollout_outcomes": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
