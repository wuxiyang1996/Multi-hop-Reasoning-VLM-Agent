#!/usr/bin/env python3
"""Freeze a fresh outcome-blind ALFWorld transformation contrast pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.contracts import stable_hash

from freeze_parameterized_alfworld_v7_manifest import (
    _iter_text_files,
    _sha256_bytes,
    _task_ids_in_text,
)
from run_slot_aware_alfworld_v8 import _read, _sha256, _validate_hash


FAMILIES = (
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
)
MAX_EXCLUSION_FILE_BYTES = 16 * 1024 * 1024


def _family(task_id: str) -> str:
    family = task_id.split("-", 1)[0]
    if family not in FAMILIES:
        raise ValueError(f"unsupported V14 family: {family}")
    return family


def _rank(seed: int, task_id: str) -> tuple[str, str]:
    return hashlib.sha256(
        f"{seed}:{task_id}".encode("utf-8")
    ).hexdigest(), task_id


def _receipt(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "file_sha256": _sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-candidate", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument(
        "--exclude-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--exclude-snapshot", type=Path)
    parser.add_argument("--enumerator-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100101)
    parser.add_argument("--per-family", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite V14 pool: {output}")
    if args.per_family != 16 or args.max_steps != 60:
        raise SystemExit("V14 freezes 16 tasks/family and a 60-step endpoint")
    parent = _read(args.parent_candidate)
    _validate_hash(parent, "candidate_sha256")
    if parent.get("experiment_version") != "v12":
        raise SystemExit("V14 requires the audited V12 dependency bundle")
    if parent.get("candidate_authority") != "FRESH_ADAPTATION":
        raise SystemExit("V14 parent lacks frozen target dependency authority")
    train_root = args.train_root.resolve()
    task_ids = sorted(
        path.resolve().relative_to(train_root).as_posix()
        for path in train_root.rglob("game.tw-pddl")
    )
    if not task_ids:
        raise SystemExit("V14 train root contains no ALFWorld tasks")
    exclusion_bytes_read = 0
    if args.exclude_snapshot:
        snapshot = _read(args.exclude_snapshot)
        excluded = set(map(str, snapshot["excluded_task_ids"]))
        excluded_payload = ("\n".join(sorted(excluded)) + "\n").encode()
        if _sha256_bytes(excluded_payload) != snapshot[
            "excluded_task_ids_sha256"
        ]:
            raise SystemExit("V14 exclusion snapshot task-ID hash mismatch")
        exclusion_roots = list(map(str, snapshot["exclusion_roots"]))
        scanned_files = int(snapshot["exclusion_files_scanned"])
        exclusion_bytes_read = int(snapshot["exclusion_bytes_read"])
    else:
        if not args.exclude_root:
            raise SystemExit("provide V14 exclusion roots or snapshot")
        excluded: set[str] = set()
        scanned_files = 0
        for path in _iter_text_files(
            tuple(root.resolve() for root in args.exclude_root), output
        ):
            size = path.stat().st_size
            if size > MAX_EXCLUSION_FILE_BYTES:
                raise SystemExit(
                    "V14 exclusion artifact exceeds fail-closed size limit: "
                    f"{path} ({size} bytes)"
                )
            scanned_files += 1
            exclusion_bytes_read += size
            try:
                excluded.update(_task_ids_in_text(
                    path.read_text(encoding="utf-8")
                ))
            except UnicodeDecodeError:
                continue
        exclusion_roots = [
            str(root.resolve()) for root in args.exclude_root
        ]
    selected = []
    selected_by_family = {}
    eligible_counts = {}
    for family in FAMILIES:
        eligible = [
            task_id for task_id in task_ids
            if task_id.startswith(family + "-") and task_id not in excluded
        ]
        eligible_counts[family] = len(eligible)
        ranked = sorted(eligible, key=lambda task_id: _rank(args.seed, task_id))
        if len(ranked) < args.per_family:
            raise SystemExit(f"V14 family {family} lacks fresh identities")
        rows = ranked[:args.per_family]
        selected.extend(rows)
        selected_by_family[family] = rows
    if set(selected) & excluded:
        raise RuntimeError("V14 selected a previously consumed identity")
    excluded_rows = sorted(excluded)
    excluded_payload = ("\n".join(excluded_rows) + "\n").encode()
    body = {
        "schema_version": "transformation-action-contrast-pool-v14",
        "status": "FROZEN_BEFORE_ANY_V14_SELECTED_TASK_RESET",
        "claim_boundary": (
            "FRESH_TRAIN_IDENTITIES_FOR_OUTCOME_BLIND_ACTION_CONTRAST_"
            "ENUMERATION_ONLY; TASKS_BECOME_DEVELOPMENT_CONSUMED_ON_RESET; "
            "NO_SUCCESS_OR_REWARD_RECORDED; CONFIRMATION_AND_EXISTING_"
            "VALID_UNSEEN_UNREAD"
        ),
        "parent_candidate": _receipt(args.parent_candidate) | {
            "candidate_sha256": parent["candidate_sha256"],
            "use_authority": (
                "TARGET_GROUNDER_PROPERTY_ROUTER_SOURCE_IR_THRESHOLDS_ONLY"
            ),
        },
        "implementation": {
            "pool_freezer": _receipt(Path(__file__)),
            "contrast_enumerator": _receipt(args.enumerator_code),
        },
        "train_root": str(train_root),
        "train_task_count": len(task_ids),
        "target_families": list(FAMILIES),
        "seed": args.seed,
        "rank_function": "SHA256(seed:relative_task_id)",
        "per_family": args.per_family,
        "max_steps": args.max_steps,
        "selection_authority": "PATH_IDENTITIES_AND_PRIOR_IDENTITIES_ONLY",
        "selection_used_task_file_contents": False,
        "selection_used_target_rollout_outcomes": False,
        "selection_used_prior_task_outcomes": False,
        "exclusion_roots": exclusion_roots,
        "exclusion_files_scanned": scanned_files,
        "exclusion_bytes_read": exclusion_bytes_read,
        "maximum_exclusion_file_bytes": MAX_EXCLUSION_FILE_BYTES,
        "excluded_task_count": len(excluded_rows),
        "excluded_task_ids_sha256": _sha256_bytes(excluded_payload),
        "excluded_task_ids": excluded_rows,
        "eligible_counts": eligible_counts,
        "selected_by_family": selected_by_family,
        "splits": {"outcome_blind_contrast_preflight": selected},
        "selected_task_count": len(selected),
        "selected_intersection_with_prior_logs": 0,
        "main_path_policy": "TARGET_NATIVE_SAFETY_ONLY_NO_SOURCE_GRAPH",
        "shadow_policy": (
            "AUTHENTIC_EXECUTABLE_BIND_MUTATE_RELATE_SOURCE_GRAPH"
        ),
        "active_required_properties": [
            "CLEAN", "COOL", "HEAT", "LIGHT"
        ],
        "allowed_source_effects": ["BIND", "MUTATE", "RELATE"],
        "contrast_gate": {
            "minimum_tasks_with_edge_action_contrast": 32,
            "minimum_families_with_four_contrast_tasks": 3,
            "minimum_mutate_contrast_tasks": 16,
            "minimum_relate_contrast_tasks": 8,
            "zero_outcomes_recorded": True,
            "zero_identity_or_receipt_failures": True,
        },
        "confirmation_read": False,
        "existing_valid_unseen_heldout_read": False,
    }
    result = body | {"pool_sha256": stable_hash(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "pool_sha256": result["pool_sha256"],
        "selected_task_count": len(selected),
        "selected_by_family": selected_by_family,
        "excluded_task_count": len(excluded_rows),
        "selection_used_target_rollout_outcomes": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
