#!/usr/bin/env python3
"""Freeze a fresh V11 relation-edge gate and preserve confirmation."""

from __future__ import annotations

import argparse
import hashlib
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


FAMILIES = ("pick_and_place_simple", "pick_two_obj_and_place")


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(*, default_manifest_version: str = "v11") -> int:
    if default_manifest_version not in {"v11", "v12"}:
        raise ValueError("unsupported default manifest version")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument(
        "--exclude-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--exclude-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptation-seed", type=int, default=99601)
    parser.add_argument("--adaptation-per-family", type=int, default=8)
    parser.add_argument(
        "--manifest-version",
        choices=("v11", "v12"),
        default=default_manifest_version,
    )
    args = parser.parse_args()
    version_label = args.manifest_version.upper()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(
            f"refusing to overwrite {version_label} manifest: {output}"
        )
    parent = _read(args.parent_manifest)
    parent_body = dict(parent)
    parent_hash = str(parent_body.pop("manifest_sha256", ""))
    if stable_hash(parent_body) != parent_hash:
        raise SystemExit("parent manifest hash mismatch")
    expected_parent_schema = (
        "budgeted-relation-edge-alfworld-manifest-v11"
        if args.manifest_version == "v12"
        else "budgeted-executable-source-graph-alfworld-manifest-v10"
    )
    if parent.get("schema_version") != expected_parent_schema:
        raise SystemExit("wrong parent manifest for requested version")
    confirmation = tuple(map(
        str, parent["splits"]["fresh_confirmation"]
    ))
    train_root = args.train_root.resolve()
    task_ids = sorted(
        path.resolve().relative_to(train_root).as_posix()
        for path in train_root.rglob("game.tw-pddl")
    )
    if args.exclude_snapshot:
        snapshot = _read(args.exclude_snapshot)
        excluded = set(map(str, snapshot["excluded_task_ids"]))
        payload = ("\n".join(sorted(excluded)) + "\n").encode()
        if _sha256_bytes(payload) != snapshot["excluded_task_ids_sha256"]:
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
    if not set(confirmation).issubset(excluded):
        raise SystemExit("preserved confirmation missing from audit snapshot")
    adaptation: list[str] = []
    selected_by_family: dict[str, list[str]] = {}
    eligible_counts: dict[str, int] = {}
    for family in FAMILIES:
        eligible = [
            task_id for task_id in task_ids
            if _family(task_id) == family and task_id not in excluded
        ]
        eligible_counts[family] = len(eligible)
        ranked = sorted(
            eligible,
            key=lambda task_id: _rank(args.adaptation_seed, task_id),
        )
        if len(ranked) < args.adaptation_per_family:
            raise SystemExit(
                f"family {family} lacks fresh {version_label} tasks"
            )
        selected = ranked[:args.adaptation_per_family]
        adaptation.extend(selected)
        selected_by_family[family] = selected
    if set(adaptation) & excluded:
        raise RuntimeError(
            f"{version_label} adaptation overlaps consumed identities"
        )
    if set(adaptation) & set(confirmation):
        raise RuntimeError(
            f"{version_label} adaptation overlaps confirmation"
        )
    excluded_rows = sorted(excluded)
    excluded_payload = (
        "\n".join(excluded_rows) + "\n"
    ).encode()
    is_v12 = args.manifest_version == "v12"
    body = {
        "schema_version": (
            "selective-budgeted-relation-edge-alfworld-manifest-v12"
            if is_v12
            else "budgeted-relation-edge-alfworld-manifest-v11"
        ),
        "status": (
            "FROZEN_BEFORE_ANY_V12_ADAPTATION_RESET"
            if is_v12
            else "FROZEN_BEFORE_ANY_V11_ADAPTATION_RESET"
        ),
        "claim_boundary": (
            (
                "SELECTIVE_STEP_NINE_RULE_FIXED_AFTER_CONSUMED_V9_V10_"
                "V11_GROUPED_AUDIT_AND_CLOSED_LOOP_REPLAY; SIXTY_STEP_"
                "ENDPOINT_UNCHANGED; V12_ADAPTATION_FRESH; CONFIRMATION_"
                "PRESERVED_UNREAD; EXISTING_VALID_UNSEEN_HELDOUT_UNREAD"
            )
            if is_v12
            else (
                "RELATE_ONLY_CLAIM_FIXED_AFTER_CONSUMED_V10_GATE; "
                "SIXTY_STEP_ENDPOINT_UNCHANGED; V11_ADAPTATION_FRESH; "
                "CONFIRMATION_PRESERVED_UNREAD; EXISTING_VALID_UNSEEN_"
                "HELDOUT_UNREAD"
            )
        ),
        "parent_manifest": {
            "path": str(args.parent_manifest.resolve()),
            "file_sha256": _sha256(args.parent_manifest),
            "manifest_sha256": parent_hash,
            "preserved_confirmation_task_count": len(confirmation),
            (
                "preserved_confirmation_reset_before_v12"
                if is_v12
                else "preserved_confirmation_reset_before_v11"
            ): False,
        },
        "development_authority": (
            (
                "V9_V10_V11_CONSUMED_GROUPED_AUDIT_AND_CLOSED_LOOP_"
                "REPLAY_PASSED_BEFORE_V12_SELECTION"
            )
            if is_v12
            else "V10_ADAPTATION_CONSUMED_FOR_CLAIM_COVERAGE_FIX_ONLY"
        ),
        "train_root": str(train_root),
        "train_task_count": len(task_ids),
        "target_families": list(FAMILIES),
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
        "rank_function": "SHA256(seed:relative_task_id)",
        "adaptation_per_family": args.adaptation_per_family,
        "eligible_counts": eligible_counts,
        "adaptation_selected_by_family": selected_by_family,
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
