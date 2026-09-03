#!/usr/bin/env python3
"""Freeze all remaining eligible train identities for V21 requalification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


FAMILIES = ("pick_and_place_simple", "pick_two_obj_and_place")
ROLE = "utility_requalification"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "file_sha256": _sha256(path)}


def _validate_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    claimed = str(body.pop(field, ""))
    if stable_hash(body) != claimed:
        raise ValueError(f"invalid stable hash: {field}")
    return claimed


def _task_ids(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, str) and value.endswith("/game.tw-pddl"):
        output.add(value)
    elif isinstance(value, dict):
        for child in value.values():
            output.update(_task_ids(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_task_ids(child))
    return output


def _rank(seed: int, family: str, task_id: str) -> tuple[str, str]:
    payload = f"v21:{seed}:{family}:{task_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), task_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-candidate", type=Path, required=True)
    parser.add_argument("--v20-manifest", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument(
        "--exclusion-artifact", type=Path, action="append", required=True
    )
    parser.add_argument("--enumerator-code", type=Path, required=True)
    parser.add_argument("--plan-freezer-code", type=Path, required=True)
    parser.add_argument("--runner-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20210812)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V21 manifest: {args.output}")
    candidate = _read(args.parent_candidate)
    candidate_hash = _validate_hash(candidate, "candidate_sha256")
    if candidate.get("status") != (
        "PROSPECTIVE_UTILITY_REQUALIFICATION_AUTHORIZED"
    ) or not candidate.get("utility_requalification_authorized"):
        raise SystemExit("V21 candidate did not authorize requalification")
    v20 = _read(args.v20_manifest)
    v20_hash = _validate_hash(v20, "manifest_sha256")
    excluded = _task_ids(v20["splits"])
    receipts = [_receipt(args.v20_manifest) | {
        "manifest_sha256": v20_hash,
        "task_ids_recovered": len(_task_ids(v20["splits"])),
    }]
    for path in args.exclusion_artifact:
        artifact = _read(path)
        ids = _task_ids(artifact)
        excluded.update(ids)
        receipts.append(_receipt(path) | {"task_ids_recovered": len(ids)})
    train_root = args.train_root.resolve()
    all_ids = sorted(
        path.resolve().relative_to(train_root).as_posix()
        for path in train_root.glob("*/*/game.tw-pddl")
    )
    selected_by_family = {}
    eligible_counts = {}
    selected = []
    for family in FAMILIES:
        eligible = [
            task_id for task_id in all_ids
            if task_id.startswith(family + "-") and task_id not in excluded
        ]
        eligible_counts[family] = len(eligible)
        rows = sorted(eligible, key=lambda task_id: _rank(
            args.seed, family, task_id
        ))
        selected_by_family[family] = rows
        selected.extend(rows)
    if len(selected) < 300:
        raise SystemExit("V21 requires at least 300 fresh remaining identities")
    if len(selected) != len(set(selected)) or set(selected) & excluded:
        raise RuntimeError("V21 selected identities are not fresh and disjoint")
    body = {
        "schema_version": "real-source-relation-requalification-manifest-v21",
        "status": "FROZEN_BEFORE_ANY_V21_SELECTED_TASK_RESET",
        "claim_boundary": (
            "ALL_REMAINING_ELIGIBLE_SIMPLE_AND_TWO_OBJECT_ALFWORLD_TRAIN_"
            "IDENTITIES_AFTER_V20_AND_PRIOR_EXCLUSIONS; PATH_HASH_ORDER_ONLY; "
            "V20_DEVELOPMENT_CONFIRMATION_AND_VALID_UNSEEN_UNREAD"
        ),
        "selection_candidate": _receipt(args.parent_candidate) | {
            "candidate_sha256": candidate_hash,
        },
        "parent_candidate": candidate["parent_candidate"],
        "implementation": {
            "manifest_freezer": _receipt(Path(__file__)),
            "outcome_blind_enumerator": _receipt(args.enumerator_code),
            "eval_plan_freezer": _receipt(args.plan_freezer_code),
            "eval_runner": _receipt(args.runner_code),
        },
        "train_root": str(train_root),
        "target_families": list(FAMILIES),
        "eligible_counts": eligible_counts,
        "selected_by_family": selected_by_family,
        "splits": {ROLE: selected},
        "selected_task_count": len(selected),
        "excluded_task_count": len(excluded),
        "exclusion_artifacts": receipts,
        "seed": args.seed,
        "rank_function": "sha256(v21:seed:family:relative_task_id)",
        "selection_used_task_file_contents": False,
        "selection_used_target_rollout_outcomes": False,
        "max_steps": 60,
        "main_path_policy": "TARGET_NATIVE_SAFETY_ONLY_SOURCE_GRAPH_DISABLED",
        "shadow_policy": "AUTHENTIC_EXECUTABLE_REAL_SOURCE_BIND_TO_RELATE_EDGE",
        "allowed_source_effects": ["BIND", "RELATE"],
        "active_required_properties": ["NONE"],
        "role_permissions": {
            ROLE: "OUTCOME_BLIND_ENUMERATION_THEN_FROZEN_PROSPECTIVE_EVALUATION"
        },
        "v20_development_or_confirmation_read_or_run": False,
        "existing_valid_unseen_read_or_run": False,
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "selected_task_count": len(selected),
        "eligible_counts": eligible_counts,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
