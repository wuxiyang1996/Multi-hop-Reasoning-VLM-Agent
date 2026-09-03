#!/usr/bin/env python3
"""Freeze disjoint ALFWorld-train roles for scalable real-source relation transfer."""

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
ROLE_COUNTS_PER_FAMILY = {
    "causal_adaptation": 200,
    "causal_calibration": 100,
    "development_gate": 100,
    "sealed_confirmation": 100,
}


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
    payload = f"v20:{seed}:{family}:{task_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), task_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-candidate", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument(
        "--exclusion-artifact", type=Path, action="append", required=True
    )
    parser.add_argument("--enumerator-code", type=Path, required=True)
    parser.add_argument("--fork-plan-code", type=Path, required=True)
    parser.add_argument("--fork-runner-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20200812)
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V20 manifest: {args.output}")
    if args.max_steps != 60:
        raise SystemExit("V20 freezes the endpoint at 60 total actions")

    parent = _read(args.parent_candidate)
    parent_hash = _validate_hash(parent, "candidate_sha256")
    if parent.get("schema_version") != "budgeted-relation-edge-alfworld-candidate-v11":
        raise SystemExit("V20 requires the operationally audited V11 source graph")
    if parent.get("status") != "ADAPTATION_GATE_ONLY":
        raise SystemExit("V20 parent lacks frozen adaptation dependency authority")
    if parent["slot_source_ir"].get("execution_authority") != (
        "SYMBOLIC_EFFECT_ROUTING_ONLY"
    ):
        raise SystemExit("V20 source IR has unexpected execution authority")

    exclusion_receipts = []
    excluded: set[str] = set()
    for path in args.exclusion_artifact:
        artifact = _read(path)
        excluded.update(_task_ids(artifact))
        if "excluded_task_ids" in artifact:
            excluded.update(map(str, artifact["excluded_task_ids"]))
        exclusion_receipts.append(_receipt(path) | {
            "task_ids_recovered": len(_task_ids(artifact)),
        })

    train_root = args.train_root.resolve()
    all_ids = sorted(
        path.resolve().relative_to(train_root).as_posix()
        for path in train_root.glob("*/*/game.tw-pddl")
    )
    if not all_ids:
        raise SystemExit("ALFWorld train root has no task identities")
    splits: dict[str, list[str]] = {
        role: [] for role in ROLE_COUNTS_PER_FAMILY
    }
    selected_by_family_and_role: dict[str, dict[str, list[str]]] = {}
    eligible_counts = {}
    for family in FAMILIES:
        eligible = [
            task_id for task_id in all_ids
            if task_id.startswith(family + "-") and task_id not in excluded
        ]
        eligible_counts[family] = len(eligible)
        ranked = sorted(eligible, key=lambda row: _rank(args.seed, family, row))
        needed = sum(ROLE_COUNTS_PER_FAMILY.values())
        if len(ranked) < needed:
            raise SystemExit(
                f"V20 {family} has {len(ranked)} eligible tasks; need {needed}"
            )
        cursor = 0
        family_roles = {}
        for role, count in ROLE_COUNTS_PER_FAMILY.items():
            rows = ranked[cursor:cursor + count]
            cursor += count
            family_roles[role] = rows
            splits[role].extend(rows)
        selected_by_family_and_role[family] = family_roles
    selected = [task for rows in splits.values() for task in rows]
    if len(selected) != len(set(selected)):
        raise RuntimeError("V20 roles are not task-disjoint")
    if set(selected) & excluded:
        raise RuntimeError("V20 selected a previously consumed identity")

    body = {
        "schema_version": "real-source-relation-causal-manifest-v20",
        "status": "FROZEN_BEFORE_ANY_V20_SELECTED_TASK_RESET",
        "claim_boundary": (
            "REAL_MINIGRID_MINIWORLD_SOURCE_GRAPH_TO_DISJOINT_ALFWORLD_TRAIN_"
            "TASKS; PATH_HASH_SELECTION_ONLY; ADAPTATION_AND_CALIBRATION_MAY_"
            "TRAIN_TARGET_NATIVE_CAUSAL_GROUNDING; DEVELOPMENT_REQUIRES_FROZEN_"
            "CANDIDATE; CONFIRMATION_REQUIRES_PASSED_DEVELOPMENT_GATE; EXISTING_"
            "VALID_UNSEEN_PROHIBITED"
        ),
        "parent_candidate": _receipt(args.parent_candidate) | {
            "candidate_sha256": parent_hash,
            "source_ir_sha256": parent["slot_source_ir"]["ir_sha256"],
            "source_lineage": parent["slot_source_ir"]["source_lineage"],
            "use_authority": (
                "REAL_SOURCE_GRAPH_TARGET_GROUNDER_ROUTER_AND_THRESHOLDS_ONLY"
            ),
        },
        "implementation": {
            "manifest_freezer": _receipt(Path(__file__)),
            "outcome_blind_enumerator": _receipt(args.enumerator_code),
            "fork_plan_freezer": _receipt(args.fork_plan_code),
            "fork_runner": _receipt(args.fork_runner_code),
        },
        "train_root": str(train_root),
        "train_task_count": len(all_ids),
        "target_families": list(FAMILIES),
        "role_counts_per_family": ROLE_COUNTS_PER_FAMILY,
        "eligible_counts": eligible_counts,
        "selected_by_family_and_role": selected_by_family_and_role,
        "splits": splits,
        "selected_task_count": len(selected),
        "excluded_task_count": len(excluded),
        "exclusion_artifacts": exclusion_receipts,
        "seed": args.seed,
        "rank_function": "sha256(v20:seed:family:relative_task_id)",
        "selection_used_task_file_contents": False,
        "selection_used_target_rollout_outcomes": False,
        "selection_used_prior_task_outcomes": False,
        "max_steps": args.max_steps,
        "main_path_policy": "TARGET_NATIVE_SAFETY_ONLY_SOURCE_GRAPH_DISABLED",
        "shadow_policy": "AUTHENTIC_EXECUTABLE_REAL_SOURCE_BIND_TO_RELATE_EDGE",
        "allowed_source_effects": ["BIND", "RELATE"],
        "active_required_properties": ["NONE"],
        "role_permissions": {
            "causal_adaptation": "ENUMERATE_AND_COLLECT_MATCHED_FORK_OUTCOMES",
            "causal_calibration": "ENUMERATE_AND_COLLECT_MATCHED_FORK_OUTCOMES",
            "development_gate": "SEALED_UNTIL_TARGET_CANDIDATE_FROZEN",
            "sealed_confirmation": "SEALED_UNTIL_DEVELOPMENT_GATE_PASSES",
        },
        "confirmation_read_or_run": False,
        "existing_valid_unseen_read_or_run": False,
    }
    manifest = body | {"manifest_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "selected_task_count": len(selected),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "eligible_counts": eligible_counts,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
