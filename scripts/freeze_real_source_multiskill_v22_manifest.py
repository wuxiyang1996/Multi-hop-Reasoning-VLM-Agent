#!/usr/bin/env python3
"""Freeze disjoint ALFWorld transformation roles for V22 multi-skill transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


FAMILIES = (
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
)
ROLE_COUNTS = {
    "causal_adaptation": {
        "pick_clean_then_place_in_recep": 200,
        "pick_heat_then_place_in_recep": 140,
    },
    "causal_calibration": {
        "pick_clean_then_place_in_recep": 100,
        "pick_heat_then_place_in_recep": 70,
    },
    "prospective_requalification": {
        "pick_clean_then_place_in_recep": 100,
        "pick_heat_then_place_in_recep": 70,
    },
}
MAX_EXCLUSION_BYTES = 16 * 1024 * 1024
TASK_RE = re.compile(
    r"(?:look_at_obj_in_light|pick_(?:clean|cool|heat)_then_place_in_recep)-"
    r"[^\"\s]+?/game\.tw-pddl"
)


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


def _rank(seed: int, family: str, task_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"v22:{seed}:{family}:{task_id}".encode("utf-8")
    ).hexdigest()
    return digest, task_id


def _scan_exclusions(roots: list[Path]) -> tuple[set[str], dict[str, Any]]:
    excluded: set[str] = set()
    receipts = []
    total_bytes = 0
    for root in roots:
        for relative in ("configs", "docs/results"):
            directory = root.resolve() / relative
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.json")):
                size = path.stat().st_size
                if size > MAX_EXCLUSION_BYTES:
                    raise SystemExit(f"oversized exclusion artifact: {path}")
                text = path.read_text(encoding="utf-8", errors="ignore")
                recovered = set(TASK_RE.findall(text))
                excluded.update(recovered)
                total_bytes += size
                receipts.append({
                    "path": str(path.resolve()),
                    "file_sha256": _sha256(path),
                    "task_ids_recovered": len(recovered),
                })
    return excluded, {
        "roots": [str(path.resolve()) for path in roots],
        "files": receipts,
        "file_count": len(receipts),
        "bytes_read": total_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-candidate", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--exclude-root", type=Path, action="append", required=True)
    parser.add_argument("--enumerator-code", type=Path, required=True)
    parser.add_argument("--plan-freezer-code", type=Path, required=True)
    parser.add_argument("--runner-code", type=Path, required=True)
    parser.add_argument("--model-code", type=Path, required=True)
    parser.add_argument("--trainer-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20220812)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V22 manifest: {args.output}")

    parent = _read(args.parent_candidate)
    parent_hash = _validate_hash(parent, "candidate_sha256")
    if parent.get("candidate_authority") != "FRESH_ADAPTATION":
        raise SystemExit("V22 requires the audited fresh V12 dependency bundle")
    source = _read(args.source_summary)
    if source.get("status") != "SOURCE_TYPED_GATE_PASSED":
        raise SystemExit("V22 source typed gate did not pass")
    edges = {
        (str(row["from"]), str(row["to"]))
        for row in source["effect_ir"]["edges"]
    }
    if not {("BIND", "MUTATE"), ("BIND", "RELATE")}.issubset(edges):
        raise SystemExit("V22 source IR lacks both causal successor skills")

    excluded, exclusion_audit = _scan_exclusions(args.exclude_root)
    train_root = args.train_root.resolve()
    all_ids = sorted(
        path.resolve().relative_to(train_root).as_posix()
        for path in train_root.glob("*/*/game.tw-pddl")
    )
    splits = {role: [] for role in ROLE_COUNTS}
    selected_by_family_and_role: dict[str, dict[str, list[str]]] = {}
    eligible_counts = {}
    for family in FAMILIES:
        eligible = [
            task_id for task_id in all_ids
            if task_id.startswith(family + "-") and task_id not in excluded
        ]
        eligible_counts[family] = len(eligible)
        ranked = sorted(eligible, key=lambda row: _rank(args.seed, family, row))
        needed = sum(ROLE_COUNTS[role][family] for role in ROLE_COUNTS)
        if len(ranked) < needed:
            raise SystemExit(f"V22 {family} has {len(ranked)} eligible; need {needed}")
        cursor = 0
        family_roles = {}
        for role in ROLE_COUNTS:
            count = ROLE_COUNTS[role][family]
            rows = ranked[cursor:cursor + count]
            cursor += count
            family_roles[role] = rows
            splits[role].extend(rows)
        selected_by_family_and_role[family] = family_roles
    selected = [task for rows in splits.values() for task in rows]
    if len(selected) != len(set(selected)) or set(selected) & excluded:
        raise RuntimeError("V22 role identities are not fresh and disjoint")

    body = {
        "schema_version": "real-source-multiskill-manifest-v22",
        "status": "FROZEN_BEFORE_ANY_V22_SELECTED_TASK_RESET",
        "claim_boundary": (
            "REAL_MINIGRID_SOURCE_BIND_TO_MUTATE_AND_CROSS_ENGINE_BIND_TO_"
            "RELATE; TARGET_NATIVE_ALFWORLD_NEURAL_GROUNDING; PATH_HASH_"
            "SELECTION; ADAPTATION_CALIBRATION_AND_PROSPECTIVE_"
            "REQUALIFICATION_DISJOINT; FUTURE_DEVELOPMENT_CONFIRMATION_AND_"
            "EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "parent_candidate": _receipt(args.parent_candidate) | {
            "candidate_sha256": parent_hash,
            "use_authority": "TARGET_GROUNDER_ROUTER_THRESHOLDS_ONLY",
        },
        "source_summary": _receipt(args.source_summary) | {
            "ir_sha256": source["effect_ir"]["ir_sha256"],
            "source_tasks": source["tasks"],
            "simulator_families": source["simulator_families"],
            "effect_ir": source["effect_ir"],
        },
        "implementation": {
            "manifest_freezer": _receipt(Path(__file__)),
            "outcome_blind_enumerator": _receipt(args.enumerator_code),
            "fork_plan_freezer": _receipt(args.plan_freezer_code),
            "fork_runner": _receipt(args.runner_code),
            "target_native_model": _receipt(args.model_code),
            "candidate_trainer": _receipt(args.trainer_code),
        },
        "train_root": str(train_root),
        "target_families": list(FAMILIES),
        "role_counts_by_family": ROLE_COUNTS,
        "eligible_counts": eligible_counts,
        "selected_by_family_and_role": selected_by_family_and_role,
        "splits": splits,
        "selected_task_count": len(selected),
        "excluded_task_count": len(excluded),
        "exclusion_audit": exclusion_audit,
        "seed": args.seed,
        "rank_function": "sha256(v22:seed:family:relative_task_id)",
        "selection_used_task_file_contents": False,
        "selection_used_target_rollout_outcomes": False,
        "max_steps": 60,
        "main_path_policy": "TARGET_NATIVE_SAFETY_ONLY_SOURCE_GRAPH_DISABLED",
        "shadow_policy": "AUTHENTIC_EXECUTABLE_TYPED_SOURCE_GRAPH",
        "allowed_source_effects": ["BIND", "MUTATE", "RELATE"],
        "active_required_properties": ["CLEAN", "HEAT"],
        "role_permissions": {
            "causal_adaptation": "ENUMERATE_THEN_MATCHED_FORK_MODEL_DEVELOPMENT",
            "causal_calibration": "ENUMERATE_THEN_MATCHED_FORK_CALIBRATION",
            "prospective_requalification": "SEALED_UNTIL_V22_CANDIDATE_FROZEN",
        },
        "prospective_requalification_read_or_run": False,
        "future_development_confirmation_read_or_run": False,
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
        "split_counts": {key: len(value) for key, value in splits.items()},
        "eligible_counts": eligible_counts,
        "excluded_task_count": len(excluded),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
