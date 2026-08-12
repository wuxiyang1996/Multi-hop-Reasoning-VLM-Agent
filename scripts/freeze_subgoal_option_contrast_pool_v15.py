#!/usr/bin/env python3
"""Freeze fresh two-object ALFWorld identities for an outcome-blind V15 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


DESTINATIONS = (
    "Drawer",
    "Cabinet",
    "ArmChair",
    "Toilet",
    "Desk",
    "Shelf",
    "Sofa",
    "CoffeeTable",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "file_sha256": _sha256(path)}


def _destination(task_id: str) -> str:
    directory = task_id.split("/", 1)[0]
    fields = directory.split("-")
    if len(fields) < 5 or fields[0] != "pick_two_obj_and_place":
        raise ValueError(f"not a two-object ALFWorld identity: {task_id}")
    return fields[3]


def _rank(seed: int, task_id: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest(), task_id


def select_tasks(
    task_ids: list[str],
    *,
    excluded: set[str],
    seed: int,
    per_destination: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    selected: dict[str, list[str]] = {}
    eligible_counts: dict[str, int] = {}
    for destination in DESTINATIONS:
        eligible = [
            task_id
            for task_id in task_ids
            if task_id not in excluded and _destination(task_id) == destination
        ]
        eligible_counts[destination] = len(eligible)
        ranked = sorted(eligible, key=lambda task_id: _rank(seed, task_id))
        if len(ranked) < per_destination:
            raise ValueError(
                f"destination {destination} has {len(ranked)} eligible tasks; "
                f"need {per_destination}"
            )
        selected[destination] = ranked[:per_destination]
    return selected, eligible_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--prior-pool", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--enumerator-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=151501)
    parser.add_argument("--per-destination", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V15 pool: {args.output}")
    if args.per_destination != 8 or args.max_steps != 60:
        raise SystemExit("V15 freezes 8 tasks/destination and a 60-step endpoint")

    artifact = _read(args.source_artifact)
    source_config = _read(args.source_config)
    source_result = _read(args.source_result)
    prior = _read(args.prior_pool)
    if artifact.get("status") != "QUALIFICATION_AUTHORIZED":
        raise SystemExit("V15 source artifact is not qualification-authorized")
    if (
        source_result.get("status") != "FINAL_HELDOUT_PASSED"
        or not source_result.get("cross_domain_transfer_supported")
    ):
        raise SystemExit("V15 requires the historically passed V4 source controller")
    if source_config.get("status") != "FINAL_FROZEN_HELDOUT_EVALUATION":
        raise SystemExit("V15 source config is not the frozen V4 evaluation")
    if _sha256(args.source_config) != source_result.get("config_sha256"):
        raise SystemExit("V15 source config differs from the V4 result receipt")
    expected_artifact_hash = source_config["development_evidence"][
        "frozen_artifact_sha256"
    ]
    if _sha256(args.source_artifact) != expected_artifact_hash:
        raise SystemExit("V15 source artifact differs from the frozen V4 dependency")
    if _sha256(args.source_artifact) != source_result.get("artifact_sha256"):
        raise SystemExit("V15 source artifact differs from the V4 result receipt")
    required_models = {
        "authentic_source_plus_target",
        "phase_permuted_source_plus_target",
    }
    if not required_models <= set(artifact["source"]["models"]):
        raise SystemExit("V15 source artifact lacks required source/control models")
    if prior.get("status") != "FROZEN_BEFORE_ANY_V14_SELECTED_TASK_RESET":
        raise SystemExit("V15 prior exclusion pool has unexpected authority")
    if stable_hash({
        key: value for key, value in prior.items() if key != "pool_sha256"
    }) != prior.get("pool_sha256"):
        raise SystemExit("V15 prior exclusion pool hash mismatch")

    train_root = args.train_root.resolve()
    task_ids = sorted(
        path.resolve().relative_to(train_root).as_posix()
        for path in train_root.glob("pick_two_obj_and_place-*/*/game.tw-pddl")
    )
    if not task_ids:
        raise SystemExit("V15 train root contains no two-object tasks")
    excluded = set(map(str, prior["excluded_task_ids"]))
    excluded.update(map(str, prior["splits"]["outcome_blind_contrast_preflight"]))
    selected_by_destination, eligible_counts = select_tasks(
        task_ids,
        excluded=excluded,
        seed=args.seed,
        per_destination=args.per_destination,
    )
    selected = [
        task_id
        for destination in DESTINATIONS
        for task_id in selected_by_destination[destination]
    ]
    if set(selected) & excluded:
        raise RuntimeError("V15 selected a previously consumed identity")

    body = {
        "schema_version": "subgoal-option-contrast-pool-v15",
        "status": "FROZEN_BEFORE_ANY_V15_SELECTED_TASK_RESET",
        "claim_boundary": (
            "CONTROLLED_MULTISOURCE_OPTION_STRUCTURE_TO_NEW_ALFWORLD_TRAIN_"
            "IDENTITIES; OUTCOME_BLIND_SHADOW_CONTRAST_FEASIBILITY_ONLY; "
            "NOT_REAL_GAME_TRANSFER; TASKS_BECOME CONSUMED_DEVELOPMENT_ON_"
            "RESET; CONFIRMATION_AND_EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "source_controller": _receipt(args.source_artifact) | {
            "historical_config": _receipt(args.source_config),
            "historical_result": _receipt(args.source_result),
            "controller_authority": "ABSTRACT_OPTION_SELECTION_ONLY",
            "concrete_action_authority": "TARGET_NATIVE_GROUNDER_ONLY",
            "policy": {
                key: source_config["policy"][key]
                for key in (
                    "controller",
                    "uncertainty_scale",
                    "decision_margin",
                )
            },
        },
        "prior_exclusion_pool": _receipt(args.prior_pool) | {
            "pool_sha256": prior["pool_sha256"]
        },
        "implementation": {
            "pool_freezer": _receipt(Path(__file__)),
            "contrast_enumerator": _receipt(args.enumerator_code),
        },
        "train_root": str(train_root),
        "train_two_object_task_count": len(task_ids),
        "excluded_task_count": len(excluded),
        "selection_authority": "IDENTITY_PATH_HASH_AND_FIXED_DESTINATION_STRATA_ONLY",
        "selection_used_task_file_contents": False,
        "selection_used_target_rollout_outcomes": False,
        "selection_used_prior_task_outcomes": False,
        "seed": args.seed,
        "rank_function": "sha256(seed:task_id)",
        "destinations": list(DESTINATIONS),
        "per_destination": args.per_destination,
        "eligible_counts": eligible_counts,
        "selected_by_destination": selected_by_destination,
        "splits": {"outcome_blind_subgoal_contrast_preflight": selected},
        "selected_task_count": len(selected),
        "selected_intersection_with_prior_logs": [],
        "max_steps": args.max_steps,
        "main_path_policy": "SOURCE_DISABLED_TARGET_NATIVE_CONTROL",
        "shadow_policies": [
            "AUTHENTIC_SOURCE_OPTION_CONTROLLER",
            "PHASE_PERMUTED_SOURCE_OPTION_CONTROL",
        ],
        "contrast_gate": {
            "minimum_tasks_with_authentic_action_contrast": 32,
            "minimum_tasks_with_authentic_phase_action_contrast": 16,
            "minimum_tasks_with_second_cycle_authentic_contrast": 16,
            "minimum_destination_groups_with_four_authentic_contrasts": 4,
        },
        "reward_serialized": False,
        "official_success_serialized": False,
        "confirmation_read": False,
        "existing_valid_unseen_heldout_read": False,
    }
    payload = body | {"pool_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "pool": str(args.output.resolve()),
        "pool_sha256": payload["pool_sha256"],
        "selected_tasks": len(selected),
        "eligible_counts": eligible_counts,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
