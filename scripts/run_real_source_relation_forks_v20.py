#!/usr/bin/env python3
"""Execute exact-state V20 source-edge/target-abstain causal forks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_masked_effect_grounder import (  # noqa: E402
    validate_artifact as validate_target_artifact,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.parameterized_alfworld_harness import (  # noqa: E402
    validate_property_router,
)
from motif_transfer.slot_aware_alfworld_harness import (  # noqa: E402
    validate_slot_source_ir,
)
from run_relation_edge_intervention_forks_v13 import (  # noqa: E402
    TREATMENTS,
    _relative_game_matches,
    _run_branch,
)
from run_slot_aware_alfworld_v8 import (  # noqa: E402
    _read,
    _sha256,
    _validate_dependency,
    _validate_file_receipt,
    _validate_hash,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alfworld-config", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite V20 fork report: {args.output}")
    plan = _read(args.plan)
    _validate_hash(plan, "plan_sha256")
    if plan.get("status") != "FROZEN_BEFORE_ANY_V20_MATCHED_FORK_OUTCOME":
        raise SystemExit("V20 fork plan has unexpected authority")
    if tuple(plan["treatments"]) != TREATMENTS:
        raise SystemExit("V20 fork treatments changed after freezing")
    for receipt in plan["implementation"].values():
        _validate_file_receipt(receipt)
    parent_receipt = plan["parent_candidate"]
    parent = _read(Path(str(parent_receipt["path"])))
    _validate_hash(parent, "candidate_sha256")
    if _sha256(Path(str(parent_receipt["path"]))) != parent_receipt["file_sha256"]:
        raise SystemExit("V20 parent candidate file changed")
    target = _validate_dependency(parent["target_grounder"])
    validate_target_artifact(target)
    router = dict(parent["property_router"])
    validate_property_router(router)
    source_ir = dict(parent["slot_source_ir"])
    validate_slot_source_ir(source_ir)
    thresholds = dict(parent["thresholds"])
    allowed_source_effects = tuple(map(
        str, parent["transfer_scope"]["allowed_source_effects"]
    ))
    active_required_properties = tuple(map(
        str, parent["transfer_scope"]["active_required_properties"]
    ))
    max_steps = int(plan["max_steps"])
    by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in plan["opportunities"]:
        by_role[str(row["role"])].append(row)
    result_by_fork: dict[str, dict[str, Any]] = {}
    for role in ("causal_adaptation", "causal_calibration"):
        opportunities = by_role[role]
        seed = int(plan["contrast_reports"][role]["seed"])
        for treatment in TREATMENTS:
            environment = ALFWorldTextBatchEnvironment(
                config_path=str(args.alfworld_config.resolve()),
                data_path=str(args.alfworld_data.resolve()),
                split="train",
                seed=seed,
                game_ids=tuple(str(row["task_id"]) for row in opportunities),
                max_steps=max_steps,
            )
            seen: set[str] = set()
            try:
                for index in range(len(opportunities)):
                    observation = environment.reset()
                    matches = [
                        row for row in opportunities
                        if _relative_game_matches(
                            environment.resolved_game_file, str(row["task_id"])
                        )
                    ]
                    if len(matches) != 1:
                        raise RuntimeError("V20 fork reset identity mismatch")
                    opportunity = matches[0]
                    task_id = str(opportunity["task_id"])
                    if task_id in seen:
                        raise RuntimeError("V20 fork repeated a task identity")
                    seen.add(task_id)
                    branch = _run_branch(
                        environment=environment,
                        observation=observation,
                        opportunity=opportunity,
                        treatment=treatment,
                        target=target,
                        router=router,
                        source_ir=source_ir,
                        thresholds=thresholds,
                        allowed_source_effects=allowed_source_effects,
                        active_required_properties=active_required_properties,
                        max_steps=max_steps,
                    )
                    fork_id = str(opportunity["fork_id"])
                    result_by_fork.setdefault(fork_id, dict(opportunity) | {
                        "branches": {}
                    })["branches"][treatment] = branch
                    print(json.dumps({
                        "role": role,
                        "treatment": treatment,
                        "branch_index": index,
                        "branch_count": len(opportunities),
                        "task_id": task_id,
                        "success": branch["official_success"],
                        "steps": branch["steps"],
                    }), flush=True)
            finally:
                environment.close()
            if seen != {str(row["task_id"]) for row in opportunities}:
                raise RuntimeError("V20 fork treatment did not execute every task")
    forks = []
    for opportunity in plan["opportunities"]:
        row = result_by_fork[str(opportunity["fork_id"])]
        branches = row["branches"]
        if set(branches) != set(TREATMENTS):
            raise RuntimeError("V20 fork lacks a matched branch")
        source = branches["SOURCE_EDGE"]
        control = branches["TARGET_ABSTAIN"]
        invariants = {
            "fork_state_match": (
                source["fork_state_sha256"]
                == control["fork_state_sha256"]
                == opportunity["expected_fork_state_sha256"]
            ),
            "source_action_match": (
                source["source_action"] == control["source_action"]
                == opportunity["expected_source_action"]
            ),
            "control_action_match": (
                source["control_action"] == control["control_action"]
                == opportunity["expected_fallback_action"]
            ),
            "feature_match": source["features_sha256"] == control["features_sha256"],
            "action_contrast": source["source_action"] != source["control_action"],
        }
        if not all(invariants.values()):
            raise RuntimeError("V20 matched-fork invariant failed")
        row["invariants"] = invariants
        row["features"] = source["features"]
        row["features_sha256"] = source["features_sha256"]
        row["fork_sha256"] = stable_hash(row)
        forks.append(row)
    body = {
        "schema_version": "real-source-relation-causal-fork-report-v20",
        "status": "MATCHED_CAUSAL_ADAPTATION_CALIBRATION_FORKS_COMPLETE",
        "claim_boundary": (
            "TARGET_NATIVE_MATCHED_COUNTERFACTUAL_RECEIPTS_FOR_TRAINING_AND_"
            "CALIBRATION_ONLY; NOT DEVELOPMENT_OR_CONFIRMATION_TRANSFER_"
            "EVIDENCE; EXISTING_VALID_UNSEEN_UNREAD"
        ),
        "plan": {
            "path": str(args.plan.resolve()),
            "file_sha256": _sha256(args.plan),
            "plan_sha256": plan["plan_sha256"],
        },
        "max_steps": max_steps,
        "forks": forks,
        "fork_count": len(forks),
        "fork_counts_by_role": {
            role: sum(str(row["role"]) == role for row in forks)
            for role in by_role
        },
        "matched_cell_count": len(forks) * 2,
        "all_matched_invariants_passed": all(
            all(row["invariants"].values()) for row in forks
        ),
        "development_or_confirmation_read_or_run": False,
        "existing_valid_unseen_read_or_run": False,
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_sha256": report["report_sha256"],
        "fork_count": len(forks),
        "fork_counts_by_role": report["fork_counts_by_role"],
        "all_matched_invariants_passed": report["all_matched_invariants_passed"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
