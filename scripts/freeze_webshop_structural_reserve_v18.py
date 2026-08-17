#!/usr/bin/env python3
"""Freeze a versioned option-relation reserve disjoint from earlier goals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.real_game_multitarget_manifest import file_sha256  # noqa: E402
from motif_transfer.webshop_semantic_reserve import require_semantic_reserve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, default=Path(
        "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/WebShop"
    ))
    parser.add_argument("--prior-manifests", type=Path, nargs="+", default=[
        REPO / "configs/webshop_synthetic_unique_v14_frozen.json",
        REPO / "configs/webshop_structural_v17_frozen.json",
    ])
    parser.add_argument("--goal-seed", type=int, default=719)
    parser.add_argument("--reserve-version", type=int, default=18)
    parser.add_argument("--qualification-size", type=int, default=4)
    parser.add_argument("--formal-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=REPO / (
        "configs/webshop_structural_v18_frozen.json"
    ))
    args = parser.parse_args()
    if args.reserve_version < 18:
        raise SystemExit("deterministic transport reserves start at V18")
    version = f"V{args.reserve_version}"
    version_lower = version.lower()

    products_path = args.vendor_root / "data/items_shuffle_1000.json"
    human_path = args.vendor_root / "data/items_human_ins.json"
    raw_products = json.loads(products_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))
    human_asins = {
        str(row["asin"]) for row in raw_products if str(row.get("asin")) in human
    }
    prior_artifacts = [json.loads(path.read_text()) for path in args.prior_manifests]
    prior_rows = [
        row for artifact in prior_artifacts
        for role in artifact["roles"].values() for row in role
    ]
    excluded_asins = human_asins | {str(row["asin"]) for row in prior_rows}
    excluded_instructions = {
        str(row["instruction_text"]).strip().lower() for row in prior_rows
    }

    sys.path.insert(0, str(args.vendor_root))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals
    random.seed(args.goal_seed)
    products, _, prices, _ = load_products(
        filepath=str(products_path), human_goals=False,
    )
    random.seed(args.goal_seed)
    goals = get_goals(products, prices, human_goals=False)
    random.seed(args.goal_seed)
    random.shuffle(goals)

    required = args.qualification_size + args.formal_size
    selected = []
    used_asins, used_instructions = set(excluded_asins), set(excluded_instructions)
    for index, goal in enumerate(goals):
        asin = str(goal["asin"])
        instruction = str(goal["instruction_text"]).strip().lower()
        if not dict(goal.get("goal_options") or {}):
            continue
        if asin in used_asins or instruction in used_instructions:
            continue
        selected.append({
            "task_id": f"webshop.{index}", "server_goal_index": index,
            "asin": asin, "instruction_text": goal["instruction_text"],
            "goal": goal, "goal_sha256": stable_hash(goal),
        })
        used_asins.add(asin)
        used_instructions.add(instruction)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise SystemExit(f"only found {len(selected)} eligible goals; need {required}")
    qualification = selected[: args.qualification_size]
    formal = selected[args.qualification_size :]
    consumed = [
        {"instruction_text": row["instruction_text"], "asin": row["asin"]}
        for row in prior_rows
    ] + [
        {"instruction_text": f"human:{asin}", "asin": asin}
        for asin in sorted(human_asins)
    ]
    all_audit = require_semantic_reserve(
        selected, consumed_rows=consumed, required_unique_goals=required,
        require_asin_disjointness=True, require_unique_candidate_asins=True,
    )
    formal_audit = require_semantic_reserve(
        formal, consumed_rows=[*consumed, *qualification],
        required_unique_goals=args.formal_size,
        require_asin_disjointness=True, require_unique_candidate_asins=True,
    )
    body = {
        "schema_version": f"webshop-structural-{version_lower}-reserve-v1",
        "artifact_role": (
            f"MATCHED_TRANSPORT_OPTION_RELATION_WEBSHOP_{version}_RESERVE"
        ),
        "status": f"FROZEN_BEFORE_ANY_{version}_PROVIDER_CALL_OR_OUTCOME",
        "claim_boundary": (
            "Native synthetic relation goals, unique ASIN/semantics, disjoint "
            "from human and every supplied prior manifest. Exact-query result "
            "ordering is replayed across matched arms."
        ),
        "goal_seed": args.goal_seed,
        "server_goal_count": len(goals),
        "number_of_registered_tasks_required": 1 + max(
            row["server_goal_index"] for row in selected
        ),
        "roles": {"transport_qualification": qualification,
                  "formal_reserve": formal},
        "preflight": {
            "all_selected_vs_all_prior_and_human": all_audit,
            "formal_vs_qualification_all_prior_and_human": formal_audit,
            "all_have_nonempty_relation_schema": all(
                bool(row["goal"].get("goal_options")) for row in selected
            ),
        },
        "prior_manifest_artifact_sha256s": [
            artifact["artifact_sha256"] for artifact in prior_artifacts
        ],
        "transport_contract": {
            "selected_goal_snapshot_is_content_addressed": True,
            "native_goal_identity_checked_before_price_snapshot_replay": True,
            "first_native_query_result_is_authoritative": True,
            "identical_query_replays_identical_asin_order": True,
            "source_free_controls_must_have_exact_trajectory_match": True,
            "authentic_and_target_ceiling_must_have_exact_trajectory_match": True,
        },
        "runtime_hashes": {
            "freezer": file_sha256(Path(__file__)),
            "server_launcher": file_sha256(REPO / (
                "scripts/run_webshop_structural_server_v18.py"
            )),
            "transport_adapter": file_sha256(REPO / (
                "src/motif_transfer/webshop_deterministic_transport_v18.py"
            )),
            "frozen_goal_adapter": file_sha256(REPO / (
                "src/motif_transfer/webshop_frozen_goal_transport.py"
            )),
            "prior_manifests": {
                str(path): file_sha256(path) for path in args.prior_manifests
            },
        },
        "formal_outcomes_read_or_run": False,
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": artifact["status"], "goal_seed": args.goal_seed,
        "reserve_version": version,
        "qualification": len(qualification), "formal": len(formal),
        "registered_tasks_required": artifact["number_of_registered_tasks_required"],
        "preflight_passed": all_audit["passed"] and formal_audit["passed"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
