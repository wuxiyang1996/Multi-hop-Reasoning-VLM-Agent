#!/usr/bin/env python3
"""Freeze a product-disjoint option-relation WebShop V17 reserve."""

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
from motif_transfer.webshop_semantic_reserve import (  # noqa: E402
    require_semantic_reserve,
)


def _human_asins(products: list[dict[str, Any]], human: dict[str, Any]) -> set[str]:
    return {str(row["asin"]) for row in products if str(row.get("asin")) in human}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-root", type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/WebShop"
        ),
    )
    parser.add_argument("--prior-manifest", type=Path, default=REPO / (
        "configs/webshop_synthetic_unique_v14_frozen.json"
    ))
    parser.add_argument("--goal-seed", type=int, default=617)
    parser.add_argument("--qualification-size", type=int, default=8)
    parser.add_argument("--formal-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=REPO / (
        "configs/webshop_structural_v17_frozen.json"
    ))
    args = parser.parse_args()

    products_path = args.vendor_root / "data/items_shuffle_1000.json"
    human_path = args.vendor_root / "data/items_human_ins.json"
    sys.path.insert(0, str(args.vendor_root))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals

    raw_products = json.loads(products_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_manifest.read_text(encoding="utf-8"))
    prior_rows = [row for rows in prior["roles"].values() for row in rows]
    excluded_asins = _human_asins(raw_products, human) | {
        str(row["asin"]) for row in prior_rows
    }
    excluded_instructions = {
        str(row["instruction_text"]).strip().lower() for row in prior_rows
    }

    random.seed(args.goal_seed)
    products, _, prices, _ = load_products(
        filepath=str(products_path), human_goals=False,
    )
    random.seed(args.goal_seed)
    goals = get_goals(products, prices, human_goals=False)
    random.seed(args.goal_seed)
    random.shuffle(goals)

    selected = []
    used_asins = set(excluded_asins)
    used_instructions = set(excluded_instructions)
    required = args.qualification_size + args.formal_size
    for server_goal_index, goal in enumerate(goals):
        asin = str(goal["asin"])
        instruction = str(goal["instruction_text"]).strip().lower()
        # V17 is explicitly the relation-coverage structural family.  Empty
        # goal_options are outside that family and are not silently coerced.
        if not dict(goal.get("goal_options") or {}):
            continue
        if asin in used_asins or instruction in used_instructions:
            continue
        selected.append({
            "task_id": f"webshop.{server_goal_index}",
            "server_goal_index": server_goal_index,
            "asin": asin,
            "instruction_text": goal["instruction_text"],
            "goal": goal,
            "goal_sha256": stable_hash(goal),
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
        for asin in sorted(_human_asins(raw_products, human))
    ]
    all_audit = require_semantic_reserve(
        selected, consumed_rows=consumed,
        required_unique_goals=required,
        require_asin_disjointness=True,
        require_unique_candidate_asins=True,
    )
    formal_audit = require_semantic_reserve(
        formal, consumed_rows=[*consumed, *qualification],
        required_unique_goals=args.formal_size,
        require_asin_disjointness=True,
        require_unique_candidate_asins=True,
    )
    body = {
        "schema_version": "webshop-structural-v17-reserve-v1",
        "artifact_role": "OUTCOME_BLIND_OPTION_RELATION_WEBSHOP_V17_RESERVE",
        "status": "FROZEN_BEFORE_ANY_V17_PROVIDER_CALL_OR_OUTCOME",
        "claim_boundary": (
            "Native synthetic WebShop goals with nonempty option relations; unique "
            "ASIN and semantics; product-disjoint from human goals and every V14 "
            "development/formal goal. Formal outcomes remain sealed."
        ),
        "goal_seed": args.goal_seed,
        "server_goal_count": len(goals),
        "number_of_registered_tasks_required": 1 + max(
            row["server_goal_index"] for row in selected
        ),
        "roles": {
            "transport_qualification": qualification,
            "formal_reserve": formal,
        },
        "preflight": {
            "all_selected_vs_prior_and_human": all_audit,
            "formal_vs_qualification_prior_and_human": formal_audit,
            "all_goals_have_nonempty_target_relation_schema": all(
                bool((row.get("goal") or {}).get("goal_options")) for row in selected
            ),
        },
        "prior_manifest_sha256": prior["artifact_sha256"],
        "selection_rule": (
            "Seed native synthetic generator, then take first unique-ASIN, "
            "unique-instruction goals with nonempty goal_options after exclusions."
        ),
        "runtime_contract": {
            "server_launcher": "scripts/run_webshop_structural_server_v17.py",
            "formal_requires_development_gate": (
                "PHASE4_WEBSHOP_DEVELOPMENT_GATE_PASSED"
            ),
            "formal_run_once": True,
        },
        "runtime_hashes": {
            "products": file_sha256(products_path),
            "human_instructions": file_sha256(human_path),
            "prior_manifest": file_sha256(args.prior_manifest),
            "freezer": file_sha256(Path(__file__)),
            "server_launcher": file_sha256(
                REPO / "scripts/run_webshop_structural_server_v17.py"
            ),
        },
        "formal_outcomes_read_or_run": False,
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": artifact["status"],
        "goal_seed": args.goal_seed,
        "transport_qualification": len(qualification),
        "formal_reserve": len(formal),
        "registered_tasks_required": artifact["number_of_registered_tasks_required"],
        "all_preflight_passed": (
            all_audit["passed"] and formal_audit["passed"]
            and artifact["preflight"]["all_goals_have_nonempty_target_relation_schema"]
        ),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
