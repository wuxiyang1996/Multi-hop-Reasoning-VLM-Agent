#!/usr/bin/env python3
"""Freeze outcome-blind, product-disjoint WebShop synthetic goal splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.webshop_semantic_reserve import (  # noqa: E402
    require_semantic_reserve,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _human_asins(products_path: Path, human_path: Path) -> set[str]:
    products = json.loads(products_path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))
    return {str(row["asin"]) for row in products if str(row.get("asin")) in human}


def _select_one_goal_per_product(
    goals: list[dict[str, Any]],
    *,
    excluded_asins: set[str],
    count: int,
) -> list[dict[str, Any]]:
    selected = []
    used_asins = set(excluded_asins)
    used_instructions = set()
    for server_goal_index, goal in enumerate(goals):
        asin = str(goal["asin"])
        instruction = str(goal["instruction_text"]).strip().lower()
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
        if len(selected) == count:
            return selected
    raise ValueError(f"only found {len(selected)} unique product goals; need {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/emnlp2026_download/workspace/vendor/WebShop"
        ),
    )
    parser.add_argument("--goal-seed", type=int, default=233)
    parser.add_argument("--development-size", type=int, default=24)
    parser.add_argument("--formal-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "configs/webshop_synthetic_unique_v14_frozen.json",
    )
    args = parser.parse_args()

    products_path = args.vendor_root / "data/items_shuffle_1000.json"
    human_path = args.vendor_root / "data/items_human_ins.json"
    engine_path = args.vendor_root / "web_agent_site/engine/engine.py"
    goal_path = args.vendor_root / "web_agent_site/engine/goal.py"
    app_path = args.vendor_root / "web_agent_site/app.py"
    for path in (products_path, human_path, engine_path, goal_path, app_path):
        if not path.exists():
            raise SystemExit(f"missing WebShop runtime file: {path}")

    sys.path.insert(0, str(args.vendor_root))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals

    # The vendor loader samples prices, so the seed must precede loading.  The
    # upstream app historically seeded only before goal generation, which made
    # some price-bucket instruction text vary across restarts.
    random.seed(args.goal_seed)
    products, _, prices, _ = load_products(
        filepath=str(products_path), human_goals=False,
    )
    random.seed(args.goal_seed)
    goals = get_goals(products, prices, human_goals=False)
    random.seed(args.goal_seed)
    random.shuffle(goals)

    excluded_asins = _human_asins(products_path, human_path)
    selected = _select_one_goal_per_product(
        goals,
        excluded_asins=excluded_asins,
        count=args.development_size + args.formal_size,
    )
    development = selected[: args.development_size]
    formal = selected[args.development_size :]
    consumed = [{"instruction_text": "excluded human goal", "asin": asin}
                for asin in sorted(excluded_asins)]
    all_audit = require_semantic_reserve(
        selected,
        consumed_rows=consumed,
        required_unique_goals=len(selected),
        require_asin_disjointness=True,
        require_unique_candidate_asins=True,
    )
    formal_audit = require_semantic_reserve(
        formal,
        consumed_rows=[*consumed, *development],
        required_unique_goals=len(formal),
        require_asin_disjointness=True,
        require_unique_candidate_asins=True,
    )
    artifact = {
        "schema_version": 1,
        "artifact_role": "OUTCOME_BLIND_WEBSHOP_SYNTHETIC_UNIQUE_V14_SPLIT",
        "status": "FROZEN_BEFORE_ANY_PROVIDER_CALL_OR_OUTCOME",
        "claim_boundary": (
            "Synthetic WebShop goals use the benchmark's native products, search, option "
            "controls, reward, and executor. Goal semantics and product ASIN are unique "
            "across development/formal and disjoint from the 13-goal human pool."
        ),
        "selection_rule": (
            "After the benchmark's seeded goal generation and shuffle, take the first "
            "eligible goal for each previously unused ASIN; development precedes formal."
        ),
        "goal_seed": args.goal_seed,
        "server_goal_count": len(goals),
        "number_of_registered_tasks_required": 1 + max(
            row["server_goal_index"] for row in selected
        ),
        "excluded_human_goal_asins": sorted(excluded_asins),
        "roles": {
            "development": development,
            "formal_reserve": formal,
        },
        "preflight": {
            "all_selected": all_audit,
            "formal_vs_development_and_human": formal_audit,
        },
        "runtime_contract": {
            "server_launcher": "scripts/run_webshop_unique_goal_server_v14.py",
            "goal_mode": "native_synthetic",
            "human_goal_mode_forbidden": True,
            "semantic_preflight_before_provider": True,
            "primary_inference_unit": "unique_asin_goal",
        },
        "runtime_hashes": {
            "products": file_sha256(products_path),
            "human_instructions": file_sha256(human_path),
            "vendor_engine": file_sha256(engine_path),
            "vendor_goal": file_sha256(goal_path),
            "vendor_app": file_sha256(app_path),
            "freezer": file_sha256(Path(__file__)),
            "server_launcher": file_sha256(
                REPO / "scripts/run_webshop_unique_goal_server_v14.py"
            ),
            "server_adapter": file_sha256(
                REPO / "src/motif_transfer/webshop_unique_goal_server_v14.py"
            ),
        },
    }
    artifact["artifact_sha256"] = stable_hash(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": artifact["status"],
        "server_goal_count": len(goals),
        "development": len(development),
        "formal_reserve": len(formal),
        "registered_tasks_required": artifact["number_of_registered_tasks_required"],
        "preflight_passed": all_audit["passed"] and formal_audit["passed"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
