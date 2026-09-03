#!/usr/bin/env python3
"""Build compact target-binding packs from already-consumed development receipts."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "runs/cross_domain_target_pilot_v1/adaptation"


def read(path: str):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def write(domain: str, examples: list[dict]) -> None:
    payload = {
        "schema_version": 1,
        "target_domain": domain,
        "split_role": "adaptation",
        "pilot_only": True,
        "examples": examples,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{domain}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def webshop() -> list[dict]:
    rows = []
    root = "runs/real_game_multitarget_neurosymbolic_v6_fixed/webshop_smoke2_all4"
    for task in ("webshop.6", "webshop.31"):
        episode = read(f"{root}/{task}.target_only.json")
        rows.append({
            "example_id": f"consumed-{task}",
            "split_role": "adaptation",
            "task": episode["goal"],
            "trajectory": [
                {
                    "step": step["step"],
                    "selected_action": step["selected_action"],
                    "available_action_count": len(step["candidates"]),
                    "state_changed": step["before_hash"] != step["after_hash"],
                }
                for step in episode["steps"]
            ],
            "outcome": {
                "official_reward": episode["official_reward"],
                "strict_success": episode["strict_success"],
            },
            "native_actions": [step["selected_action"] for step in episode["steps"]],
            "source_receipt_sha256": episode["receipt_sha256"],
        })
    return rows


def alfworld() -> list[dict]:
    source = read("runs/multisource_alfworld_neurosymbolic_v1/adaptation_expert_receipts.json")
    successful = [episode for episode in source["episodes"] if episode["official_success"]][:2]
    return [{
        "example_id": f"expert-{episode['task_id']}",
        "split_role": "adaptation",
        "task": episode["transitions"][0]["goal"],
        "trajectory": [
            {
                "step": transition["step"],
                "observation": transition["before_observation"][:900],
                "selected_action": transition["expert_action"],
                "next_observation": transition["after_observation"][:900],
            }
            for transition in episode["transitions"][:20]
        ],
        "outcome": {"official_success": episode["official_success"]},
        "native_actions": list(dict.fromkeys(
            transition["expert_action"] for transition in episode["transitions"][:20]
        )),
        "source_receipt_sha256": episode["transitions"][-1]["receipt_sha256"],
    } for episode in successful]


def discoveryworld() -> list[dict]:
    paths = [
        "runs/phase3_discoveryworld_consumed_development_v8_portfolio_typed/proteomics.easy.seed45/matched_result.json",
        "runs/phase3_discoveryworld_consumed_development_v8_portfolio_typed/proteomics.easy.seed46/matched_result.json",
    ]
    rows = []
    for path in paths:
        result = read(path)
        condition = result["conditions"]["target_only_recorded"]
        recovery = condition.get("recovery") or []
        rows.append({
            "example_id": f"consumed-{result['task']['task_id']}",
            "split_role": "adaptation",
            "task": result["task"],
            "observable_binding": {
                key: value for key, value in result["target_binding"].items()
                if key not in {"commit_action"}
            },
            "trajectory": [{
                "step": step["recovery_step"],
                "observable_state": step.get("before_target_native_facts"),
                "selected_action_type": (step.get("action") or {}).get("action"),
                "action_succeeded": (step.get("transition") or {}).get("action_succeeded"),
                "terminal": (step.get("transition") or {}).get("terminal"),
            } for step in recovery[:8]],
            "outcome": {
                "official_success": any(
                    bool((step.get("transition") or {}).get("official_success"))
                    for step in recovery
                )
            },
            "native_actions": list(dict.fromkeys(
                str((step.get("action") or {}).get("action"))
                for step in recovery if (step.get("action") or {}).get("action")
            )),
            "source_receipt_sha256": result["result_sha256"],
        })
    return rows


def tirbench() -> list[dict]:
    receipts = read(
        "runs/phase3_tir_visual_search_v11_verified_development/development_train_receipts.json"
    )[:2]
    return [{
        "example_id": f"development-train-{receipt['sample_id']}",
        "split_role": "adaptation",
        "task_family": receipt["family"],
        "baseline_reason": receipt["baseline"].get("reason"),
        "candidate_programs": [{
            "actions": candidate.get("actions", [])[:4],
            "target_hypothesis": candidate.get("target_hypothesis"),
            "planner_score": candidate.get("planner_score"),
            "observed_persistence_fraction": candidate.get("observed_persistence_fraction"),
        } for candidate in receipt["candidates"][:3]],
        "outcome": {
            "baseline_answer": receipt["baseline"].get("answer"),
            "gold_answer": receipt["gold_answer"],
        },
        "native_actions": ["zoom_region"],
        "source_receipt_sha256": receipt["receipt_sha256"],
    } for receipt in receipts]


def main() -> int:
    builders = {
        "webshop": webshop,
        "alfworld": alfworld,
        "discoveryworld": discoveryworld,
        "tirbench": tirbench,
    }
    for domain, builder in builders.items():
        examples = builder()
        write(domain, examples)
        print(json.dumps({"domain": domain, "examples": len(examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
