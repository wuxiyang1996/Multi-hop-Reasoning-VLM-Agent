#!/usr/bin/env python3
"""Build the receipt-backed four-domain cross-domain-memory pilot summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/cross_domain_target_pilot_v1"


def load(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    domains = ["webshop", "alfworld", "discoveryworld", "tirbench"]
    methods = ["expel", "awm", "reasoning_bank"]
    admission = {}
    for domain in domains:
        admission[domain] = {}
        for method in methods:
            relative = f"runs/cross_domain_target_pilot_v1/bound/{domain}/{method}.json"
            artifact = load(relative)
            admission[domain][method] = {
                "admitted_items": len(artifact.get("items") or []),
                "binding_status": artifact["target_binding"]["binding_status"],
                "artifact_sha256": artifact["artifact_sha256"],
                "path": relative,
            }

    ws0 = load("runs/cross_domain_target_pilot_v1/eval/webshop/task13_target_only_v2/webshop.13.target_only.json")
    wsr = load("runs/cross_domain_target_pilot_v1/eval/webshop/task13_reasoning_bank/webshop.13.reasoning_bank.json")
    wsr_report = load("runs/cross_domain_target_pilot_v1/eval/webshop/task13_reasoning_bank/report.reasoning_bank.json")
    af0 = load("runs/cross_domain_target_pilot_v1/eval/alfworld/task4.target_only.v2.json")
    afr = load("runs/cross_domain_target_pilot_v1/eval/alfworld/task4.reasoning_bank.v2.json")
    dw = load("runs/discoveryworld_target_only_v2_development/proteomics.easy.seed0.json")
    tir0 = load("runs/cross_domain_target_pilot_v1/eval/tirbench/sample58.target_only.json")
    tirr = load("runs/cross_domain_target_pilot_v1/eval/tirbench/sample58.reasoning_bank.json")
    tir_rows = load("/fs/gamma-projects/vlm-robot/datasets/TIR-Bench/TIR-Bench.json")
    tir_row = next(row for row in tir_rows if str(row["id"]) == "58")
    tir_prompt_sha = hashlib.sha256(str(tir_row.get("prompt") or "").encode()).hexdigest()
    source_episode = json.loads(next(iter((RUN.parent / "cross_domain_shared_source_v1_smoke_l40s/authentic_skill_loaded/candy_crush/evidence/episodes.jsonl").read_text().splitlines())))

    payload = {
        "schema_version": 1,
        "status": "PILOT_RESULTS_NOT_A_POWERED_CLAIM",
        "scope": "non-video targets; already-consumed/development data only",
        "model": "qwen/qwen3.5-35b-a3b",
        "admission_matrix": admission,
        "outcomes": {
            "webshop": {
                "task_id": "webshop.13",
                "paired_input_goal_equal": ws0["goal"] == wsr["goal"],
                "target_only": {"success": ws0["strict_success"], "reward": ws0["official_reward"], "steps": ws0["step_count"]},
                "reasoning_bank": {"success": wsr["strict_success"], "reward": wsr["official_reward"], "steps": wsr["step_count"], "augmented_calls": wsr_report["memory_receipt"]["decision_calls_augmented"]},
                "interpretation": "no observed benefit in one development episode",
            },
            "alfworld": {
                "task_id": af0["task_id"],
                "paired_game_file_equal": af0["resolved_game_file"] == afr["resolved_game_file"],
                "paired_goal_equal": af0["records"][0]["before"]["state"]["task_goal"] == afr["records"][0]["before"]["state"]["task_goal"],
                "target_only": af0["metrics"],
                "reasoning_bank": afr["metrics"] | {"retrieval_calls": len(afr["memory_retrieval_receipts"])},
                "interpretation": "both succeeded; ReasoningBank required 7 more steps and 3 more repeated actions",
            },
            "discoveryworld": {
                "task_id": dw["task_id"],
                "target_only": {"success": dw["evaluation"]["official_success"], "normalized_score": dw["evaluation"]["scorecard"][0]["scoreNormalized"], "steps": len(dw["steps"])},
                "memory_arms": "strict target-only no-op because every verified artifact is empty",
                "interpretation": "execution receipt exists; no cross-domain item passed target evidence verification",
            },
            "tirbench": {
                "sample_id": "58",
                "effective_prompt_sha256_both_arms": tir_prompt_sha,
                "target_only": {"answer": tir0["result"].get("answer"), "correct": tir0["result"].get("correct"), "tool_events": len(tir0["result"].get("tool_trace") or [])},
                "reasoning_bank_empty_noop": {"answer": tirr["result"].get("answer"), "correct": tirr["result"].get("correct"), "tool_events": len(tirr["result"].get("tool_trace") or []), "retrieved_items": len((tirr.get("memory_retrieval") or {}).get("retrieved") or [])},
                "interpretation": "identical prompt but provider/tool-loop outputs differed; no difference is attributable to memory",
            },
        },
        "empty_artifact_semantics": "no advisory field/block is added; tested as byte-equivalent to target-only",
        "excluded_runs": [
            "initial WebShop run: local server connection refused",
            "initial ALFWorld pair: legacy split+skip selected different actual games",
        ],
        "l40s_probe": {
            "slurm_job_id": "7433578_0",
            "node": "gammagpu19",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "episode": source_episode,
            "evidence_manifest": "runs/cross_domain_shared_source_v1_smoke_l40s/authentic_skill_loaded/candy_crush/evidence/manifest.json",
        },
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["report_sha256"] = hashlib.sha256(body).hexdigest()
    output = RUN / "nonvideo_pilot_report.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
