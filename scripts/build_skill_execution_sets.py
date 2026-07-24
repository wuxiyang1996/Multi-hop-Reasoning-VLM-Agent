#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

from motif_transfer.instrumented_import import import_native_source_batch
from motif_transfer.skill_internal import build_execution_sets, load_skill_hypotheses


DEFAULT_BANK_ROOT = Path(
    "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/"
    "runs/b2_best_checkpoints"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build receipt-grounded, per-skill execution sets from old Phase-1 logs"
    )
    parser.add_argument("evidence_dirs", nargs="+")
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for evidence_dir_text in args.evidence_dirs:
        evidence_dir = Path(evidence_dir_text)
        episodes = import_native_source_batch(evidence_dir)
        if not episodes:
            continue
        games = {episode.game for episode in episodes}
        if len(games) != 1:
            raise ValueError(f"mixed games in {evidence_dir}: {sorted(games)}")
        game = next(iter(games))
        bank_path = args.bank_root / game / "banks" / game / "skill_bank.jsonl"
        hypotheses = load_skill_hypotheses(bank_path) if bank_path.is_file() else {}
        sets = build_execution_sets(game, episodes, hypotheses)
        for execution_set in sets:
            split_counts = {
                split: sum(row.split == split for row in execution_set.executions)
                for split in ("discovery", "qualification", "held_out")
            }
            rows.append({
                "evidence_dir": str(evidence_dir),
                "bank_path": str(bank_path) if bank_path.is_file() else None,
                "bank_skill_found": execution_set.skill_id in hypotheses,
                "execution_set": asdict(execution_set),
                "summary": {
                    "executions": len(execution_set.executions),
                    "transitions": len(execution_set.transition_receipt_ids),
                    "split_executions": split_counts,
                    "eligible_for_agent_discovery": split_counts["discovery"] > 0,
                    "eligible_for_three_way_evaluation": all(split_counts.values()),
                },
            })

    report = {
        "schema_version": 1,
        "authority": "RECORDED_EXECUTION_MEMBERSHIP_ONLY",
        "notes": [
            "selected_skill_id defines membership, never internal graph boundaries",
            "historical bank content is an untrusted hypothesis sidecar",
            "episode-level splits are frozen before graph proposal",
        ],
        "execution_sets": rows,
        "totals": {
            "games": len({row["execution_set"]["game"] for row in rows}),
            "skills": len(rows),
            "executions": sum(row["summary"]["executions"] for row in rows),
            "transitions": sum(row["summary"]["transitions"] for row in rows),
            "bank_skills_found": sum(row["bank_skill_found"] for row in rows),
            "three_way_eligible": sum(
                row["summary"]["eligible_for_three_way_evaluation"] for row in rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
