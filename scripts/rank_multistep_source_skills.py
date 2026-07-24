#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


SPLITS = ("discovery", "qualification", "heldout")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank recorded source skill IDs using lineage only, without semantics."
    )
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    seeds = {
        str(row["episode_id"]): int(row["payload"]["requested_seed"])
        for row in rows if row.get("kind") == "RESET"
    }
    split_by_seed = {
        seed: SPLITS[index % 3] for index, seed in enumerate(sorted(seeds.values()))
    }
    selected: dict[str, dict[int, str]] = defaultdict(dict)
    for row in rows:
        if row.get("kind") != "AGENT_PROPOSAL_SET":
            continue
        payload = row.get("payload") or {}
        skill_id = payload.get("selected_skill_id")
        if skill_id:
            selected[str(row["episode_id"])][int(payload["step"])] = str(skill_id)
    skill_ids = sorted({value for episode in selected.values() for value in episode.values()})
    ranking = []
    for skill_id in skill_ids:
        split_stats = {}
        for split in SPLITS:
            steps = edges = spans = max_span = 0
            for episode_id, episode in selected.items():
                if split_by_seed[seeds[episode_id]] != split:
                    continue
                occurrences = sorted(step for step, value in episode.items() if value == skill_id)
                steps += len(occurrences)
                lengths = []
                if occurrences:
                    length = 1
                    for left, right in zip(occurrences, occurrences[1:]):
                        if right == left + 1:
                            edges += 1
                            length += 1
                        else:
                            lengths.append(length)
                            length = 1
                    lengths.append(length)
                spans += len(lengths)
                max_span = max(max_span, max(lengths, default=0))
            split_stats[split] = {
                "steps": steps, "continuous_edges": edges,
                "spans": spans, "max_span": max_span,
            }
        ranking.append({
            "skill_id": skill_id,
            "split_stats": split_stats,
            "minimum_split_edges": min(
                split_stats[split]["continuous_edges"] for split in SPLITS
            ),
            "total_continuous_edges": sum(
                split_stats[split]["continuous_edges"] for split in SPLITS
            ),
        })
    ranking.sort(
        key=lambda row: (
            -row["minimum_split_edges"], -row["total_continuous_edges"], row["skill_id"]
        )
    )
    report = {
        "schema_version": 1,
        "selection_rule": (
            "maximize minimum continuous edges across frozen splits, then total edges; "
            "skill names and action/reward contents are not inspected"
        ),
        "split_by_seed": {str(key): value for key, value in split_by_seed.items()},
        "ranking": ranking,
        "selected_skill_id": ranking[0]["skill_id"] if ranking else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_skill_id": report["selected_skill_id"],
        "candidates": len(ranking), "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
