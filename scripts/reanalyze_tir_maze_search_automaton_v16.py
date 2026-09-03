#!/usr/bin/env python3
"""Re-execute target-native TIR maze candidates under source V16 control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
)
from motif_transfer.tir_search_automaton_v16 import (  # noqa: E402
    CONDITIONS,
    evaluate_tir_maze_search,
    execute_tir_maze_search,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=REPO / "runs/sokoban_search_automaton_v16/artifact.json",
    )
    parser.add_argument(
        "--input-receipts",
        type=Path,
        default=REPO / "runs/tir_maze_topology_v2_frozen/heldout_receipts.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/TIR-Bench/TIR-Bench.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "runs/tir_search_automaton_v16/reanalysis_report.json",
    )
    args = parser.parse_args()

    artifact = json.loads(args.source_artifact.read_text(encoding="utf-8"))
    source = SourceSearchAutomaton(artifact)
    receipts = json.loads(args.input_receipts.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    prompts = {str(row["id"]): str(row["prompt"]) for row in dataset}
    replay_rows = []
    for receipt in receipts:
        sample_id = str(receipt["sample_id"])
        conditions = {}
        with Image.open(str(receipt["image_path"])) as handle:
            image = handle.convert("RGB")
        for condition in CONDITIONS:
            conditions[condition] = execute_tir_maze_search(
                image=image,
                prompt=prompts[sample_id],
                sample_id=sample_id,
                baseline_answer=str(receipt["baseline_answer"]),
                neural_binding=receipt["neural_binding"],
                source=source,
                condition=condition,
            )
        replay_rows.append({
            "sample_id": sample_id,
            "baseline_answer": str(receipt["baseline_answer"]),
            "neural_binding_valid": bool(receipt["neural_binding_valid"]),
            "conditions": conditions,
            # Evaluator-only attachment happens after all conditions emit.
            "gold_answer_evaluator_only": str(
                receipt["gold_answer_evaluator_only"]
            ),
        })
    report = evaluate_tir_maze_search(
        replay_rows,
        source_artifact_sha256=source.artifact_sha256,
        evidence_tier="PREVIOUSLY_CONSUMED_FRESH_FORMAL_REANALYSIS",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "summaries": report["summaries"],
        "paired": report["paired"],
        "gates": report["gates"],
        "report_sha256": report["report_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
