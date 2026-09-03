from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from motif_transfer.search_automaton_transfer_v16 import SourceSearchAutomaton
from motif_transfer.sokoban_search_automaton_v16 import BACKTRACK, COMMIT, EXPLORE
from motif_transfer.tir_search_automaton_v16 import (
    AUTHENTIC,
    COMMIT_AVAILABLE,
    EXHAUSTIVE,
    LEDGER_BLIND,
    PERMUTED,
    RAW,
    execute_tir_maze_search,
)


REPO = Path(__file__).resolve().parents[1]
REAL_RECEIPTS = REPO / "runs/tir_maze_topology_v2_frozen/heldout_receipts.json"
REAL_DATASET = Path(
    "/fs/gamma-projects/vlm-robot/datasets/TIR-Bench/TIR-Bench.json"
)


@pytest.mark.skipif(
    not REAL_RECEIPTS.is_file() or not REAL_DATASET.is_file(),
    reason="real TIR receipt/dataset not materialized",
)
def test_real_tir_receipt_routes_all_three_source_actions() -> None:
    artifact = json.loads(
        (REPO / "runs/sokoban_search_automaton_v16/artifact.json").read_text()
    )
    source = SourceSearchAutomaton(artifact)
    receipt = json.loads(
        REAL_RECEIPTS.read_text()
    )[0]
    dataset = json.loads(REAL_DATASET.read_text())
    prompt = next(
        str(row["prompt"]) for row in dataset
        if str(row["id"]) == str(receipt["sample_id"])
    )
    with Image.open(receipt["image_path"]) as handle:
        image = handle.convert("RGB")
    results = {
        condition: execute_tir_maze_search(
            image=image,
            prompt=prompt,
            sample_id=str(receipt["sample_id"]),
            baseline_answer=str(receipt["baseline_answer"]),
            neural_binding=receipt["neural_binding"],
            source=source,
            condition=condition,
        )
        for condition in (
            RAW, AUTHENTIC, PERMUTED, LEDGER_BLIND, COMMIT_AVAILABLE, EXHAUSTIVE,
        )
    }
    actions = {
        row["source_action"] for row in results[AUTHENTIC]["source_decisions"]
    }
    assert actions == {BACKTRACK, COMMIT, EXPLORE}
    assert results[AUTHENTIC]["selected_answer"] == results[EXHAUSTIVE]["selected_answer"]
    assert results[PERMUTED]["selected_answer"] == receipt["baseline_answer"]
    assert results[LEDGER_BLIND]["selected_answer"] == receipt["baseline_answer"]
