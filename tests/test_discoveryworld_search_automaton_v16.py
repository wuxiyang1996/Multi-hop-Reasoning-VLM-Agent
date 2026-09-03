from __future__ import annotations

import json
from pathlib import Path

import pytest

from motif_transfer.discoveryworld_search_automaton_v16 import (
    relineage_discovery_episode,
)
from motif_transfer.search_automaton_transfer_v16 import SourceSearchAutomaton
from motif_transfer.sokoban_search_automaton_v16 import COMMIT, EXPLORE


REPO = Path(__file__).resolve().parents[1]
REAL_RECEIPT = (
    REPO / "runs/discoveryworld_replication_v1_matched/"
    "proteomics.easy.seed14.json"
)


@pytest.mark.skipif(not REAL_RECEIPT.is_file(), reason="real receipt not materialized")
def test_real_discoveryworld_receipt_is_v16_route_compatible() -> None:
    source = SourceSearchAutomaton(json.loads(
        (REPO / "runs/sokoban_search_automaton_v16/artifact.json").read_text()
    ))
    result = json.loads(REAL_RECEIPT.read_text())
    replay = relineage_discovery_episode(result, source=source)
    actions = set(replay["v16_source_action_counts"])
    assert EXPLORE in actions
    assert COMMIT in actions
    assert replay["v16_route_reproduced_every_recorded_action"]
