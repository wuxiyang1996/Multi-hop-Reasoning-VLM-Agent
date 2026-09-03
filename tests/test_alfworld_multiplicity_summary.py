from __future__ import annotations

import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


REPO = Path(__file__).resolve().parents[1]


def test_consumed_development_uses_task_level_multiplicity_gate() -> None:
    report = json.loads((
        REPO / "docs/results/alfworld_multiplicity_v1_consumed_development_summary.json"
    ).read_text(encoding="utf-8"))
    body = dict(report)
    claimed = body.pop("summary_sha256")
    assert stable_hash(body) == claimed
    assert report["status"] == "CONSUMED_DEVELOPMENT_MULTIPLICITY_GATE_PASSED"
    assert report["raw_runner_status_preserved"] == "QUALIFICATION_CANDIDATE_FAILED"
    assert report["task_level_changed_option_tasks"] == 11
    assert report["paired_authentic_vs_target"]["wins"] == 3
    assert report["paired_authentic_vs_target"]["losses"] == 1
    assert all(report["gates"].values())


def test_formal_reserve_is_the_previously_locked_six_task_set() -> None:
    master = json.loads((
        REPO / "configs/four_domain_replication_v1_manifest.json"
    ).read_text(encoding="utf-8"))
    formal = json.loads((
        REPO / "configs/alfworld_multiplicity_v1_formal_manifest.json"
    ).read_text(encoding="utf-8"))
    heldout = formal["cells"]["alfworld_valid_unseen"]["splits"]["held_out"]
    assert heldout == master["locked_future_alfworld_multiplicity_ids"]
    assert len(heldout) == len(set(heldout)) == 6
    assert formal["development_outcomes_used_for_task_selection"] is False
