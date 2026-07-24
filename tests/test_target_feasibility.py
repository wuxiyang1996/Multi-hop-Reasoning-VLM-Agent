from __future__ import annotations

import pytest

from motif_transfer.target_feasibility import CellAudit, summarize_matrix


CELLS = (
    ("visual_reasoning", "visual_toolbench"),
    ("visual_reasoning", "tir_bench"),
    ("video", "video_holmes"),
    ("browser", "miniwob"),
    ("browser", "webshop"),
    ("alfworld", "alfworld_valid_seen"),
    ("alfworld", "alfworld_valid_unseen"),
)


def _row(domain: str, cell: str, **overrides) -> CellAudit:
    values = dict(
        data_ready=True,
        official_evaluator_ready=True,
        real_executor_ready=True,
        adaptation_split_ready=True,
        test_split_ready=True,
        stub_fallback_possible=False,
        target_content_in_source_treatment=False,
    )
    values.update(overrides)
    return CellAudit(domain, cell, **values)


def test_frozen_matrix_has_four_domains_and_seven_cells() -> None:
    summary = summarize_matrix(_row(domain, cell) for domain, cell in CELLS)
    assert summary["matrix"] == "4-domain/7-cell"
    assert summary["runnable_cells"] == 7
    assert len(summary["domains"]) == 4


def test_stub_and_contamination_fail_closed() -> None:
    rows = [_row(domain, cell) for domain, cell in CELLS]
    rows[0] = _row(*CELLS[0], stub_fallback_possible=True)
    rows[1] = _row(*CELLS[1], target_content_in_source_treatment=True)
    summary = summarize_matrix(rows)
    assert summary["runnable_cells"] == 5
    assert summary["cells"][0]["status"] == "STUB_FALLBACK_BLOCKED"
    assert summary["cells"][1]["status"] == "CONTAMINATED"


def test_siv_or_missing_cell_is_rejected() -> None:
    rows = [_row(domain, cell) for domain, cell in CELLS[:-1]]
    rows.append(_row("video", "siv_bench"))
    with pytest.raises(ValueError):
        summarize_matrix(rows)
