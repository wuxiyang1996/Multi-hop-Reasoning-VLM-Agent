from __future__ import annotations

import json
from pathlib import Path

import pytest

from motif_transfer.source_video_operator_algebra import induce_source_video_algebra


ROOT = Path(__file__).resolve().parents[1]


def _catalog() -> dict:
    return json.loads((ROOT / "configs/full_video_source_catalog_v1.json").read_text())


def test_full_source_catalog_qualifies_without_target_inputs() -> None:
    result = induce_source_video_algebra(root=ROOT, catalog=_catalog())
    assert result["status"] == "SOURCE_VIDEO_OPERATOR_ALGEBRA_QUALIFIED"
    assert result["target_data_read"] is False
    assert result["target_dsl_tokens_used_for_induction"] is False
    names = set(result["primitive_names"])
    assert {
        "STATE_EQUIVALENCE",
        "MEASURE_EQUIVALENCE",
        "LOGICAL_AND",
        "PRESENCE",
        "GOAL_RELATION_TEST",
        "COUNTERFACTUAL_REFUTATION",
        "ORDERED_ENDPOINTS",
    } <= names
    assert {x["source"] for x in result["source_abstentions"]} == {
        "gymv_streets_of_rage_2",
        "gymv_strider",
    }


def test_catalog_rejects_target_authorized_selection() -> None:
    catalog = _catalog()
    catalog["selection_disclosure"]["new_target_reserves_read"] = True
    with pytest.raises(ValueError, match="target blind"):
        induce_source_video_algebra(root=ROOT, catalog=catalog)


def test_primitive_names_are_target_independent() -> None:
    result = induce_source_video_algebra(root=ROOT, catalog=_catalog())
    forbidden = {"QUERY", "VERIFY", "CHOOSE", "AGQA", "CLEVRER"}
    assert not (forbidden & set(result["primitive_names"]))
    assert all(row["support"] > 0 for row in result["primitives"])


def test_measure_equality_uses_matched_repeats_and_deranged_control() -> None:
    result = induce_source_video_algebra(root=ROOT, catalog=_catalog())
    equality = next(row for row in result["primitives"] if row["name"] == "MEASURE_EQUIVALENCE")
    assert equality["support"] == 576
    assert "authentic=576/576" in equality["control"]
    assert "deranged=312/576" in equality["control"]
