from __future__ import annotations

import pytest

from motif_transfer.vtb_evaluator import (
    VTBRubricVerdict,
    official_judge_prompt,
    parse_official_judge_response,
    parse_rubric_blob,
    score_vtb_task,
)


def test_official_critical_rule_uses_weight_not_dataset_flag() -> None:
    rubrics = parse_rubric_blob({
        "r1": {"description": "critical by weight", "weight": 4, "critical": "no"},
        "r2": {"description": "not critical by weight", "weight": 3, "critical": "yes"},
    })
    assert [row.critical for row in rubrics] == [True, False]


def test_official_ars_and_apr_across_multiple_turns() -> None:
    rubric_turns = (
        parse_rubric_blob({
            "a": {"description": "answer", "weight": 5},
            "b": {"description": "style", "weight": 2},
        }),
        parse_rubric_blob({"c": {"description": "follow-up", "weight": 4}}),
    )
    verdicts = (
        (VTBRubricVerdict("a", True, "ok"), VTBRubricVerdict("b", False, "missing")),
        (VTBRubricVerdict("c", False, "wrong"),),
    )
    score = score_vtb_task("task", rubric_turns, verdicts)
    assert score.ars == pytest.approx(5 / 11)
    assert score.apr_pass is False
    assert [row.critical_pass for row in score.turns] == [True, False]


def test_missing_turn_or_fabricated_rubric_id_fails_closed() -> None:
    rubrics = (parse_rubric_blob({"a": {"description": "x", "weight": 5}}),)
    with pytest.raises(ValueError):
        score_vtb_task("task", rubrics, ())
    with pytest.raises(ValueError):
        score_vtb_task("task", rubrics, ((VTBRubricVerdict("fake", True, "x"),),))


def test_judge_response_and_prompt_contract() -> None:
    rubric = parse_rubric_blob({"a": {"description": "states four boxes", "weight": 5}})[0]
    prompt = official_judge_prompt("How many?", "Four", rubric, "4")
    assert "states four boxes" in prompt
    verdict = parse_official_judge_response("a", {"judge_result": "Met", "explanation": "Equivalent"})
    assert verdict.met is True
    with pytest.raises(ValueError):
        parse_official_judge_response("a", {"judge_result": "maybe", "explanation": "unclear"})
