from scripts.freeze_agqa_query_grounder_v2_strict_boundary_preoutcome import (
    _prediction_disagreement_opportunity,
)


def _row(source: str, neural: str) -> dict:
    return {"predictions": {"source_induced": source, "neural_only": neural}}


def test_prediction_opportunity_rejects_commits_that_copy_fallback() -> None:
    rows = [_row("chair", "chair") for _ in range(98)] + [
        _row("door", "blanket"),
        _row("sofa", "television"),
    ]
    result = _prediction_disagreement_opportunity(rows, minimum_fraction=0.05)
    assert result["source_neural_prediction_disagreements"] == 2
    assert result["source_neural_prediction_disagreement_fraction"] == 0.02
    assert not result["passes"]


def test_prediction_opportunity_passes_without_reading_outcomes() -> None:
    rows = [_row("chair", "chair") for _ in range(95)] + [
        _row(f"source-{index}", f"neural-{index}") for index in range(5)
    ]
    result = _prediction_disagreement_opportunity(rows, minimum_fraction=0.05)
    assert result["passes"]
