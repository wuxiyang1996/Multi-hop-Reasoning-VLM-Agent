import json
from pathlib import Path

from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_source_portfolio import (
    permute_selected_effect_binding,
    select_source_program_portfolio,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "configs/phase3_source_induction_v3/frozen_reserve/programs"


def artifacts():
    return [json.loads(path.read_text()) for path in sorted(PROGRAMS.glob("*.json"))]


def test_portfolio_selects_content_not_source_identity():
    rows = artifacts()
    effects = [
        {
            "EFFECT_BY_TRANSITION_1": 0.2,
            "EFFECT_BY_TRANSITION_4": 0.95,
            "EFFECT_BY_TRANSITION_8": 0.5,
            "EXECUTABLE_TRANSITION_PERSISTENCE": 0.4,
        },
        {
            "EFFECT_BY_TRANSITION_1": 0.3,
            "EFFECT_BY_TRANSITION_4": 0.1,
            "EFFECT_BY_TRANSITION_8": 0.3,
            "EXECUTABLE_TRANSITION_PERSISTENCE": 0.5,
        },
        {
            "EFFECT_BY_TRANSITION_1": 0.4,
            "EFFECT_BY_TRANSITION_4": 0.2,
            "EFFECT_BY_TRANSITION_8": 0.4,
            "EXECUTABLE_TRANSITION_PERSISTENCE": 0.6,
        },
    ]
    receipt = select_source_program_portfolio(
        rows, candidate_ids=("a", "b", "c"), candidate_effects=effects,
        target_grounding_sha256="g",
    )
    assert receipt["selected_effect_type"] == "EFFECT_BY_TRANSITION_4"
    assert receipt["source_identity_used_as_feature"] is False
    assert receipt["target_outcome_read"] is False

    # Renaming and reordering the outer source records cannot change content
    # selection because only frozen program measurements enter the score.
    renamed = []
    for index, artifact in enumerate(reversed(rows)):
        row = dict(artifact)
        row["source_game"] = f"anonymous-{index}"
        body = dict(row); body.pop("artifact_sha256")
        row["artifact_sha256"] = stable_hash(body)
        renamed.append(row)
    second = select_source_program_portfolio(
        renamed, candidate_ids=("a", "b", "c"), candidate_effects=effects,
        target_grounding_sha256="g",
    )
    assert second["selected_program_sha256"] == receipt["selected_program_sha256"]
    assert second["selected_effect_type"] == receipt["selected_effect_type"]


def test_target_effect_binding_control_is_nonidentity_and_outcome_blind():
    artifact = json.loads((PROGRAMS / "gymv_columns.json").read_text())
    program = artifact["typed_effect_program"]
    effects = [
        {"EFFECT_BY_TRANSITION_4": 0.9},
        {"EFFECT_BY_TRANSITION_4": 0.3},
        {"EFFECT_BY_TRANSITION_4": 0.1},
    ]
    permuted, receipt = permute_selected_effect_binding(
        program, candidate_ids=("a", "b", "c"), candidate_effects=effects,
    )
    assert [row["EFFECT_BY_TRANSITION_4"] for row in permuted] != [0.9, 0.3, 0.1]
    assert receipt["nonidentity"] is True
    assert receipt["target_outcome_read"] is False
