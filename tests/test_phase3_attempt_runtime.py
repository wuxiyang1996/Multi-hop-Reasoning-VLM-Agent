import json
from pathlib import Path

import pytest

from motif_transfer.contracts import stable_hash
from motif_transfer.phase3_attempt_runtime import AnonymousAttemptRuntime


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "configs/phase3_source_induction_v1/frozen_confirmation/programs"
TYPED_PROGRAMS = ROOT / "configs/phase3_source_induction_v3/frozen_reserve/programs"


def artifact(game: str = "tetris"):
    return json.loads((PROGRAMS / f"{game}.json").read_text())


def runtime(game: str = "tetris", candidates=("a", "b", "c")):
    return AnonymousAttemptRuntime(
        artifact=artifact(game), candidate_ids=candidates,
        target_grounding_sha256=stable_hash(list(candidates)),
    )


def test_authentic_effects_execute_anonymous_deltas_and_validate_receipts():
    item = runtime()
    first = item.start()
    assert first.kind == "TRIAL"
    assert first.validate()
    second = item.observe("LOW")
    assert second.kind == "TRIAL"
    assert second.candidate_id != first.candidate_id
    assert len(second.operator_ids) == 2  # induced release, then activation
    final = item.observe("HIGH")
    assert final.kind == "TERMINATE"
    assert final.candidate_id == second.candidate_id
    assert final.validate()


def test_unknown_effect_fails_closed_under_frozen_abstention_rule():
    item = runtime()
    item.start()
    result = item.observe("UNKNOWN")
    assert result.kind == "ABSTAIN"
    assert result.reason == "NO_UNIQUE_QUALIFIED_OPERATOR"


def test_observed_acquisition_prefix_can_terminate_or_release_without_fake_trial():
    high = runtime()
    decision = high.resume_observed_prefix("HIGH")
    assert decision.kind == "TERMINATE"
    assert decision.candidate_id is None
    low = runtime()
    decision = low.resume_observed_prefix("LOW")
    assert decision.kind == "TRIAL"
    assert len(low.tried) == 1


@pytest.mark.parametrize("candidates", [(), ("one",), ("same", "same")])
def test_non_multiple_or_duplicate_target_candidates_abstain(candidates):
    item = runtime(candidates=candidates)
    assert item.admitted is False
    assert item.start().kind == "ABSTAIN"


def test_six_source_profiles_change_only_order_not_anonymous_operators():
    games = [path.stem for path in sorted(PROGRAMS.glob("*.json"))]
    items = {game: runtime(game, ("a", "b", "c", "d")) for game in games}
    operator_sets = {
        tuple((row.operator_id, row.state_delta) for row in item.operators)
        for item in items.values()
    }
    orders = {item.order for item in items.values()}
    assert len(operator_sets) == 1
    assert len(orders) >= 3


def test_profile_clone_not_source_name_controls_behavior():
    original = artifact("tetris")
    clone = dict(original)
    clone["source_game"] = "renamed_without_identity_feature"
    clone_body = dict(clone)
    clone_body.pop("artifact_sha256")
    clone["artifact_sha256"] = stable_hash(clone_body)
    left = AnonymousAttemptRuntime(
        artifact=original, candidate_ids=("a", "b", "c"),
        target_grounding_sha256="g",
    )
    right = AnonymousAttemptRuntime(
        artifact=clone, candidate_ids=("a", "b", "c"),
        target_grounding_sha256="g",
    )
    assert left.order == right.order
    assert left.prior.prior_sha256 == right.prior.prior_sha256


def test_v3_runtime_orders_target_candidates_by_source_selected_typed_effect():
    typed = json.loads((TYPED_PROGRAMS / "tetris.json").read_text())
    effects = [
        {"EXECUTABLE_TRANSITION_PERSISTENCE": 0.2},
        {"EXECUTABLE_TRANSITION_PERSISTENCE": 0.9},
        {"EXECUTABLE_TRANSITION_PERSISTENCE": 0.4},
    ]
    item = AnonymousAttemptRuntime(
        artifact=typed, candidate_ids=("a", "b", "c"),
        target_grounding_sha256="g", candidate_effects=effects,
    )
    assert item.admitted is True
    assert item.order == (1, 2, 0)
    assert item.start().candidate_id == "b"
    assert item.applicability_receipt["target_outcome_read"] is False


def test_v3_runtime_rebinds_current_operands_without_changing_program():
    typed = json.loads((TYPED_PROGRAMS / "tetris.json").read_text())
    item = AnonymousAttemptRuntime(
        artifact=typed, candidate_ids=("old-a", "old-b"),
        target_grounding_sha256="initial",
        candidate_effects=[
            {"EXECUTABLE_TRANSITION_PERSISTENCE": 0.1},
            {"EXECUTABLE_TRANSITION_PERSISTENCE": 0.9},
        ],
    )
    first = item.start()
    assert first.candidate_id == "old-b"
    original_program = item.program_sha256
    receipt = item.rebind_candidates(
        candidate_ids=("new-a", "new-b"),
        target_grounding_sha256="successor",
        candidate_effects=[
            {"EXECUTABLE_TRANSITION_PERSISTENCE": 0.8},
            {"EXECUTABLE_TRANSITION_PERSISTENCE": 0.2},
        ],
    )
    second = item.observe("LOW")
    assert second.kind == "TRIAL"
    assert second.candidate_id == "new-a"
    assert item.program_sha256 == original_program
    assert item.tried == {"old-b", "new-a"}
    assert receipt["target_outcome_read"] is False


def test_v3_cross_batch_unqualified_source_abstains_on_target():
    typed = json.loads((TYPED_PROGRAMS / "candy_crush.json").read_text())
    effects = [
        {"EFFECT_BY_TRANSITION_8": 0.2},
        {"EFFECT_BY_TRANSITION_8": 0.9},
    ]
    item = AnonymousAttemptRuntime(
        artifact=typed, candidate_ids=("a", "b"),
        target_grounding_sha256="g", candidate_effects=effects,
    )
    assert item.admitted is False
    result = item.start()
    assert result.kind == "ABSTAIN"
    assert result.reason == "SOURCE_TYPED_EFFECT_PROGRAM_NOT_QUALIFIED"


def test_v3_source_identity_rename_cannot_change_typed_program_behavior():
    original = json.loads((TYPED_PROGRAMS / "gymv_columns.json").read_text())
    renamed = dict(original)
    renamed["source_game"] = "anonymous_source_identity"
    renamed_body = dict(renamed)
    renamed_body.pop("artifact_sha256")
    renamed["artifact_sha256"] = stable_hash(renamed_body)
    effects = [
        {"EFFECT_BY_TRANSITION_4": 0.2},
        {"EFFECT_BY_TRANSITION_4": 0.9},
        {"EFFECT_BY_TRANSITION_4": 0.5},
    ]
    left = AnonymousAttemptRuntime(
        artifact=original, candidate_ids=("a", "b", "c"),
        target_grounding_sha256="g", candidate_effects=effects,
    )
    right = AnonymousAttemptRuntime(
        artifact=renamed, candidate_ids=("a", "b", "c"),
        target_grounding_sha256="g", candidate_effects=effects,
    )
    assert left.order == right.order == (1, 2, 0)
    assert left.prior.prior_sha256 == right.prior.prior_sha256
    assert left.start().candidate_id == right.start().candidate_id == "b"
