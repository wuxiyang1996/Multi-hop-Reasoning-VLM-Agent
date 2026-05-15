"""Tests for cross-domain predicate filtering in seed bank loading.

Locks in the May-2026 fix for the silent contamination of the
``frontier_data/output/per_task_banks`` seed bank.  Investigation of
the TF3 co-evolution runs revealed:

  * Skills seeded from web / board / QA / video frontier tasks carried
    predicates like ``dom_changed=true``, ``element_clicked=true``,
    ``board_transformed=true``, ``answer_confirmed=true`` in their
    ``step_checks``, ``predicate_success``, and ``predicate_abort``.

  * The legacy ``repair_step_checks_against_registry`` used a fuzzy
    substring match on the game name to look up the per-game closed
    effect set.  Runtime names like ``gymv_thunder_force_iii`` did not
    substring-match the registry key ``temporal_thunderforceiii-v0``,
    so the function fell back to the *global*
    ``EFFECT_REGISTRY`` (47 predicates) which advertised the
    cross-domain keys as valid — silently passing every contaminated
    protocol through unchanged.

  * Consequence: ``StepTracker.intrinsic_bonus`` fired 0 / 8204 times
    for the entire TF3 run, starving the ``skill_selection`` GRPO of
    learning signal and freezing reward at the SFT cold-start
    plateau.

The fix has three layers (this test exercises all three):

  1. ``canonicalize_game_key`` — explicit alias table mapping
     ``gymv_*`` wrapper names to the canonical ``temporal_*-v0``
     registry keys, plus a clean fallback hierarchy.

  2. ``repair_step_checks_against_registry`` — uses the canonical key
     and now treats every predicate whose key falls outside the
     game-specific subset as needing repair (the old code accepted
     anything in the global registry).

  3. ``filter_predicates_against_registry`` — new sibling that drops
     cross-domain predicates from ``predicate_success`` /
     ``predicate_abort`` lists.  Used by
     ``trainer.coevolution.skillbank_pipeline._sanitize_skill_in_place``
     and the LoRA protocol-synthesis path in
     ``skill_agents.pipeline``.
"""

from __future__ import annotations

import pytest

from decision_agents.protocol_utils import (
    TASK_EFFECT_SUBSET,
    canonicalize_game_key,
    filter_predicates_against_registry,
    get_valid_effects,
    repair_step_checks_against_registry,
)


# ---------------------------------------------------------------------------
# canonicalize_game_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrapper_name, expected_canonical", [
    ("gymv_thunder_force_iii",  "temporal_thunderforceiii-v0"),
    ("gymv_altered_beast",      "temporal_alteredbeast-v0"),
    ("gymv_streets_of_rage_2",  "temporal_streetsofrage2-v0"),
    ("gymv_strider",            "temporal_strider-v0"),
    ("gymv_space_harrier_ii",   "temporal_spaceharrierii-v0"),
    ("gymv_dynamite_headdy",    "temporal_dynamiteheaddy-v0"),
    ("gymv_airstriker",         "temporal_airstriker-v0"),
    ("gymv_columns",            "temporal_columns-v0"),
])
def test_canonicalize_gymv_wrapper_names(wrapper_name, expected_canonical):
    """Every gymv_* wrapper name must alias to its temporal_*-v0 registry
    key.  Without this the per-game closed-set lookup falls through to
    the global EFFECT_REGISTRY and silently accepts cross-domain
    predicates."""
    assert canonicalize_game_key(wrapper_name) == expected_canonical
    # The aliased key must actually exist in TASK_EFFECT_SUBSET — alias
    # to a missing registry entry would be just as bad as the original
    # fuzzy-match failure.
    assert expected_canonical in TASK_EFFECT_SUBSET


def test_canonicalize_native_keys_pass_through():
    """Names already matching a TASK_EFFECT_SUBSET key are returned
    unchanged."""
    for native in ("tetris", "candy_crush", "twenty_forty_eight",
                   "temporal_thunderforceiii-v0", "miniwob", "webshop"):
        assert canonicalize_game_key(native) == native


def test_canonicalize_unknown_game_returns_input():
    """Unknown games fall through to the input (callers then use
    ``EFFECT_REGISTRY`` as a last-resort fallback)."""
    assert canonicalize_game_key("foobar_unknown_game") == "foobar_unknown_game"
    assert canonicalize_game_key("") == ""


# ---------------------------------------------------------------------------
# get_valid_effects (TF3 specifically — the bug originally surfaced here)
# ---------------------------------------------------------------------------


def test_get_valid_effects_tf3_excludes_cross_domain():
    """TF3 (gymv shooter) must NOT return web/board/QA predicates.

    Pre-fix this returned all 47 keys from EFFECT_REGISTRY because
    the fuzzy substring match between ``gymv_thunder_force_iii`` and
    ``temporal_thunderforceiii-v0`` failed silently."""
    effects = set(get_valid_effects("gymv_thunder_force_iii"))

    # Shooter-specific predicates that SHOULD be present
    expected_present = {"enemy_hit", "projectile_fired", "damage_taken",
                        "state_observed", "action_taken"}
    missing = expected_present - effects
    assert not missing, f"TF3 closed-set missing core shooter predicates: {missing}"

    # Cross-domain predicates that MUST NOT be present
    forbidden = {"dom_changed", "element_clicked", "board_transformed",
                 "board_reshuffled", "answer_confirmed", "page_navigated",
                 "form_filled"}
    leaked = forbidden & effects
    assert not leaked, (
        f"TF3 closed-set leaked cross-domain predicates: {leaked} "
        "(the contamination bug has regressed)"
    )


# ---------------------------------------------------------------------------
# repair_step_checks_against_registry — the contamination fix
# ---------------------------------------------------------------------------


def test_repair_drops_web_predicates_for_tf3():
    """Pre-fix this returned ``was_repaired=False`` and left
    ``dom_changed=true`` in place; lock in the new behaviour."""
    bad_checks = ["dom_changed=true", "element_clicked=true", "enemy_hit=true"]
    steps = ["observe", "click target", "hit enemy"]
    repaired, was_repaired = repair_step_checks_against_registry(
        bad_checks, steps, game_name="gymv_thunder_force_iii",
    )
    assert was_repaired, (
        "repair must trigger on cross-domain predicates "
        "(web ``dom_changed``/``element_clicked`` in TF3 protocol)"
    )
    assert len(repaired) == len(steps)
    # Repaired keys must come from the TF3 closed-set or be empty.
    allowed = set(get_valid_effects("gymv_thunder_force_iii"))
    for chk in repaired:
        if not chk:
            continue
        key = chk.split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip()
        assert key in allowed, (
            f"Repaired check {chk!r} still uses off-subset key {key!r}"
        )


def test_repair_drops_board_predicates_for_tf3():
    """Board-game predicates (``board_transformed``,
    ``answer_confirmed``) leaking into a TF3 protocol must be
    repaired."""
    bad_checks = ["board_transformed=true", "answer_confirmed=true"]
    steps = ["transform the board", "confirm answer"]
    _repaired, was_repaired = repair_step_checks_against_registry(
        bad_checks, steps, game_name="gymv_thunder_force_iii",
    )
    assert was_repaired


def test_repair_passes_through_clean_protocol():
    """A protocol whose step_checks all reference the game's closed
    set must pass through unchanged."""
    clean = ["state_observed=true", "enemy_hit=true", "damage_taken=false"]
    steps = ["observe", "fire", "evade"]
    repaired, was_repaired = repair_step_checks_against_registry(
        clean, steps, game_name="gymv_thunder_force_iii",
    )
    assert not was_repaired
    assert repaired == clean


def test_repair_handles_empty_step_checks():
    repaired, was_repaired = repair_step_checks_against_registry(
        [], ["any"], game_name="gymv_thunder_force_iii",
    )
    assert not was_repaired
    assert repaired == []


# ---------------------------------------------------------------------------
# filter_predicates_against_registry — predicate_success/predicate_abort fix
# ---------------------------------------------------------------------------


def test_filter_drops_cross_domain_keeps_shooter():
    bad = ["dom_changed=true", "element_clicked=true",
           "enemy_hit=true", "board_transformed=true"]
    filtered, n_dropped = filter_predicates_against_registry(
        bad, game_name="gymv_thunder_force_iii",
    )
    assert filtered == ["enemy_hit=true"]
    assert n_dropped == 3


def test_filter_preserves_clean_predicates():
    clean = ["enemy_hit=true", "damage_taken=false", "reward_positive=true"]
    filtered, n_dropped = filter_predicates_against_registry(
        clean, game_name="gymv_thunder_force_iii",
    )
    assert filtered == clean
    assert n_dropped == 0


def test_filter_passes_through_unknown_game():
    """For games with no closed-set registered we can't safely filter
    (we'd risk dropping everything).  Leave the list intact."""
    weird = ["foo=bar", "baz=qux"]
    filtered, n_dropped = filter_predicates_against_registry(
        weird, game_name="totally_unknown_game",
    )
    assert filtered == weird
    assert n_dropped == 0


def test_filter_handles_empty_list():
    filtered, n_dropped = filter_predicates_against_registry(
        [], game_name="gymv_thunder_force_iii",
    )
    assert filtered == []
    assert n_dropped == 0


# ---------------------------------------------------------------------------
# Integration: full sanitization of a contaminated seed-bank skill
# ---------------------------------------------------------------------------


def test_sanitize_skill_in_place_for_tf3():
    """End-to-end: a minimal Skill object carrying every flavour of
    cross-domain contamination must come out with only TF3-valid
    predicates after ``_sanitize_skill_in_place``."""
    from trainer.coevolution.skillbank_pipeline import _sanitize_skill_in_place
    from skill_agents.stage3_mvp.schemas import Protocol, Skill

    proto = Protocol(
        preconditions=["player visible"],
        steps=["observe enemy", "fire weapon", "evade"],
        success_criteria=["enemy destroyed"],
        abort_criteria=["damage taken"],
        # 1 valid + 2 cross-domain (web, board)
        step_checks=["enemy_hit=true", "dom_changed=true", "board_transformed=true"],
        # 1 valid + 2 cross-domain (web, QA)
        predicate_success=["enemy_hit=true",
                           "element_clicked=true",
                           "answer_confirmed=true"],
        # 1 valid + 1 cross-domain (web)
        predicate_abort=["damage_taken=true", "page_navigated=true"],
        action_vocab=["FIRE", "UP", "LEFT", "RIGHT"],
        source="seed",
    )
    skill = Skill(
        skill_id="TEST_TF3_SHOOT",
        version=1,
        name="test_shoot",
        protocol=proto,
    )

    stats = _sanitize_skill_in_place(skill, game_name="gymv_thunder_force_iii")
    assert stats.get("step_checks_repaired") == 1, (
        f"step_checks should have been repaired; got stats={stats}"
    )
    assert stats.get("success_predicates_dropped") == 2, (
        f"2 cross-domain success predicates should have been dropped; got stats={stats}"
    )
    assert stats.get("abort_predicates_dropped") == 1, (
        f"1 cross-domain abort predicate should have been dropped; got stats={stats}"
    )
    # Survivors must all reference TF3 closed-set keys.
    allowed = set(get_valid_effects("gymv_thunder_force_iii"))
    for pred in (list(skill.protocol.predicate_success)
                 + list(skill.protocol.predicate_abort)):
        key = pred.split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip()
        assert key in allowed, (
            f"Sanitized skill still carries off-subset predicate {pred!r}"
        )


def test_sanitize_skill_no_op_for_unknown_game():
    """No closed-set registered ⇒ sanitization is a no-op (safety:
    we'd rather keep noise than drop a domain we don't understand)."""
    from trainer.coevolution.skillbank_pipeline import _sanitize_skill_in_place
    from skill_agents.stage3_mvp.schemas import Protocol, Skill

    proto = Protocol(
        preconditions=[],
        steps=["do a thing"],
        success_criteria=[],
        abort_criteria=[],
        step_checks=["weird_predicate=true"],
        predicate_success=["foo=bar"],
        predicate_abort=["baz=qux"],
        action_vocab=[],
        source="seed",
    )
    skill = Skill(skill_id="TEST_UNKNOWN", version=1,
                  name="test", protocol=proto)
    stats = _sanitize_skill_in_place(skill, game_name="some_unknown_game")
    assert stats == {}, "Unknown game must produce empty stats dict (no-op)"
    assert skill.protocol.step_checks == ["weird_predicate=true"]
    assert skill.protocol.predicate_success == ["foo=bar"]
    assert skill.protocol.predicate_abort == ["baz=qux"]
