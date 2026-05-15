"""Tests for per-game episode-count overrides.

Originally locked in the n=16 bump for high-variance gymv games (TF3,
Altered Beast, etc.), motivated by a bootstrap analysis showing the
sampling-noise floor on bimodal-success games drops from ~22%
(P(zero-mean) at n=8) to ~4% at n=16.

As of May-2026 the bump was rolled back to n=8: investigation of the
TF3 co-evolution runs showed the dominant source of mean_reward
variance was the cross-domain ``step_checks`` predicates contaminating
the seeded skill bank, not bimodal sampling noise.  n=8 halves rollout
wall-clock; the skill-bank contamination fix is the right place to
address the variance.  The
``HIGH_VARIANCE_GYMV_EPISODES`` registry is retained as
authoritative documentation of *which* games to re-bump (and the
``--no-high-variance-defaults`` CLI flag remains the escape hatch),
but the values are aligned with the global default.

These tests still pin the override-resolution contract end-to-end so
re-bumping (or further rollback) is a one-line edit with zero
behavioural surprises.
"""

from __future__ import annotations

from trainer.coevolution.config import (
    CoEvolutionConfig,
    EPISODES_PER_GAME_MULTIROLE,
    HIGH_VARIANCE_GYMV_EPISODES,
)
from trainer.coevolution.rollout_collector import build_lpt_schedule


# ---------------------------------------------------------------------------
# Default factory now bakes in the high-variance gymv games
# ---------------------------------------------------------------------------


def test_default_overrides_bake_in_high_variance_gymv_games():
    cfg = CoEvolutionConfig()
    for g, expected_n in HIGH_VARIANCE_GYMV_EPISODES.items():
        assert cfg.episodes_per_game_overrides.get(g) == expected_n, (
            f"{g} should default to {expected_n} eps/step "
            "(see HIGH_VARIANCE_GYMV_EPISODES)"
        )


def test_default_overrides_include_multirole_constants():
    cfg = CoEvolutionConfig()
    for g, n in EPISODES_PER_GAME_MULTIROLE.items():
        # MULTIROLE constants are merged FIRST so any HIGH_VARIANCE
        # entry with the same key wins (intentional precedence).
        if g in HIGH_VARIANCE_GYMV_EPISODES:
            continue
        assert cfg.episodes_per_game_overrides.get(g) == n


# ---------------------------------------------------------------------------
# get_episodes_for_game honors overrides regardless of mode
# ---------------------------------------------------------------------------


def test_get_episodes_for_game_uses_override_in_legacy_mode():
    cfg = CoEvolutionConfig(
        episodes_per_game=8,
        unified_role_rollouts=False,
        episodes_per_game_overrides={"gymv_thunder_force_iii": 16},
    )
    assert cfg.get_episodes_for_game("gymv_thunder_force_iii") == 16
    # Games not in the override dict fall back to the global.
    assert cfg.get_episodes_for_game("tetris") == 8


def test_get_episodes_for_game_uses_override_in_unified_mode():
    cfg = CoEvolutionConfig(
        episodes_per_game=4,
        unified_role_rollouts=True,
        episodes_per_game_overrides={"avalon": 5, "diplomacy": 7},
    )
    assert cfg.get_episodes_for_game("avalon") == 5
    assert cfg.get_episodes_for_game("diplomacy") == 7
    assert cfg.get_episodes_for_game("tetris") == 4   # unrelated game ⇒ global


# ---------------------------------------------------------------------------
# build_lpt_schedule honors overrides in BOTH modes (this is the bug fix)
# ---------------------------------------------------------------------------


def _count_episodes_per_game(specs):
    counts: dict = {}
    for s in specs:
        counts[s.game] = counts.get(s.game, 0) + 1
    return counts


def test_lpt_schedule_legacy_mode_honors_overrides():
    """The original bug: under the legacy code this assertion failed
    because ``build_lpt_schedule`` only looked at overrides when
    ``unified_role_rollouts=True``.  Pin the post-fix behaviour using
    an *explicit* per-game override (n=16) so the test is robust to
    future changes to the HIGH_VAR registry defaults."""
    games = ["tetris", "gymv_thunder_force_iii"]
    schedule = build_lpt_schedule(
        games,
        episodes_per_game=8,
        episodes_per_game_overrides={"gymv_thunder_force_iii": 16},
        unified_role_rollouts=False,    # ← legacy mode
    )
    counts = _count_episodes_per_game(schedule)
    assert counts == {"tetris": 8, "gymv_thunder_force_iii": 16}


def test_lpt_schedule_legacy_mode_with_full_high_variance_default():
    """End-to-end: hand build_lpt_schedule the same overrides dict
    CoEvolutionConfig produces by default, and confirm every game uses
    the registry-resolved count.  Post-May-2026 the HIGH_VAR registry
    matches the global default (n=8), so every game returns 8 — but
    the resolution path is still exercised so a future re-bump remains
    a one-line edit."""
    cfg = CoEvolutionConfig(episodes_per_game=8)
    games = ["tetris", "candy_crush", "gymv_thunder_force_iii", "gymv_altered_beast"]
    schedule = build_lpt_schedule(
        games,
        episodes_per_game=cfg.episodes_per_game,
        episodes_per_game_overrides=cfg.episodes_per_game_overrides,
        unified_role_rollouts=False,
    )
    counts = _count_episodes_per_game(schedule)
    assert counts["tetris"] == 8
    assert counts["candy_crush"] == 8
    # HIGH_VAR registry currently aligned with global default; values
    # come *from the registry* not the fallback path (registry hit).
    for g in ("gymv_thunder_force_iii", "gymv_altered_beast"):
        assert counts[g] == HIGH_VARIANCE_GYMV_EPISODES[g], (
            f"{g} count must come from HIGH_VARIANCE_GYMV_EPISODES registry"
        )


def test_lpt_schedule_unified_mode_still_honors_overrides():
    """Unified-roles parity test (this path always worked)."""
    schedule = build_lpt_schedule(
        ["avalon"],
        episodes_per_game=4,
        episodes_per_game_overrides={"avalon": 5},
        unified_role_rollouts=True,
    )
    counts = _count_episodes_per_game(schedule)
    assert counts == {"avalon": 5}


def test_lpt_schedule_no_overrides_uses_global():
    schedule = build_lpt_schedule(
        ["tetris", "gymv_thunder_force_iii"],
        episodes_per_game=8,
        episodes_per_game_overrides=None,
        unified_role_rollouts=False,
    )
    counts = _count_episodes_per_game(schedule)
    assert counts == {"tetris": 8, "gymv_thunder_force_iii": 8}


# ---------------------------------------------------------------------------
# CLI-side resolution: ``--no-high-variance-defaults`` and ``{}`` empty-dict
# semantics (regression for the gotcha where the legacy ``{**dict, **{}}``
# merge silently kept the built-in HIGH_VAR bumps despite the help string
# claiming the opposite).
# ---------------------------------------------------------------------------


import argparse as _argparse  # noqa: E402  (import-after-tests is intentional)

from scripts.run_coevolution import resolve_episode_overrides  # noqa: E402


def _mk_args(
    *,
    episodes_per_game: int = 8,
    overrides_json=None,
    no_high_variance_defaults: bool = False,
    unified_roles: bool = False,
) -> _argparse.Namespace:
    return _argparse.Namespace(
        episodes_per_game=episodes_per_game,
        episodes_per_game_overrides=overrides_json,
        no_high_variance_defaults=no_high_variance_defaults,
        unified_roles=unified_roles,
    )


def test_resolve_default_populates_high_variance_registry():
    """Sanity: the default CLI invocation populates the HIGH_VAR
    registry entries (currently n=8) into the override map.  This pins
    the resolution path so a future re-bump (or further rollback) is a
    one-line edit in ``HIGH_VARIANCE_GYMV_EPISODES`` with no surprises
    downstream."""
    args = _mk_args()
    out = resolve_episode_overrides(args, ["gymv_thunder_force_iii", "tetris"])
    assert "gymv_thunder_force_iii" in out, (
        "HIGH_VAR registry games must appear in the override map by default"
    )
    assert out["gymv_thunder_force_iii"] == HIGH_VARIANCE_GYMV_EPISODES[
        "gymv_thunder_force_iii"
    ]
    # ``tetris`` not in the registry ⇒ not present in override map ⇒
    # caller falls back to global ``--episodes-per-game``.
    assert "tetris" not in out


def test_resolve_no_high_variance_defaults_drops_registry():
    """The ``--no-high-variance-defaults`` flag clears the HIGH_VAR
    registry from the override map so every game uses the global
    ``--episodes-per-game``.  Currently a no-op behaviourally (registry
    values match the global default) but remains the documented escape
    hatch in case the registry is re-bumped."""
    args = _mk_args(no_high_variance_defaults=True)
    out = resolve_episode_overrides(args, ["gymv_thunder_force_iii", "tetris"])
    assert "gymv_thunder_force_iii" not in out, (
        "--no-high-variance-defaults must remove every HIGH_VAR entry"
    )


def test_resolve_no_high_variance_defaults_keeps_multirole():
    """``--no-high-variance-defaults`` is narrowly scoped: it drops
    HIGH_VAR but *keeps* the Avalon/Diplomacy MULTIROLE constants since
    those exist for role-coverage fan-out, not bimodal-success
    variance reduction."""
    from trainer.coevolution.config import EPISODES_PER_GAME_MULTIROLE
    if not EPISODES_PER_GAME_MULTIROLE:
        return  # nothing to assert against
    args = _mk_args(no_high_variance_defaults=True)
    out = resolve_episode_overrides(args, list(EPISODES_PER_GAME_MULTIROLE.keys()))
    for g, n in EPISODES_PER_GAME_MULTIROLE.items():
        assert out.get(g) == n, (
            f"MULTIROLE override for {g} must survive --no-high-variance-defaults"
        )


def test_resolve_empty_dict_overrides_clears_everything():
    """Regression: under the old merge logic
    ``{**eps_overrides, **{}} == eps_overrides``, so passing an explicit
    empty dict silently kept the HIGH_VAR/MULTIROLE bumps despite the
    help string promising the opposite.  Lock in the documented
    contract."""
    args = _mk_args(overrides_json="{}")
    out = resolve_episode_overrides(args, ["gymv_thunder_force_iii", "tetris"])
    assert out == {}, "Empty JSON ``{}`` must clear every override"


def test_resolve_explicit_overrides_win_over_defaults():
    """User-supplied per-game overrides take precedence over both
    HIGH_VAR and MULTIROLE defaults."""
    args = _mk_args(overrides_json='{"gymv_thunder_force_iii": 4}')
    out = resolve_episode_overrides(args, ["gymv_thunder_force_iii"])
    assert out["gymv_thunder_force_iii"] == 4


def test_resolve_invalid_json_raises_systemexit():
    """Malformed JSON should fail loudly at startup, not silently."""
    import pytest
    args = _mk_args(overrides_json="not-json")
    with pytest.raises(SystemExit):
        resolve_episode_overrides(args, ["tetris"])

    args = _mk_args(overrides_json='["a", "b"]')   # list, not object
    with pytest.raises(SystemExit):
        resolve_episode_overrides(args, ["tetris"])
