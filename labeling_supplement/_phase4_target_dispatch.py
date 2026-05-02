#!/usr/bin/env python
"""Target-domain dispatch for the Phase-4/5/6 cross-domain transfer cycle.

Phase-4 (Day-5b) shipped game->game (within-gymv) transfer through
``_phase4_transfer_cycle.py`` with the target adapter, executor,
schema producer, and demos all hardcoded for ``target_domain='gymv'``.

Phase 5 of the cross-domain measurement plan extends this to the
remaining four target domains:

    visual_reasoning  -- VTB + TIR-Bench (image-VR)
    video             -- Video-Holmes + SIV-Bench (video-VR)
    osworld           -- desktop tasks
    browser           -- BrowserGym

Rather than rewriting ``_phase4_transfer_cycle._run_transfer`` per
domain, this module provides:

  * ``TargetBuild`` dataclass: the (adapter, harness, demos,
    success_fn_factory) bundle one transfer cell needs.
  * ``_TARGET_BUILDERS``: a per-target_domain dict of builder
    callables that each produce a ``TargetBuild`` given the parsed
    CLI args (the parent driver's ``argparse.Namespace``).
  * ``build_target(target_domain, args) -> TargetBuild``: the public
    entry point. Raises ``UnsupportedTargetDomain`` on unknown names
    and ``NotImplementedError`` for stages that haven't shipped yet.

Phase-5 sub-agents (Stages 1-4) add their per-domain builder here.
The gymv builder is the canonical reference; mirror its shape.

See ``implementation_notes/phase5-cross-domain-measurement.md`` for
the per-stage scope (which target_domain each sub-agent owns).
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("phase4_target_dispatch")


# ---------------------------------------------------------------------------
# Public API: TargetBuild + dispatcher
# ---------------------------------------------------------------------------


@dataclass
class TargetBuild:
    """Per-cell bundle returned by a target-domain builder.

    Fields:
      target_domain:
        The canonical domain string registered with
        ``common.enums.TRANSFER_TARGET_DOMAINS`` (e.g. ``'gymv'``,
        ``'visual_reasoning'``, ``'video'``, ``'osworld'``,
        ``'browser'``).
      adapter:
        The ``SkillAdapter`` (or subclass) instance with its
        executor already bound (via ``adapter.set_executor(...)`` or
        a domain-specific ``bind_<domain>_executor`` helper).
      harness:
        ``SkillHarness`` built around an ``AdapterRegistry`` that has
        ``adapter`` registered. The driver loops calls
        ``few_shot.adapt(...)`` against this harness.
      demos:
        List of ``FewShotDemo``s pre-loaded from the target-domain's
        cold-start corpus. The driver passes these to
        ``FewShotAdapter.adapt(demos=...)`` per skill.
      success_fn_factory:
        Callable taking ``(pass_rate_threshold: float,
        require_episode_success: bool) -> SuccessFn``. The driver
        constructs a fresh per-skill ``SuccessFn`` by calling this
        factory inside the skill loop. For gymv this is
        ``harness.gymv_success.make_per_step_success_fn``; for VR /
        video / osworld / browser the per-domain ``*_success.py``
        modules export an analogous factory.
    """

    target_domain: str
    adapter: Any
    harness: Any  # SkillHarness; loose-typed to avoid import cycles
    demos: List[Any]  # List[FewShotDemo]
    success_fn_factory: Callable[..., Any]


class UnsupportedTargetDomain(SystemExit):
    """Raised by ``build_target`` for unknown ``--target-domain`` names."""


# Type of a per-domain builder. Each stage sub-agent registers their
# domain's builder by adding an entry to ``_TARGET_BUILDERS`` below.
TargetBuilder = Callable[[argparse.Namespace], TargetBuild]


def build_target(target_domain: str, args: argparse.Namespace) -> TargetBuild:
    """Dispatch to the per-domain builder for ``target_domain``.

    Raises:
      UnsupportedTargetDomain: if no builder is registered.
      NotImplementedError: if the builder is the placeholder stub for
        a stage that hasn't shipped yet (Stages 1-4 land progressively
        per ``implementation_notes/phase5-cross-domain-measurement.md``).
    """
    builder = _TARGET_BUILDERS.get(target_domain)
    if builder is None:
        raise UnsupportedTargetDomain(
            f"unknown --target-domain={target_domain!r}; "
            f"registered: {sorted(_TARGET_BUILDERS)}"
        )
    return builder(args)


# ---------------------------------------------------------------------------
# Builder: gymv (canonical reference)
# ---------------------------------------------------------------------------


def _build_gymv_target(args: argparse.Namespace) -> TargetBuild:
    """Build the gymv (within-game) transfer cell.

    This is the existing Day-5b path extracted verbatim out of
    ``_phase4_transfer_cycle._run_transfer``. Mirror this shape for
    all new domains:

      1. Build the target env (deterministic where possible).
      2. Build a schema producer that round-trips through
         ``parse_schema_canonical`` -> ``StateSchema.facts``.
      3. Build the executor that the adapter calls per hop.
      4. Construct the adapter; widen ``supported_types`` so
         REASONING / GROUNDING skills also dispatch.
      5. Wire ``adapter.set_executor(executor)`` (or
         ``bind_<domain>_executor(adapter, ...)`` for VR / video).
      6. Build a ``SkillHarness`` around the adapter via
         ``AdapterRegistry``.
      7. Build target-task demos (``List[FewShotDemo]``).
      8. Pick the per-domain success_fn factory
         (``make_per_step_success_fn`` for gymv;
         ``make_qa_success_fn`` for VR/video; etc.).
    """
    # Imports are deferred so importing this module (e.g. for the
    # dispatcher table introspection) doesn't trigger heavy harness
    # imports unless we actually build a target.
    from common.enums import SkillType
    from harness import (
        AdapterRegistry,
        HarnessConfig,
        SkillHarness,
        make_gaming_env_producer,
        make_gymv_executor,
        make_per_step_success_fn,
    )
    from harness.adapters.gymv_adapter import GymvAdapter
    from harness.few_shot_demos_gymv import build_demos_from_episodes

    # Reuse the Phase-2 driver's env-builder.
    from labeling_supplement._phase2_real_env_skill_smoke import build_env

    target_game = args.target  # for gymv, --target *is* the task name
    target_env, target_env_source = build_env(target_game)
    logger.info("target env: %s (source=%s)", target_game, target_env_source)
    schema_producer = make_gaming_env_producer(target_game)
    executor, _holder = make_gymv_executor(
        target_env,
        domain="gymv",
        task=target_game,
        on_unresolved="skip",
        schema_producer=schema_producer,
    )

    adapter = GymvAdapter()
    adapter.supported_types = (
        SkillType.ACTION, SkillType.MIXED,
        SkillType.GROUNDING, SkillType.REASONING,
    )
    adapter.set_executor(executor)

    registry = AdapterRegistry()
    registry.register(adapter)
    harness = SkillHarness(registry, config=HarnessConfig(
        seed=0,
        default_budget_hops=12,
        default_budget_ms=30_000.0,
    ))

    demos = build_demos_from_episodes(
        Path(args.actions_root),
        corpus="env_wrappers",
        game=target_game,
        max_episodes=args.max_episodes,
        max_demos_per_episode=args.max_demos_per_episode,
    )
    logger.info(
        "loaded %d target-task demo(s) from %s cold-start episodes",
        len(demos), target_game,
    )

    return TargetBuild(
        target_domain="gymv",
        adapter=adapter,
        harness=harness,
        demos=demos,
        success_fn_factory=make_per_step_success_fn,
    )


# ---------------------------------------------------------------------------
# Builders: cross-domain stubs (Stages 1-4 fill in)
# ---------------------------------------------------------------------------


def _stage_not_shipped(stage: str, target_domain: str, sub_section: str) -> Callable[
    [argparse.Namespace], TargetBuild
]:
    """Factory for a placeholder builder that fails clean with a
    pointer at the stage's design-memo section."""

    def _stub(args: argparse.Namespace) -> TargetBuild:  # noqa: ARG001
        raise NotImplementedError(
            f"{stage} ({target_domain!r}) not implemented yet -- see "
            f"implementation_notes/phase5-cross-domain-measurement.md "
            f"{sub_section}. Land the per-domain builder by replacing "
            f"_TARGET_BUILDERS[{target_domain!r}] with a real "
            f"`_build_{target_domain}_target` callable that mirrors "
            f"`_build_gymv_target`'s shape."
        )

    return _stub


def _build_visual_reasoning_target(args: argparse.Namespace) -> TargetBuild:
    """Stage-1 image-VR transfer cell — VTB + TIR-Bench.

    Adapter is left on its inherited stub executor;
    `bind_visual_reasoning_executor` requires a per-sample PIL.Image
    we don't yet load. The stub still exercises dispatch + success_fn
    end-to-end so the chain runs without raising.
    """
    from common.enums import SkillType
    from harness import AdapterRegistry, HarnessConfig, SkillHarness
    from harness.adapters.visual_reasoning_adapter import VisualReasoningAdapter
    from harness.few_shot_demos_vr import build_demos_from_vr_samples
    import harness.qa_success  # noqa: F401  (registers success_fn factory)
    from harness.qa_success import make_qa_success_fn

    cold_start_root = Path(
        args.cold_start_root or "Cold-start-out-visual-reasoning"
    )
    if not cold_start_root.exists():
        raise SystemExit(
            f"cold_start_root missing: {cold_start_root} "
            f"(expected Cold-start-out-visual-reasoning/"
            f"{args.target}/sample_*.json)"
        )

    sub_corpus = args.target
    if sub_corpus not in ("visual_toolbench", "tir_bench"):
        raise SystemExit(
            f"--target {sub_corpus!r} not a known visual_reasoning "
            f"sub-corpus; expected one of: visual_toolbench, tir_bench"
        )

    adapter = VisualReasoningAdapter()
    adapter.supported_types = (
        SkillType.ACTION, SkillType.MIXED,
        SkillType.GROUNDING, SkillType.REASONING,
    )

    registry = AdapterRegistry()
    registry.register(adapter)
    harness_obj = SkillHarness(registry, config=HarnessConfig(
        seed=0, default_budget_hops=8, default_budget_ms=30_000.0,
    ))

    demos = build_demos_from_vr_samples(
        cold_start_root,
        sub_corpus=sub_corpus,
        max_demos=int(getattr(args, "max_episodes", 8))
        * int(getattr(args, "max_demos_per_episode", 1)),
    )
    logger.info("loaded %d VR demo(s) from %s/%s",
                len(demos), cold_start_root, sub_corpus)

    return TargetBuild(
        target_domain="visual_reasoning",
        adapter=adapter,
        harness=harness_obj,
        demos=demos,
        success_fn_factory=make_qa_success_fn,
    )

# Placeholder until Stage 2 lands `_build_video_target`.
_build_video_target: TargetBuilder = _stage_not_shipped(
    "Stage 2 (video-VR)", "video", "Section 5",
)

# Placeholder until Stage 3 lands `_build_osworld_target`.
_build_osworld_target: TargetBuilder = _stage_not_shipped(
    "Stage 3 (osworld)", "osworld", "Section 6",
)


# Placeholder until Stage 4 lands `_build_browser_target`.
_build_browser_target: TargetBuilder = _stage_not_shipped(
    "Stage 4 (browsergym)", "browser", "Section 7",
)


# ---------------------------------------------------------------------------
# Registry (the table sub-agents extend)
# ---------------------------------------------------------------------------


_TARGET_BUILDERS: Dict[str, TargetBuilder] = {
    "gymv": _build_gymv_target,
    "visual_reasoning": _build_visual_reasoning_target,
    "video": _build_video_target,
    "osworld": _build_osworld_target,
    "browser": _build_browser_target,
}


def registered_target_domains() -> List[str]:
    """Sorted view of currently-registered target domains."""
    return sorted(_TARGET_BUILDERS.keys())


__all__ = [
    "TargetBuild",
    "TargetBuilder",
    "UnsupportedTargetDomain",
    "build_target",
    "registered_target_domains",
]
