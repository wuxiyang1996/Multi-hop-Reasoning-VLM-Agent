"""Bridge between :mod:`skill_bank.shared_abstract_bank` and
:class:`harness.few_shot_adapter.FewShotAdapter`.

The bridge implements **Path A** of the cold-start validation design:
when a new :class:`BoundConcreteSkill` is forward-bound to a target
task, we don't ship it as ``binding_status="PENDING"`` and hope —
instead we pull every prior natively-mined binding of the parent
abstract from the per-task bank, synthesise a :class:`FewShotDemo`
from each (using the source binding's contract preconditions as
``state.facts`` and the highest-quality :class:`SubEpisodeRef` as the
evidence anchor), then run the candidate binding through
:meth:`FewShotAdapter.adapt`.  The returned ``pass_rate`` decides
whether the binding lands as ``VALIDATED`` (≥ ``pass_rate_min``),
``REJECTED`` (no demo passed), or ``PENDING`` (no demos available).

This is the same machinery the orchestrator's Stage-3a few-shot
gate uses for ACTIVE-promotion in production — re-purposed for
cold-start seed validation.

Coverage: every task currently in ``shared_skill_bank/_latest/by_task``
maps onto one of the canonical harness domains
(``gymv | browser | alfworld | video | visual_reasoning``) via
:func:`task_to_harness_domain`.  ``env_wr_game`` cohort tasks
(``candy_crush``, ``tetris``, ``super_mario``, ``twenty_forty_eight``)
ride the ``gymv`` adapter as intra-domain task transfers — the
:class:`harness.adapters.gymv_adapter.GymvAdapter`'s deterministic
hop-walker doesn't care about the underlying engine, it just
exercises the protocol structure against the demo state.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from common.enums import SkillSourceType, SkillStatus, SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.adapters.alfworld_adapter import AlfworldAdapter
from harness.adapters.browser_adapter import BrowserAdapter
from harness.adapters.gymv_adapter import GymvAdapter
from harness.adapters.video_adapter import VideoAdapter
from harness.adapters.visual_reasoning_adapter import VisualReasoningAdapter
from harness.few_shot_adapter import AdaptResult, FewShotAdapter, FewShotDemo
from harness.skill_harness import SkillHarness
from skill_bank.shared_abstract_bank import (
    BoundConcreteSkill, ProtocolStep, SharedAbstractSkill, TwoLayerSkillStore,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task → harness domain map.
#
# Mirrors ``scripts/build_shared_skill_bank.COHORT_OF_KNOWN`` plus the
# "Temporal_*-v0 / gymv_*" gymv prefix rules.  Every PerTaskBank task
# present in the repo at the time of writing maps cleanly.
# ---------------------------------------------------------------------------
TASK_TO_HARNESS_DOMAIN: Dict[str, str] = {
    # env-wrapper games — gym-style 2D puzzlers / platformers / merge games.
    # These ride the gymv adapter as intra-domain task transfers.
    "candy_crush":        "gymv",
    "tetris":             "gymv",
    "super_mario":        "gymv",
    "twenty_forty_eight": "gymv",
    # Web tasks
    "miniwob":  "browser",
    "webshop":  "browser",
    # Embodied text/control tasks
    "alfworld": "alfworld",
    # Visual reasoning over images
    "tir_bench":         "visual_reasoning",
    "visual_toolbench":  "visual_reasoning",
    # Video reasoning
    "video_holmes": "video",
    "siv_bench":    "video",
}


def task_to_harness_domain(task: str) -> str:
    """Return the canonical harness domain for ``task``.

    Resolution order:

      1. Exact match in :data:`TASK_TO_HARNESS_DOMAIN`.
      2. ``Temporal_<game>-v0`` (gymv suite) and ``gymv_*`` prefixes
         → ``gymv``.
      3. Default → ``gymv``: every gym-style task we currently track
         (and any future Temporal_* additions) reuses the gymv
         adapter's hop-walker; the F2′ task-axis veto keeps cross-
         task bleed contained.
    """
    if task in TASK_TO_HARNESS_DOMAIN:
        return TASK_TO_HARNESS_DOMAIN[task]
    if task.startswith("Temporal_") and task.endswith("-v0"):
        return "gymv"
    if task.startswith("gymv_"):
        return "gymv"
    return "gymv"


# ---------------------------------------------------------------------------
# Adapter registry construction (cached, source plus active targets).
# ---------------------------------------------------------------------------
_REGISTRY: Optional[AdapterRegistry] = None


def _get_registry() -> AdapterRegistry:
    """Build (once) and return a registry with every harness adapter
    registered.  Adapters are stateless, so a process-wide singleton
    is safe."""
    global _REGISTRY
    if _REGISTRY is None:
        reg = AdapterRegistry()
        reg.register(GymvAdapter())
        reg.register(BrowserAdapter())
        reg.register(AlfworldAdapter())
        reg.register(VideoAdapter())
        reg.register(VisualReasoningAdapter())
        _REGISTRY = reg
    return _REGISTRY


# ---------------------------------------------------------------------------
# BoundConcreteSkill -> SkillRecord (the candidate the harness validates)
# ---------------------------------------------------------------------------
def _protocol_steps_to_hop_dicts(
    steps: List[ProtocolStep],
) -> List[Dict[str, Any]]:
    """Convert :class:`ProtocolStep`\\ s into the typed hop-dict shape
    that :func:`harness.adapters._common.iter_hops` /
    :func:`normalize_hop_action` consume.  The hop-walker reads
    ``hop["op"|"action"]`` for the action verb and ``hop["payload"]``
    for slot bindings."""
    hops: List[Dict[str, Any]] = []
    for s in steps:
        hops.append({
            "action":            s.op or "EXEC",
            "op":                s.op or "EXEC",
            "payload":           dict(s.payload or {}),
            "slot_types":        dict(s.slot_types or {}),
            "preconditions":     list(s.preconditions or []),
            "effects_add":       list(s.effects_add or []),
            "effects_del":       list(s.effects_del or []),
            "evidence_role":     s.evidence_role or "COMMIT",
            "notes":             s.notes or "",
            "success_criteria":  list(s.success_criteria or []),
            "abort_criteria":    list(s.abort_criteria or []),
        })
    return hops


def binding_to_skill_record(
    binding: BoundConcreteSkill,
    *,
    target_domain: str,
    source_domains: Optional[List[str]] = None,
) -> SkillRecord:
    """Wrap a :class:`BoundConcreteSkill` as a :class:`SkillRecord`
    that :class:`FewShotAdapter` and the harness adapters consume.

    ``source_domains`` defaults to ``["gymv"]`` because that's where
    the abstract-skeleton evidence almost always originates (mining
    over Temporal_*-v0 + env-wrapper games).  Overriding it is
    rarely needed; the harness uses it only for the asymmetric
    transfer thesis (skill must have a foundry lineage to claim
    cross-domain support).
    """
    contract = SkillContract(
        preconditions=list(binding.contract.get("preconditions") or []),
        effects_add=list(binding.contract.get("eff_add") or []),
        effects_del=list(binding.contract.get("eff_del") or []),
        success_criteria=list(binding.contract.get("postconditions") or []),
        abort_criteria=[
            c for s in binding.protocol for c in (s.abort_criteria or [])
        ][:6],
        expected_evidence_roles=["COMMIT"],
    )
    rec = SkillRecord.new(
        name=binding.name or binding.concrete_skill_id,
        skill_type=SkillType.MIXED,
        source_type=SkillSourceType.MINED,
        feasible_domains=[target_domain],
        feasible_tasks=[binding.task],
        source_domains=list(source_domains or ["gymv"]),
        protocol=_protocol_steps_to_hop_dicts(binding.protocol),
        contract=contract,
    )
    # Override the freshly-minted UUID with the bank's stable skill_id
    # so any rejection-log / validate audit row references the right
    # binding.  Same pattern ``_record_from_bank_entry`` uses.
    object.__setattr__(rec, "skill_id", binding.concrete_skill_id)
    object.__setattr__(rec, "status", SkillStatus.PROVISIONAL)
    return rec


# ---------------------------------------------------------------------------
# SharedAbstractSkill.lineage -> List[FewShotDemo]
# ---------------------------------------------------------------------------
def _facts_from_preconditions(preconditions: List[Any]) -> Dict[str, float]:
    """Turn a binding's contract preconditions into a flat
    ``{predicate: 1.0}`` ``state.facts`` dict.  Both string-form
    (legacy) and dict-form (Day-2 lifted) predicates are accepted."""
    facts: Dict[str, float] = {}
    for p in preconditions or []:
        if isinstance(p, str):
            key = p.strip()
            if key:
                facts[key] = 1.0
        elif isinstance(p, dict):
            t = p.get("type") or p.get("predicate") or p.get("name")
            if t:
                facts[str(t)] = 1.0
    return facts


def _best_sub_episode(binding: BoundConcreteSkill) -> Any:
    """Pick the highest-quality successful sub-episode (ties broken
    by ``cumulative_reward``).  Returns ``None`` if the binding has
    no sub-episode receipts."""
    candidates = [s for s in (binding.sub_episodes or [])
                  if s.outcome == "success"]
    if not candidates:
        candidates = list(binding.sub_episodes or [])
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda s: (
            float(getattr(s, "quality_score", 0.0) or 0.0),
            float(getattr(s, "cumulative_reward", 0.0) or 0.0),
        ),
    )


def lineage_to_demos(
    abstract: SharedAbstractSkill,
    bank: TwoLayerSkillStore,
    *,
    max_demos: int = 8,
    require_native: bool = True,
    require_distinct_tasks: bool = True,
    exclude_task: Optional[str] = None,
) -> List[FewShotDemo]:
    """Materialise a :class:`FewShotDemo` per cross-task lineage entry
    on ``abstract``.

    For each lineage entry we look up the prior :class:`BoundConcreteSkill`
    in PerTaskBank, build a :class:`StateSchema` whose ``facts`` are
    the source binding's contract preconditions, anchor the state's
    ``inner_step`` / ``outer_step`` to the highest-quality
    :class:`SubEpisodeRef`'s ``seg_start``, and attach the
    sub-episode id as evidence so the harness can correlate the
    validation back to the original rollout.

    Filters:

    * ``require_native``  — drop ``discovered_via in {binding,
      translation}`` lineage entries (forward-bound or translated
      records aren't first-hand evidence).
    * ``require_distinct_tasks`` — at most one demo per source task
      (prefer breadth over depth for cross-game generalisation).
    * ``exclude_task``    — skip lineage entries whose ``task`` matches
      (typically the target task we're binding *to*; we don't want
      to seed it as its own demo).
    """
    demos: List[FewShotDemo] = []
    seen_tasks: set[str] = set()
    for L in abstract.lineage:
        if exclude_task and L.task == exclude_task:
            continue
        if require_native:
            if not L.is_native:
                continue
            if L.discovered_via in {"binding", "translation"}:
                continue
        if require_distinct_tasks and L.task in seen_tasks:
            continue
        per_task = bank.per_task(L.task)
        src = per_task.by_concrete_id(L.concrete_skill_id)
        if src is None:
            # The lineage entry references a concrete_skill_id we
            # don't have a binding record for (older bank rotation
            # or a translation-only entry).  Skip — we'd be guessing.
            continue
        sub = _best_sub_episode(src)
        facts = _facts_from_preconditions(src.contract.get("preconditions", []))
        demo_state = StateSchema(
            task=L.task,
            domain=task_to_harness_domain(L.task),
            inner_step=int(getattr(sub, "seg_start", 0)) if sub else 0,
            outer_step=int(getattr(sub, "seg_start", 0)) if sub else 0,
            facts=facts,
            evidence=[
                EvidenceRef(
                    source=f"prior_binding:{L.task}/{L.concrete_skill_id}",
                    locator=(
                        f"episode={sub.episode_id}#{sub.seg_start}-{sub.seg_end}"
                        if sub else f"binding={L.concrete_skill_id}"
                    ),
                    role="REASON",
                    confidence=float(getattr(sub, "quality_score", 0.5) or 0.5)
                                if sub else 0.5,
                )
            ],
        )
        # Bindings the source binding's protocol expected to bind —
        # passed down so the harness's hop-walker can resolve slot
        # references at validate time.  Best-effort: we copy any
        # ``payload`` the prior binding's first step had.
        bindings: Dict[str, Any] = {}
        if src.protocol and src.protocol[0].payload:
            bindings.update(src.protocol[0].payload)
        demo = FewShotDemo(
            state=demo_state,
            bindings=bindings,
            expected={
                "outcome":        "success" if (sub and sub.outcome == "success") else "unknown",
                "source_task":    L.task,
                "source_binding": L.concrete_skill_id,
                "quality_score":  float(getattr(sub, "quality_score", 0.0) or 0.0)
                                  if sub else 0.0,
            },
        )
        demos.append(demo)
        seen_tasks.add(L.task)
        if len(demos) >= max_demos:
            break
    return demos


# ---------------------------------------------------------------------------
# Public API: validate a forward-bound binding against its abstract's
# cross-game evidence.
# ---------------------------------------------------------------------------
def _source_ops_vocabulary(
    abstract: SharedAbstractSkill, bank: TwoLayerSkillStore,
    *, exclude_task: Optional[str] = None,
) -> set:
    """Collect the set of action verbs that the source bindings of
    ``abstract`` actually used.  Used by the bridge's success_fn to
    score candidate bindings on action-vocabulary overlap.
    """
    vocab: set = set()
    for L in abstract.lineage:
        if exclude_task and L.task == exclude_task:
            continue
        if L.discovered_via in {"binding", "translation"}:
            continue
        src = bank.per_task(L.task).by_concrete_id(L.concrete_skill_id)
        if src is None:
            continue
        for s in src.protocol:
            if s.op:
                vocab.add(s.op.upper())
    # Always allow the abstract's own template ops as a fallback —
    # they're the canonical evidence-role verbs.
    for ts in abstract.protocol_steps:
        if ts.op:
            vocab.add(ts.op.upper())
    return vocab


def _make_structural_success_fn(
    skill_record: SkillRecord, source_vocab: set,
):
    """Build a success_fn closure that scores candidate bindings on:

      1. The harness episode succeeded (deterministic walker
         completed) AND contract_satisfied is True.
      2. The candidate uses at least one action verb the source
         family used (vocabulary overlap).

    Returns ``1.0`` for full pass, ``0.5`` for partial (ran but
    no vocab overlap), ``0.0`` for any fail.
    """
    cand_ops = {
        str(h.get("op") or h.get("action") or "").upper()
        for h in skill_record.protocol
    } - {""}

    def fn(episode, demo) -> float:                                   # noqa: ANN001
        out = episode.outcome
        if out is None or not (out.success and out.contract_satisfied):
            return 0.0
        if not cand_ops:
            return 0.0
        if not source_vocab:
            # No vocab to compare against — fall back to base scoring.
            return 1.0
        overlap = cand_ops & source_vocab
        if not overlap:
            return 0.5
        return 1.0

    return fn


def validate_binding_via_harness(
    *,
    candidate_binding: BoundConcreteSkill,
    abstract: SharedAbstractSkill,
    bank: TwoLayerSkillStore,
    max_demos: int = 8,
    pass_rate_min: float = 0.5,
    min_protocol_len: int = 2,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Validate ``candidate_binding`` (just-produced by the LLM
    forward-bind path) against demos pulled from ``abstract.lineage``.

    Returns ``(verdict, diagnostics)`` where ``verdict`` is:

    * ``True``   — ``pass_rate >= pass_rate_min`` and ``n_total > 0``;
                   set ``binding_status = "VALIDATED"``.
    * ``False``  — ``n_total > 0`` but ``pass_rate < pass_rate_min``;
                   set ``binding_status = "REJECTED"``.
    * ``None``   — couldn't even attempt validation (no native
                   lineage demos, or adapter unavailable for the
                   target domain); leave ``binding_status="PENDING"``
                   and rely on the trainer's first-rollout feedback.

    Diagnostics include the full :class:`AdaptResult`-derived shape
    (``pass_rate / n_success / n_total / diagnostic_label /
    episode_ids``) plus the source-task list used as demos.
    """
    target_domain = task_to_harness_domain(candidate_binding.task)
    diag: Dict[str, Any] = {
        "validator":        "FewShotAdapter",
        "target_task":      candidate_binding.task,
        "target_domain":    target_domain,
        "pass_rate_min":    pass_rate_min,
        "n_protocol_steps": len(candidate_binding.protocol),
    }

    # Pre-flight: a candidate with fewer than `min_protocol_len`
    # hops is degenerate by construction — skill bank invariants
    # require a multi-hop protocol.  Reject without spending demos.
    if len(candidate_binding.protocol) < min_protocol_len:
        diag["reason"] = (
            f"protocol_too_short: len={len(candidate_binding.protocol)} "
            f"< min={min_protocol_len}"
        )
        diag["verdict"] = "REJECTED"
        return False, diag

    demos = lineage_to_demos(
        abstract, bank,
        max_demos=max_demos,
        exclude_task=candidate_binding.task,
    )
    diag["n_demos"]          = len(demos)
    diag["demo_source_tasks"] = [d.expected.get("source_task", "")
                                  for d in demos]
    if not demos:
        diag["reason"] = "no_lineage_demos"
        return None, diag

    skill_record = binding_to_skill_record(
        candidate_binding, target_domain=target_domain,
    )

    registry = _get_registry()
    if registry.get(target_domain, skill_record.skill_type) is None:
        diag["reason"] = "no_adapter_for_target_domain"
        return None, diag

    source_vocab = _source_ops_vocabulary(
        abstract, bank, exclude_task=candidate_binding.task,
    )
    diag["source_ops_vocab"] = sorted(source_vocab)[:8]
    diag["candidate_ops"]    = sorted({
        (s.op or "").upper() for s in candidate_binding.protocol if s.op
    })

    harness = SkillHarness(registry)
    fsa = FewShotAdapter(
        harness=harness,
        target_domain_pass_rate_min=pass_rate_min,
        success_fn=_make_structural_success_fn(skill_record, source_vocab),
    )
    try:
        result: AdaptResult = fsa.adapt(
            skill=skill_record,
            target_domain=target_domain,
            demos=demos,
            target_task=candidate_binding.task,
            k=len(demos),
        )
    except Exception as exc:                                          # noqa: BLE001
        diag["reason"]  = "adapter_raised"
        diag["error"]   = repr(exc)
        return None, diag

    diag["pass_rate"]         = float(result.pass_rate)
    diag["n_success"]         = int(result.n_success)
    diag["n_total"]           = int(result.n_total)
    diag["aborted"]           = int(result.aborted)
    diag["diagnostic_label"]  = result.diagnostic_label
    diag["episode_ids"]       = list(result.episode_ids[:6])
    diag["k_used"]            = int(result.k_used)

    if result.diagnostic_label == "target_domain_demo_unavailable":
        diag["reason"] = "no_adapter_for_target_domain"
        return None, diag
    if result.n_total == 0:
        diag["reason"] = "n_total=0"
        return None, diag

    verdict = result.pass_rate >= pass_rate_min
    diag["verdict"] = "VALIDATED" if verdict else "REJECTED"
    return verdict, diag


__all__ = [
    "TASK_TO_HARNESS_DOMAIN",
    "binding_to_skill_record",
    "lineage_to_demos",
    "task_to_harness_domain",
    "validate_binding_via_harness",
]
