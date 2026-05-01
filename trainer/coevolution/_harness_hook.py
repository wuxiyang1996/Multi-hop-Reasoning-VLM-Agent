"""Per-rollout `SkillHarnessHook` for the trainer's co-evolution loop.

Splices into ``trainer.coevolution.episode_runner.run_episode_async`` so
the harness's two LLM-free, deterministic surfaces (PLAN-HARNESS §5.2 +
PLAN-UNIFIED-SKILL-GATE §3.4) get applied around the actor's
``skill_selection`` LLM call:

  1. **Pre-LLM filter** — :meth:`SkillHarnessHook.filter_candidates`
     wraps the cold-start RAG output from
     :func:`scripts.qwen3_decision_agent.get_top_k_skill_candidates`
     through :meth:`harness.skill_harness.SkillHarness.select_eligible_skills`.
     Skills the harness vetoes (status / domain / task / adapter / can_handle)
     are dropped *before* the LLM sees them, and their veto reason is
     observed by an in-process :class:`harness.RejectedSkillSink`.

  2. **Post-LLM validation** — :meth:`SkillHarnessHook.validate_choice`
     wraps the LLM's chosen skill through
     :meth:`harness.skill_harness.SkillHarness.validate_invocation`.
     If vetoed, the actor falls back to the next surviving eligible
     candidate (or runs unguided when none remains).

The aggregated rejections survive across episodes within a single
trainer step. After Phase A, :meth:`SkillHarnessHook.flush_to_lifecycle`
drains the sink into a freshly-seeded :class:`SkillLifecycleManager`
inside the existing Crafter hook so each
``SkillRecord.false_binding_patterns`` lights up the Repairer's
patch-or-retire path.

Strict trainer-mode contract
----------------------------
* No live LLM call; no env binding. The hook is pure CPU (microseconds
  per step). The deterministic ``GymvAdapter`` stub is registered so
  ``F3 adapter`` and ``F4 can_handle`` do not falsely veto every
  candidate. Real ``set_executor(env_step_fn)`` wiring stays out of
  scope — that lives in ``vlm_wrapper/gymv_wrapper`` (see
  ``harness/README.md`` §16.1).
* All public methods are wrapped in ``try / except`` so a harness bug
  can never break a training rollout. On internal failure the hook
  reports a degraded result and the actor proceeds unchanged.
* Skills the live bank.jsonl exposes are *de-facto* provisional — they
  passed the legacy 4-stage pipeline. We mint them with
  ``status = SkillStatus.PROVISIONAL`` so :class:`EligibilityFilter`'s
  F1 status check admits them. The lifecycle manager is the only
  authority that may transition status; this hook never persists any
  status mutation — it just produces an in-memory snapshot for the
  filter to consume.

Cross-refs
----------
* ``harness/README.md`` §22 (trainer integration block).
* ``trainer/coevolution/_crafter_hook.py`` (consumes the rejection
  sink via ``flush_to_lifecycle``).
* ``crafter-harness-orchestrator-roles.md`` §2.2 (Harness role boundary).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from common.enums import SkillStatus, SkillType
from common.state_schema import EvidenceRef, StateSchema
from data_structure.extensions.skill_record import SkillRecord
from harness.adapter_registry import AdapterRegistry
from harness.adapters.gymv_adapter import GymvAdapter
from harness.eligibility import EligibleSkill, RejectedSkill, task_id_from_state
from harness.rejected_skill_sink import FlushReport, RejectedSkillSink
from harness.skill_harness import HarnessConfig, SkillHarness, ValidateInvocationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-step diagnostic records the orchestrator can roll up into wandb / TB
# ---------------------------------------------------------------------------


@dataclass
class HarnessStepStats:
    """Roll-up of one episode's harness activity (logged in `experiences`)."""

    n_candidates_in: int = 0
    n_candidates_admitted: int = 0
    n_candidates_rejected: int = 0
    n_validate_ok: int = 0
    n_validate_veto: int = 0
    n_validate_skipped: int = 0   # candidate not in lifecycle cache
    veto_class_distribution: Dict[str, int] = field(default_factory=dict)
    last_eligible_ids: List[str] = field(default_factory=list)
    last_rejected: List[Dict[str, Any]] = field(default_factory=list)
    last_validate: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "n_candidates_in": self.n_candidates_in,
            "n_candidates_admitted": self.n_candidates_admitted,
            "n_candidates_rejected": self.n_candidates_rejected,
            "n_validate_ok": self.n_validate_ok,
            "n_validate_veto": self.n_validate_veto,
            "n_validate_skipped": self.n_validate_skipped,
            "veto_class_distribution": dict(self.veto_class_distribution),
            "last_eligible_ids": list(self.last_eligible_ids),
            "last_rejected": list(self.last_rejected),
            "last_validate": dict(self.last_validate) if self.last_validate else None,
        }


# ---------------------------------------------------------------------------
# `SkillRecord` cache hydration from the trainer's per-game bank.jsonl
# ---------------------------------------------------------------------------


def _hydrate_records_from_bank(
    bank_path: Path,
    *,
    default_domain: str,
    runtime_status: SkillStatus = SkillStatus.PROVISIONAL,
) -> Dict[str, SkillRecord]:
    """Lift one per-game ``skill_bank.jsonl`` into ``Dict[skill_id, SkillRecord]``.

    Reuses :func:`trainer.coevolution._crafter_hook._record_from_bank_entry`
    so the hydration logic is shared with the Crafter hook (single
    source of truth for the legacy → record schema).

    Skills are mounted with ``runtime_status=PROVISIONAL`` rather than
    the default ``DRAFT`` so the eligibility filter's F1 status check
    admits them. This is a *runtime view*, not a persisted transition —
    the lifecycle manager remains the only authority that may write
    status to disk (PLAN-SKILL-BANK §0.5).

    Malformed lines are silently skipped (matching the Crafter hook's
    tolerance — the live bank is a running artefact and we'd rather
    seed N-1 records than refuse to fire).
    """
    if not bank_path.is_file():
        return {}
    # Lazy import — `_crafter_hook` pulls in heavyweight crafter deps via
    # transitive imports; we only want them when the harness hook actually
    # builds. Importing under a leading-underscore name is intentional:
    # both hook modules live in the same package and share the loader.
    from trainer.coevolution._crafter_hook import _record_from_bank_entry

    out: Dict[str, SkillRecord] = {}
    try:
        import json as _json
        with bank_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = _json.loads(raw)
                except Exception:
                    continue
                rec = _record_from_bank_entry(entry, default_domain)
                if rec is None:
                    continue
                # Override the default DRAFT status with the runtime view.
                # `SkillRecord` is a plain dataclass; assignment works,
                # but we go through `object.__setattr__` for consistency
                # with `_record_from_bank_entry`'s skill_id override and
                # to be forward-compatible if a `__setattr__` lock lands.
                object.__setattr__(rec, "status", runtime_status)
                out[rec.skill_id] = rec
    except OSError as exc:  # noqa: BLE001
        logger.debug(
            "harness_hook: failed to read bank %s: %s", bank_path, exc,
        )
    return out


# ---------------------------------------------------------------------------
# `StateSchema` synthesis from the trainer's per-step text bundle
# ---------------------------------------------------------------------------


def _state_for_step(
    *,
    game: str,
    summary_state: str,
    intention: str,
    domain: str = "gymv",
    inner_step: int = 0,
    outer_step: int = 0,
) -> StateSchema:
    """Build a minimal :class:`StateSchema` from the trainer's per-step
    text bundle so the eligibility filter can run F2/F2′/F4.

    The trainer's actor doesn't yet emit a full :class:`StateSchema`
    (that lives in the cold-start mining path). We synthesise one with
    just enough information to make the eligibility filter and
    ``validate_invocation`` produce a meaningful verdict:

    * ``domain``    — fixed to the adapter domain for the trainer's
                      games. All trainer-side games map to ``"gymv"``.
    * ``task``      — set to ``game`` so F2′ task-axis is enforced
                      consistently with the offline mirror.
    * ``facts``     — empty (no precondition predicates evaluated
                      against text). The deterministic adapter only
                      needs the domain to admit, so this is fine.
    * ``evidence``  — empty list; the live actor's evidence chain is
                      not lifted here. The harness's
                      ``validate_invocation`` only checks
                      ``evidence_in`` for ACTION skills (which require
                      it); GROUNDING / REASONING skills are exempt.

    Failures degrade to a permissive ``StateSchema(domain=domain,
    task=game)`` so the live runner is never starved by a malformed
    summary.
    """
    try:
        return StateSchema(
            domain=domain,
            task=game,
            evidence=[],
            inner_step=int(inner_step),
            outer_step=int(outer_step),
            extra={
                "summary_state": (summary_state or "")[:1024],
                "intention": (intention or "")[:256],
            },
        )
    except Exception:  # noqa: BLE001
        return StateSchema(domain=domain, task=game)


# ---------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------


class SkillHarnessHook:
    """Trainer-side façade around the harness's eligibility + validate
    surfaces.

    One instance per (game, step). The hook is cheap to build (registers
    a :class:`GymvAdapter` and hydrates the per-game bank.jsonl into a
    flat dict) and stateful only across the episodes within a single
    step — the orchestrator drains the rejection sink into the Crafter
    hook's lifecycle and discards the hook at end of step.

    Lifecycle
    ---------
    1. Orchestrator builds one ``SkillHarnessHook`` per game per step
       (``cls.for_game(...)``), passing the path to the per-game
       ``skill_bank.jsonl``.
    2. ``filter_candidates(candidates, state)`` is called inside the
       episode loop, before the ``skill_selection`` LLM. Returns the
       narrowed candidate list and an opaque diagnostic dict for
       ``experiences[].harness``.
    3. ``validate_choice(skill_id, state)`` is called after the LLM has
       picked one. Returns ``(ok, validation_dict)``.
    4. After Phase A, the orchestrator passes the hook's
       :class:`RejectedSkillSink` to the Crafter hook so the
       per-skill_id veto patterns end up on
       ``SkillRecord.false_binding_patterns`` via
       ``SkillLifecycleManager.record_false_binding_pattern``.

    All public methods catch internal failures and downgrade to
    pass-through behaviour — the trainer's rollout must never break
    because the harness disagreed with itself.
    """

    def __init__(
        self,
        *,
        domain: str = "gymv",
        records: Optional[Mapping[str, SkillRecord]] = None,
        allow_shadow: bool = True,
        adapters: Optional[List[Any]] = None,
        sink: Optional[RejectedSkillSink] = None,
    ) -> None:
        self._domain = domain
        self._records: Dict[str, SkillRecord] = dict(records or {})
        self._sink = sink if sink is not None else RejectedSkillSink()

        registry = AdapterRegistry()
        adapter_list = list(adapters) if adapters is not None else [GymvAdapter()]
        for ad in adapter_list:
            try:
                registry.register(ad)
            except ValueError:
                # Already registered (shared registry) — fine.
                pass
        self._registry = registry

        self._harness = SkillHarness(
            registry=registry,
            config=HarnessConfig(allow_shadow=allow_shadow),
        )

        self._stats = HarnessStepStats()

    # ── Construction helpers ─────────────────────────────────────────

    @classmethod
    def for_game(
        cls,
        *,
        game: str,
        bank_path: Optional[Path],
        domain: str = "gymv",
        allow_shadow: bool = True,
    ) -> "SkillHarnessHook":
        """Build a hook for one game, hydrating its bank.jsonl into the
        SkillRecord cache.

        Returns an empty-cache hook (still functional, just with no
        records to filter) when the bank file is missing — matching
        the cold-start convention where step 0's bank is empty.
        """
        records: Dict[str, SkillRecord] = {}
        if bank_path is not None:
            records = _hydrate_records_from_bank(
                Path(bank_path), default_domain=domain,
            )
        return cls(
            domain=domain,
            records=records,
            allow_shadow=allow_shadow,
        )

    # ── Properties ───────────────────────────────────────────────────

    @property
    def sink(self) -> RejectedSkillSink:
        return self._sink

    @property
    def stats(self) -> HarnessStepStats:
        return self._stats

    def n_records(self) -> int:
        return len(self._records)

    # ── Per-step API ─────────────────────────────────────────────────

    def state_for_step(
        self,
        *,
        game: str,
        summary_state: str = "",
        intention: str = "",
        inner_step: int = 0,
        outer_step: int = 0,
    ) -> StateSchema:
        return _state_for_step(
            game=game,
            summary_state=summary_state,
            intention=intention,
            domain=self._domain,
            inner_step=inner_step,
            outer_step=outer_step,
        )

    def filter_candidates(
        self,
        candidates: List[Dict[str, Any]],
        state: StateSchema,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Pre-LLM eligibility filter (PLAN-HARNESS §5.2).

        Maps cold-start RAG candidate dicts → ``SkillRecord`` (via the
        per-game cache) → :meth:`SkillHarness.select_eligible_skills`.
        Skills the cache doesn't know are passed through unchanged
        (degrade gracefully — no veto, no observe).

        Returns ``(filtered_candidates, diagnostic_dict)``. The
        diagnostic dict matches the
        ``HarnessStepStats.to_json()`` shape and is suitable for
        embedding in ``experiences[].harness``.
        """
        n_in = len(candidates or [])
        diag: Dict[str, Any] = {
            "n_in": n_in,
            "n_admitted": 0,
            "n_rejected": 0,
            "n_unknown": 0,
            "eligible_ids": [],
            "rejected": [],
            "task_match_distribution": {},
        }

        if not candidates:
            return [], diag

        # Map each candidate dict to its corresponding cached SkillRecord.
        # Unknowns are passed through (the harness has no opinion on
        # skills it doesn't know — that may happen in unit tests or
        # when the bank.jsonl was rotated mid-step).
        records_for_filter: List[SkillRecord] = []
        unknown_idx: List[int] = []
        sid_to_cand: Dict[str, Dict[str, Any]] = {}
        for i, cand in enumerate(candidates):
            sid = (cand or {}).get("skill_id")
            rec = self._records.get(sid) if sid else None
            if rec is None:
                unknown_idx.append(i)
                continue
            records_for_filter.append(rec)
            sid_to_cand[sid] = cand

        try:
            admitted, rejected = self._harness._eligibility.filter_with_rejections(
                records_for_filter, state,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "harness_hook.filter_candidates: filter raised %s — "
                "degrading to pass-through", exc,
            )
            return list(candidates), {
                **diag,
                "n_admitted": n_in,
                "harness_error": repr(exc),
            }

        # Observe rejections into the sink (keyed by the source domain
        # / task — the receiving lifecycle uses these for `false_binding_patterns`).
        if rejected:
            try:
                self._sink.observe(
                    rejected,
                    domain=state.domain,
                    task=task_id_from_state(state),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "harness_hook.filter_candidates: sink.observe raised %s",
                    exc,
                )

        # Reconstruct the filtered candidate list, preserving the order
        # the LLM expects (admitted first, then unknowns — the LLM
        # already had unknowns in its prior view, so dropping them
        # would be a behavioural change beyond the eligibility surface).
        admitted_sids = {e.skill.skill_id for e in admitted}
        filtered: List[Dict[str, Any]] = []
        eligible_ids: List[str] = []
        task_match_dist: Dict[str, int] = {}
        for e in admitted:
            cand = sid_to_cand.get(e.skill.skill_id)
            if cand is None:
                continue
            # Decorate with the eligibility verdict so the actor's
            # downstream logging / GRPO records can reason about it.
            cand2 = dict(cand)
            cand2["_harness_eligible"] = e.to_json()
            filtered.append(cand2)
            eligible_ids.append(e.skill.skill_id)
            task_match_dist[e.task_match] = task_match_dist.get(e.task_match, 0) + 1
        for i in unknown_idx:
            filtered.append(candidates[i])

        # Diagnostic block.
        diag["n_admitted"] = len(admitted)
        diag["n_rejected"] = len(rejected)
        diag["n_unknown"] = len(unknown_idx)
        diag["eligible_ids"] = eligible_ids
        diag["rejected"] = [r.to_json() for r in rejected]
        diag["task_match_distribution"] = task_match_dist

        # Step stats (per-episode rollup).
        self._stats.n_candidates_in += n_in
        self._stats.n_candidates_admitted += len(admitted)
        self._stats.n_candidates_rejected += len(rejected)
        for r in rejected:
            self._stats.veto_class_distribution[r.veto] = (
                self._stats.veto_class_distribution.get(r.veto, 0) + 1
            )
        self._stats.last_eligible_ids = list(eligible_ids)
        self._stats.last_rejected = list(diag["rejected"])

        return filtered, diag

    def validate_choice(
        self,
        skill_id: Optional[str],
        state: StateSchema,
        *,
        bindings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Post-LLM second-pass invocation veto (PLAN-UNIFIED §3.4).

        Returns ``(ok, diagnostic_dict)``. ``ok=False`` means the actor
        should fall back to the next surviving eligible candidate.
        ``ok=True`` is the default for skills the cache doesn't know
        (we can't prove a veto, so we admit — degrade gracefully).
        """
        if not skill_id:
            self._stats.n_validate_skipped += 1
            return True, {"status": "no_skill_id_supplied"}

        rec = self._records.get(skill_id)
        if rec is None:
            self._stats.n_validate_skipped += 1
            d = {"status": "skill_not_in_cache", "skill_id": skill_id}
            self._stats.last_validate = d
            return True, d

        try:
            res: ValidateInvocationResult = self._harness.validate_invocation(
                rec, state, bindings=bindings or {},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "harness_hook.validate_choice: validate_invocation raised %s — "
                "degrading to admit", exc,
            )
            self._stats.n_validate_skipped += 1
            return True, {"status": "harness_error", "error": repr(exc)}

        d = res.to_json()
        d["status"] = "ok" if res.ok else "veto"
        d["skill_id"] = skill_id
        self._stats.last_validate = dict(d)
        if res.ok:
            self._stats.n_validate_ok += 1
        else:
            self._stats.n_validate_veto += 1
        return res.ok, d

    # ── Drainage ─────────────────────────────────────────────────────

    def flush_to_lifecycle(
        self,
        lifecycle: Any,
        *,
        min_count: int = 1,
        reset: bool = True,
    ) -> FlushReport:
        """Drain the in-memory rejection sink into ``lifecycle``.

        Wraps :meth:`harness.RejectedSkillSink.flush_to`. Skills that
        the lifecycle doesn't know are skipped (and reported in the
        ``FlushReport``) — this is expected because the per-step hook
        sees the *runtime view* (PROVISIONAL records minted from
        bank.jsonl) while the Crafter hook's lifecycle holds an
        ephemeral ``SkillRepository`` seeded with CANDIDATE records;
        the two share skill_ids but the lifecycle owns the
        ``record_false_binding_pattern`` write surface.
        """
        try:
            return self._sink.flush_to(
                lifecycle, min_count=min_count, reset=reset,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "harness_hook.flush_to_lifecycle: flush raised %s", exc,
            )
            return FlushReport(
                n_skills_touched=0, n_patterns_written=0, n_errors=1,
                skipped_unknown_skill_ids=[],
                errors=[{"error": repr(exc)}],
            )


__all__ = [
    "HarnessStepStats",
    "SkillHarnessHook",
]
