#!/usr/bin/env python
"""
Per-episode Skill Crafter driver — invokes the LIVE
`SkillCrafterService.reflect_on_episode` against each episode in a
``labeling/skill_actions_out`` run, paired with that episode's
``per_episode_bank_management/bank_management_io.json``.

This is the cold-start *mirror* of the live two-tier trigger model
(see ``implementation_notes/legacy/crafter-harness-orchestrator-roles.md``
§"Two-tier trigger"): the live runtime calls
``SkillCrafterService.reflect_on_episode(EpisodeReflection)`` immediately
after the Skill Bank Agent finishes one episode. This script does the
same thing offline, so the cold-start corpus carries genuine
``BankMutationProposal`` records emitted by the *real* Crafter code,
not just rule-based stand-ins.

Per-episode flow
----------------
For each ``(corpus, source)`` pair:

  1. Load the FROZEN per-source skill bank
     (``skill_bank_run/<corpus>/<source>/skill_bank.jsonl``) and seed it
     into a temp ``SkillCrafterService`` with each skill at status
     ``CANDIDATE``.  ACTIVE promotion is intentionally skipped — most
     cold-start skills declare a single ``feasible_domain`` and the
     lifecycle invariant requires ≥2 for ACTIVE; the Failure-Reflector
     dispatch path uses ``repository.get`` which queries every store, so
     CANDIDATE seeding is sufficient for everything *except* the
     subsumption-retire heuristic (which compares CANDIDATE→ACTIVE and
     therefore stays dormant in this mirror — documented limitation,
     unblocked once Phase-2 multi-domain skills land).
  2. Walk every ``episode_*.json`` under
     ``skill_actions_run/<corpus>/<source>/`` in order.
  3. For each episode, locate the matching
     ``skill_bank_run/<corpus>/<source>/per_episode_bank_management/episode_<i>/bank_management_io.json``
     (if present) and use its ``stage_4_bank_maintenance`` outputs to
     populate ``new_candidate_skill_ids`` + ``bank_agent_actions`` on
     the :class:`EpisodeReflection`.
  4. Synthesize :class:`FailureTrace`s from on-disk per-step signals
     (see :func:`_synthesize_failures` for the rules).
  5. Build the :class:`EpisodeReflection` and call
     ``crafter.reflect_on_episode(reflection)``.
  6. Serialize the resulting ``CrafterCycleResult`` (proposals +
     subsumption count + bank-view summary) to disk.

Failure synthesis (offline approximation)
-----------------------------------------
The cold-start traces in ``labeling/skill_actions_out`` are mostly
healthy (the bank fits the rollouts cleanly, applicability ≈ 0.5 by
construction); to give the Crafter something meaningful to act on, we
synthesize :class:`FailureTrace`s from the following per-episode
signals:

* **OUTCOME_FAILURE** — ``episode["outcome"] != True``: one synthetic
  trace tagged to the most-selected skill of the episode
  (``failure_class = "INVARIANT_VIOLATION"``).
* **EMPTY_QUERY** — ``step.skill_query.empty == True``: one trace per
  occurrence (``failure_class = "MISSING_ADAPTER"``) signalling a
  bank gap that the Hypothesizer should pick up.
* **LOW_APPLICABILITY** — ``step.skills.applicability < threshold``
  (default 0.4): one trace per occurrence
  (``failure_class = "PRECONDITION_VIOLATION"``) — indicates the
  bound skill's preconditions don't hold strongly enough.
* **MISSING_EFFECTS** — ``step.skills.missing_effects`` non-empty: one
  trace per occurrence (``failure_class = "INVARIANT_VIOLATION"``) —
  the bound skill claimed effects that didn't materialize.

A per-episode cap (``--max-failures-per-episode``, default 8) keeps the
synthesized batch size bounded so a 200-step rollout doesn't blow up
``FailureMemory``.

Outputs
-------
``<output_dir>/<corpus>/<source>/episode_<NNN>/proposals.jsonl`` —
one JSON object per line, the typed proposals
``crafter.reflect_on_episode`` emitted (PLAN-SKILL-CRAFTER §2.5).

``<output_dir>/<corpus>/<source>/episode_<NNN>/reflection.json`` — the
``EpisodeReflection`` that was passed in (full audit input).

``<output_dir>/<corpus>/<source>/episode_<NNN>/result.json`` — the
``CrafterCycleResult`` minus the proposals (kind / proposer /
subsumption / bank-view-size summary).

``<output_dir>/<corpus>/<source>/_source_summary.json`` — per-source
totals.

``<output_dir>/_run_meta.json`` and ``_run_summary.json`` — run-level.

Usage
-----

    # Process every (corpus, source) pair, every episode, default thresholds.
    python labeling_supplement/reflect_per_episode_gpt54.py \
        --bank-run     labeling/skill_bank_out/run_20260430_030637 \
        --actions-run  labeling/skill_actions_out/run_20260430_064325 \
        --output-dir   labeling_supplement/episode_reflections_out/run_<ts>

    # Smoke: one source, two episodes.
    python labeling_supplement/reflect_per_episode_gpt54.py \
        --bank-run     labeling/skill_bank_out/run_20260430_030637 \
        --actions-run  labeling/skill_actions_out/run_20260430_064325 \
        --corpus       env_wrappers --source twenty_forty_eight \
        --max-episodes 2 -v

The companion bash dispatcher ``run_reflect_per_episode.sh`` fans this
out one worker per ``(corpus, source)``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup so the script runs from any cwd (mirror sibling drivers).
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for _p in (CODEBASE_ROOT, WORKSPACE_ROOT):
    _ps = str(_p)
    if _p.exists() and _ps not in sys.path:
        sys.path.insert(0, _ps)


def _bootstrap_api_keys_from_file() -> Optional["Path"]:
    """Seed ``os.environ`` from an ``api_keys.py`` sidecar.

    Mirrors the cold-start launcher's lookup order so the LLM hooks
    (``--llm-repairer`` / ``--llm-hypothesizer``) work out of the
    box from the same paths the rest of the project uses. No-op when
    the env vars are already set.
    """
    import importlib.util

    here = Path(__file__).resolve().parent
    candidates = [
        Path(os.environ.get("COSPLAY_API_KEYS_FILE", "") or ""),
        here / "api_keys.py",
        CODEBASE_ROOT / "api_keys.py",
        CODEBASE_ROOT.parent / "api_keys.py",   # /workspace/api_keys.py
    ]
    for path in candidates:
        try:
            if not path or not path.is_file():
                continue
        except OSError:
            continue
        try:
            spec = importlib.util.spec_from_file_location("_reflect_api_keys", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:                                                # noqa: BLE001
            continue
        mapping = {
            "openrouter_api_key": "OPENROUTER_API_KEY",
            "openai_api_key":     "OPENAI_API_KEY",
            "claude_api_key":     "ANTHROPIC_API_KEY",
            "gemini_api_key":     "GEMINI_API_KEY",
        }
        for attr, env_name in mapping.items():
            val = getattr(mod, attr, None)
            if isinstance(val, str) and val.strip() and not os.environ.get(env_name):
                os.environ[env_name] = val.strip()
        return path
    return None


_API_KEYS_FILE_USED = _bootstrap_api_keys_from_file()

# ---------------------------------------------------------------------------
# Project imports — these load the LIVE Crafter code.
# ---------------------------------------------------------------------------
from common.enums import (
    SkillSourceType,
    SkillStatus,
    SkillType,
    TRANSFER_TARGET_DOMAINS,
)
from crafter import SkillCrafterService
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    proposal_to_json,
)
from data_structure.extensions.episode_reflection import EpisodeReflection
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_record import SkillContract, SkillRecord
from orchestrator import ArtifactStore
from skill_bank import SkillLifecycleManager, SkillRepository, SkillStore
from skill_bank.stores import StoreName

# Transfer-target dispatch (smoke for AB / VR / video / OSWorld). Lives
# under a sibling package so the gymv-only legacy path stays intact.
try:
    from labeling_supplement._failure_synth import get_synthesizer
except Exception:                                                # noqa: BLE001
    # Defensive — keeps the gymv path importable if the optional
    # _failure_synth package is missing (e.g. partial check-out).
    get_synthesizer = None  # type: ignore[assignment]

logger = logging.getLogger("labeling_supplement.reflect_per_episode")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BANK_RUN = (
    CODEBASE_ROOT / "labeling" / "skill_bank_out" / "run_20260430_030637"
)
DEFAULT_ACTIONS_RUN = (
    CODEBASE_ROOT / "labeling" / "skill_actions_out" / "run_20260430_064325"
)
DEFAULT_OUTPUT_ROOT = (
    CODEBASE_ROOT / "labeling_supplement" / "episode_reflections_out"
)

CORPORA = ("gym_v", "env_wrappers")

# Failure-synthesis thresholds (rule-of-thumb for the cold-start corpus).
# Override via CLI.
DEFAULT_LOW_APPLICABILITY = 0.4
DEFAULT_MAX_FAILURES_PER_EPISODE = 8


# ═══════════════════════════════════════════════════════════════════════
# Bank seeding — turn on-disk skill_bank.jsonl into live SkillRecords
# ═══════════════════════════════════════════════════════════════════════

# Map between the on-disk evidence_role string and a SkillType + canonical
# expected_evidence_roles value. The bank emits roles in {GATHER, VERIFY,
# REASON, COMMIT}; the live SkillType vocabulary has different
# granularity (REASONING / ACTION / GROUNDING / MIXED). We map
# defensively and let the gate sort out semantics later.
_ROLE_TO_SKILL_TYPE = {
    "GATHER": SkillType.GROUNDING,
    "VERIFY": SkillType.REASONING,
    "REASON": SkillType.REASONING,
    "COMMIT": SkillType.ACTION,
}


def _safe_skill_id(skill_id: str) -> str:
    """Cold-start labels use ``OPERATOR/SUBGOAL`` (e.g. ``COMMIT/ATTACK``)
    which the on-disk ``SkillStore`` flat-filename layout rejects (it
    builds ``<root>/<skill_id>.json`` so ``/`` would force a nested
    path that the store doesn't pre-create). Map ``/`` → ``__`` and
    apply the same mapping to ``parent_skill_ids`` so subsumption /
    failure-resolution links keep resolving."""
    return skill_id.replace("/", "__")


def _wrap_protocol_steps(raw_steps: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convert on-disk ``protocol.steps`` (often natural-language strings
    in cold-start banks) into the live ``protocol: List[Dict[...]]``
    shape that ``Repairer._rule_repair`` and friends expect."""
    out: List[Dict[str, Any]] = []
    for s in raw_steps or []:
        if isinstance(s, dict):
            out.append(dict(s))
        elif isinstance(s, str):
            out.append({"action": "EXEC", "payload": {}, "notes": s})
        else:
            out.append({"action": "EXEC", "payload": {}, "notes": str(s)})
    return out


def _record_from_bank_entry(entry: Dict[str, Any], default_domain: str) -> SkillRecord:
    """Hydrate a `SkillRecord` from one ``skill_bank.jsonl`` line (the
    ``{"skill": ..., "report": ...}`` envelope).

    Two on-disk shapes for ``protocol`` (mirrors
    ``trainer/coevolution/_crafter_hook.py::_record_from_bank_entry``):

    * **legacy cold-start** (pre-Day-2 lift) — a dict ``{"steps": [...],
      "preconditions": [...], "success_criteria": [...], ...}``. The
      ancillary contract fields hang off the protocol dict.
    * **Day-2-lifted** — a list of typed hops ``[{"op": "READ",
      "payload": {...}, "notes": ..., ...}, ...]``. Contract fields
      have moved into ``skill["contract"]`` upstream so the protocol
      body carries no preconditions / success_criteria of its own.
    """
    skill = entry.get("skill") or {}
    contract = skill.get("contract") or {}
    role = (skill.get("evidence_role") or "COMMIT").upper()
    skill_type = _ROLE_TO_SKILL_TYPE.get(role, SkillType.MIXED)

    feasible = list(skill.get("applicable_domains") or []) or [default_domain]
    raw_protocol = skill.get("protocol")
    if isinstance(raw_protocol, list):
        protocol_steps = list(raw_protocol)
        protocol_blob: Dict[str, Any] = {}
    elif isinstance(raw_protocol, dict):
        protocol_blob = raw_protocol
        protocol_steps = list(protocol_blob.get("steps") or [])
    else:
        protocol_blob = {}
        protocol_steps = []

    sk = SkillRecord.new(
        name=skill.get("name", skill.get("skill_id", "_unknown")),
        skill_type=skill_type,
        source_type=SkillSourceType.MINED,
        feasible_domains=feasible,
        protocol=_wrap_protocol_steps(protocol_steps),
        contract=SkillContract(
            preconditions=list(protocol_blob.get("preconditions") or []),
            effects_add=list(contract.get("eff_add") or contract.get("effects_add") or []),
            effects_del=list(contract.get("eff_del") or contract.get("effects_del") or []),
            expected_evidence_roles=[role] if role else [],
            success_criteria=list(protocol_blob.get("success_criteria") or []),
            abort_criteria=list(protocol_blob.get("abort_criteria") or []),
        ),
    )
    # Force the bank-given skill_id (overrides the freshly-minted UUID)
    # so that downstream `parent_skill_ids` references resolve. We must
    # bypass `SkillRecord.__setattr__` (which guards status mutations) by
    # going through `object.__setattr__`.
    raw_id = skill.get("skill_id") or sk.skill_id
    object.__setattr__(sk, "skill_id", _safe_skill_id(raw_id))
    return sk


def _seed_bank(
    lifecycle: SkillLifecycleManager,
    bank_path: Path,
    default_domain: str,
) -> Tuple[int, int]:
    """Seed the temp bank as CANDIDATE from a ``skill_bank.jsonl`` file.

    Returns ``(n_seeded, n_skipped)`` for the run summary.
    """
    if not bank_path.exists():
        return 0, 0
    n = 0
    skipped = 0
    with bank_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            try:
                rec = _record_from_bank_entry(entry, default_domain)
                lifecycle.ingest_draft(rec)
                lifecycle.transition(
                    rec.skill_id,
                    to_status=SkillStatus.CANDIDATE,
                    rationale="seed-from-bank-snapshot",
                )
                n += 1
            except Exception as exc:                                # noqa: BLE001
                logger.debug("skip bank seed %s: %s", entry.get("skill", {}).get("skill_id"), exc)
                skipped += 1
    return n, skipped


def _seed_refines_as_candidates(
    lifecycle: SkillLifecycleManager,
    bank_mgmt: Dict[str, Any],
    default_domain: str,
) -> List[str]:
    """Seed the per-episode bank-mgmt refine_details as fresh CANDIDATE
    skills.

    These mimic the candidates the live Bank Agent would have minted in
    this episode, giving the Crafter's per-episode pass real
    ``new_candidate_skill_ids`` to chew on. The new skills carry a
    ``parent_skill_ids`` link to the original (when the refine has a
    ``skill_id`` pointer), which is what the subsumption heuristic
    actually checks.

    Returns the list of newly-seeded skill_ids (so the caller can plumb
    them into ``EpisodeReflection.new_candidate_skill_ids``).
    """
    out: List[str] = []
    s4 = (bank_mgmt.get("stage_4_bank_maintenance") or {}).get("outputs") or {}
    for r in (s4.get("refine_details") or []):
        fields = r.get("_fields") or {}
        parent_id_raw = fields.get("skill_id")
        new_contract = fields.get("new_contract") or {}
        if not parent_id_raw or not new_contract:
            continue
        # The refine's `new_contract` lacks `applicable_domains`; mirror
        # the parent. Also sanitise `parent_skill_ids` to match the
        # filename-safe form we used when seeding the bank, so the
        # subsumption / repair lookups can find the parent.
        parent_id = _safe_skill_id(parent_id_raw)
        cand = SkillRecord.new(
            name=f"{new_contract.get('name', parent_id_raw)}__refined_v{new_contract.get('version', 0)}",
            skill_type=SkillType.MIXED,
            source_type=SkillSourceType.MINED,
            feasible_domains=[default_domain],
            protocol=[],                                             # refines target the contract, not protocol
            contract=SkillContract(
                preconditions=[],
                effects_add=_clean_str_list(new_contract.get("eff_add") or []),
                effects_del=_clean_str_list(new_contract.get("eff_del") or []),
                expected_evidence_roles=[],
                success_criteria=[],
                abort_criteria=[],
            ),
            parent_skill_ids=[parent_id],
        )
        try:
            lifecycle.ingest_draft(cand)
            lifecycle.transition(
                cand.skill_id,
                to_status=SkillStatus.CANDIDATE,
                rationale=f"bank-mgmt refine for {parent_id}",
            )
            out.append(cand.skill_id)
        except Exception as exc:                                    # noqa: BLE001
            logger.debug("skip refine seed %s: %s", parent_id, exc)
    return out


def _clean_str_list(items: Iterable[Any]) -> List[str]:
    """Filter out ``<depth-limit: ...>`` placeholders that the bank-mgmt
    JSONs use when a list was truncated by the on-disk serialiser."""
    out: List[str] = []
    for it in items:
        if isinstance(it, str) and not it.startswith("<depth-limit:"):
            out.append(it)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Failure synthesis — turn step-level signals into FailureTraces
# ═══════════════════════════════════════════════════════════════════════

def _synthesize_failures(
    *,
    episode: Dict[str, Any],
    episode_id: str,
    domain: str,
    low_applicability: float,
    max_failures: int,
) -> List[FailureTrace]:
    """Walk per-step signals and emit a bounded list of FailureTraces.

    See module docstring for the four signals (OUTCOME_FAILURE,
    EMPTY_QUERY, LOW_APPLICABILITY, MISSING_EFFECTS). The traces are
    ordered by signal severity so a low ``max_failures`` cap retains
    the most informative ones first.
    """
    out: List[FailureTrace] = []
    exps = episode.get("experiences") or episode.get("steps") or []

    # NB: we route all per-step skill_ids through `_safe_skill_id` so
    # the FailureMemory keys agree with the IDs we used when seeding
    # the bank — otherwise `_resolve_base("COMMIT/ATTACK")` misses the
    # seeded `COMMIT__ATTACK` record and the dispatch silently drops
    # the patch.
    skill_select_counter: Counter[str] = Counter()
    for exp in exps:
        sk = exp.get("skills") or {}
        sid = sk.get("skill_id")
        if sid:
            skill_select_counter[_safe_skill_id(sid)] += 1

    # ── 1. OUTCOME_FAILURE — single-shot, episode-level ───────────────
    # `outcome=True` is a strong success signal and short-circuits the
    # status fallback (some rollouts have outcome=True with
    # status="running" because the trace was sliced mid-success).
    outcome = episode.get("outcome")
    status = (episode.get("episode_status") or "").lower()
    is_outcome_failure = (
        outcome is False
        or (outcome is None and status in ("failed", "timeout"))
    )
    if is_outcome_failure:
        rep_skill = (
            skill_select_counter.most_common(1)[0][0]
            if skill_select_counter
            else ""
        )
        out.append(FailureTrace(
            skill_id=rep_skill,
            skill_episode_id=f"{episode_id}#outcome",
            domain=domain,
            failed_step_index=len(exps) - 1 if exps else None,
            failure_class="INVARIANT_VIOLATION",
            abort_reason=f"episode_outcome={outcome}, status={status}",
            extra={
                "synthesis_signal": "OUTCOME_FAILURE",
                "episode_id": episode_id,
                "n_steps": len(exps),
            },
        ))

    # ── 2. EMPTY_QUERY — bank had no match for this step ─────────────
    for i, exp in enumerate(exps):
        sq = exp.get("skill_query") or {}
        if sq.get("empty") and not sq.get("error"):
            out.append(FailureTrace(
                skill_id="",
                skill_episode_id=f"{episode_id}#empty_query@{i}",
                domain=domain,
                failed_step_index=i,
                failure_class="MISSING_ADAPTER",
                abort_reason="skill_query.empty",
                extra={"synthesis_signal": "EMPTY_QUERY", "step_index": i},
            ))

    # ── 3. LOW_APPLICABILITY — bound skill has weak fit ──────────────
    for i, exp in enumerate(exps):
        sk = exp.get("skills") or {}
        if not sk:
            continue
        app = sk.get("applicability")
        if app is None or not isinstance(app, (int, float)):
            continue
        if app < low_applicability:
            out.append(FailureTrace(
                skill_id=_safe_skill_id(sk.get("skill_id", "") or ""),
                skill_episode_id=f"{episode_id}#low_app@{i}",
                domain=domain,
                failed_step_index=i,
                failure_class="PRECONDITION_VIOLATION",
                abort_reason=f"applicability={app:.3f} < {low_applicability}",
                extra={
                    "synthesis_signal": "LOW_APPLICABILITY",
                    "step_index": i,
                    "applicability": float(app),
                    "confidence": sk.get("confidence"),
                },
            ))

    # ── 4. MISSING_EFFECTS — bound skill claimed effects that
    #      didn't materialize.
    for i, exp in enumerate(exps):
        sk = exp.get("skills") or {}
        miss = sk.get("missing_effects")
        if miss:
            out.append(FailureTrace(
                skill_id=_safe_skill_id(sk.get("skill_id", "") or ""),
                skill_episode_id=f"{episode_id}#miss_eff@{i}",
                domain=domain,
                failed_step_index=i,
                failure_class="INVARIANT_VIOLATION",
                abort_reason=f"missing_effects={miss}",
                extra={
                    "synthesis_signal": "MISSING_EFFECTS",
                    "step_index": i,
                    "missing_effects": list(miss),
                },
            ))

    # Cap: severity ordering matches list order (outcome > empty_query >
    # low_app > missing_effects), so taking the head retains the most
    # informative signals.
    if len(out) > max_failures:
        out = out[:max_failures]
    return out


# ═══════════════════════════════════════════════════════════════════════
# Per-episode driver
# ═══════════════════════════════════════════════════════════════════════

def _bank_mgmt_path(bank_run: Path, corpus: str, source: str, ep_idx: int) -> Path:
    return (
        bank_run / corpus / source / "per_episode_bank_management"
        / f"episode_{ep_idx}" / "bank_management_io.json"
    )


def _process_episode(
    *,
    crafter: SkillCrafterService,
    lifecycle: SkillLifecycleManager,
    episode_path: Path,
    bank_mgmt_path: Path,
    out_dir: Path,
    corpus: str,
    source: str,
    domain: str,
    low_app: float,
    max_failures: int,
) -> Dict[str, Any]:
    """Process one episode end-to-end. Returns a row for the per-source summary."""
    episode = json.loads(episode_path.read_text())
    episode_id = (
        episode.get("episode_id")
        or f"{corpus}/{source}/{episode_path.stem}"
    )

    # ── new_candidate_skill_ids + bank_agent_actions from bank-mgmt ──
    new_cand_ids: List[str] = []
    bank_agent_actions: Dict[str, Any] = {}
    if bank_mgmt_path.exists():
        try:
            bank_mgmt = json.loads(bank_mgmt_path.read_text())
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("could not parse %s: %s", bank_mgmt_path, exc)
            bank_mgmt = {}
        if bank_mgmt:
            new_cand_ids = _seed_refines_as_candidates(
                lifecycle, bank_mgmt, default_domain=domain
            )
            s4 = (bank_mgmt.get("stage_4_bank_maintenance") or {}).get("outputs") or {}
            bank_agent_actions = {
                "n_splits": int(s4.get("n_splits") or 0),
                "n_merges": int(s4.get("n_merges") or 0),
                "n_refines": int(s4.get("n_refines") or 0),
                "alias_map": s4.get("alias_map") or {},
            }

    # ── failure traces from per-step signals ─────────────────────────
    failure_traces = _synthesize_failures(
        episode=episode,
        episode_id=episode_id,
        domain=domain,
        low_applicability=low_app,
        max_failures=max_failures,
    )

    # ── outcome summary ──────────────────────────────────────────────
    outcome_summary = {
        "outcome": episode.get("outcome"),
        "episode_status": episode.get("episode_status"),
        "n_steps": len(episode.get("experiences") or episode.get("steps") or []),
        "total_reward": _sum_reward(episode),
    }

    reflection = EpisodeReflection(
        episode_id=episode_id,
        domain=domain,
        parent_run_id=str(episode_path.parent),
        failure_traces=failure_traces,
        skill_episodes=[],                                          # offline mirror has no SkillEpisode log
        new_candidate_skill_ids=new_cand_ids,
        bank_agent_actions=bank_agent_actions,
        outcome_summary=outcome_summary,
    )

    # ── invoke the LIVE Crafter ──────────────────────────────────────
    t0 = time.time()
    result = crafter.reflect_on_episode(reflection)
    elapsed = time.time() - t0

    # ── serialise per-episode artefacts ──────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    proposals_path = out_dir / "proposals.jsonl"
    with proposals_path.open("w") as f:
        for p in result.proposals:
            f.write(json.dumps(proposal_to_json(p), ensure_ascii=False, sort_keys=True) + "\n")

    (out_dir / "reflection.json").write_text(
        json.dumps(reflection.to_json(), indent=2, ensure_ascii=False, sort_keys=True)
    )

    by_kind: Counter[str] = Counter(type(p).__name__ for p in result.proposals)
    by_proposer: Counter[str] = Counter()
    for p in result.proposals:
        # The proposer string is encoded in `source_type` for the live
        # proposals; mirror the offline driver's `proposer` label by
        # mapping back.
        by_proposer[_infer_proposer(p)] += 1

    (out_dir / "result.json").write_text(json.dumps({
        "trigger": result.trigger,
        "episode_id": result.episode_id,
        "n_failures_ingested": result.n_failures_ingested,
        "n_patterns_examined": result.n_patterns_examined,
        "n_proposals": len(result.proposals),
        "n_subsumption_retires": result.n_subsumption_retires,
        "n_patches_coalesced": result.n_patches_coalesced,
        "n_patches_skipped_cooldown": result.n_patches_skipped_cooldown,
        "by_kind": dict(by_kind),
        "by_proposer": dict(by_proposer),
        "bank_view_summary": dict(result.bank_view_summary),
        "elapsed_sec": round(elapsed, 4),
        "completed_at": _utcnow_iso(),
    }, indent=2))

    return {
        "episode_path": str(episode_path),
        "episode_id": episode_id,
        "n_failures_synthesized": len(failure_traces),
        "n_new_candidates": len(new_cand_ids),
        "bank_agent_actions": bank_agent_actions,
        "n_proposals": len(result.proposals),
        "n_subsumption_retires": result.n_subsumption_retires,
        "n_patches_coalesced": result.n_patches_coalesced,
        "n_patches_skipped_cooldown": result.n_patches_skipped_cooldown,
        "by_kind": dict(by_kind),
        "by_proposer": dict(by_proposer),
        "bank_view_summary": dict(result.bank_view_summary),
        "elapsed_sec": round(elapsed, 4),
    }


def _sum_reward(episode: Dict[str, Any]) -> float:
    exps = episode.get("experiences") or episode.get("steps") or []
    total = 0.0
    for e in exps:
        r = e.get("reward")
        if isinstance(r, (int, float)):
            total += float(r)
    return total


def _infer_proposer(p: BankMutationProposal) -> str:
    """Mirror the offline labels (composer / generalizer / hypothesizer /
    reflector) that the labeling_supplement summary tooling expects."""
    cls = type(p).__name__
    if cls == "ComposeProposal":
        return "composer"
    if cls == "GeneralizeProposal":
        return "generalizer"
    if cls == "HypothesisProposal":
        return "hypothesizer"
    # Patches and retires both come from the failure-reflection chain.
    return "reflector"


# ═══════════════════════════════════════════════════════════════════════
# Per-source driver
# ═══════════════════════════════════════════════════════════════════════

def _process_source(
    *,
    corpus: str,
    source: str,
    bank_run: Path,
    actions_run: Path,
    output_root: Path,
    max_episodes: Optional[int],
    low_app: float,
    max_failures: int,
) -> Dict[str, Any]:
    """Run reflect_on_episode across every episode of one (corpus, source)."""
    t0 = time.time()
    src_actions = actions_run / corpus / source
    src_bank = bank_run / corpus / source
    bank_path = src_bank / "skill_bank.jsonl"

    out_src = output_root / corpus / source
    out_src.mkdir(parents=True, exist_ok=True)

    eps = sorted(src_actions.glob("episode_*.json"))
    if max_episodes is not None:
        eps = eps[:max_episodes]
    if not eps:
        logger.warning("%s/%s: no episode_*.json under %s", corpus, source, src_actions)
        (out_src / "_source_summary.json").write_text(json.dumps({
            "corpus": corpus, "source": source,
            "status": "no_episodes",
            "actions_dir": str(src_actions),
        }, indent=2))
        return {
            "corpus": corpus, "source": source,
            "status": "no_episodes", "n_episodes": 0,
            "n_proposals": 0, "elapsed_sec": 0.0,
        }

    domain = _infer_domain(corpus, source)

    # ── temp bank, fresh per source ──────────────────────────────────
    temp_root = Path(tempfile.mkdtemp(prefix=f"crafter_mirror_{corpus}_{source}_"))
    try:
        repo = SkillRepository(
            draft_store=SkillStore(StoreName.DRAFT, str(temp_root / "draft")),
            candidate_store=SkillStore(StoreName.CANDIDATE, str(temp_root / "candidate")),
            active_store=SkillStore(StoreName.ACTIVE, str(temp_root / "active")),
            archive_store=SkillStore(StoreName.ARCHIVE, str(temp_root / "archive")),
        )
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(temp_root / "artifacts"))
        # Offline diagnostic driver — keep the Repairer path live so the
        # JSONL captures every proposal kind for inspection, even though
        # the live trainer's default lane-(a) flag is ``False`` (T1.3a /
        # ``implementation_notes/legacy/skill-lane-decision.md``).
        crafter = SkillCrafterService(
            lifecycle=lifecycle,
            artifact_store=artifacts,
            enable_protocol_patching=True,
        )

        n_seeded, n_seed_skipped = _seed_bank(lifecycle, bank_path, domain)

        # ── per-episode loop ────────────────────────────────────────
        rows: List[Dict[str, Any]] = []
        total_props = 0
        total_subsumes = 0
        total_coalesced = 0
        total_cooldown = 0
        by_kind: Counter[str] = Counter()
        by_proposer: Counter[str] = Counter()

        for ep_path in eps:
            # Resolve the matching bank-mgmt JSON. Naming is
            # ``per_episode_bank_management/episode_<i>/bank_management_io.json``
            # where <i> is the *integer* episode index — note that
            # skill_actions_out filenames are zero-padded
            # ``episode_NNN.json`` while the bank-mgmt directory names
            # are unpadded ``episode_<i>``. Strip leading zeros.
            try:
                ep_idx = int(ep_path.stem.split("_")[-1])
            except ValueError:
                ep_idx = -1
            bank_mgmt_path = _bank_mgmt_path(bank_run, corpus, source, ep_idx)

            ep_out = out_src / ep_path.stem
            row = _process_episode(
                crafter=crafter,
                lifecycle=lifecycle,
                episode_path=ep_path,
                bank_mgmt_path=bank_mgmt_path,
                out_dir=ep_out,
                corpus=corpus,
                source=source,
                domain=domain,
                low_app=low_app,
                max_failures=max_failures,
            )
            rows.append(row)
            total_props += row["n_proposals"]
            total_subsumes += row["n_subsumption_retires"]
            total_coalesced += row.get("n_patches_coalesced", 0)
            total_cooldown += row.get("n_patches_skipped_cooldown", 0)
            by_kind.update(row["by_kind"])
            by_proposer.update(row["by_proposer"])

        elapsed = time.time() - t0
        (out_src / "_source_summary.json").write_text(json.dumps({
            "corpus": corpus,
            "source": source,
            "domain": domain,
            "bank_path": str(bank_path),
            "actions_dir": str(src_actions),
            "status": "ok",
            "n_skills_seeded": n_seeded,
            "n_skills_skipped_in_seed": n_seed_skipped,
            "n_episodes": len(eps),
            "n_proposals": total_props,
            "n_subsumption_retires": total_subsumes,
            "n_patches_coalesced": total_coalesced,
            "n_patches_skipped_cooldown": total_cooldown,
            "by_kind": dict(by_kind),
            "by_proposer": dict(by_proposer),
            "elapsed_sec": round(elapsed, 3),
            "thresholds": {
                "low_applicability": low_app,
                "max_failures_per_episode": max_failures,
            },
            "completed_at": _utcnow_iso(),
            "per_episode": rows,
        }, indent=2))

        return {
            "corpus": corpus,
            "source": source,
            "domain": domain,
            "status": "ok",
            "n_skills_seeded": n_seeded,
            "n_episodes": len(eps),
            "n_proposals": total_props,
            "n_subsumption_retires": total_subsumes,
            "n_patches_coalesced": total_coalesced,
            "n_patches_skipped_cooldown": total_cooldown,
            "by_kind": dict(by_kind),
            "by_proposer": dict(by_proposer),
            "elapsed_sec": round(elapsed, 3),
        }
    finally:
        # Tear down the temp bank — proposals are persisted to
        # ``out_src/`` so dropping the temp dir doesn't lose data.
        shutil.rmtree(temp_root, ignore_errors=True)


def _infer_domain(corpus: str, source: str) -> str:
    """Best-effort domain id for the live Crafter / lifecycle invariants.

    The on-disk bank uses ``"gymv"`` for both gym_v envs and
    env_wrappers games (Phase-1 single-domain). We honour that.
    """
    return "gymv"


# ═══════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════

def _discover_pairs(
    bank_run: Path,
    actions_run: Path,
    corpus_filter: Optional[str],
    source_filter: Optional[str],
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for corpus in CORPORA:
        if corpus_filter and corpus != corpus_filter:
            continue
        cdir_b = bank_run / corpus
        cdir_a = actions_run / corpus
        if not cdir_b.exists() or not cdir_a.exists():
            continue
        for src_dir in sorted(cdir_b.iterdir()):
            if not src_dir.is_dir() or src_dir.name.startswith("_"):
                continue
            if source_filter and src_dir.name != source_filter:
                continue
            if not (src_dir / "skill_bank.jsonl").exists():
                continue
            actions_dir = cdir_a / src_dir.name
            if not actions_dir.exists():
                continue
            if not list(actions_dir.glob("episode_*.json")):
                continue
            out.append((corpus, src_dir.name))
    return out


# ═══════════════════════════════════════════════════════════════════════
# Transfer-target driver (VR / video / browser / osworld)
# ═══════════════════════════════════════════════════════════════════════
#
# This second driver mirrors `_process_source` but for the per-sample
# cold-start corpora produced by
# ``cold_start/generate_cold_start_actor_visual_reasoning.py`` (and the
# planned browsergym / osworld / video equivalents). The contract is:
#
#   * One temp ``SkillCrafterService`` per (domain, benchmark) pair.
#   * Optional bank seeding from a ``--seed-bank`` JSONL, tagging each
#     loaded skill with ``feasible_domains=[<domain>]`` so the
#     EligibilityFilter accepts it (gymv-tagged seeds would silently get
#     vetoed for domain mismatch).
#   * Per-sample synthesis via ``_failure_synth.get_synthesizer(domain)``.
#   * Optional nearest-neighbor binding (``--match-skill-by-token``) to
#     attach a base skill_id to each FailureTrace so the Crafter's
#     dispatch chain (repair > retire > hypothesize) reaches the
#     Repairer. Without this every VR failure has empty skill_id and
#     the dispatch falls through to the Hypothesizer — fine for a pure
#     hypothesizer smoke, but useless if the test target is the
#     Repairer / Patch path the user enabled.
#
# The output layout matches the gymv driver (per-sample
# ``proposals.jsonl`` / ``reflection.json`` / ``result.json`` plus a
# per-benchmark ``_source_summary.json``) so the existing aggregation
# tooling under ``labeling_supplement/promotion_decisions_out/`` keeps
# working without changes.

def _seed_bank_for_target_domain(
    lifecycle: SkillLifecycleManager,
    bank_jsonl: Path,
    target_domain: str,
) -> Tuple[int, int]:
    """Like ``_seed_bank`` but rewrites ``feasible_domains`` to
    ``[target_domain]`` so the EligibilityFilter admits the skill in
    the new domain. The original gymv lineage is preserved on
    ``parent_skill_ids`` (NOT on ``feasible_domains`` — that's the
    runtime-eligibility axis, not the provenance axis)."""
    if not bank_jsonl.exists():
        return 0, 0
    n = 0
    skipped = 0
    with bank_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            try:
                rec = _record_from_bank_entry(entry, default_domain=target_domain)
                # Force the runtime-eligibility domain to the target
                # domain so the EligibilityFilter doesn't veto every
                # skill on the F1 (domain) axis. The Repairer will
                # patch the skill body further.
                object.__setattr__(rec, "feasible_domains", [target_domain])
                lifecycle.ingest_draft(rec)
                lifecycle.transition(
                    rec.skill_id,
                    to_status=SkillStatus.CANDIDATE,
                    rationale=f"seed-from-bank-snapshot[retag→{target_domain}]",
                )
                n += 1
            except Exception as exc:                                # noqa: BLE001
                logger.debug("skip seed %s: %s",
                             entry.get("skill", {}).get("skill_id"), exc)
                skipped += 1
    return n, skipped


def _tokenize(text: Any) -> set:
    """Mirror ``crafter.service._tokenize_for_relatedness`` so
    nearest-neighbor binding uses the same vocabulary the
    Hypothesizer's relatedness gate uses."""
    if not text:
        return set()
    import re as _re
    return {
        w for w in _re.split(r"[^a-zA-Z0-9]+", str(text).lower())
        if len(w) >= 2
    }


def _bind_failures_to_nearest_skill(
    failures: List[FailureTrace],
    lifecycle: SkillLifecycleManager,
    *,
    min_jaccard: float = 0.05,
) -> int:
    """Tag each ``FailureTrace`` whose ``skill_id`` is empty with the
    bank's nearest-neighbor skill_id by token Jaccard over the
    failure's ``abort_reason`` + ``extra``.

    Returns the number of failures that received a binding. Skills are
    drawn from candidate + draft (the two stores
    ``_seed_bank_for_target_domain`` populates).
    """
    repo = lifecycle.repository
    bank: List[Tuple[str, set]] = []
    # Repository store accessors are ``draft`` / ``candidate`` /
    # ``active`` / ``archive`` (no ``_store`` suffix); the gymv seed
    # path lands records as CANDIDATE so that's the primary search
    # surface, with DRAFT as fallback for in-flight records.
    for store in (repo.candidate, repo.draft):
        for sk in store.all():
            tokens = _tokenize(sk.name) | _tokenize(sk.notes)
            for hop in sk.protocol or []:
                tokens |= _tokenize(hop.get("notes"))
            if tokens:
                bank.append((sk.skill_id, tokens))
    if not bank:
        return 0

    n_bound = 0
    for trace in failures:
        if trace.skill_id:
            continue
        ctx = _tokenize(trace.abort_reason) | _tokenize(trace.failure_class)
        for k, v in (trace.extra or {}).items():
            ctx |= _tokenize(v)
            ctx |= _tokenize(k)
        if not ctx:
            continue
        best_id, best_j = "", 0.0
        for sk_id, sk_tokens in bank:
            inter = len(ctx & sk_tokens)
            if inter == 0:
                continue
            j = inter / len(ctx | sk_tokens)
            if j > best_j:
                best_id, best_j = sk_id, j
        if best_id and best_j >= min_jaccard:
            object.__setattr__(trace, "skill_id", best_id)
            trace.extra.setdefault("binding", {})["nearest_skill_jaccard"] = round(best_j, 4)
            n_bound += 1
    return n_bound


def _process_target_sample(
    *,
    crafter: SkillCrafterService,
    lifecycle: SkillLifecycleManager,
    sample_path: Path,
    out_dir: Path,
    benchmark: str,
    domain: str,
    max_failures: int,
    match_skill_by_token: bool,
    binding_jaccard_min: float,
) -> Dict[str, Any]:
    """Process one cold-start per-sample JSON end-to-end (transfer-target mode)."""
    sample = json.loads(sample_path.read_text())
    sid = str(sample.get("sample_id") or sample.get("task_id") or sample_path.stem)
    sample_id_full = f"{benchmark}/{sid}"

    if get_synthesizer is None:
        raise RuntimeError(
            "labeling_supplement._failure_synth is not importable; "
            "transfer-target mode requires the synthesiser package."
        )
    synth = get_synthesizer(domain)
    failure_traces = synth(
        sample,
        domain=domain,
        sample_id=sample_id_full,
        max_failures=max_failures,
    )

    # Optional: bind empty-skill_id failures to the nearest seeded
    # bank skill so the Crafter's dispatch reaches the Repairer.
    n_bound = 0
    if match_skill_by_token and failure_traces:
        n_bound = _bind_failures_to_nearest_skill(
            failure_traces, lifecycle, min_jaccard=binding_jaccard_min,
        )

    outcome_summary = {
        "benchmark": benchmark,
        "sample_id": sid,
        "correct": bool(sample.get("correct")),
        "judge_verdict": (sample.get("judge") or {}).get("verdict"),
        "is_mcq": bool(sample.get("is_mcq")),
        "schema_recovery": sample.get("schema_recovery"),
    }

    reflection = EpisodeReflection(
        episode_id=sample_id_full,
        domain=domain,
        parent_run_id=str(sample_path.parent),
        failure_traces=failure_traces,
        skill_episodes=[],
        new_candidate_skill_ids=[],
        bank_agent_actions={},
        outcome_summary=outcome_summary,
    )

    t0 = time.time()
    result = crafter.reflect_on_episode(reflection)
    elapsed = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    proposals_path = out_dir / "proposals.jsonl"
    with proposals_path.open("w") as f:
        for p in result.proposals:
            f.write(json.dumps(proposal_to_json(p), ensure_ascii=False, sort_keys=True) + "\n")

    (out_dir / "reflection.json").write_text(
        json.dumps(reflection.to_json(), indent=2, ensure_ascii=False, sort_keys=True)
    )

    by_kind: Counter[str] = Counter(type(p).__name__ for p in result.proposals)
    by_proposer: Counter[str] = Counter(_infer_proposer(p) for p in result.proposals)

    (out_dir / "result.json").write_text(json.dumps({
        "trigger": result.trigger,
        "episode_id": result.episode_id,
        "n_failures_synthesized": len(failure_traces),
        "n_failures_ingested": result.n_failures_ingested,
        "n_failures_bound_by_token": n_bound,
        "n_patterns_examined": result.n_patterns_examined,
        "n_proposals": len(result.proposals),
        "n_subsumption_retires": result.n_subsumption_retires,
        "n_patches_coalesced": result.n_patches_coalesced,
        "n_patches_skipped_cooldown": result.n_patches_skipped_cooldown,
        "by_kind": dict(by_kind),
        "by_proposer": dict(by_proposer),
        "bank_view_summary": dict(result.bank_view_summary),
        "elapsed_sec": round(elapsed, 4),
        "completed_at": _utcnow_iso(),
    }, indent=2))

    return {
        "sample_path": str(sample_path),
        "benchmark": benchmark,
        "sample_id": sid,
        "n_failures_synthesized": len(failure_traces),
        "n_failures_bound_by_token": n_bound,
        "n_proposals": len(result.proposals),
        "by_kind": dict(by_kind),
        "by_proposer": dict(by_proposer),
        "elapsed_sec": round(elapsed, 4),
    }


def _process_target_benchmark(
    *,
    benchmark: str,
    domain: str,
    samples_dir: Path,
    output_root: Path,
    seed_bank_path: Optional[Path],
    max_samples: Optional[int],
    sample_id_filter: Optional[List[str]],
    max_failures: int,
    match_skill_by_token: bool,
    binding_jaccard_min: float,
    enable_protocol_patching: bool,
    hot_pattern_threshold: int,
    hypothesize_min_recurrences: int,
    llm_repairer: bool,
    llm_hypothesizer: bool,
    llm_diagnoser: bool,
    llm_model: str,
) -> Dict[str, Any]:
    """Run reflect across all per-sample JSONs for one benchmark."""
    t0 = time.time()
    out_src = output_root / domain / benchmark
    out_src.mkdir(parents=True, exist_ok=True)

    samples = sorted(samples_dir.glob("sample_*.json"))
    if sample_id_filter is not None:
        keep = set(sample_id_filter)
        # Match either by file stem or by the sample's own sample_id.
        kept: List[Path] = []
        for sp in samples:
            if sp.stem in keep:
                kept.append(sp)
                continue
            try:
                blob = json.loads(sp.read_text())
                if str(blob.get("sample_id") or "") in keep:
                    kept.append(sp)
            except Exception:                                       # noqa: BLE001
                continue
        samples = kept
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        logger.warning("%s/%s: no sample_*.json (or filter dropped all)",
                       domain, benchmark)
        (out_src / "_source_summary.json").write_text(json.dumps({
            "domain": domain, "benchmark": benchmark,
            "status": "no_samples", "samples_dir": str(samples_dir),
        }, indent=2))
        return {"domain": domain, "benchmark": benchmark, "status": "no_samples",
                "n_samples": 0, "n_proposals": 0, "elapsed_sec": 0.0}

    temp_root = Path(tempfile.mkdtemp(prefix=f"crafter_target_{domain}_{benchmark}_"))
    try:
        repo = SkillRepository(
            draft_store=SkillStore(StoreName.DRAFT, str(temp_root / "draft")),
            candidate_store=SkillStore(StoreName.CANDIDATE, str(temp_root / "candidate")),
            active_store=SkillStore(StoreName.ACTIVE, str(temp_root / "active")),
            archive_store=SkillStore(StoreName.ARCHIVE, str(temp_root / "archive")),
        )
        lifecycle = SkillLifecycleManager(repo)
        artifacts = ArtifactStore(str(temp_root / "artifacts"))
        crafter = SkillCrafterService(
            lifecycle=lifecycle,
            artifact_store=artifacts,
            enable_protocol_patching=enable_protocol_patching,
            hot_pattern_threshold=hot_pattern_threshold,
            hypothesize_min_recurrences=hypothesize_min_recurrences,
        )

        # Optional LLM hooks (Step-0 of the README's integration roadmap).
        llm_status: Dict[str, Any] = {"installed": False}
        if llm_repairer or llm_hypothesizer or llm_diagnoser:
            try:
                from crafter._llm_runtime import install_llm_hooks
                llm_status = install_llm_hooks(
                    crafter,
                    model=llm_model,
                    audit_sink=artifacts.append_audit,
                    enable_diagnoser=llm_diagnoser,
                    enable_repairer=llm_repairer,
                    enable_hypothesizer=llm_hypothesizer,
                )
                llm_status["installed"] = True
            except Exception as exc:                                # noqa: BLE001
                logger.error("LLM hook install failed: %s", exc)
                llm_status = {"installed": False, "error": str(exc)}

        n_seeded = n_seed_skipped = 0
        if seed_bank_path is not None:
            n_seeded, n_seed_skipped = _seed_bank_for_target_domain(
                lifecycle, seed_bank_path, target_domain=domain,
            )

        rows: List[Dict[str, Any]] = []
        total_props_per_episode = 0
        total_subsumes = 0
        total_coalesced = 0
        total_cooldown = 0
        total_bound = 0
        by_kind: Counter[str] = Counter()
        by_proposer: Counter[str] = Counter()

        for sp in samples:
            row = _process_target_sample(
                crafter=crafter,
                lifecycle=lifecycle,
                sample_path=sp,
                out_dir=out_src / sp.stem,
                benchmark=benchmark,
                domain=domain,
                max_failures=max_failures,
                match_skill_by_token=match_skill_by_token,
                binding_jaccard_min=binding_jaccard_min,
            )
            rows.append(row)
            total_props_per_episode += row["n_proposals"]
            total_bound += row.get("n_failures_bound_by_token", 0)
            by_kind.update(row["by_kind"])
            by_proposer.update(row["by_proposer"])

        # Per-batch reflective pass (PLAN-SKILL-CRAFTER §6.5) — VR is a
        # one-call-per-sample modality, so the per-episode pass above
        # always sees pattern.count==1 and the Hypothesizer recurrence
        # gate (default 3) blocks every proposal. The cross-sample
        # aggregation needs `cycle()`: FailureMemory accumulates across
        # `reflect_on_episode` calls (it's the same SkillCrafterService),
        # so calling `cycle()` here with no new failures simply re-runs
        # dispatch over the now-populated memory at the proper
        # `hot_pattern_threshold`.
        cycle_t0 = time.time()
        cycle_result = crafter.cycle(new_failures=None)
        cycle_elapsed = time.time() - cycle_t0
        total_props_cycle = len(cycle_result.proposals)
        cycle_by_kind: Counter[str] = Counter(
            type(p).__name__ for p in cycle_result.proposals
        )
        cycle_by_proposer: Counter[str] = Counter(
            _infer_proposer(p) for p in cycle_result.proposals
        )
        by_kind.update(cycle_by_kind)
        by_proposer.update(cycle_by_proposer)
        total_subsumes += cycle_result.n_subsumption_retires
        total_coalesced += cycle_result.n_patches_coalesced
        total_cooldown += cycle_result.n_patches_skipped_cooldown

        # Persist the cycle proposals separately so the per-sample
        # vs. cross-sample attribution is preserved on disk.
        cycle_out = out_src / "_cycle"
        cycle_out.mkdir(parents=True, exist_ok=True)
        with (cycle_out / "proposals.jsonl").open("w") as f:
            for p in cycle_result.proposals:
                f.write(json.dumps(proposal_to_json(p), ensure_ascii=False, sort_keys=True) + "\n")
        (cycle_out / "result.json").write_text(json.dumps({
            "trigger": cycle_result.trigger,
            "n_failures_ingested": cycle_result.n_failures_ingested,
            "n_patterns_examined": cycle_result.n_patterns_examined,
            "n_proposals": total_props_cycle,
            "n_subsumption_retires": cycle_result.n_subsumption_retires,
            "n_patches_coalesced": cycle_result.n_patches_coalesced,
            "n_patches_skipped_cooldown": cycle_result.n_patches_skipped_cooldown,
            "by_kind": dict(cycle_by_kind),
            "by_proposer": dict(cycle_by_proposer),
            "elapsed_sec": round(cycle_elapsed, 4),
            "completed_at": _utcnow_iso(),
        }, indent=2))

        total_props = total_props_per_episode + total_props_cycle

        elapsed = time.time() - t0
        (out_src / "_source_summary.json").write_text(json.dumps({
            "domain": domain,
            "benchmark": benchmark,
            "samples_dir": str(samples_dir),
            "status": "ok",
            "n_samples": len(samples),
            "n_skills_seeded": n_seeded,
            "n_skills_skipped_in_seed": n_seed_skipped,
            "n_failures_bound_by_token": total_bound,
            "n_proposals": total_props,
            "n_proposals_per_episode": total_props_per_episode,
            "n_proposals_cross_sample_cycle": total_props_cycle,
            "by_kind": dict(by_kind),
            "by_proposer": dict(by_proposer),
            "thresholds": {
                "max_failures_per_sample": max_failures,
                "binding_jaccard_min": binding_jaccard_min if match_skill_by_token else None,
                "hot_pattern_threshold": hot_pattern_threshold,
                "hypothesize_min_recurrences": hypothesize_min_recurrences,
            },
            "knobs": {
                "enable_protocol_patching": enable_protocol_patching,
                "match_skill_by_token": match_skill_by_token,
                "seed_bank": str(seed_bank_path) if seed_bank_path else None,
                "llm_status": llm_status,
            },
            "elapsed_sec": round(elapsed, 3),
            "completed_at": _utcnow_iso(),
            "per_sample": rows,
        }, indent=2))

        return {
            "domain": domain,
            "benchmark": benchmark,
            "status": "ok",
            "n_samples": len(samples),
            "n_proposals": total_props,
            "n_subsumption_retires": total_subsumes,
            "n_patches_coalesced": total_coalesced,
            "n_patches_skipped_cooldown": total_cooldown,
            "n_failures_bound_by_token": total_bound,
            "by_kind": dict(by_kind),
            "by_proposer": dict(by_proposer),
            "elapsed_sec": round(elapsed, 3),
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bank-run", type=Path, default=DEFAULT_BANK_RUN,
                   help="Skill-bank snapshot directory.")
    p.add_argument("--actions-run", type=Path, default=DEFAULT_ACTIONS_RUN,
                   help="Skill-actions snapshot directory (per-episode JSONs).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output root; defaults to "
                        "labeling_supplement/episode_reflections_out/run_<ts>.")
    p.add_argument("--corpus", choices=CORPORA, default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Per-source cap on episodes processed (smoke testing).")
    p.add_argument("--low-applicability", type=float,
                   default=DEFAULT_LOW_APPLICABILITY,
                   help="Threshold below which a per-step applicability "
                        "is treated as a PRECONDITION_VIOLATION.")
    p.add_argument("--max-failures-per-episode", type=int,
                   default=DEFAULT_MAX_FAILURES_PER_EPISODE,
                   help="Hard cap on synthesised FailureTraces per episode.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print discovered pairs and exit without invoking the Crafter.")
    p.add_argument("-v", "--verbose", action="store_true")

    # ── Transfer-target mode (VR / video / browser / osworld smoke) ──
    # When --domain is set to a transfer-target domain (NOT 'gymv'),
    # the legacy gymv (corpus, source) discovery is bypassed and the
    # script switches to the per-sample driver in
    # `_process_target_benchmark`. The gymv path is byte-identical
    # whenever --domain is unset or is 'gymv'.
    target_grp = p.add_argument_group(
        "Transfer-target mode (VR / video / browser / osworld)",
        "Activated when --domain != 'gymv'. Mutually exclusive with the "
        "gymv (corpus, source) discovery flags.",
    )
    target_grp.add_argument(
        "--domain", default="gymv",
        choices=("gymv",) + tuple(TRANSFER_TARGET_DOMAINS),
        help="Which target domain to process. Default 'gymv' = legacy path.",
    )
    target_grp.add_argument(
        "--samples-root", type=Path, default=None,
        help="Root holding per-benchmark cold-start dirs, e.g. "
             "Cold-start-out-visual-reasoning/. Required when --domain != gymv.",
    )
    target_grp.add_argument(
        "--benchmarks", nargs="+", default=None,
        help="Subdirectory names under --samples-root to process, e.g. "
             "visual_toolbench tir_bench. Default = every subdir with sample_*.json.",
    )
    target_grp.add_argument(
        "--max-samples-per-benchmark", type=int, default=None,
        help="Per-benchmark cap (smoke testing).",
    )
    target_grp.add_argument(
        "--sample-ids-file", type=Path, default=None,
        help="Optional file with one sample_id (or sample_NNN stem) per "
             "line; restricts processing to those samples. Pair with the "
             "manifests under cold_start/evaluation_dataset/{pool,holdout}/.",
    )
    target_grp.add_argument(
        "--seed-bank", type=Path, default=None,
        help="Optional skill_bank.jsonl to seed as CANDIDATE (re-tagged "
             "feasible_domains=[<domain>]). Required for the Repairer "
             "path; without it the dispatch falls through to the "
             "Hypothesizer for every failure.",
    )
    target_grp.add_argument(
        "--match-skill-by-token", action="store_true",
        help="Token-Jaccard nearest-neighbor: bind every empty-skill_id "
             "FailureTrace to the closest seeded skill so the Repairer "
             "(rather than the Hypothesizer) handles it.",
    )
    target_grp.add_argument(
        "--binding-jaccard-min", type=float, default=0.05,
        help="Minimum Jaccard for token-binding (default 0.05).",
    )

    # ── Crafter knobs ────────────────────────────────────────────────
    knob_grp = p.add_argument_group("Crafter knobs (apply to both modes)")
    knob_grp.add_argument(
        "--enable-protocol-patching", action="store_true", default=False,
        help="Lane-(b): allow Repairer to mint PatchProposal records. "
             "Default OFF (lane-(a)) — matches live trainer behaviour.",
    )
    knob_grp.add_argument(
        "--hot-pattern-threshold", type=int, default=3,
        help="Per-batch dispatch threshold (per-episode pass uses 1).",
    )
    knob_grp.add_argument(
        "--hypothesize-min-recurrences", type=int, default=3,
        help="Hypothesizer fallthrough gate: pattern.count must reach "
             "this many before the Hypothesizer fires. Set to a large "
             "number to effectively disable Hypothesizer for repair-only "
             "smoke tests.",
    )

    # ── LLM hooks (Step-0 of crafter/README.md teacher-LLM roadmap) ──
    llm_grp = p.add_argument_group(
        "Crafter LLM hooks",
        "Wire the dormant Repairer / Hypothesizer / FailureDiagnoser "
        "hooks to a real LLM via API_func.ask_model. Defaults preserve "
        "the deterministic rule path.",
    )
    llm_grp.add_argument("--llm-repairer", action="store_true", default=False,
                        help="Replace Repairer rule path with LLMRepairer (gpt-5.4 by default).")
    llm_grp.add_argument("--llm-hypothesizer", action="store_true", default=False)
    llm_grp.add_argument("--llm-diagnoser", action="store_true", default=False)
    llm_grp.add_argument("--llm-model", default="gpt-5.4",
                        help="Model id for the LLM hooks (default gpt-5.4).")
    return p


def _main_target(args: argparse.Namespace, output_root: Path) -> int:
    """Driver for transfer-target benchmarks (VR / video / browser / osworld).

    Activated when ``--domain != gymv``. Reads cold-start per-sample
    JSONs from ``--samples-root/<benchmark>/`` and emits the same
    per-sample artefact layout the gymv driver does, plus a top-level
    ``_run_summary.json``.
    """
    if args.samples_root is None:
        logger.error("--samples-root is required when --domain != gymv")
        return 2
    samples_root: Path = args.samples_root.resolve()
    if not samples_root.is_dir():
        logger.error("samples-root not a directory: %s", samples_root)
        return 2

    # Discover benchmarks.
    if args.benchmarks:
        benchmarks = list(args.benchmarks)
    else:
        benchmarks = sorted(
            p.name for p in samples_root.iterdir()
            if p.is_dir()
            and not p.name.startswith("_")
            and any(p.glob("sample_*.json"))
        )
    if not benchmarks:
        logger.error("no benchmark subdirs with sample_*.json under %s", samples_root)
        return 2

    sample_id_filter: Optional[List[str]] = None
    if args.sample_ids_file is not None:
        ids = []
        for line in args.sample_ids_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
        sample_id_filter = ids
        logger.info("loaded %d sample-id filter entries from %s",
                    len(ids), args.sample_ids_file)

    seed_bank_path: Optional[Path] = None
    if args.seed_bank is not None:
        seed_bank_path = args.seed_bank.resolve()
        if not seed_bank_path.is_file():
            logger.error("seed-bank not a file: %s", seed_bank_path)
            return 2

    started_at = _utcnow_iso()
    logger.info("reflect_per_episode (target): domain=%s benchmarks=%s",
                args.domain, ",".join(benchmarks))
    logger.info("  samples_root: %s", samples_root)
    logger.info("  seed_bank   : %s", seed_bank_path)
    logger.info("  knobs       : protocol_patching=%s match_token=%s "
                "llm_repairer=%s llm_hypothesizer=%s model=%s",
                args.enable_protocol_patching, args.match_skill_by_token,
                args.llm_repairer, args.llm_hypothesizer, args.llm_model)
    if args.dry_run:
        for b in benchmarks:
            n = len(list((samples_root / b).glob("sample_*.json")))
            print(f"  {args.domain}/{b}: {n} sample(s)")
        return 0

    rows: List[Dict[str, Any]] = []
    total_samples = total_props = total_bound = 0
    by_kind_total: Counter[str] = Counter()
    by_proposer_total: Counter[str] = Counter()

    for bench in benchmarks:
        bench_dir = samples_root / bench
        if not bench_dir.is_dir():
            logger.warning("skipping missing benchmark dir: %s", bench_dir)
            continue
        logger.info("processing %s/%s", args.domain, bench)
        row = _process_target_benchmark(
            benchmark=bench,
            domain=args.domain,
            samples_dir=bench_dir,
            output_root=output_root,
            seed_bank_path=seed_bank_path,
            max_samples=args.max_samples_per_benchmark,
            sample_id_filter=sample_id_filter,
            max_failures=args.max_failures_per_episode,
            match_skill_by_token=args.match_skill_by_token,
            binding_jaccard_min=args.binding_jaccard_min,
            enable_protocol_patching=args.enable_protocol_patching,
            hot_pattern_threshold=args.hot_pattern_threshold,
            hypothesize_min_recurrences=args.hypothesize_min_recurrences,
            llm_repairer=args.llm_repairer,
            llm_hypothesizer=args.llm_hypothesizer,
            llm_diagnoser=args.llm_diagnoser,
            llm_model=args.llm_model,
        )
        rows.append(row)
        total_samples += row.get("n_samples", 0)
        total_props += row.get("n_proposals", 0)
        total_bound += row.get("n_failures_bound_by_token", 0)
        by_kind_total.update(row.get("by_kind") or {})
        by_proposer_total.update(row.get("by_proposer") or {})
        logger.info("  %s/%s -> %d sample(s), %d proposal(s) (%s)",
                    args.domain, bench,
                    row.get("n_samples", 0),
                    row.get("n_proposals", 0),
                    ", ".join(f"{k}={v}" for k, v in (row.get("by_kind") or {}).items()) or "-")

    (output_root / "_run_meta.json").write_text(json.dumps({
        "domain":        args.domain,
        "samples_root":  str(samples_root),
        "benchmarks":    benchmarks,
        "seed_bank":     str(seed_bank_path) if seed_bank_path else None,
        "max_samples_per_benchmark": args.max_samples_per_benchmark,
        "sample_ids_file": str(args.sample_ids_file) if args.sample_ids_file else None,
        "knobs": {
            "enable_protocol_patching": args.enable_protocol_patching,
            "match_skill_by_token": args.match_skill_by_token,
            "binding_jaccard_min": args.binding_jaccard_min,
            "hot_pattern_threshold": args.hot_pattern_threshold,
            "hypothesize_min_recurrences": args.hypothesize_min_recurrences,
            "llm_repairer": args.llm_repairer,
            "llm_hypothesizer": args.llm_hypothesizer,
            "llm_diagnoser": args.llm_diagnoser,
            "llm_model": args.llm_model,
        },
        "started_at": started_at,
    }, indent=2))

    (output_root / "_run_summary.json").write_text(json.dumps({
        "domain":           args.domain,
        "samples_root":     str(samples_root),
        "n_benchmarks":     len(benchmarks),
        "n_samples":        total_samples,
        "n_proposals":      total_props,
        "n_failures_bound_by_token": total_bound,
        "by_kind":          dict(by_kind_total),
        "by_proposer":      dict(by_proposer_total),
        "per_benchmark":    rows,
        "started_at":       started_at,
        "completed_at":     _utcnow_iso(),
    }, indent=2))

    logger.info("DONE: %d sample(s) across %d benchmark(s), %d proposal(s)",
                total_samples, len(benchmarks), total_props)
    logger.info("  by kind     : %s",
                ", ".join(f"{k}={v}" for k, v in by_kind_total.most_common()) or "-")
    logger.info("  by proposer : %s",
                ", ".join(f"{k}={v}" for k, v in by_proposer_total.most_common()) or "-")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    output_root: Path = (
        args.output_dir.resolve() if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / f"run_{_utc_run_stamp()}").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # Transfer-target dispatch — bypasses the gymv (corpus, source) loop.
    if args.domain != "gymv":
        return _main_target(args, output_root)

    bank_run: Path = args.bank_run.resolve()
    actions_run: Path = args.actions_run.resolve()
    if not bank_run.exists():
        logger.error("bank-run does not exist: %s", bank_run)
        return 2
    if not actions_run.exists():
        logger.error("actions-run does not exist: %s", actions_run)
        return 2

    pairs = _discover_pairs(
        bank_run, actions_run,
        corpus_filter=args.corpus,
        source_filter=args.source,
    )
    if not pairs:
        logger.error("no (corpus, source) pairs discovered")
        return 2

    logger.info("reflect_per_episode: %d pair(s) under bank=%s actions=%s",
                len(pairs), bank_run, actions_run)
    logger.info("  output_root : %s", output_root)
    logger.info("  thresholds  : low_app=%s, max_failures=%s",
                args.low_applicability, args.max_failures_per_episode)
    if args.dry_run:
        for c, s in pairs:
            print(f"  {c} / {s}")
        return 0

    started_at = _utcnow_iso()

    per_pair_summaries: List[Dict[str, Any]] = []
    total_props = 0
    total_subsumes = 0
    total_coalesced = 0
    total_cooldown = 0
    total_eps = 0
    by_kind_total: Counter[str] = Counter()
    by_proposer_total: Counter[str] = Counter()

    for corpus, source in pairs:
        logger.info("processing %s / %s", corpus, source)
        row = _process_source(
            corpus=corpus, source=source,
            bank_run=bank_run, actions_run=actions_run,
            output_root=output_root,
            max_episodes=args.max_episodes,
            low_app=args.low_applicability,
            max_failures=args.max_failures_per_episode,
        )
        per_pair_summaries.append(row)
        total_props += row.get("n_proposals", 0)
        total_subsumes += row.get("n_subsumption_retires", 0)
        total_coalesced += row.get("n_patches_coalesced", 0)
        total_cooldown += row.get("n_patches_skipped_cooldown", 0)
        total_eps += row.get("n_episodes", 0)
        by_kind_total.update(row.get("by_kind") or {})
        by_proposer_total.update(row.get("by_proposer") or {})
        logger.info("  %s/%s -> %d ep(s), %d proposal(s), subsumes=%d (%s)",
                    corpus, source,
                    row.get("n_episodes", 0),
                    row.get("n_proposals", 0),
                    row.get("n_subsumption_retires", 0),
                    ", ".join(f"{k}={v}" for k, v in (row.get("by_kind") or {}).items()) or "-")

    (output_root / "_run_meta.json").write_text(json.dumps({
        "bank_run":      str(bank_run),
        "actions_run":   str(actions_run),
        "output_root":   str(output_root),
        "thresholds": {
            "low_applicability": args.low_applicability,
            "max_failures_per_episode": args.max_failures_per_episode,
        },
        "max_episodes":  args.max_episodes,
        "pairs":         [{"corpus": c, "source": s} for c, s in pairs],
        "started_at":    started_at,
        "argv":          [str(a) for a in (argv or sys.argv)],
    }, indent=2))

    (output_root / "_run_summary.json").write_text(json.dumps({
        "bank_run":                  str(bank_run),
        "actions_run":               str(actions_run),
        "output_root":               str(output_root),
        "n_pairs":                   len(pairs),
        "n_pairs_ok":                sum(1 for r in per_pair_summaries if r.get("status") == "ok"),
        "n_episodes":                total_eps,
        "n_proposals":               total_props,
        "n_subsumption_retires":     total_subsumes,
        "n_patches_coalesced":       total_coalesced,
        "n_patches_skipped_cooldown":total_cooldown,
        "by_kind":                   dict(by_kind_total),
        "by_proposer":               dict(by_proposer_total),
        "per_pair":                  per_pair_summaries,
        "started_at":                started_at,
        "completed_at":              _utcnow_iso(),
    }, indent=2))

    logger.info(
        "DONE: %d pair(s), %d episode(s), %d proposal(s), "
        "%d subsumption-retire(s), %d coalesced, %d cooldown-skipped",
        len(pairs), total_eps, total_props, total_subsumes,
        total_coalesced, total_cooldown,
    )
    logger.info("  by kind     : %s",
                ", ".join(f"{k}={v}" for k, v in by_kind_total.most_common()) or "-")
    logger.info("  by proposer : %s",
                ", ".join(f"{k}={v}" for k, v in by_proposer_total.most_common()) or "-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
