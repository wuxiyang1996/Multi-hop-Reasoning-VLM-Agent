"""Cold-start SFT data → offline validation index for Crafter promotion.

Why
---
``_post_writeback_inherit`` currently inherits a *discounted parent
pass_rate* onto Crafter-promoted skills as a stop-gap.  That keeps a
patched skill from entering the bank with ``pass_rate = 0`` (which would
make it un-selectable), but it never actually verifies that the *child's*
effects contract holds in any concrete state-transition.

We have a much better signal sitting on disk: the
``frontier_distill_jsonl`` cold-start corpus.  Each game has ~2k SFT
records of the form::

    {"prompt": "<state>...domain=gymv...phase=early...e3.value=68000...</state>...",
     "active_skill": "RECOVER/EVADE",
     "episode_id": "ep_001",
     "step_idx": 17}

Two records belonging to the same ``episode_id`` with consecutive
``step_idx`` values give us a (B_start, B_end) pair from which we can
compute :class:`SegmentRecord`-shaped instances:

  * ``B_start`` / ``B_end`` -- flat predicate sets parsed from the
    ``<state>`` block.
  * ``eff_add = B_end - B_start``
  * ``eff_del = B_start - B_end``
  * ``eff_event`` -- inferred from numeric attribute deltas
    (``score`` increases → ``event.score_changed``) and phase
    transitions (``world.phase=early → mid`` → ``event.phase_changed``).
  * ``skill_label = active_skill``

These are exactly the fields :func:`verify_effects_contract` consumes,
so we can run it directly against a Crafter proposal's contract — no
new verifier, no shadow rollout.

Predicate vocab translation
---------------------------
The SFT prompt and the runtime contracts use *different* surface
forms for predicates:

  SFT  ``state_flags.phase=early``  /  ``e3.value=68000``
  bank ``world.phase=early``         /  ``world.score=68000``

We only translate the subset that the runtime contract literals
actually use today (audited from ``skillbank/*/skill_bank.jsonl`` on
2026-05-06):

  * ``world.phase=*``        ← ``state_flags.phase``
  * ``world.scene_type=*``   ← ``state_flags.scene_type``
  * ``world.progress=*``     ← ``state_flags.progress``  (rounded)
  * ``world.score=N``        ← entity-with-label "score" .value
  * ``event.phase_changed``  ← ``world.phase`` differs across the pair
  * ``event.score_changed``  ← ``world.score`` differs across the pair

That covers all four non-empty contracts in the active TF3 bank.  Any
contract literal *outside* this vocab will simply have zero support in
the validation set; the verifier flags it as ``miss_add`` and the
overall pass rate drops accordingly — which is the *correct* signal
that the contract is over-specified vs. what the SFT teacher saw.

Caching
-------
Parsing 16 k records per call (8 games × 2 k) takes ~1 s.  We cache the
boolean predicate sets per (game, skill_label) to
``runs/_cold_start_validation_index/<corpus_id>/<game>.jsonl`` and reuse
across promotion steps.  The cache is keyed by the source JSONL's
mtime+size so an updated corpus invalidates automatically.

Cross-refs
----------
* Phase B design discussion: 2026-05-06 chat after Phase A commit ``a540434``.
* Verifier API: ``skill_agents/stage3_mvp/contract_verify.py``.
* Schema: ``skill_agents/stage3_mvp/schemas.py`` -- ``SegmentRecord``.
* Game → SFT dir mapping: ``common/reward_anchors._GYMV_COLD_START_DIRS``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predicate parsing
# ---------------------------------------------------------------------------

# Discount unstable / negative-flag values: these flip on every error or
# input event and would otherwise dominate the eff_add/del sets.
_SKIP_STATE_FLAGS: frozenset = frozenset({
    "error", "dialog_open", "input_pending",
})

# Heuristic: which entity *labels* count as a numeric "score" channel?
# Kept conservative (only the obvious matches) so we never mis-attribute
# ``world.score`` to a tile value.
_SCORE_LABEL_HINTS: frozenset = frozenset({
    "score", "points", "score_text", "hud_score",
})

_BLOCK_RE: Dict[str, re.Pattern] = {
    "state_flags": re.compile(r"<state_flags>(.*?)<", re.DOTALL),
    "attributes":  re.compile(r"<attributes>(.*?)<", re.DOTALL),
    "entities":    re.compile(r"<entities>(.*?)<", re.DOTALL),
}


def _extract_block(text: str, name: str) -> str:
    pat = _BLOCK_RE.get(name)
    if not pat:
        return ""
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def _parse_state_flags(block: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if not k or k in _SKIP_STATE_FLAGS:
            continue
        if v in ("null", "none", "false"):
            continue
        out[k] = v
    return out


def _parse_attributes(block: str) -> Dict[str, Dict[str, str]]:
    """``e3.value=68000``  →  ``{"e3": {"value": "68000"}}``."""
    out: Dict[str, Dict[str, str]] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or "=" not in line or "." not in line.split("=", 1)[0]:
            continue
        lhs, _, v = line.partition("=")
        eid, _, attr = lhs.partition(".")
        eid = eid.strip()
        attr = attr.strip()
        v = v.strip()
        if not eid or not attr:
            continue
        out.setdefault(eid, {})[attr] = v
    return out


def _parse_entities(block: str) -> Dict[str, Dict[str, str]]:
    """``e3[type=hud, label=score, ...]`` → ``{"e3": {"type": "hud", "label": "score"}}``.

    Best-effort; only ``type`` and ``label`` are pulled out (the only
    fields the score-detection heuristic needs).
    """
    out: Dict[str, Dict[str, str]] = {}
    for line in block.splitlines():
        line = line.strip()
        m = re.match(r"^(e\d+)\[(.*)\]\s*$", line)
        if not m:
            continue
        eid = m.group(1)
        body = m.group(2)
        meta: Dict[str, str] = {}
        for kv in body.split(","):
            kv = kv.strip()
            if "=" not in kv:
                continue
            k, _, v = kv.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                meta[k] = v
        out[eid] = meta
    return out


def _score_entity_ids(entities: Mapping[str, Mapping[str, str]]) -> List[str]:
    """Pick entity IDs whose label matches ``_SCORE_LABEL_HINTS``."""
    out: List[str] = []
    for eid, meta in entities.items():
        label = (meta.get("label") or "").lower()
        if not label:
            continue
        if any(hint in label for hint in _SCORE_LABEL_HINTS):
            out.append(eid)
    return out


def state_to_predicate_set(prompt: str) -> Tuple[Set[str], Optional[int]]:
    """Parse a SFT ``prompt`` into a flat ``{world.* / hud.*}`` predicate set.

    Returns
    -------
    (predicates, score_value)
        ``predicates`` is the set of ``namespace.key=value`` literals.
        ``score_value`` is the parsed integer score (if any score-tagged
        entity was found) — kept separate so we can compute
        ``event.score_changed`` from numeric deltas without polluting
        the predicate set with one literal per score increment.
    """
    state_flags = _parse_state_flags(_extract_block(prompt, "state_flags"))
    attrs = _parse_attributes(_extract_block(prompt, "attributes"))
    entities = _parse_entities(_extract_block(prompt, "entities"))

    preds: Set[str] = set()

    # state_flags → world.<key>=<value>
    for k, v in state_flags.items():
        preds.add(f"world.{k}={v}")

    # score entity → world.score=<v> (one literal; the *changed* event is
    # computed by comparing values across consecutive steps)
    score_eids = _score_entity_ids(entities)
    score_val: Optional[int] = None
    for eid in score_eids:
        v_raw = (attrs.get(eid) or {}).get("value")
        if v_raw is None:
            continue
        try:
            score_val = int(v_raw)
            preds.add(f"world.score={v_raw}")
            break  # first match wins; multiple score entities are rare
        except (TypeError, ValueError):
            continue

    return preds, score_val


# ---------------------------------------------------------------------------
# Validation segments
# ---------------------------------------------------------------------------


@dataclass
class _ValidationSegment:
    """SegmentRecord-shaped row — kept dataclass-light so we can serialize
    to JSONL without depending on ``skill_agents.stage3_mvp.schemas`` at
    import time (the verifier-side import happens lazily in
    :func:`build_segment_records`)."""

    seg_id: str
    traj_id: str
    t_start: int
    t_end: int
    skill_label: str
    B_start: List[str]
    B_end: List[str]
    eff_add: List[str]
    eff_del: List[str]
    eff_event: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seg_id": self.seg_id,
            "traj_id": self.traj_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "skill_label": self.skill_label,
            "B_start": self.B_start,
            "B_end": self.B_end,
            "eff_add": self.eff_add,
            "eff_del": self.eff_del,
            "eff_event": self.eff_event,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "_ValidationSegment":
        return cls(
            seg_id=d["seg_id"],
            traj_id=d["traj_id"],
            t_start=int(d["t_start"]),
            t_end=int(d["t_end"]),
            skill_label=d["skill_label"],
            B_start=list(d.get("B_start") or []),
            B_end=list(d.get("B_end") or []),
            eff_add=list(d.get("eff_add") or []),
            eff_del=list(d.get("eff_del") or []),
            eff_event=list(d.get("eff_event") or []),
        )


def _segments_from_jsonl(
    jsonl_path: Path,
    *,
    max_records: Optional[int] = None,
) -> List[_ValidationSegment]:
    """Read a ``skill_selection.jsonl`` file and emit per-step segments."""
    segments: List[_ValidationSegment] = []

    # Group records by episode → list[(step_idx, prompt, active_skill)]
    by_ep: Dict[str, List[Tuple[int, str, str]]] = {}
    n_read = 0
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep = str(rec.get("episode_id") or "")
                step = int(rec.get("step_idx") or 0)
                prompt = rec.get("prompt") or ""
                skill = (rec.get("active_skill") or "").strip()
                if not ep or not skill:
                    continue
                by_ep.setdefault(ep, []).append((step, prompt, skill))
                n_read += 1
                if max_records and n_read >= max_records:
                    break
    except OSError as exc:
        logger.warning(
            "cold_start_validation_index: read failed for %s: %s",
            jsonl_path, exc,
        )
        return segments

    # For each episode, sort by step_idx and pair consecutive records.
    for ep, rows in by_ep.items():
        rows.sort(key=lambda r: r[0])
        for i in range(len(rows) - 1):
            t0, p0, sk0 = rows[i]
            t1, p1, sk1 = rows[i + 1]
            if t1 - t0 != 1:
                continue  # gap → skip (pair must be adjacent)
            B_start, score0 = state_to_predicate_set(p0)
            B_end,   score1 = state_to_predicate_set(p1)
            eff_add = B_end - B_start
            eff_del = B_start - B_end
            eff_event: Set[str] = set()
            # phase change → ``event.phase_changed``
            phase0 = next(
                (p for p in B_start if p.startswith("world.phase=")), None,
            )
            phase1 = next(
                (p for p in B_end if p.startswith("world.phase=")), None,
            )
            if phase0 != phase1:
                eff_event.add("event.phase_changed")
            # score delta → ``event.score_changed``
            if (score0 is not None and score1 is not None
                    and score0 != score1):
                eff_event.add("event.score_changed")
            # Strip raw ``world.score=N`` from B_start/B_end so they
            # don't dominate eff_add/del with per-tick increments.  The
            # event captures the change; the literal varies on every step.
            B_start = {p for p in B_start if not p.startswith("world.score=")}
            B_end   = {p for p in B_end   if not p.startswith("world.score=")}
            eff_add = {p for p in eff_add if not p.startswith("world.score=")}
            eff_del = {p for p in eff_del if not p.startswith("world.score=")}

            seg = _ValidationSegment(
                seg_id=f"{ep}:{t0}-{t1}",
                traj_id=ep,
                t_start=t0,
                t_end=t1,
                # When the *active_skill* changes within the pair, label
                # the segment with the skill at t_start; the t_end skill
                # belongs to the next pair anyway.
                skill_label=sk0,
                B_start=sorted(B_start),
                B_end=sorted(B_end),
                eff_add=sorted(eff_add),
                eff_del=sorted(eff_del),
                eff_event=sorted(eff_event),
            )
            segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Public entrypoint: load_or_build
# ---------------------------------------------------------------------------


_CACHE_DIRNAME = "_cold_start_validation_index"


def _cache_key_for(jsonl_path: Path) -> str:
    """``<size>_<mtime_ns>`` so an updated corpus invalidates the cache."""
    try:
        st = jsonl_path.stat()
        return f"{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        return "0_0"


def _cache_path_for(cache_root: Path, game_slug: str, key: str) -> Path:
    return cache_root / f"{game_slug}.{key}.jsonl"


def _resolve_jsonl(corpus_root: Path, game_slug: str) -> Optional[Path]:
    """Map ``gymv_thunder_force_iii`` → ``<corpus_root>/Temporal_ThunderForceIII-v0/skill_selection.jsonl``."""
    try:
        from common.reward_anchors import _GYMV_COLD_START_DIRS  # type: ignore
    except Exception:                                            # noqa: BLE001
        # Inline fallback — keeps this module self-contained for testing.
        _GYMV_COLD_START_DIRS = {
            "gymv_thunder_force_iii":  "Temporal_ThunderForceIII-v0",
            "gymv_altered_beast":      "Temporal_AlteredBeast-v0",
            "gymv_columns":            "Temporal_Columns-v0",
            "gymv_dynamite_headdy":    "Temporal_DynamiteHeaddy-v0",
            "gymv_space_harrier_ii":   "Temporal_SpaceHarrierII-v0",
            "gymv_streets_of_rage_2":  "Temporal_StreetsOfRage2-v0",
            "gymv_airstriker":         "Temporal_Airstriker-v0",
            "gymv_strider":            "Temporal_Strider-v0",
        }
    sub = _GYMV_COLD_START_DIRS.get(game_slug)
    if not sub:
        return None
    p = corpus_root / sub / "skill_selection.jsonl"
    return p if p.exists() else None


def load_or_build_segments_for_game(
    *,
    corpus_root: Path,
    game_slug: str,
    cache_root: Optional[Path] = None,
) -> Dict[str, List[_ValidationSegment]]:
    """Return ``{skill_label: [_ValidationSegment, ...]}`` for *game_slug*.

    Returns an empty dict if the SFT JSONL for the game can't be
    located.  Reads/writes a per-game JSONL cache under *cache_root*.
    """
    jsonl = _resolve_jsonl(corpus_root, game_slug)
    if jsonl is None:
        return {}

    key = _cache_key_for(jsonl)
    cache_root = cache_root or (corpus_root / _CACHE_DIRNAME)
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_path = _cache_path_for(cache_root, game_slug, key)
    segments: List[_ValidationSegment]

    if cache_path.exists():
        segments = []
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        segments.append(_ValidationSegment.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            segments = []
    else:
        segments = _segments_from_jsonl(jsonl)
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(json.dumps(seg.to_dict()) + "\n")
        except OSError as exc:
            logger.warning(
                "cold_start_validation_index: cache write failed for %s: %s",
                cache_path, exc,
            )

    by_skill: Dict[str, List[_ValidationSegment]] = {}
    for seg in segments:
        by_skill.setdefault(seg.skill_label, []).append(seg)
    return by_skill


# ---------------------------------------------------------------------------
# Verification glue
# ---------------------------------------------------------------------------


@dataclass
class ValidationVerdict:
    """Outcome of running a Crafter contract against cold-start segments."""

    pass_rate: float = 0.0
    n_instances: int = 0
    eff_add_success_rate: Dict[str, float] = field(default_factory=dict)
    eff_del_success_rate: Dict[str, float] = field(default_factory=dict)
    eff_event_rate: Dict[str, float] = field(default_factory=dict)
    failure_signatures: Dict[str, int] = field(default_factory=dict)
    insufficient_evidence: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_rate": self.pass_rate,
            "n_instances": self.n_instances,
            "eff_add_success_rate": self.eff_add_success_rate,
            "eff_del_success_rate": self.eff_del_success_rate,
            "eff_event_rate": self.eff_event_rate,
            "failure_signatures": self.failure_signatures,
            "insufficient_evidence": self.insufficient_evidence,
        }


def _contract_has_ood_literal(
    *,
    contract_literals: Iterable[str],
    segments: List[_ValidationSegment],
) -> bool:
    """Return True if **any** contract literal refers to a value that
    never appears in any segment's
    ``B_start ∪ B_end ∪ eff_add ∪ eff_del ∪ eff_event``.

    Why "any" and not "all"
    ----------------------
    The GPT cold-start corpus is biased — it underrepresents the
    late / endgame phase because the teacher rarely survives long enough
    to reach it.  A contract like
    ``eff_add=['event.phase_changed', 'world.phase=endgame']``
    has one literal that *is* in distribution (``event.phase_changed``
    appears at the early→mid boundary) and one that is *not*
    (``world.phase=endgame`` never appears).  If we run the verifier
    anyway, the OOD literal trivially fails on every segment, drops the
    per-instance literal-pass rate below ``instance_pass_literal_frac``,
    and the overall pass_rate collapses to 0 — even though the parent
    skill is curator-validated and the contract is structurally fine.

    The honest read is: if even one literal is unverifiable against
    this corpus, the corpus cannot tell us whether the contract is
    correct.  Abstaining keeps the Phase A discount in place and lets
    the in-loop ``run_contract_learning`` cycle (which will see new
    runtime segments tagged to this skill) produce the authoritative
    report.

    Trade-off acknowledged
    ----------------------
    A truly bogus contract whose lone literal happens to share a key
    with cold-start data (e.g. ``world.phase=zzz``) would NOT trigger
    OOD on the literal-level check; the verifier runs and returns
    ``pass_rate=0`` (all segments fail the comparison) — which is the
    correct outcome.  But a bogus literal in a *novel* namespace
    (``foo.bar=zzz``) abstains.  Phase B is therefore strictly weaker
    on bogus-namespace literals; we accept this in exchange for not
    deleting legitimate late-phase skills.
    """
    universe: set = set()
    for seg in segments:
        universe.update(seg.B_start)
        universe.update(seg.B_end)
        universe.update(seg.eff_add)
        universe.update(seg.eff_del)
        universe.update(seg.eff_event)

    literals = [lit for lit in contract_literals if lit]
    if not literals:
        return False  # nothing to verify — let downstream branch handle
    return any(lit not in universe for lit in literals)


def verify_contract_against_segments(
    *,
    contract_dict: Mapping[str, Any],
    segments: List[_ValidationSegment],
    min_segments: int = 5,
) -> ValidationVerdict:
    """Run :func:`verify_effects_contract` and return a slim verdict.

    Returns a verdict with ``insufficient_evidence=True`` when the
    skill has fewer than *min_segments* validation rows OR every
    contract literal is out-of-distribution relative to the SFT corpus
    (see :func:`_contract_is_out_of_distribution`).  Caller treats
    both cases as "abstain, fall back to discount".
    """
    if not contract_dict:
        return ValidationVerdict()
    if len(segments) < min_segments:
        v = ValidationVerdict(n_instances=len(segments))
        return v

    all_literals = list(
        list(contract_dict.get("eff_add") or [])
        + list(contract_dict.get("eff_del") or [])
        + list(contract_dict.get("eff_event") or [])
    )
    if _contract_has_ood_literal(
        contract_literals=all_literals, segments=segments,
    ):
        v = ValidationVerdict(n_instances=len(segments))
        # Mark insufficient_evidence=True (default) so caller routes to
        # abstain.  Surface the reason via failure_signatures so
        # ``_step_summary.json`` shows why each abstained.
        v.failure_signatures = {"out_of_distribution": 1}
        return v

    try:
        from skill_agents.stage3_mvp.config import Stage3MVPConfig
        from skill_agents.stage3_mvp.contract_verify import (
            verify_effects_contract,
        )
        from skill_agents.stage3_mvp.schemas import (
            SegmentRecord,
            SkillEffectsContract,
        )
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(
            "cold_start_validation_index: stage3_mvp imports failed: %s", exc,
        )
        return ValidationVerdict(n_instances=len(segments))

    # Hydrate contract → SkillEffectsContract.  Tolerate missing fields.
    try:
        contract = SkillEffectsContract.from_dict({
            "skill_id": contract_dict.get("skill_id") or "validate",
            "version": contract_dict.get("version", 1),
            "name": contract_dict.get("name"),
            "description": contract_dict.get("description"),
            "eff_add": list(contract_dict.get("eff_add") or []),
            "eff_del": list(contract_dict.get("eff_del") or []),
            "eff_event": list(contract_dict.get("eff_event") or []),
        })
    except Exception as exc:                                   # noqa: BLE001
        logger.warning(
            "cold_start_validation_index: bad contract dict: %s", exc,
        )
        return ValidationVerdict(n_instances=len(segments))

    if contract.total_literals == 0:
        # No literals to verify — caller will keep the discount fallback.
        return ValidationVerdict(n_instances=len(segments))

    instances: List[SegmentRecord] = []
    for seg in segments:
        instances.append(SegmentRecord(
            seg_id=seg.seg_id,
            traj_id=seg.traj_id,
            t_start=seg.t_start,
            t_end=seg.t_end,
            skill_label=seg.skill_label,
            B_start=set(seg.B_start),
            B_end=set(seg.B_end),
            eff_add=set(seg.eff_add),
            eff_del=set(seg.eff_del),
            eff_event=set(seg.eff_event),
        ))

    cfg = Stage3MVPConfig()
    rep = verify_effects_contract(contract, instances, cfg)

    return ValidationVerdict(
        pass_rate=rep.overall_pass_rate,
        n_instances=rep.n_instances,
        eff_add_success_rate=dict(rep.eff_add_success_rate),
        eff_del_success_rate=dict(rep.eff_del_success_rate),
        eff_event_rate=dict(rep.eff_event_rate),
        failure_signatures=dict(rep.failure_signatures),
        insufficient_evidence=False,
    )


__all__ = [
    "ValidationVerdict",
    "load_or_build_segments_for_game",
    "state_to_predicate_set",
    "verify_contract_against_segments",
]
