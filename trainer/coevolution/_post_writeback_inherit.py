"""Post-writeback evidence-inheritance sweep for Crafter-promoted skills.

Hook contract
-------------

After ``_promotion_hook`` calls :func:`writeback_promotion` to project the
gate's promoted skills back into the trainer's per-game
``skill_bank.jsonl``, the new on-disk skill records are *structurally
correct* (skill_id / name / contract / protocol) but *evidentially empty*:

    * ``sub_episodes = []``
    * ``strategic_description = ""``
    * ``execution_hint = null``
    * ``n_instances = 0``
    * ``report = null``  (no VerificationReport)

Empirically (audit 2026-05-06) this is the dominant failure mode for the
Crafter loop:

    SkillBankSelector._build_index pulls relevance tokens from
    ``contract.eff_*`` and ``skill.{name, strategic_description}``.  An
    empty strategic_description halves the token surface; an empty
    sub_episodes list means there's no evidence ledger for the next
    ``run_contract_learning`` cycle to pick up.  And
    ``_compute_confidence`` falls back to ``pass_rate=0.5`` when no
    report exists — six points below typical curator-trained skills.
    Result: Crafter skills enter the bank technically, but *never* get
    selected by the actor.

This sweep closes that gap by inheriting evidence from the proposal's
parent (for PATCH / TRANSFER / COMPOSE / GENERALIZE — anything with a
non-empty ``parent_skill_ids``).  The inheritance is **discounted** so
the new skill enters with non-trivial credibility but does not claim
the parent's full track record:

    * sub_episodes  → up to ``MAX_INHERITED_SUB_EPISODES`` from parent
    * strategic_description → parent's verbatim
    * execution_hint → parent's verbatim
    * n_instances → ``max(1, parent.n_instances // INHERIT_DISCOUNT)``
    * report.overall_pass_rate → ``parent.report.overall_pass_rate
                                  * INHERIT_PASS_RATE_FACTOR``

For HYPOTHESIS proposals (no ``parent_skill_ids``), no inheritance is
done.  The hypothesis enters the bank with empty fields and must rely
on UCB exploration to earn its first selection — the existing path,
just without the field-inheritance escape hatch.

Phase B: cold-start validation gate (T2.18, 2026-05-06)
-------------------------------------------------------

Phase A's discount is a soft trial — it *credits* the patched skill
with parent evidence but never tests whether the patched contract
holds.  Phase B closes that loop by running
:func:`verify_effects_contract` against segments derived from the
SFT cold-start corpus (``frontier_distill_jsonl``):

    * For each PATCH-shaped Crafter skill that has a non-empty
      ``contract.eff_*``, look up the parent's name in the per-game
      validation index.
    * Run the verifier against those segments.
    * If pass_rate ≥ ``crafter_validation_pass_thresh`` (PATCH=0.6 by
      default), keep the skill and **replace** the discounted parent
      report with the *real* per-segment report.
    * If pass_rate < threshold, the skill is **removed** from the
      bank — the patched contract demonstrably fails on teacher data,
      so it has no business influencing the actor's top-K.
    * If fewer than ``crafter_validation_min_segments`` SFT rows
      exist for the parent skill, abstain → keep Phase A's discount.

Module ``_cold_start_validation_index`` owns the parsing /
predicate-vocab translation (SFT ``state_flags.phase=early`` →
``world.phase=early``) so this module stays focused on bank rewrite
mechanics.

Atomicity
---------

The sweep rewrites ``skill_bank.jsonl`` atomically (write-then-rename)
so a crash mid-sweep leaves the original file intact.  Existing
records that aren't in the inserted set are passed through verbatim.

Cross-refs
----------
* Audit summary: ``runs/_launch_logs/2026-05-06_crafter_audit.md``
  (empirical: 195/195 hypothesis skills with zero evidence;
  0/0 patch skills ever in bank across 31 banks).
* Selector logic: ``skill_agents/query.py`` ``_compute_relevance`` and
  ``_compute_confidence``.
* Writeback projector: ``skill_bank/legacy_writeback.py``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from trainer.coevolution._cold_start_validation_index import (
    ValidationVerdict,
    load_or_build_segments_for_game,
    verify_contract_against_segments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# How many of the parent's sub_episodes we copy onto a child skill.
# Capped to avoid massive duplication when the parent has hundreds of
# instances (e.g. ``early:SETUP`` with 194 instances).  Ten is enough
# to seed the next contract_learning cycle's relevance index without
# exploding the bank file size.
MAX_INHERITED_SUB_EPISODES: int = 10

# n_instances inheritance discount.  ``new_n = max(1, parent_n // K)``.
# K=4 gives a 200-instance parent a 50-instance child — credible enough
# to compete in top-K but well below "fully validated".  K=4 chosen so
# that single-instance parents (n=1) still produce a non-zero child.
INHERIT_DISCOUNT: int = 4

# pass_rate inheritance factor.  child.pass_rate = parent.pass_rate * F.
# F=0.7 means a 0.85 parent passes 0.595 to the child — competitive but
# not rubber-stamping.  We don't go all the way to 1.0 because the
# patched contract has not actually been verified yet against the
# inherited evidence.
INHERIT_PASS_RATE_FACTOR: float = 0.7

# Source-types we treat as "Crafter-origin" for inheritance.  Anything
# else is left alone (curator-evolved skills already carry their own
# evidence; any mistake we make here would corrupt their reports).
CRAFTER_SOURCE_TYPES: frozenset = frozenset({
    "REPAIRED",          # PatchProposal subject
    "TEACHER",           # HypothesisProposal subject (rarely useful — no parent)
    "CRAFTED",           # ComposeProposal / generic Crafter
    "FEW_SHOT_ADAPTED",  # TransferProposal / GeneralizeProposal subject
})


# Phase B (T2.18) — cold-start validation thresholds.  PATCH starts at
# 0.6 because the parent skill already passed Stage 3 once and the patch
# is supposed to repair a known failure mode; HYPOTHESIS is set lower
# (0.4) because it's a *novel* skill that has no prior verification and
# we want to give it a chance to earn evidence on the first run-loop.
DEFAULT_VALIDATION_PASS_THRESH_PATCH: float = 0.6
DEFAULT_VALIDATION_PASS_THRESH_HYPOTHESIS: float = 0.4
DEFAULT_VALIDATION_PASS_THRESH_OTHER: float = 0.5

# Minimum number of SFT segments required before we run the verifier.
# Below this we abstain and fall through to Phase A inheritance, on the
# theory that 1-4 segments are too few to distinguish "contract is bad"
# from "contract happens not to fire on this slice".
DEFAULT_VALIDATION_MIN_SEGMENTS: int = 5


# ---------------------------------------------------------------------------
# Public report
# ---------------------------------------------------------------------------


@dataclass
class InheritReport:
    """Per-game stats produced by one ``inherit_evidence_for_inserted``
    sweep.  Mirrors :class:`WritebackReport`'s style so dashboards can
    surface both side by side."""

    bank_path: Path
    n_inserted_skills_examined: int = 0
    n_inherited: int = 0
    n_no_parent: int = 0
    n_parent_missing: int = 0
    n_already_filled: int = 0
    n_curator_skipped: int = 0
    inherited_skill_ids: List[str] = field(default_factory=list)

    # Phase B — cold-start validation outcomes.
    n_validation_attempted: int = 0
    n_validation_passed: int = 0
    n_validation_rejected: int = 0
    n_validation_abstained: int = 0  # below min_segments or no contract
    rejected_skill_ids: List[str] = field(default_factory=list)
    validation_per_skill: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bank_path": str(self.bank_path),
            "n_inserted_skills_examined": self.n_inserted_skills_examined,
            "n_inherited": self.n_inherited,
            "n_no_parent": self.n_no_parent,
            "n_parent_missing": self.n_parent_missing,
            "n_already_filled": self.n_already_filled,
            "n_curator_skipped": self.n_curator_skipped,
            "inherited_skill_ids": list(self.inherited_skill_ids),
            "n_validation_attempted": self.n_validation_attempted,
            "n_validation_passed": self.n_validation_passed,
            "n_validation_rejected": self.n_validation_rejected,
            "n_validation_abstained": self.n_validation_abstained,
            "rejected_skill_ids": list(self.rejected_skill_ids),
            "validation_per_skill": dict(self.validation_per_skill),
        }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def inherit_evidence_for_inserted(
    *,
    bank_path: Path,
    inserted_skill_ids: Sequence[str],
    max_inherited_sub_episodes: int = MAX_INHERITED_SUB_EPISODES,
    inherit_discount: int = INHERIT_DISCOUNT,
    pass_rate_factor: float = INHERIT_PASS_RATE_FACTOR,
    validation_segments_by_skill: Optional[Mapping[str, Any]] = None,
    validation_pass_thresh_by_kind: Optional[Mapping[str, float]] = None,
    validation_min_segments: int = DEFAULT_VALIDATION_MIN_SEGMENTS,
) -> InheritReport:
    """Inherit evidence from each inserted skill's parent in *bank_path*.

    Parameters
    ----------
    bank_path
        Per-game ``skill_bank.jsonl`` to sweep (one file per call).
    inserted_skill_ids
        Skill IDs that ``writeback_promotion`` just INSERTed.  Existing
        skills (UPDATEs) are left alone — their evidence is canonical.
    max_inherited_sub_episodes
        Cap on the number of sub_episode entries copied from parent.
    inherit_discount
        ``n_instances`` divisor.  child = max(1, parent // K).
    pass_rate_factor
        Multiplier on parent's ``overall_pass_rate``.
    validation_segments_by_skill
        Phase B — ``{skill_label: [_ValidationSegment, ...]}`` keyed by
        the *teacher* skill name (e.g. ``"RECOVER/EVADE"``).  When
        supplied, every Crafter-origin skill with a non-trivial
        ``contract.eff_*`` is verified against the segments tagged
        with its parent's name (or its own name for HYPOTHESIS).  See
        module docstring "Phase B" section.
    validation_pass_thresh_by_kind
        Per-source-type threshold dict, e.g.
        ``{"REPAIRED": 0.6, "TEACHER": 0.4}``.  Defaults to
        ``DEFAULT_VALIDATION_PASS_THRESH_*``.
    validation_min_segments
        Below this we abstain from verification (insufficient evidence).

    Returns
    -------
    InheritReport with per-skill counts.  When the bank has no inserted
    skills (or the file doesn't exist), returns an empty report and
    skips the rewrite.
    """
    report = InheritReport(bank_path=bank_path)

    if not inserted_skill_ids:
        return report
    if not bank_path.exists():
        logger.warning(
            "post_writeback_inherit: bank_path missing: %s", bank_path,
        )
        return report

    inserted_set = set(inserted_skill_ids)

    # Resolve thresholds — caller may override per-kind, otherwise we
    # use the module-level defaults (PATCH=0.6, HYPOTHESIS=0.4, …).
    thresh_by_kind: Dict[str, float] = {
        "REPAIRED": DEFAULT_VALIDATION_PASS_THRESH_PATCH,
        "TEACHER": DEFAULT_VALIDATION_PASS_THRESH_HYPOTHESIS,
        "CRAFTED": DEFAULT_VALIDATION_PASS_THRESH_OTHER,
        "FEW_SHOT_ADAPTED": DEFAULT_VALIDATION_PASS_THRESH_OTHER,
    }
    if validation_pass_thresh_by_kind:
        for k, v in validation_pass_thresh_by_kind.items():
            thresh_by_kind[k.upper()] = float(v)

    # Read the entire bank file.  At ~30 skills/game (steady state) and
    # ~2 KB/skill the file is well under 100 KB; full read is fine.
    rows: List[Dict[str, Any]] = []
    try:
        with bank_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "post_writeback_inherit: skipping bad line %d "
                        "in %s: %s", line_no, bank_path, exc,
                    )
    except OSError as exc:
        logger.warning(
            "post_writeback_inherit: read failed for %s: %s",
            bank_path, exc,
        )
        return report

    # Build skill_id → row index for parent lookup.
    by_id: Dict[str, int] = {}
    for i, row in enumerate(rows):
        sk = row.get("skill", row)
        sid = sk.get("skill_id")
        if sid:
            by_id[sid] = i

    any_modified = False

    # Track row indices to drop after Phase B validation.  We collect
    # indices instead of mutating during iteration so the rewrite step
    # can build the surviving list cleanly.
    rows_to_drop: set = set()

    for i, row in enumerate(rows):
        sk = row.get("skill", row)
        sid = sk.get("skill_id")
        if sid not in inserted_set:
            continue
        report.n_inserted_skills_examined += 1

        # Curator-evolved skills with non-empty source_type that *isn't*
        # in CRAFTER_SOURCE_TYPES are passed through; we don't want to
        # mutate skills that already carry their own evidence.  Empty /
        # missing source_type is treated as "unclear" → still process,
        # but only if the skill has parent_skill_ids (which is the real
        # crafter signal anyway).
        src = (sk.get("source_type") or "").upper()
        parents = sk.get("parent_skill_ids") or []

        if src and src not in CRAFTER_SOURCE_TYPES and not parents:
            report.n_curator_skipped += 1
            continue

        # ── Phase A inheritance (parent-driven) ──────────────────────
        # HYPOTHESIS proposals (no parent) skip inheritance entirely
        # but are still eligible for Phase B validation below.
        base: Optional[Dict[str, Any]] = None
        base_report: Dict[str, Any] = {}
        if parents:
            base_id = parents[0]
            base_idx = by_id.get(base_id)
            if base_idx is None:
                report.n_parent_missing += 1
            else:
                base_row = rows[base_idx]
                base = base_row.get("skill", base_row)
                base_report = base_row.get("report") or {}
        else:
            report.n_no_parent += 1

        # Idempotency: if the skill already has all the evidence fields
        # filled, skip — this lets the sweep run safely on re-promotion.
        already_filled = (
            sk.get("sub_episodes")
            and sk.get("strategic_description")
            and sk.get("n_instances", 0) > 0
            and (row.get("report") or {}).get("overall_pass_rate") is not None
        )
        if already_filled:
            report.n_already_filled += 1
        elif base is not None:
            if not sk.get("sub_episodes"):
                sk["sub_episodes"] = list(base.get("sub_episodes") or [])[
                    :max_inherited_sub_episodes
                ]
            if not sk.get("strategic_description"):
                sk["strategic_description"] = base.get("strategic_description") or ""
            if not sk.get("execution_hint"):
                sk["execution_hint"] = base.get("execution_hint")
            if not sk.get("expected_tag_pattern"):
                sk["expected_tag_pattern"] = list(
                    base.get("expected_tag_pattern") or [],
                )

            # n_instances — discounted
            cur_n = int(sk.get("n_instances") or 0)
            if cur_n == 0:
                base_n = int(base.get("n_instances") or 0)
                sk["n_instances"] = max(1, base_n // max(1, inherit_discount)) if base_n else 0
                # Mirror onto the contract sub-object — _build_index reads
                # ``contract.n_instances`` independently of the top-level.
                c = sk.get("contract")
                if isinstance(c, dict) and not c.get("n_instances"):
                    c["n_instances"] = sk["n_instances"]

            # report — discounted pass_rate (Phase B may overwrite this)
            if row.get("report") is None and base_report:
                new_report = dict(base_report)
                base_pr = float(base_report.get("overall_pass_rate") or 0.0)
                new_report["overall_pass_rate"] = base_pr * float(pass_rate_factor)
                new_report["skill_id"] = sid
                # Reset per-segment evidence so the next finalize_update
                # cycle doesn't double-count parent's worst_segments as
                # this skill's worst — they're not the patched skill's
                # actual failures.
                new_report["worst_segments"] = []
                new_report["failure_signatures"] = {}
                new_report["n_instances"] = sk["n_instances"]
                row["report"] = new_report

            report.n_inherited += 1
            report.inherited_skill_ids.append(sid)
            any_modified = True

        # ── Phase B: cold-start validation ───────────────────────────
        if validation_segments_by_skill is not None:
            verdict = _maybe_validate_skill(
                sk=sk,
                base=base,
                segments_by_skill=validation_segments_by_skill,
                min_segments=validation_min_segments,
            )
            if verdict is not None:
                report.validation_per_skill[sid] = verdict.to_dict()
                if verdict.insufficient_evidence:
                    report.n_validation_abstained += 1
                else:
                    report.n_validation_attempted += 1
                    thresh = thresh_by_kind.get(
                        src, DEFAULT_VALIDATION_PASS_THRESH_OTHER,
                    )
                    if verdict.pass_rate >= thresh:
                        report.n_validation_passed += 1
                        # Replace the inherited (discounted) report with
                        # the *real* per-segment one.  The actor's
                        # confidence calculator will now see a verified
                        # pass_rate instead of a rubber-stamped 0.7×.
                        new_report = dict(row.get("report") or {})
                        new_report["skill_id"] = sid
                        new_report["overall_pass_rate"] = verdict.pass_rate
                        new_report["n_instances"] = verdict.n_instances
                        new_report["eff_add_success_rate"] = verdict.eff_add_success_rate
                        new_report["eff_del_success_rate"] = verdict.eff_del_success_rate
                        new_report["eff_event_rate"] = verdict.eff_event_rate
                        new_report["failure_signatures"] = verdict.failure_signatures
                        new_report["worst_segments"] = []
                        new_report["validated_against"] = "cold_start"
                        row["report"] = new_report
                        any_modified = True
                    else:
                        report.n_validation_rejected += 1
                        report.rejected_skill_ids.append(sid)
                        rows_to_drop.add(i)
                        any_modified = True

    if any_modified or rows_to_drop:
        surviving = [r for i, r in enumerate(rows) if i not in rows_to_drop]
        _atomic_rewrite(bank_path, surviving)
        logger.info(
            "post_writeback_inherit %s: examined=%d inherited=%d "
            "no_parent=%d parent_missing=%d already_filled=%d "
            "curator_skipped=%d val_attempt=%d val_pass=%d "
            "val_reject=%d val_abstain=%d",
            bank_path.name,
            report.n_inserted_skills_examined,
            report.n_inherited,
            report.n_no_parent,
            report.n_parent_missing,
            report.n_already_filled,
            report.n_curator_skipped,
            report.n_validation_attempted,
            report.n_validation_passed,
            report.n_validation_rejected,
            report.n_validation_abstained,
        )

    return report


def _maybe_validate_skill(
    *,
    sk: Mapping[str, Any],
    base: Optional[Mapping[str, Any]],
    segments_by_skill: Mapping[str, Any],
    min_segments: int,
) -> Optional[ValidationVerdict]:
    """Look up segments for the *parent* (or self) name and verify.

    Returns ``None`` when the skill has no contract literals to test
    (in which case the caller falls through to Phase A only).
    """
    contract = sk.get("contract") or {}
    if not isinstance(contract, Mapping):
        return None
    has_literals = bool(
        contract.get("eff_add") or contract.get("eff_del") or contract.get("eff_event"),
    )
    if not has_literals:
        return None

    # Skill label to look up: parent's name first (PATCH/COMPOSE/TRANSFER),
    # else the skill's own name (HYPOTHESIS).
    label_candidates: List[str] = []
    if base is not None:
        for key in ("name", "skill_id"):
            v = base.get(key)
            if v:
                label_candidates.append(v)
    for key in ("name", "skill_id"):
        v = sk.get(key)
        if v and v not in label_candidates:
            label_candidates.append(v)

    segments: List[Any] = []
    for label in label_candidates:
        # Normalise: SFT data uses bare names like "RECOVER/EVADE";
        # bank skills sometimes carry phase prefixes like
        # "early:RECOVER/EVADE".  Try both.
        for try_label in (label, label.split(":", 1)[-1] if ":" in label else label):
            hit = segments_by_skill.get(try_label)
            if hit:
                segments = list(hit)
                break
        if segments:
            break

    return verify_contract_against_segments(
        contract_dict=contract,
        segments=segments,
        min_segments=min_segments,
    )


def inherit_evidence_per_game(
    *,
    legacy_bank_paths: Mapping[str, Path],
    writeback_per_game: Mapping[str, Mapping[str, Any]],
    validation_corpus_root: Optional[Path] = None,
    validation_pass_thresh_by_kind: Optional[Mapping[str, float]] = None,
    validation_min_segments: int = DEFAULT_VALIDATION_MIN_SEGMENTS,
) -> Dict[str, Dict[str, Any]]:
    """Run :func:`inherit_evidence_for_inserted` for every game whose
    writeback report has a non-empty ``inserted_skill_ids``.

    Parameters
    ----------
    legacy_bank_paths
        Mapping ``game_slug -> skill_bank.jsonl Path`` (one per game).
    writeback_per_game
        Per-game writeback report dict (``inserted_skill_ids`` etc.).
    validation_corpus_root
        Phase B — root of the SFT cold-start corpus, e.g.
        ``labeling/frontier_distill_jsonl/run_<...>_with_labeled``.
        When ``None``, Phase B is skipped (Phase A inheritance only).
    validation_pass_thresh_by_kind
        Optional override for per-source-type pass thresholds.
    validation_min_segments
        Below this we abstain from per-skill verification.

    Returns
    -------
    Per-game dict matching the shape of ``writeback_per_game`` so the
    caller can merge it into the promotion-step summary.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for game, info in writeback_per_game.items():
        inserted = list(info.get("inserted_skill_ids") or [])
        if not inserted:
            out[game] = {"n_inherited": 0}
            continue
        bank_path = legacy_bank_paths.get(game)
        if bank_path is None:
            logger.warning(
                "post_writeback_inherit: no bank_path for game=%s; "
                "skipping inheritance sweep", game,
            )
            out[game] = {"n_inherited": 0, "error": "no bank_path"}
            continue

        # Phase B — load validation segments for this game (lazy; cached
        # per-game on disk so steady-state cost is a single ``mtime`` stat).
        segments_by_skill: Optional[Dict[str, Any]] = None
        if validation_corpus_root is not None:
            try:
                segments_by_skill = load_or_build_segments_for_game(
                    corpus_root=Path(validation_corpus_root),
                    game_slug=game,
                )
                if not segments_by_skill:
                    logger.info(
                        "post_writeback_inherit: no validation segments "
                        "for game=%s under %s; Phase B disabled for this game",
                        game, validation_corpus_root,
                    )
                    segments_by_skill = None
            except Exception as exc:                            # noqa: BLE001
                logger.warning(
                    "post_writeback_inherit: validation index load "
                    "failed for game=%s: %s", game, exc,
                )
                segments_by_skill = None

        try:
            rep = inherit_evidence_for_inserted(
                bank_path=Path(bank_path),
                inserted_skill_ids=inserted,
                validation_segments_by_skill=segments_by_skill,
                validation_pass_thresh_by_kind=validation_pass_thresh_by_kind,
                validation_min_segments=validation_min_segments,
            )
            out[game] = rep.to_dict()
        except Exception as exc:                                # noqa: BLE001
            logger.exception(
                "post_writeback_inherit: sweep failed for game=%s: %s",
                game, exc,
            )
            out[game] = {"n_inherited": 0, "error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_rewrite(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write *rows* to *path* atomically via a sibling tmp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp_inherit")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


__all__ = [
    "InheritReport",
    "inherit_evidence_for_inserted",
    "inherit_evidence_per_game",
]
