"""Consolidate every skill source the repo currently has into a single
:class:`SharedAbstractBank` + per-task :class:`PerTaskBank`.

Sources, in order of authority (later writes take precedence on
matching ``stable_key``):

  1. **Layer-C lifted templates** (``labeling/skill_templates/run_*``)
     — gives ``template_signature`` + ``template_steps`` for every
     mined skill across 5 cohorts.

  2. **Mining skill banks** — gives ``protocol_steps`` and the
     concrete contract that each lineage entry's ``contract_hash``
     points at.  We read from BOTH:

     * ``labeling/skill_bank_out/run_repair_20260510_051643/gym_v/...``
       (repaired contracts after May 10).
     * ``labeling/skill_bank_envwrappers/run_20260506_201030/env_wrappers/...``.
     * ``labeling/skill_bank_qa/run_20260506_184439/{miniwob,
       siv_bench, tir_bench, video_holmes, visual_toolbench}``.
     * ``labeling/skill_bank_qa/run_webshop_20260510_044000/webshop``.

  3. **Production transfer logs** (``runs/*/transfer_log/usage.jsonl``)
     — these contain phase-prefixed (``early:`` / ``mid:`` / ``late:``)
     IDs, crafter ``#v2`` suffixes, and ``__translated_to__`` cross-
     game translations that NEVER appear in the lift output.  This is
     where the 6.6 % → 70 %+ coverage uplift comes from.  Each
     production observation gets ``LineageEntry`` with
     ``discovered_via="production_usage"``, n_uses / n_success / etc.

  4. **Crafter v2 offline outputs** (``runs/*/crafter_v2_offline/...``)
     — proposed but not-yet-mined skills.  We DO NOT attempt to
     promote them here (that's the orchestrator's job); we simply
     record their existence as lineage with
     ``discovered_via="crafter_proposal"`` and ``is_native=True``
     for the task they were mined against.

The script never calls an LLM — for sources 1+2 the template
signature is already present; for sources 3+4 we attach lineage to
existing abstracts (matching by ``abstract_skill_id``) and emit a
"NO_TEMPLATE" placeholder abstract for orphans.  The next pass
(``scripts/lift_orphan_abstracts.py``) will fill those in via
GPT-5.4.

Invocation::

    python scripts/build_shared_skill_bank.py \\
        --out shared_skill_bank/run_<utc-ts>

Default output dir: ``shared_skill_bank/run_<UTC>``.  Symlinks
``shared_skill_bank/_latest`` to the new dir on success.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from skill_bank.shared_abstract_bank import (                          # noqa: E402
    BoundConcreteSkill, LineageEntry, ProtocolStep, SharedAbstractSkill,
    SubEpisodeRef, TemplateStep, TwoLayerSkillStore, hash_contract,
    normalise_skill_id, parse_skill_id_decorations,
)

logger = logging.getLogger("build_shared_skill_bank")

# ── Default sources (override via CLI) ───────────────────────────
DEFAULT_TEMPLATE_RUN = REPO / "labeling/skill_templates/run_20260510_053121"

DEFAULT_BANK_SOURCES: List[Tuple[str, str, Path]] = [
    # (cohort, task-glob-relative-to-root, root)
    # gym_v repaired
    ("gymv_game", "*",
     REPO / "labeling/skill_bank_out/run_repair_20260510_051643/gym_v"),
    # env_wrappers
    ("env_wr_game", "*",
     REPO / "labeling/skill_bank_envwrappers/run_20260506_201030/env_wrappers"),
    # QA-style banks (vr_image, vr_video, miniwob)
    ("qa_mixed", "*",
     REPO / "labeling/skill_bank_qa/run_20260506_184439"),
    # webshop (separate run because it landed on May 10)
    ("web", "webshop",
     REPO / "labeling/skill_bank_qa/run_webshop_20260510_044000"),
]

DEFAULT_PROD_GLOB = "runs/*/transfer_log/usage.jsonl"
DEFAULT_TF3_GLOB  = "runs/tf3_to_candy_crush_*/transfer_log/usage.jsonl"

# Cohort heuristic: derived from task name.  Mirrors lift_skill_templates_gpt54.
COHORT_OF_KNOWN: Dict[str, str] = {
    "tetris": "env_wr_game", "twenty_forty_eight": "env_wr_game",
    "super_mario": "env_wr_game", "candy_crush": "env_wr_game",
    "miniwob": "web", "webshop": "web",
    "video_holmes": "vr_video", "siv_bench": "vr_video",
    "tir_bench": "vr_image", "visual_toolbench": "vr_image",
}


def cohort_for(task: str) -> str:
    if task in COHORT_OF_KNOWN:
        return COHORT_OF_KNOWN[task]
    if task.startswith("Temporal_") and task.endswith("-v0"):
        return "gymv_game"
    if task.startswith("gymv_"):
        return "gymv_game"
    return "unknown"


# ---------------------------------------------------------------------------
# Source 1+2: lift bank + mining bank → SharedAbstractSkill records
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _peek_inner(rec: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(rec, dict) and isinstance(rec.get("skill"), dict):
        return rec["skill"]
    return rec


def _bank_skills(root: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Walk ``<root>/<task>/skill_bank.jsonl`` (or
    ``<root>/<task>/episode_snapshots/episode_*/skill_bank.jsonl`` for
    legacy gym_v layouts) and yield (task, raw_record) pairs."""
    if not root.exists():
        return
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        # Direct layout
        bank = task_dir / "skill_bank.jsonl"
        if bank.exists():
            for rec in _load_jsonl(bank):
                yield (task_dir.name, rec)
            continue
        # Legacy gym_v episode_snapshots layout
        ep_dir = task_dir / "episode_snapshots"
        if ep_dir.exists():
            for ep in sorted(ep_dir.iterdir()):
                bank = ep / "skill_bank.jsonl"
                if bank.exists():
                    for rec in _load_jsonl(bank):
                        yield (task_dir.name, rec)


def collect_lift_templates(template_run: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return ``{(task, skill_id): template_record}`` for every lifted
    template (gives us template_signature + template_steps)."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not template_run.exists():
        return out
    for cohort_dir in sorted(template_run.iterdir()):
        if not cohort_dir.is_dir() or cohort_dir.name.startswith("_"):
            continue
        for task_dir in sorted(cohort_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            bank = task_dir / "template_bank.jsonl"
            for rec in _load_jsonl(bank):
                sid = rec.get("skill_id", "")
                if sid:
                    out[(task_dir.name, sid)] = rec
    return out


def consolidate_mining_banks(
    *,
    bank_sources: List[Tuple[str, str, Path]],
    lift_templates: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[SharedAbstractSkill], List[BoundConcreteSkill]]:
    """Walk every mining bank, emit one SharedAbstractSkill per unique
    (skill_id stem, template_signature) and one BoundConcreteSkill per
    (task, skill_id) it appears in."""
    abstracts_by_key: Dict[Tuple[str, str], SharedAbstractSkill] = {}
    bindings: List[BoundConcreteSkill] = []
    n_recs = 0
    n_no_template = 0

    for cohort_hint, _, root in bank_sources:
        for task, raw in _bank_skills(root):
            inner = _peek_inner(raw)
            sid = inner.get("skill_id") or ""
            if not sid:
                continue
            n_recs += 1

            stem = normalise_skill_id(sid)
            cohort = cohort_for(task)
            template_rec = lift_templates.get((task, sid))
            if template_rec is None:
                # No lift output → use a NO_TEMPLATE placeholder.
                # We still keep the skill so its concrete contract is
                # available for forward-bind / harness validation.
                signature = "NO_TEMPLATE"
                template_steps: List[TemplateStep] = []
                n_no_template += 1
            else:
                signature = template_rec.get("template_signature", "NO_TEMPLATE")
                template_steps = [
                    TemplateStep.from_dict(s)
                    for s in (template_rec.get("template_steps") or [])
                ]

            # ── Protocol — the executable plan ──────────────────
            # Mining emits two flavours:
            #
            #   Type A (list[dict]): full per-step structure with
            #     op / payload / slot_types / preconditions /
            #     effects_add / effects_del / evidence_role / notes.
            #     We keep ALL of those — that's what the agent will
            #     run at runtime.
            #
            #   Type B (dict with "steps" list[str]): the crafter-
            #     style high-level protocol.  Steps are free-text
            #     ("Identify the matching cluster of three"); we
            #     wrap each as a ProtocolStep with op="?" + notes
            #     so the abstract retains the multi-hop outline,
            #     and stash success_criteria / abort_criteria on
            #     the LAST step so they survive lifting.
            proto_raw = inner.get("protocol")
            protocol_steps: List[ProtocolStep] = []
            if isinstance(proto_raw, list):
                for s in proto_raw:
                    if isinstance(s, dict):
                        protocol_steps.append(ProtocolStep.from_dict(s))
            elif isinstance(proto_raw, dict):
                steps_b = proto_raw.get("steps") or []
                pre_b   = proto_raw.get("preconditions") or []
                succ_b  = proto_raw.get("success_criteria") or []
                abort_b = proto_raw.get("abort_criteria") or []
                for i, s in enumerate(steps_b):
                    if not isinstance(s, str):
                        continue
                    is_last = (i == len(steps_b) - 1)
                    protocol_steps.append(ProtocolStep(
                        op="?",
                        notes=s,
                        preconditions=([{"type": "free_text", "args": {"text": p}}
                                        for p in pre_b if isinstance(p, str)]
                                        if i == 0 else []),
                        success_criteria=([str(c) for c in succ_b]
                                          if is_last else []),
                        abort_criteria=([str(c) for c in abort_b]
                                        if is_last else []),
                    ))

            # ── Sub-episodes — pointers to PRIOR ACTUAL ROLLOUTS ─
            sub_eps_raw = inner.get("sub_episodes") or []
            sub_episodes = [SubEpisodeRef.from_dict(s)
                             for s in sub_eps_raw if isinstance(s, dict)]
            # Mining-pipeline sub_episodes don't always tag the
            # source task; backfill so harness re-runs know which
            # env to load.
            for s in sub_episodes:
                if not s.task:
                    s.task = task

            # The abstract gets a stripped (slot-name + semantic-type
            # only) view of the protocol; the concrete binding keeps
            # the full task-vocabulary version.
            abstract_protocol_steps = [s.abstract_view() for s in protocol_steps]

            # ── upsert the abstract ───────────────────────────────
            key = (stem, signature)
            abs_rec = abstracts_by_key.get(key)
            if abs_rec is None:
                abs_rec = SharedAbstractSkill(
                    abstract_skill_id=stem,
                    name=inner.get("name", "") or stem,
                    template_signature=signature,
                    template_steps=list(template_steps),
                    protocol_steps=list(abstract_protocol_steps),
                    discovered_via="mining",
                )
                abstracts_by_key[key] = abs_rec
            else:
                # Merge: prefer first-seen template_steps and protocol_steps,
                # but accept later ones if the first was empty.
                if not abs_rec.template_steps and template_steps:
                    abs_rec.template_steps = list(template_steps)
                if not abs_rec.protocol_steps and abstract_protocol_steps:
                    abs_rec.protocol_steps = list(abstract_protocol_steps)

            contract = inner.get("contract") or {}
            chash = hash_contract(contract)

            # Lineage entry — this concrete skill bound to this task.
            lineage = LineageEntry(
                task=task,
                concrete_skill_id=stem,
                raw_skill_id=sid,
                cohort=cohort,
                discovered_via="mining",
                is_native=True,
                contract_hash=chash,
                n_uses=sum(1 for s in sub_episodes
                           if s.outcome in ("success", "failure", "partial")),
                n_success=sum(1 for s in sub_episodes if s.outcome == "success"),
            )
            abs_rec.upsert_lineage(lineage)

            # Concrete binding — the executable plan + receipts.
            binding = BoundConcreteSkill(
                concrete_skill_id=stem,
                task=task,
                abstract_skill_id=stem,
                name=inner.get("name", "") or stem,
                protocol=protocol_steps,        # PRIMARY: structured plan
                sub_episodes=sub_episodes,      # EVIDENCE: rollout pointers
                contract=contract,              # DERIVED: static summary
                binding_status="VALIDATED",     # mined skills are validated by construction
                binding_source="mining",
                raw_skill_id=sid,
                n_episodes_verified=len(sub_episodes),
            )
            # Prefer empirical pass rate when sub_episodes carry
            # outcome labels; falls back to 0.0 for QA-style skills
            # where rollouts are partial-only.
            er = binding.empirical_success_rate
            if er > 0.0 or binding.n_sub_episodes_failure > 0:
                binding.pass_rate = er
            bindings.append(binding)

    logger.info(
        "consolidate_mining_banks: %d records → %d abstracts, %d bindings (%d had NO_TEMPLATE)",
        n_recs, len(abstracts_by_key), len(bindings), n_no_template,
    )
    return list(abstracts_by_key.values()), bindings


# ---------------------------------------------------------------------------
# Source 3: production transfer logs — PHASE-PREFIX / V2 / TRANSLATED IDs
# ---------------------------------------------------------------------------
def fold_production_usage(
    *,
    usage_glob: str,
    abstracts_by_stem: Dict[str, List[SharedAbstractSkill]],
    bank: TwoLayerSkillStore,
) -> Dict[str, int]:
    """Replay every ``transfer_log/usage.jsonl`` file matching the
    glob and stitch each (decorated) skill_id into its stem-form
    abstract's lineage.  When the abstract doesn't exist, we mint a
    NO_TEMPLATE placeholder so its uses are visible.

    Returns counts ``{n_records, n_decorated, n_translated, n_orphan_abstracts_minted}``.
    """
    stats = Counter()
    file_paths = sorted(REPO.glob(usage_glob))
    if not file_paths:
        logger.warning("no production usage logs match %s", usage_glob)
        return dict(stats)

    # Aggregate by (stem, target_task) so we don't emit one lineage
    # entry per usage record (there are tens of thousands).
    agg: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for fp in file_paths:
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                stats["n_records"] += 1

                sid_raw = r.get("skill_id", "")
                if not sid_raw:
                    continue
                stem = normalise_skill_id(sid_raw)
                deco = parse_skill_id_decorations(sid_raw)
                if any(k in deco for k in ("phase", "crafter_version", "translated_to")):
                    stats["n_decorated"] += 1
                if "translated_to" in deco:
                    stats["n_translated"] += 1

                target_task = r.get("game") or ""

                key = (stem, sid_raw, target_task)
                bucket = agg.setdefault(key, {
                    "n_uses": 0, "n_success": 0, "n_translated_uses": 0,
                    "name": r.get("skill_name", ""),
                    "decorations": deco,
                    "first_run": fp.parent.parent.name,
                })
                bucket["n_uses"] += 1
                v = r.get("harness_verdict", "")
                if isinstance(v, str) and v.startswith("success"):
                    bucket["n_success"] += 1
                if r.get("is_cross_game_translated"):
                    bucket["n_translated_uses"] += 1

    # Now emit lineage entries.
    for (stem, raw_sid, task), bucket in agg.items():
        cohort = cohort_for(task)

        # Find the abstract this stem belongs to.
        candidates = abstracts_by_stem.get(stem, [])
        if candidates:
            # Use the first candidate (most-common signature).
            target_abstract = candidates[0]
        else:
            # Mint a NO_TEMPLATE placeholder so the lineage isn't lost.
            target_abstract = SharedAbstractSkill(
                abstract_skill_id=stem,
                name=bucket.get("name") or stem,
                template_signature="NO_TEMPLATE",
                discovered_via="production_usage",
            )
            abstracts_by_stem.setdefault(stem, []).append(target_abstract)
            stats["n_orphan_abstracts_minted"] += 1

        # is_native: a translation arriving via __translated_to__<task>
        # is BY DEFINITION foreign in <task> — not native.  Otherwise
        # treat the production usage as native (the trainer was using
        # the skill directly).  When the same abstract has a
        # mining-source lineage in the SAME task, the discovery=
        # "mining" entry already has is_native=True, so we won't
        # double-flag.
        is_native = "translated_to" not in bucket["decorations"]

        lineage = LineageEntry(
            task=task,
            concrete_skill_id=stem,
            raw_skill_id=raw_sid,
            cohort=cohort,
            discovered_via=(
                "translation" if "translated_to" in bucket["decorations"]
                else "production_usage"
            ),
            is_native=is_native,
            n_uses=bucket["n_uses"],
            n_success=bucket["n_success"],
            n_translated_uses=bucket["n_translated_uses"],
            decorations=bucket["decorations"],
            notes=f"prod_run={bucket['first_run']}",
        )
        target_abstract.upsert_lineage(lineage)
        if cohort and cohort not in target_abstract.cohorts_seen:
            target_abstract.cohorts_seen.append(cohort)

    return dict(stats)


# ---------------------------------------------------------------------------
# Source 4: crafter_v2_offline proposals (placeholder — light-weight)
# ---------------------------------------------------------------------------
def fold_crafter_v2_outputs(
    *,
    runs_glob: str,
    abstracts_by_stem: Dict[str, List[SharedAbstractSkill]],
) -> Dict[str, int]:
    """Walk ``runs/*/crafter_v2_offline/proposals/<game>/accepted.jsonl``
    and add lineage entries for any *accepted* proposal.  This catches
    the ``#v2`` evolutions that don't yet have lift output."""
    stats = Counter()
    for accept in REPO.glob(runs_glob):
        run_id = accept.parent.parent.parent.parent.name  # runs/<run>/...
        try:
            for p in _load_jsonl(accept):
                stats["n_proposals_seen"] += 1
                inner = _peek_inner(p)
                sid = inner.get("skill_id") or inner.get("name") or ""
                if not sid:
                    continue
                stem = normalise_skill_id(sid)
                deco = parse_skill_id_decorations(sid)
                # accepted proposals are most often versioned (#v2)
                cands = abstracts_by_stem.get(stem, [])
                target = cands[0] if cands else SharedAbstractSkill(
                    abstract_skill_id=stem,
                    name=inner.get("name") or stem,
                    template_signature="NO_TEMPLATE",
                    discovered_via="crafter_proposal",
                )
                if not cands:
                    abstracts_by_stem.setdefault(stem, []).append(target)
                    stats["n_orphan_minted"] += 1
                # lineage attaches to the source game where the proposal
                # was generated.
                game = accept.parent.name  # crafter_v2_offline/proposals/<game>/...
                target.upsert_lineage(LineageEntry(
                    task=game,
                    concrete_skill_id=stem,
                    raw_skill_id=sid,
                    cohort=cohort_for(game),
                    discovered_via="crafter_proposal",
                    is_native=True,
                    decorations={**deco, "from_run": run_id},
                ))
                stats["n_lineages_added"] += 1
        except Exception as exc:                                       # pragma: no cover
            logger.warning("could not parse crafter accept %s: %s", accept, exc)
    return dict(stats)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="Output dir.  Default: shared_skill_bank/run_<utc>")
    ap.add_argument("--template-run", default=str(DEFAULT_TEMPLATE_RUN))
    ap.add_argument("--prod-glob",    default=DEFAULT_PROD_GLOB)
    ap.add_argument("--crafter-accept-glob",
                    default="runs/*/crafter_v2_offline/proposals/*/accepted.jsonl")
    ap.add_argument("--no-prod", action="store_true",
                    help="Skip production usage folding (debug).")
    ap.add_argument("--no-crafter", action="store_true",
                    help="Skip crafter v2 folding (debug).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    out_dir = Path(args.out or REPO / "shared_skill_bank" /
                   ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("out_dir=%s", out_dir)

    # ── Source 1+2 — mining banks anchored on lift templates ────────
    template_run = Path(args.template_run)
    logger.info("loading lift templates from %s", template_run)
    lift_tpls = collect_lift_templates(template_run)
    logger.info("  loaded %d lifted templates", len(lift_tpls))

    abstracts, bindings = consolidate_mining_banks(
        bank_sources=DEFAULT_BANK_SOURCES,
        lift_templates=lift_tpls,
    )

    # Build a stem → [abstracts] index for source 3+4.
    abstracts_by_stem: Dict[str, List[SharedAbstractSkill]] = defaultdict(list)
    for a in abstracts:
        abstracts_by_stem[a.abstract_skill_id].append(a)

    # ── Source 3 — production transfer logs ─────────────────────────
    if not args.no_prod:
        logger.info("folding production transfer logs (%s)", args.prod_glob)
        prod_stats = fold_production_usage(
            usage_glob=args.prod_glob,
            abstracts_by_stem=abstracts_by_stem,
            bank=None,  # we'll wire upserts at the final write step
        )
        logger.info("  %s", prod_stats)

    # ── Source 4 — crafter v2 accepted proposals ────────────────────
    if not args.no_crafter:
        crafter_stats = fold_crafter_v2_outputs(
            runs_glob=args.crafter_accept_glob,
            abstracts_by_stem=abstracts_by_stem,
        )
        logger.info("  crafter: %s", crafter_stats)

    # ── Persist ─────────────────────────────────────────────────────
    bank = TwoLayerSkillStore(out_dir)
    bank.abstract.load()  # ensure index is initialised

    n_new = 0
    n_merged = 0
    for abs_rec in [a for L in abstracts_by_stem.values() for a in L]:
        v = bank.abstract.upsert_abstract(abs_rec)
        if v == "new":
            n_new += 1
        else:
            n_merged += 1

    n_bindings = 0
    for b in bindings:
        bank.per_task(b.task).upsert_binding(b)
        n_bindings += 1

    # ── Summary report ──────────────────────────────────────────────
    # Sub-episode coverage stats — how many bindings carry receipts.
    n_bindings_with_subeps = 0
    n_subeps_total = 0
    n_subeps_success = 0
    for task in bank.list_tasks():
        for b in bank.per_task(task).records:
            if b.sub_episodes:
                n_bindings_with_subeps += 1
                n_subeps_total += b.n_sub_episodes
                n_subeps_success += b.n_sub_episodes_success

    summary: Dict[str, Any] = {
        "out_dir": str(out_dir),
        "n_abstracts_total":   bank.abstract.size,
        "n_abstracts_new":     n_new,
        "n_abstracts_merged":  n_merged,
        "n_per_task_bindings": n_bindings,
        "n_bindings_with_sub_episodes": n_bindings_with_subeps,
        "n_sub_episodes_total":   n_subeps_total,
        "n_sub_episodes_success": n_subeps_success,
        "n_tasks_with_bindings": len(bank.list_tasks()),
        "lift_templates_loaded": len(lift_tpls),
        "no_template_abstracts": sum(
            1 for r in bank.abstract.records if r.template_signature == "NO_TEMPLATE"
        ),
        "abstracts_with_translation_lineage": sum(
            1 for r in bank.abstract.records
            if any(L.discovered_via == "translation" for L in r.lineage)
        ),
        "abstracts_with_production_lineage": sum(
            1 for r in bank.abstract.records
            if any(L.discovered_via in ("production_usage", "translation") for L in r.lineage)
        ),
        "total_production_uses": sum(
            r.total_production_uses for r in bank.abstract.records
        ),
        "total_production_successes": sum(
            r.total_production_successes for r in bank.abstract.records
        ),
    }
    (out_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
    )
    logger.info("summary: %s", json.dumps(summary, indent=2))

    # Symlink shared_skill_bank/_latest → this run.
    latest_link = REPO / "shared_skill_bank" / "_latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(out_dir.relative_to(latest_link.parent))
        logger.info("→ %s", latest_link)
    except Exception as exc:                                           # pragma: no cover
        logger.warning("could not update _latest symlink: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
