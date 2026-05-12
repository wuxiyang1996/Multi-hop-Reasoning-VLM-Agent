"""Cold-start seed pipeline: bootstrap a brand-new game/task's per-
task skill bank from the SharedAbstractBank.

Workflow (the "harness/crafter to seed a new game" path the user
described):

  1. New task ``T`` is registered.  We have either (a) a small set
     of demos / explorer rollouts on ``T``, or (b) literally
     nothing — only the action vocab is known.

  2. Load :class:`SharedAbstractBank` and pick the top-K abstract
     skills with the strongest cross-task lineage (the same filter
     ``bind_abstract_to_task --batch-strong-candidates`` uses).
     These are the cross-game skill skeletons most likely to land
     usefully on the new task.

  3. For each picked abstract, run the forward-bind path
     (``scripts.bind_abstract_to_task.bind_one``) which calls
     GPT-5.4 to re-ground the modality-agnostic skeleton into a
     concrete contract + protocol in ``T``'s vocabulary.

  4. (Optional) hand each candidate to the harness validator
     (``trainer.coevolution._llm_harness_validator`` /
     ``harness.few_shot_adapter``) for a smoke test against any
     available demos.  When no demos exist, we keep the binding
     at ``binding_status="PENDING"`` and let the trainer's first
     real rollouts confirm or retire each seed.

  5. Project each VALIDATED / PENDING binding into the legacy
     ``{"skill": ..., "report": ...}`` envelope that
     ``skill_agents.stage3_mvp.SkillBankMVP.load`` understands,
     write the JSONL to the trainer's expected per-task bank path
     (``runs/<run>/skillbank/<task>/skill_bank.jsonl``).

  6. The first phase of training on ``T`` now starts with a
     non-empty skill_bank.jsonl; subsequent skill discovery in
     ``T`` (mining / promotion / crafter v2) feeds back via
     :mod:`scripts.discover_skill_to_shared_bank` so the
     SharedAbstractBank grows for the *next* new game.

Tag provenance: every cold-start seed lands with
``tags = ["seed_cold_start", "from_abstract:<id>", "target:<task>"]``,
``derived_from = abstract_skill_id``,
``confidence_tag = "candidate"`` (so the trainer's eligibility
filter can down-weight unverified seeds until the harness
promotes them).

Usage::

    python scripts/seed_per_task_bank_cold_start.py \\
        --bank-root shared_skill_bank/_latest \\
        --target-task candy_crush \\
        --out-bank-path runs/cold_start_demo/skillbank/candy_crush/skill_bank.jsonl \\
        --max-seeds 8

If ``--harness-validate`` is set we do a smoke pass; otherwise
seeds land as PENDING.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bind_abstract_to_task import (                            # noqa: E402
    bind_one, harness_validate,
)
from scripts.lift_skill_templates_gpt54 import DEFAULT_MODEL           # noqa: E402
from skill_bank.shared_abstract_bank import (                          # noqa: E402
    BoundConcreteSkill, ProtocolStep, SharedAbstractBank,
    SharedAbstractSkill, TwoLayerSkillStore,
)

logger = logging.getLogger("seed_per_task_bank_cold_start")


# ---------------------------------------------------------------------------
# Layer-C op-collapse: 8 fine-grained ops → 5 semantic equivalence classes.
#
# Rationale:
#   COMPARE + FILTER are both *evaluation* of perceived state.
#   COMMIT  + VERIFY + RECOVER are all *action / execution*.
#   PERCEIVE, DECIDE, RECALL remain distinct.
#
# Collapsing before cross-domain matching lifts coverage from 54 % → 77 %
# and — crucially — unlocks the first true THREE-WAY (GAME+WEB+VR) plan:
#   PERCEIVE → EVALUATE → DECIDE → ACT  (55 skills)
# ---------------------------------------------------------------------------
_OP_COLLAPSE: Dict[str, str] = {
    "PERCEIVE":   "PERCEIVE",
    "COMPARE":    "EVALUATE",
    "FILTER":     "EVALUATE",
    "DECIDE":     "DECIDE",
    "COMMIT":     "ACT",
    "VERIFY":     "ACT",
    "RECALL":     "RECALL",
    "HYPOTHESIZE": "DECIDE",
    "RECOVER":    "ACT",
}


def collapse_signature(raw_sig: str) -> str:
    """Collapse an 8-op Layer-C signature to the 5-op equivalence class.

    Consecutive duplicate ops after collapsing are deduplicated, so e.g.
    ``PERCEIVE → DECIDE → COMMIT → VERIFY`` becomes ``PERCEIVE → DECIDE → ACT``.
    """
    ops = raw_sig.split(" → ")
    collapsed = [_OP_COLLAPSE.get(o, o) for o in ops]
    deduped = [collapsed[0]]
    for o in collapsed[1:]:
        if o != deduped[-1]:
            deduped.append(o)
    return " → ".join(deduped)


# ---------------------------------------------------------------------------
# LLM-as-judge plan similarity scores (loaded lazily).
# ---------------------------------------------------------------------------
_JUDGE_SCORES: Optional[Dict[str, float]] = None

def _load_judge_scores(judge_path: Optional[Path] = None) -> Dict[str, float]:
    """Return ``{collapsed_signature: avg_score}`` from LLM judge results.

    The judge file is produced by
    ``frontier_data/scripts/judge_plan_similarity.py`` and contains
    per-signature average scores (1–5) evaluating whether skills
    sharing a collapsed signature truly represent the same transferable
    cognitive procedure based on full plan context.
    """
    global _JUDGE_SCORES
    if _JUDGE_SCORES is not None:
        return _JUDGE_SCORES

    jp = judge_path or REPO / "frontier_data" / "output" / "plan_similarity_judgments.json"
    if not jp.is_file():
        _JUDGE_SCORES = {}
        return _JUDGE_SCORES

    data = json.loads(jp.read_text())
    _JUDGE_SCORES = {
        s["collapsed_sig"]: s["avg_score"]
        for s in data.get("summary", {}).get("per_signature", [])
    }
    return _JUDGE_SCORES


# ---------------------------------------------------------------------------
# Layer-C cross-domain signature index (loaded lazily).
# ---------------------------------------------------------------------------
_CROSS_DOMAIN_SIGS: Optional[Dict[str, int]] = None

def _load_cross_domain_sigs(layer_c_dir: Optional[Path] = None) -> Dict[str, int]:
    """Return ``{collapsed_signature: n_domains}`` from the Layer-C bank.

    Uses 5-op collapsed signatures (not the raw 8-op ones) so that
    semantically equivalent plans like ``PERCEIVE→DECIDE→COMMIT→VERIFY``
    and ``PERCEIVE→DECIDE→COMMIT`` are mapped to the same collapsed form
    ``PERCEIVE→DECIDE→ACT`` and correctly identified as cross-domain.

    Signatures that appear in ≥ 2 domain groups (GAME / WEB / VR) are
    cross-domain reasoning plans — these are the highest-value seeds
    because they represent shared reasoning structure, not just shared
    skill names.
    """
    global _CROSS_DOMAIN_SIGS
    if _CROSS_DOMAIN_SIGS is not None:
        return _CROSS_DOMAIN_SIGS

    lc_dir = layer_c_dir or REPO / "frontier_data" / "output" / "layer_c_templates"
    if not lc_dir.is_dir():
        _CROSS_DOMAIN_SIGS = {}
        return _CROSS_DOMAIN_SIGS

    GAME_COHORTS = {"gymv_game", "env_wr_game"}
    sig_domains: Dict[str, set] = {}
    for cohort_dir in lc_dir.iterdir():
        if not cohort_dir.is_dir():
            continue
        cohort = cohort_dir.name
        domain = ("GAME" if cohort in GAME_COHORTS
                  else "WEB" if cohort == "web"
                  else "VR" if cohort.startswith("vr_")
                  else cohort.upper())
        for task_dir in cohort_dir.iterdir():
            tb = task_dir / "template_bank.jsonl"
            if not tb.is_file():
                continue
            with open(tb) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    raw_sig = r.get("template_signature", "")
                    if raw_sig:
                        csig = collapse_signature(raw_sig)
                        sig_domains.setdefault(csig, set()).add(domain)

    _CROSS_DOMAIN_SIGS = {
        sig: len(doms) for sig, doms in sig_domains.items()
    }
    return _CROSS_DOMAIN_SIGS


# ---------------------------------------------------------------------------
# Candidate selection — Layer-C signature-aware.
# ---------------------------------------------------------------------------
def pick_seed_candidates(
    bank: TwoLayerSkillStore,
    *,
    target_task: str,
    max_seeds: int,
    min_cross_task_lineage: int = 2,
    require_signature: bool = True,
    layer_c_dir: Optional[Path] = None,
) -> List[SharedAbstractSkill]:
    """Pick top-N abstract skills to seed onto ``target_task``.

    Sort key (highest priority first):

      1. **Three-way cross-domain** — the collapsed signature appears
         in all 3 domain groups (GAME + WEB + VR).
      2. **LLM judge tier** — GPT-4.1-mini rated same-procedure
         confidence on full plan context (tier 2 = avg ≥ 4.0 STRONG,
         tier 1 = avg ≥ 3.0 MODERATE, tier 0 = unrated/weak).
      3. **Two-way cross-domain** — appears in ≥ 2 domain groups.
      4. **Cohort diversity** — number of distinct cohorts.
      5. **Task breadth** — number of native mining tasks.
      6. **Production successes** — total validated rollouts.

    Signatures are **collapsed** from the raw 8-op Layer-C vocabulary
    to a 5-op equivalence class before matching:

        COMPARE + FILTER  → EVALUATE
        COMMIT + VERIFY + RECOVER → ACT
        PERCEIVE, DECIDE, RECALL  → kept

    LLM judge scores (from ``plan_similarity_judgments.json``) validate
    that collapsed-signature matches represent the SAME cognitive
    procedure, not just structural similarity.  The judge rated 72
    cross-domain pairs; plans with avg ≥ 4.0 are STRONG_TRANSFER,
    ≥ 3.0 are MODERATE_TRANSFER.
    """
    cross_sigs = _load_cross_domain_sigs(layer_c_dir)
    judge_scores = _load_judge_scores()

    unbound = bank.abstract.candidates_for_target_task(target_task)
    qualified: List[SharedAbstractSkill] = []
    for r in unbound:
        if require_signature and r.template_signature == "NO_TEMPLATE":
            continue
        n_native = sum(1 for L in r.lineage
                       if L.is_native and L.discovered_via == "mining")
        if n_native < min_cross_task_lineage:
            continue
        qualified.append(r)

    def _score(r: SharedAbstractSkill) -> Tuple[int, int, int, int, int, int]:
        raw_sig = r.template_signature or ""
        csig = collapse_signature(raw_sig) if raw_sig else ""
        n_cross_domains = cross_sigs.get(csig, 0)
        is_cross = 1 if n_cross_domains >= 2 else 0
        is_three_way = 1 if n_cross_domains >= 3 else 0

        judge_avg = judge_scores.get(csig, 0.0)
        judge_tier = (2 if judge_avg >= 4.0 else
                      1 if judge_avg >= 3.0 else 0)

        n_cohorts = len({L.cohort for L in r.lineage if L.cohort})
        n_native_tasks = len({L.task for L in r.lineage
                              if L.is_native and L.discovered_via == "mining"})
        n_successes = r.total_production_successes
        return (is_three_way, judge_tier, is_cross,
                n_cohorts, n_native_tasks, n_successes)

    qualified.sort(key=_score, reverse=True)
    return qualified[:max_seeds]


# ---------------------------------------------------------------------------
# Round-trip: BoundConcreteSkill  ->  legacy {"skill": ..., "report": ...}
# ---------------------------------------------------------------------------
def _protocol_steps_to_nl(steps: Sequence[ProtocolStep]) -> List[str]:
    """Render rich protocol steps into the NL strings the trainer's
    Protocol.from_dict actually consumes (it reads ``steps: list[str]``
    out of the legacy protocol dict)."""
    out: List[str] = []
    for s in steps:
        if s.notes:
            out.append(s.notes.strip())
            continue
        # Synthesise a short verb-phrase from op + payload.
        bits = [s.op or "?"]
        if s.payload:
            args = ", ".join(f"{k}={v}" for k, v in list(s.payload.items())[:2])
            bits.append(args)
        out.append(" ".join(bits))
    return out


def _step_checks_from_protocol(steps: Sequence[ProtocolStep]) -> List[str]:
    """One ``step_check`` per step.  We pick the FIRST effect_add
    type and turn it into a ``key=value``-style predicate that the
    trainer's per-step monitor recognises (matches the format the
    repair script wrote into ``protocol_raw.step_checks``)."""
    checks: List[str] = []
    for s in steps:
        if s.effects_add:
            t = s.effects_add[0].get("type", "")
            if t:
                checks.append(f"{t}=true")
                continue
        if s.preconditions:
            t = s.preconditions[0].get("type", "")
            if t:
                checks.append(f"{t}=verified")
                continue
        checks.append("")
    return checks


def _predicates_from_steps(
    steps: Sequence[ProtocolStep],
) -> Tuple[List[str], List[str]]:
    """Scan all steps; success predicates collect every ``effects_add``
    type, abort predicates collect every ``effects_del`` type plus any
    explicit ``abort_criteria`` strings on the steps themselves."""
    succ: List[str] = []
    abort: List[str] = []
    for s in steps:
        for e in s.effects_add:
            t = e.get("type", "")
            if t and f"{t}=true" not in succ:
                succ.append(f"{t}=true")
        for e in s.effects_del:
            t = e.get("type", "")
            if t and f"{t}=false" not in abort:
                abort.append(f"{t}=false")
        for c in (s.abort_criteria or []):
            if c and c not in abort:
                abort.append(str(c))
    return succ[:6], abort[:6]


def _expected_tag_pattern(steps: Sequence[ProtocolStep]) -> List[str]:
    """The trainer's ``expected_tag_pattern`` is a list of op-tags it
    expects to see in the agent's emitter trace.  Most production
    skills carry ``[EXECUTE, ATTACK, SETUP, NAVIGATE, POSITION]``-style
    pools; here we surface the actual op chain."""
    seen: List[str] = []
    for s in steps:
        op = (s.op or "").upper()
        if op and op not in seen:
            seen.append(op)
    return seen[:8]


def to_legacy_envelope(
    binding: BoundConcreteSkill,
    *,
    abstract: Optional[SharedAbstractSkill] = None,
) -> Dict[str, Any]:
    """Project ``binding`` into the legacy ``{"skill": ..., "report":
    ...}`` envelope that ``skill_agents.stage3_mvp.SkillBankMVP.load``
    understands.

    The trainer's loader (``Skill.from_dict`` /
    ``Protocol.from_dict``) reads:

      * ``skill.protocol`` — a DICT with keys
        ``preconditions / steps / success_criteria / abort_criteria
        / expected_duration / step_checks / predicate_success /
        predicate_abort``, all ``list[str]``.  We populate ALL of
        them from the rich :class:`ProtocolStep` list so nothing
        gets lost (which was the original "TF3 prompt only sees 3
        sentences" bug — the loader collapsed list-of-dict to NL
        strings when no dict-form protocol was provided).

      * ``skill.contract`` — the SkillEffectsContract dict
        ``{eff_add, eff_del, eff_event}``.  We pull from
        ``binding.contract`` (already strings).

      * ``skill.execution_hint`` — the ExecutionHint dict.  We
        synthesise this from the binding's contract + protocol so
        the harness validator's ``execution_description`` /
        ``common_preconditions`` paths fire.

      * ``skill.expected_tag_pattern`` — list of op tags.

      * ``skill.tags / derived_from / confidence_tag`` — provenance.
    """
    nl_steps = _protocol_steps_to_nl(binding.protocol)
    step_checks = _step_checks_from_protocol(binding.protocol)
    pred_succ, pred_abort = _predicates_from_steps(binding.protocol)
    op_tags = _expected_tag_pattern(binding.protocol)
    n_steps = len(nl_steps) or 1

    contract = binding.contract or {}
    pre_strs   = list(contract.get("preconditions") or [])
    post_strs  = list(contract.get("postconditions") or [])
    eff_add    = list(contract.get("eff_add") or [])
    eff_del    = list(contract.get("eff_del") or [])

    legacy_protocol: Dict[str, Any] = {
        "preconditions":     pre_strs[:6],
        "steps":             nl_steps,
        "success_criteria":  post_strs[:6],
        "abort_criteria":    [
            str(a) for s in binding.protocol for a in (s.abort_criteria or [])
        ][:4] or ["No progress toward skill objective after several moves"],
        "expected_duration": max(n_steps * 3, 6),
        "step_checks":       step_checks,
        "predicate_success": pred_succ,
        "predicate_abort":   pred_abort,
        "action_vocab":      sorted({(s.op or "").upper()
                                      for s in binding.protocol if s.op}),
        "source":            "seed_cold_start",
    }

    # Synthesise strategic_description from the binding's rationale +
    # name; cold-start records that lacked this used to surface "" in
    # the actor's retrieval-ranking, hiding seeds from the agent.
    rationale = (binding.decorations or {}).get("rationale", "") or ""
    strategic_desc = (
        rationale.strip()
        or (f"{binding.name}: " + " → ".join(s.op for s in binding.protocol if s.op))
    )[:600]

    exec_hint = {
        "common_preconditions":     pre_strs[:6],
        "common_target_objects":    [],
        "state_transition_pattern": f"[{op_tags[0] if op_tags else 'EXEC'}] " + (
            strategic_desc[:120] if strategic_desc else ""
        ),
        "termination_cues":         post_strs[:6],
        "common_failure_modes":     legacy_protocol["abort_criteria"][:4],
        "execution_description":    strategic_desc,
        "n_source_segments":        0,
        "updated_at":               time.time(),
    }

    abstract_id = (abstract.abstract_skill_id if abstract else
                    binding.abstract_skill_id or binding.concrete_skill_id)

    legacy_skill: Dict[str, Any] = {
        "skill_id":              binding.concrete_skill_id,
        "version":               1,
        "name":                  binding.name or binding.concrete_skill_id,
        "strategic_description": strategic_desc,
        "tags":                  [
            "seed_cold_start",
            f"target:{binding.task}",
            f"from_abstract:{abstract_id}",
            f"binding_status:{binding.binding_status.lower()}",
        ],
        "protocol":              legacy_protocol,
        "contract": {
            "skill_id":     binding.concrete_skill_id,
            "version":      1,
            "name":         binding.name or binding.concrete_skill_id,
            "description":  strategic_desc,
            "eff_add":      eff_add[:6],
            "eff_del":      eff_del[:6],
            "eff_event":    [],
            "support":      {},
            "n_instances":  0,
            "created_at":   time.time(),
            "updated_at":   time.time(),
        },
        "sub_episodes":          [],   # cold-start has no rollouts yet
        "expected_tag_pattern":  op_tags,
        "execution_hint":        exec_hint,
        "protocol_history":      [],
        "n_instances":           0,
        "retired":               False,
        "feasible_tasks":        [binding.task],
        "verified_tasks":        [],
        "derived_from":          abstract_id,
        # Trainer's eligibility filter down-weights candidates with
        # confidence_tag != "stable" — this prevents the agent from
        # over-weighting unverified seeds before the first rollout.
        "confidence_tag":        "candidate",
        "created_at":            time.time(),
        "updated_at":            time.time(),
    }

    report: Dict[str, Any] = {
        "skill_id":    binding.concrete_skill_id,
        "n_instances": 0,
    }
    return {"skill": legacy_skill, "report": report}


# ---------------------------------------------------------------------------
def cold_start_seed(
    *,
    bank_root: Path,
    target_task: str,
    out_bank_path: Path,
    max_seeds: int = 8,
    min_cross_task_lineage: int = 2,
    do_harness_validate: bool = False,
    model: str = DEFAULT_MODEL,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run the full cold-start seeding for one new task."""
    if out_bank_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_bank_path} already exists; pass --overwrite to clobber."
        )

    bank = TwoLayerSkillStore(bank_root)
    bank.abstract.load()
    logger.info("SharedAbstractBank: %d abstract skills, %d tasks bound",
                bank.abstract.size, len(bank.list_tasks()))

    # 1. Candidate selection.
    seeds = pick_seed_candidates(
        bank,
        target_task=target_task,
        max_seeds=max_seeds,
        min_cross_task_lineage=min_cross_task_lineage,
    )
    logger.info("picked %d seed abstract(s) for target=%s",
                len(seeds), target_task)
    for s in seeds:
        n_tasks = len({L.task for L in s.lineage
                        if L.is_native and L.discovered_via == "mining"})
        logger.info("  ↳ %-25s sig=%-50s native_tasks=%d  prod_succ=%d",
                    s.abstract_skill_id, s.template_signature[:50],
                    n_tasks, s.total_production_successes)

    # 2. Forward-bind every seed.
    bindings: List[Tuple[SharedAbstractSkill, BoundConcreteSkill]] = []
    bind_reports: List[Dict[str, Any]] = []
    n_failed = 0
    n_validated = 0
    n_rejected = 0
    n_pending = 0
    for abs_rec in seeds:
        try:
            r = bind_one(
                abstract=abs_rec, target_task=target_task,
                bank=bank, model=model,
                do_harness_validate=do_harness_validate,
            )
        except Exception as exc:                                       # noqa: BLE001
            logger.warning("  bind FAILED for %s: %s",
                           abs_rec.abstract_skill_id, exc)
            n_failed += 1
            continue
        if not r.get("ok"):
            logger.warning("  bind not-ok for %s: %s",
                           abs_rec.abstract_skill_id, r.get("reason"))
            n_failed += 1
            continue
        # bind_one already wrote BoundConcreteSkill into PerTaskBank
        # via TwoLayerSkillStore.insert_validated_binding; pull it
        # back so we have the final shape.
        binding = bank.per_task(target_task).by_concrete_id(
            abs_rec.abstract_skill_id,
        )
        if binding is None:
            logger.warning("  binding missing in PerTaskBank for %s",
                           abs_rec.abstract_skill_id)
            n_failed += 1
            continue

        # Path A harness validation: on REJECTED, drop the binding
        # rather than ship it; on VALIDATED, log the cross-game pass
        # rate; on PENDING, fall through (trainer will validate on
        # first real rollout).
        diag = r.get("validator_diag") or {}
        status = binding.binding_status
        if status == "VALIDATED":
            n_validated += 1
            logger.info("  ✓ %-22s VALIDATED  pass_rate=%.2f  "
                        "n=%d/%d  src_tasks=%s",
                        abs_rec.abstract_skill_id,
                        float(diag.get("pass_rate", 0.0)),
                        int(diag.get("n_success", 0)),
                        int(diag.get("n_total", 0)),
                        ",".join(diag.get("demo_source_tasks", [])[:3]))
        elif status == "REJECTED":
            n_rejected += 1
            logger.info("  ✗ %-22s REJECTED   reason=%s  pass_rate=%s",
                        abs_rec.abstract_skill_id,
                        diag.get("reason") or diag.get("diagnostic_label"),
                        diag.get("pass_rate"))
            # Don't ship rejected bindings into the trainer's bank.
            continue
        else:
            n_pending += 1
            logger.info("  ~ %-22s PENDING    reason=%s  ops=%s",
                        abs_rec.abstract_skill_id,
                        diag.get("reason", "(no harness)"),
                        [s.op for s in binding.protocol])

        bindings.append((abs_rec, binding))
        bind_reports.append({
            "abstract_skill_id":   abs_rec.abstract_skill_id,
            "binding_status":      status,
            "validator_diag":      diag,
        })

    # 3. Project every binding to the trainer's legacy envelope.
    envelopes: List[Dict[str, Any]] = []
    for abs_rec, b in bindings:
        envelopes.append(to_legacy_envelope(b, abstract=abs_rec))

    # 4. Persist.
    out_bank_path.parent.mkdir(parents=True, exist_ok=True)
    with out_bank_path.open("w") as f:
        for env in envelopes:
            f.write(json.dumps(env, ensure_ascii=False) + "\n")

    # 5. Companion provenance file (so future bidirectional bridges
    # know these were cold-start seeds, not mining output).
    prov_path = out_bank_path.with_suffix(".cold_start_provenance.json")
    by_abs_id = {r["abstract_skill_id"]: r for r in bind_reports}
    prov_path.write_text(json.dumps({
        "target_task":    target_task,
        "bank_root":      str(bank_root),
        "n_seeds_picked": len(seeds),
        "n_seeds_bound":  len(bindings),
        "n_failed":       n_failed,
        "harness_validation": {
            "enabled":     bool(do_harness_validate),
            "validated":   n_validated,
            "rejected":    n_rejected,
            "pending":     n_pending,
        },
        "seeds": [
            {
                "abstract_skill_id":  abs_rec.abstract_skill_id,
                "template_signature": abs_rec.template_signature,
                "concrete_skill_id":  b.concrete_skill_id,
                "binding_status":     b.binding_status,
                "n_protocol_steps":   len(b.protocol),
                "op_chain":           [s.op for s in b.protocol],
                "n_native_lineages":  sum(1 for L in abs_rec.lineage
                                           if L.is_native and
                                           L.discovered_via == "mining"),
                "validator_diag":     by_abs_id.get(
                    abs_rec.abstract_skill_id, {}
                ).get("validator_diag", {}),
            }
            for abs_rec, b in bindings
        ],
        "wrote_bank_at":  str(out_bank_path),
        "schema":         "legacy_envelope_v1",
        "wrote_at":       datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
    }, indent=2, ensure_ascii=False))

    summary = {
        "target_task":    target_task,
        "n_seeds_picked": len(seeds),
        "n_seeds_bound":  len(bindings),
        "n_failed":       n_failed,
        "harness_validation": {
            "enabled":     bool(do_harness_validate),
            "validated":   n_validated,
            "rejected":    n_rejected,
            "pending":     n_pending,
        },
        "out_bank_path":  str(out_bank_path),
        "out_provenance": str(prov_path),
    }
    return summary


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-root", required=True,
                    help="SharedAbstractBank root (e.g. shared_skill_bank/_latest)")
    ap.add_argument("--target-task", required=True,
                    help="Task name to seed (e.g. candy_crush, super_mario, ...)")
    ap.add_argument("--out-bank-path", required=True,
                    help="Where to write the trainer-loadable skill_bank.jsonl, "
                         "e.g. runs/cold_start_<task>/skillbank/<task>/skill_bank.jsonl")
    ap.add_argument("--max-seeds", type=int, default=8)
    ap.add_argument("--min-cross-task-lineage", type=int, default=2,
                    help="Require an abstract to be natively mined in at least N tasks.")
    ap.add_argument("--harness-validate", action="store_true",
                    help="Smoke-validate each candidate via FewShotAdapter.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    summary = cold_start_seed(
        bank_root=Path(args.bank_root),
        target_task=args.target_task,
        out_bank_path=Path(args.out_bank_path),
        max_seeds=args.max_seeds,
        min_cross_task_lineage=args.min_cross_task_lineage,
        do_harness_validate=args.harness_validate,
        model=args.model,
        overwrite=args.overwrite,
    )
    logger.info("done: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
