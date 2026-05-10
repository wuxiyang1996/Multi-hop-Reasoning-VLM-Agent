"""End-to-end Layer-C cross-domain demo.

Walks one concrete skill through the full pipeline:

  Step 1.  Pick a gym_v skill (default: Temporal_AlteredBeast-v0 /
           INSPECT/SETUP — a beat-em-up scouting skill).
  Step 2.  Load it from the repaired gym_v skill bank
           (run_repair_20260510_051643).
  Step 3.  Live-call GPT-5.4 with the same prompt
           ``scripts/lift_skill_templates_gpt54.py`` uses, to produce
           the modality-agnostic abstract template.
  Step 4.  Compare the live abstract against the GROUND TRUTH that
           shipped in the canonical lift run (template_bank.jsonl
           entry for the same skill).  Reports signature equality
           and per-step operator overlap.
  Step 5.  Use the live abstract's signature to query
           :class:`TemplateIndex` for the **intended skill in another
           domain** — pick a top match from a *different* cohort.
  Step 6.  Pull that target skill's full record from its skill bank
           so we can show the surface-level (modality-specific)
           description side-by-side with the gym_v source.
  Step 7.  Render a clean side-by-side table: source description vs
           target description, both abstracts, and a step-by-step
           alignment.

Usage::

    python scripts/demo_layerc_cross_domain.py
    python scripts/demo_layerc_cross_domain.py \
        --gymv-game Temporal_ThunderForceIII-v0 \
        --gymv-skill-id COMMIT/EVADE \
        --target-cohort web

Flags --no-llm skips the live GPT-5.4 call and just shows the
ground-truth abstract — useful for offline runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
WORK = REPO.parent
for p in [str(WORK), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the canonical lift prompt + LLM client from the production lift.
from scripts.lift_skill_templates_gpt54 import (                     # noqa: E402
    SYSTEM_PROMPT, _build_prompt, _coerce_template, _extract_json,
    _get_openai_client, cohort_of, DEFAULT_MODEL,
)
from scripts.template_index import TemplateIndex                    # noqa: E402


# ---------------------------------------------------------------------------
DEFAULT_GYMV_BANK = (
    REPO / "labeling/skill_bank_out/run_repair_20260510_051643/gym_v"
)
DEFAULT_TEMPLATE_RUN = REPO / "labeling/skill_templates/run_20260510_053121"
DEFAULT_GYMV_GAME = "Temporal_AlteredBeast-v0"
DEFAULT_GYMV_SKILL_ID = "INSPECT/SETUP"

# Cross-cohort target bank locations (read for surface descriptions).
TARGET_BANK_PATHS: Dict[str, Dict[str, Path]] = {
    "web": {
        "webshop": REPO / "labeling/skill_bank_qa/run_webshop_20260510_044000/webshop/skill_bank.jsonl",
        "miniwob": REPO / "labeling/skill_bank_qa/run_20260506_184439/miniwob/skill_bank.jsonl",
    },
    "vr_image": {
        "tir_bench":         REPO / "labeling/skill_bank_qa/run_20260506_184439/tir_bench/skill_bank.jsonl",
        "visual_toolbench":  REPO / "labeling/skill_bank_qa/run_20260506_184439/visual_toolbench/skill_bank.jsonl",
    },
    "vr_video": {
        "video_holmes": REPO / "labeling/skill_bank_qa/run_20260506_184439/video_holmes/skill_bank.jsonl",
        "siv_bench":   REPO / "labeling/skill_bank_qa/run_20260506_184439/siv_bench/skill_bank.jsonl",
    },
    "env_wr_game": {
        "tetris":             REPO / "labeling/skill_bank_envwrappers/run_20260506_201030/env_wrappers/tetris/skill_bank.jsonl",
        "twenty_forty_eight": REPO / "labeling/skill_bank_envwrappers/run_20260506_201030/env_wrappers/twenty_forty_eight/skill_bank.jsonl",
        "super_mario":        REPO / "labeling/skill_bank_envwrappers/run_20260506_201030/env_wrappers/super_mario/skill_bank.jsonl",
        "candy_crush":        REPO / "labeling/skill_bank_envwrappers/run_20260506_201030/env_wrappers/candy_crush/skill_bank.jsonl",
    },
}


# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
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


def _find_skill_in_bank(
    bank_records: List[Dict[str, Any]], skill_id: str,
) -> Optional[Dict[str, Any]]:
    """Skill banks have two on-disk shapes:
       (a) flat dict with ``skill_id`` at top level, or
       (b) wrapped { "skill": {...}, "report": {...} }.
    This helper returns the inner skill dict regardless."""
    for rec in bank_records:
        if isinstance(rec, dict):
            if rec.get("skill_id") == skill_id:
                return rec
            inner = rec.get("skill") if isinstance(rec.get("skill"), dict) else None
            if inner is not None and inner.get("skill_id") == skill_id:
                return inner
    return None


def _peek_skill_record(bank_records: List[Dict[str, Any]]) -> List[str]:
    """Return the available skill IDs from a bank, for diagnostics."""
    ids: List[str] = []
    for rec in bank_records:
        if isinstance(rec, dict):
            sid = rec.get("skill_id") or (
                rec.get("skill", {}).get("skill_id") if isinstance(rec.get("skill"), dict) else None
            )
            if sid:
                ids.append(sid)
    return ids


def _normalize_skill_for_prompt(skill: Dict[str, Any]) -> Dict[str, Any]:
    """Bridge the on-disk shapes used by the gym_v bank vs the QA bank
    so the lift prompt is fed a consistent dict.  Mirrors the
    normalisation done by ``lift_skill_templates_gpt54.py``."""
    if "skill" in skill and isinstance(skill["skill"], dict):
        skill = skill["skill"]
    return {
        "skill_id":    skill.get("skill_id", ""),
        "name":        skill.get("name", ""),
        "strategic_description": skill.get("strategic_description", ""),
        "contract":    skill.get("contract") or {},
        "protocol":    skill.get("protocol"),
    }


# ---------------------------------------------------------------------------
def lift_with_gpt54(
    skill: Dict[str, Any], cohort: str, task: str, model: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """One-shot live lift.  Returns (template_dict, raw_response).
    template_dict is None on call/parse failure."""
    client = _get_openai_client()
    prompt = _build_prompt(skill=skill, cohort=cohort, task=task)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        max_tokens=600,
    )
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    parsed = _extract_json(text)
    if parsed is None:
        return None, text
    coerced = _coerce_template(parsed)
    return coerced, text


# ---------------------------------------------------------------------------
def _signature_diff(live: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
    """Per-step operator-level diff between the live lift and ground truth."""
    live_ops = [s["op"] for s in (live.get("template_steps") or [])]
    gt_ops   = [s["op"] for s in (gt.get("template_steps")   or [])]
    return {
        "signature_match":      live.get("template_signature") == gt.get("template_signature"),
        "live_signature":       live.get("template_signature"),
        "ground_truth_signature": gt.get("template_signature"),
        "live_ops":             live_ops,
        "ground_truth_ops":     gt_ops,
        "n_steps_match":        live_ops == gt_ops,
        "set_overlap":          (
            len(set(live_ops) & set(gt_ops)) / max(1, len(set(live_ops) | set(gt_ops)))
        ),
    }


# ---------------------------------------------------------------------------
def render_side_by_side(
    *,
    src_skill: Dict[str, Any], src_template: Dict[str, Any],
    src_cohort: str, src_task: str,
    tgt_skill: Optional[Dict[str, Any]], tgt_template: Dict[str, Any],
    tgt_cohort: str, tgt_task: str,
    width: int = 56,
) -> str:
    """Build a two-column comparison string."""
    def _wrap(s: str, w: int) -> List[str]:
        return textwrap.wrap(s or "(empty)", width=w) or ["(empty)"]

    def _col(label: str, s: str, w: int) -> List[str]:
        out = [f"{label}:"]
        out.extend("  " + ln for ln in _wrap(s, w - 2))
        return out

    L: List[str] = []
    L.append("─" * (width * 2 + 5))
    L.append(
        f"{'SOURCE (gym_v)'.center(width)}  │  {'TARGET (cross-cohort)'.center(width)}"
    )
    L.append(
        f"{(src_cohort+' / '+src_task).center(width)}  │  "
        f"{(tgt_cohort+' / '+tgt_task).center(width)}"
    )
    L.append("─" * (width * 2 + 5))

    src_lines = []
    src_lines += _col("skill_id", src_skill.get("skill_id", ""), width)
    src_lines += _col("name",     src_skill.get("name", ""), width)
    src_lines += _col("strategic_description",
                      src_skill.get("strategic_description", ""), width)
    src_lines += _col("preconditions",
                      "; ".join((src_skill.get("contract") or {}).get("preconditions", [])),
                      width)
    src_lines += _col("postconditions",
                      "; ".join((src_skill.get("contract") or {}).get("postconditions", [])),
                      width)

    tgt_lines = []
    if tgt_skill is None:
        tgt_lines += _col("skill_id", "(target bank not available)", width)
    else:
        tgt_lines += _col("skill_id", tgt_skill.get("skill_id", ""), width)
        tgt_lines += _col("name",     tgt_skill.get("name", ""), width)
        tgt_lines += _col("strategic_description",
                          tgt_skill.get("strategic_description", ""), width)
        tgt_lines += _col("preconditions",
                          "; ".join((tgt_skill.get("contract") or {}).get("preconditions", [])),
                          width)
        tgt_lines += _col("postconditions",
                          "; ".join((tgt_skill.get("contract") or {}).get("postconditions", [])),
                          width)

    n = max(len(src_lines), len(tgt_lines))
    src_lines += [""] * (n - len(src_lines))
    tgt_lines += [""] * (n - len(tgt_lines))
    for s, t in zip(src_lines, tgt_lines):
        L.append(f"{s.ljust(width)}  │  {t.ljust(width)}")

    L.append("─" * (width * 2 + 5))
    L.append(
        f"{('signature: ' + (src_template.get('template_signature','') or '?')).center(width)}"
        f"  │  "
        f"{('signature: ' + (tgt_template.get('template_signature','') or '?')).center(width)}"
    )
    L.append("─" * (width * 2 + 5))

    src_steps = src_template.get("template_steps") or []
    tgt_steps = tgt_template.get("template_steps") or []
    n_steps = max(len(src_steps), len(tgt_steps))
    for i in range(n_steps):
        ss = src_steps[i] if i < len(src_steps) else {"op": "", "predicate": ""}
        ts = tgt_steps[i] if i < len(tgt_steps) else {"op": "", "predicate": ""}
        s_line = f"[{i+1}] {ss.get('op','-'):<9} {ss.get('predicate','')}"
        t_line = f"[{i+1}] {ts.get('op','-'):<9} {ts.get('predicate','')}"
        for sx, tx in zip(_wrap(s_line, width), _wrap(t_line, width)):
            L.append(f"{sx.ljust(width)}  │  {tx.ljust(width)}")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gymv-bank-root", default=str(DEFAULT_GYMV_BANK))
    ap.add_argument("--template-run",   default=str(DEFAULT_TEMPLATE_RUN))
    ap.add_argument("--gymv-game",      default=DEFAULT_GYMV_GAME)
    ap.add_argument("--gymv-skill-id",  default=DEFAULT_GYMV_SKILL_ID)
    ap.add_argument("--target-cohort",  default="web",
                    choices=sorted(TARGET_BANK_PATHS.keys()))
    ap.add_argument("--model",          default=DEFAULT_MODEL)
    ap.add_argument("--no-llm",         action="store_true",
                    help="Skip the live GPT-5.4 call and just show ground truth.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    print("=" * 100)
    print("LAYER-C CROSS-DOMAIN DEMO")
    print("=" * 100)

    # ── Step 1+2 ── load source skill ──────────────────────────────
    src_bank_path = (
        Path(args.gymv_bank_root) / args.gymv_game / "skill_bank.jsonl"
    )
    if not src_bank_path.exists():
        print(f"ERROR: gym_v bank not found: {src_bank_path}")
        return 2
    src_records = _load_jsonl(src_bank_path)
    src_skill = _find_skill_in_bank(src_records, args.gymv_skill_id)
    if src_skill is None:
        ids = _peek_skill_record(src_records)
        print(f"ERROR: skill_id={args.gymv_skill_id!r} not in {src_bank_path}")
        print(f"available IDs (first 10): {ids[:10]}  total={len(ids)}")
        return 2
    src_skill = _normalize_skill_for_prompt(src_skill)
    print(f"\n[Step 1+2] Source gym_v skill")
    print(f"  bank      : {src_bank_path.relative_to(REPO)}")
    print(f"  task      : {args.gymv_game}")
    print(f"  skill_id  : {src_skill['skill_id']}")
    print(f"  name      : {src_skill.get('name', '')}")
    desc = src_skill.get("strategic_description", "") or "(empty)"
    print(f"  description (first 200 chars): {desc[:200]}")

    # ── Step 3 ── live lift ────────────────────────────────────────
    live: Optional[Dict[str, Any]] = None
    if not args.no_llm:
        print(f"\n[Step 3] Live lift via {args.model} (single LLM call)…")
        try:
            live, raw = lift_with_gpt54(
                src_skill, "gymv_game", args.gymv_game, args.model,
            )
        except Exception as exc:
            print(f"  LLM call failed: {exc}")
            live = None
        if live is None:
            print(f"  LLM did not return a usable template.  Skipping live diff.")
        else:
            print(f"  → live signature : {live['template_signature']}")
            for i, step in enumerate(live["template_steps"], 1):
                print(f"      [{i}] {step['op']:<9} {step['predicate']}")

    # ── Step 4 ── ground truth ─────────────────────────────────────
    idx = TemplateIndex.from_run(Path(args.template_run))
    gt_path = Path(args.template_run) / "gymv_game" / args.gymv_game / "template_bank.jsonl"
    gt_records = _load_jsonl(gt_path)
    gt = next(
        (r for r in gt_records if r.get("skill_id") == args.gymv_skill_id),
        None,
    )
    if gt is None:
        print(f"\nERROR: ground-truth template not found at {gt_path}")
        return 2
    print(f"\n[Step 4] Ground truth (canonical lift run)")
    print(f"  bank      : {gt_path.relative_to(REPO)}")
    print(f"  signature : {gt['template_signature']}")
    for i, step in enumerate(gt["template_steps"], 1):
        print(f"      [{i}] {step['op']:<9} {step['predicate']}")
    print(f"  self-reported transferable_to_cohorts: {gt.get('transferable_to_cohorts', [])}")

    if live is not None:
        diff = _signature_diff(live, gt)
        print(f"\n  ── live vs ground-truth diff ──")
        print(f"    signature_match : {diff['signature_match']}")
        print(f"    op_sequence_match : {diff['n_steps_match']}")
        print(f"    op_set_overlap    : {diff['set_overlap']:.2f}")

    use_template = live or gt
    sig = use_template["template_signature"]

    # ── Step 5 ── cross-cohort lookup ─────────────────────────────
    print(f"\n[Step 5] Retrieve cross-cohort 'intended skill in another domain'")
    print(f"  query signature : {sig}")
    cross = idx.lookup_by_signature(
        sig, exclude_task=args.gymv_game, exclude_cohort="gymv_game",
        prefer_cross_cohort=True, k=20,
    )
    cross_in_target = [r for r in cross if r.cohort == args.target_cohort]
    if not cross_in_target:
        print(f"  no candidates in cohort={args.target_cohort!r}; falling "
              f"back to any non-gymv cohort.")
        cross_in_target = cross
    if not cross_in_target:
        print("  ERROR: no cross-cohort candidates at all.")
        return 2

    chosen = cross_in_target[0]
    print(f"  candidates in target cohort '{args.target_cohort}': "
          f"{len(cross_in_target)}")
    print(f"  → CHOSEN: {chosen.cohort}/{chosen.task}/{chosen.skill_id}  "
          f"({chosen.skill_name})")

    # ── Step 6 ── load target's full bank record ─────────────────
    tgt_skill: Optional[Dict[str, Any]] = None
    tgt_paths = TARGET_BANK_PATHS.get(chosen.cohort, {})
    bank_path = tgt_paths.get(chosen.task)
    if bank_path is not None and bank_path.exists():
        tgt_records = _load_jsonl(bank_path)
        rec = _find_skill_in_bank(tgt_records, chosen.skill_id)
        if rec is not None:
            tgt_skill = _normalize_skill_for_prompt(rec)
    print(f"\n[Step 6] Loaded target skill bank record: "
          f"{'YES' if tgt_skill else 'NO (will only show abstract)'}")
    if tgt_skill is not None and bank_path is not None:
        print(f"  bank: {bank_path.relative_to(REPO)}")
        d = tgt_skill.get("strategic_description", "") or "(empty)"
        print(f"  description (first 200 chars): {d[:200]}")

    tgt_template = {
        "template_signature": chosen.template_signature,
        "template_steps": [dict(s) for s in chosen.template_steps],
    }

    # ── Step 7 ── side-by-side ───────────────────────────────────
    print(f"\n[Step 7] Side-by-side comparison")
    print(render_side_by_side(
        src_skill=src_skill, src_template=use_template,
        src_cohort="gymv_game", src_task=args.gymv_game,
        tgt_skill=tgt_skill, tgt_template=tgt_template,
        tgt_cohort=chosen.cohort, tgt_task=chosen.task,
    ))

    print("\n[Summary]")
    print(f"  source : {args.gymv_game} / {args.gymv_skill_id}")
    print(f"  target : {chosen.cohort} / {chosen.task} / {chosen.skill_id}")
    print(f"  shared signature: {sig}")
    if live is not None:
        print(f"  live abstract matches ground-truth signature: "
              f"{live['template_signature'] == gt['template_signature']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
