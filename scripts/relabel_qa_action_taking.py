"""Relabel ``intention_operator``/``intention_subgoal`` on QA + MiniWob + WebShop action_taking SFT rows.

Why
---
The existing QA + browser ``action_taking.jsonl`` rows in
``labeling/decision_sft_jsonl/run_multimodal_*/`` have their (operator,
subgoal) pair collapsed to ``EXECUTE/EXECUTE`` (or ``COMMIT/EXECUTE`` for
miniwob/webshop) — the original builder hard-coded
``intention_subgoal="EXECUTE"`` because at the time we did not have
hop-level intention labels.

We now have those labels:

* QA samples: ``labeling/qa_multihop_out/run_<ts>/<source>/<model>/samples_with_hops.jsonl``
  Each sample is decomposed into atomic hops, each tagged with
  ``operator``/``subgoal``/``note``.

* MiniWob / WebShop steps:
  ``labeling/qa_miniwob_labeled/run_<ts>/<browser_source>/<model>/<game>/rollouts.jsonl``
  Each experience carries ``intention_operator``/``intention_subgoal``/``intention_note``.
  The two browser sources sit side-by-side under the same labeled-run dir
  (peers ``miniwob/`` and ``webshop/``).

This script *patches the existing SFT rows in place* (or writes to a
sibling output) so the trainer sees per-instance intent signal:

* QA: each action_taking row corresponds to a single (sample_id, model)
  (we now key by ``episode_id`` which contains the sample_id).  We pick the
  *most informative* hop's (op, sg) — preferring REASON > COMPARE > VERIFY
  > INSPECT > COMMIT > others — and copy its ``operator``, ``subgoal`` and
  ``note`` into the row.  We also persist ``intention_hops`` with the full
  hop sequence so later trainers can use multi-hop signal if desired.

* Browser (miniwob, webshop): each row carries ``episode_id`` + ``step_idx``.
  We look up the matching experience in the labeled rollout (per-source
  subtree) and patch
  ``intention_operator``/``intention_subgoal``/``intention``/``intention_full``.

Usage
-----
    python -m scripts.relabel_qa_action_taking \\
        --sft-dir labeling/decision_sft_jsonl/run_multimodal_20260506_055105 \\
        --multihop-run labeling/qa_multihop_out/run_20260506_181625 \\
        --miniwob-run labeling/qa_miniwob_labeled/run_20260506_070722 \\
        --inplace
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("scripts.relabel_qa_action_taking")

QA_SOURCES = ("video_holmes", "siv_bench", "tir_bench", "visual_toolbench")
MINIWOB_SOURCE = "miniwob"
WEBSHOP_SOURCE = "webshop"
BROWSER_SOURCES = (MINIWOB_SOURCE, WEBSHOP_SOURCE)
DEFAULT_MODELS = ("gpt-5.4", "claude", "gemini", "qwen")

# The SFT row's ``source_model`` field uses the long version-specific name
# (``claude-4.6``/``gemini-3.1-pro``/``qwen3-vl-235b``), while the labeled
# rollout directories under ``qa_miniwob_labeled/<source>/<model>/`` use the
# short family name.  Mapping bridges the two so per-row lookups hit the
# direct (eid, step, model) entry instead of falling back to a cross-model
# proxy.  Identity is preserved when the SFT name already matches.
SFT_MODEL_TO_LABEL_DIR: Dict[str, str] = {
    "gpt-5.4": "gpt-5.4",
    "gpt5.4": "gpt-5.4",
    "claude-4.6": "claude",
    "claude4.6": "claude",
    "gemini-3.1-pro": "gemini",
    "gemini3.1-pro": "gemini",
    "qwen3-vl-235b": "qwen",
}

# Operator priority for choosing the "headline" (op, sg) for a QA row that
# represents a single MCQ commit. Higher = preferred.
_OP_PRIORITY: Dict[str, int] = {
    "REASON": 90,
    "COMPARE": 80,
    "VERIFY": 70,
    "INSPECT": 60,
    "COMMIT": 50,
    "TRACK": 40,
    "RECOVER": 30,
    "EXECUTE": 10,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# QA: build sample_id -> headline (op, sg, note) + full hop sequence
# ---------------------------------------------------------------------------

def _pick_headline_hop(hops: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the most informative hop to represent a QA sample's intent."""
    if not hops:
        return None
    best = None
    best_score = -1
    for h in hops:
        op = (h.get("operator") or "").upper()
        sg = (h.get("subgoal") or "").upper()
        prio = _OP_PRIORITY.get(op, 0)
        # Boost terminal hops slightly — they're usually the commit step.
        is_last = (h is hops[-1])
        # Boost hops with a non-empty note.
        has_note = bool((h.get("note") or "").strip())
        score = prio + (3 if is_last else 0) + (2 if has_note else 0)
        if score > best_score:
            best_score = score
            best = h
    return best


def _load_qa_intent_map(
    multihop_run: Path, sources: List[str], models: List[str],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Returns ``(source, sample_id) -> {op, sg, note, hops_brief}``.

    We collapse across models — multiple frontier models often disagree on
    the exact decomposition, so we pick the *best* hops list (operator
    priority on the last hop) across models for a given sample_id.
    """
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for source in sources:
        if source not in QA_SOURCES:
            continue
        for model in models:
            in_path = multihop_run / source / model / "samples_with_hops.jsonl"
            if not in_path.exists():
                continue
            for sample in _iter_jsonl(in_path):
                sid = str(
                    sample.get("sample_id") or sample.get("task_id")
                    or sample.get("question_id") or ""
                )
                if not sid:
                    continue
                hops = list(sample.get("hops") or [])
                if not hops:
                    continue
                headline = _pick_headline_hop(hops)
                if headline is None:
                    continue
                op = (headline.get("operator") or "EXECUTE").upper()
                sg = (headline.get("subgoal") or "EXECUTE").upper()
                note = (headline.get("note") or "").strip()
                hops_brief = [
                    {
                        "step": h.get("step"),
                        "operator": (h.get("operator") or "").upper(),
                        "subgoal": (h.get("subgoal") or "").upper(),
                        "note": (h.get("note") or "")[:140],
                    }
                    for h in hops
                ]
                key = (source, sid)
                # If we already have a candidate from an earlier model, keep
                # the one with higher operator priority (headline op).
                prev = out.get(key)
                if prev is not None:
                    prev_prio = _OP_PRIORITY.get(prev.get("operator", ""), 0)
                    new_prio = _OP_PRIORITY.get(op, 0)
                    if new_prio <= prev_prio:
                        continue
                out[key] = {
                    "operator": op,
                    "subgoal": sg,
                    "note": note,
                    "intention_hops": hops_brief,
                    "source_model": model,
                }
    return out


# ---------------------------------------------------------------------------
# MiniWob: build (episode_id, step_idx) -> {op, sg, note}
# ---------------------------------------------------------------------------

def _load_browser_intent_map(
    labeled_run: Path, models: List[str], *, source: str,
) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """Returns ``(episode_id, step_idx, model_dir) -> {op, sg, note}``.

    *source* is one of :data:`BROWSER_SOURCES`.  Both browser sources use
    the same labeled-rollout layout
    (``<run>/<source>/<model>/<game>/rollouts.jsonl``); only the
    sub-directory name differs.

    ``episode_id`` here is the **base** browser episode_id (UUID) *without*
    the ``__model`` suffix.  We additionally key by model so the relabel
    driver can prefer a same-model match before falling back to a different
    model that ran the same task seed.
    """
    out: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    src_root = labeled_run / source
    if not src_root.is_dir():
        return out
    for model in models:
        mdir = src_root / model
        if not mdir.is_dir():
            continue
        for game_dir in sorted(mdir.iterdir()):
            if not game_dir.is_dir():
                continue
            for f in sorted(game_dir.glob("*.jsonl")):
                for ep in _iter_jsonl(f):
                    base_eid = str(ep.get("episode_id") or f.stem)
                    for i, exp in enumerate(ep.get("experiences") or []):
                        op = (exp.get("intention_operator") or "").upper()
                        sg = (exp.get("intention_subgoal") or "").upper()
                        note = (exp.get("intention_note") or "").strip()
                        if not op:
                            continue
                        idx = exp.get("idx")
                        try:
                            idx_i = int(idx) if idx is not None else i
                        except Exception:
                            idx_i = i
                        out[(base_eid, idx_i, model)] = {
                            "operator": op,
                            "subgoal": sg or "EXECUTE",
                            "note": note,
                        }
    return out


def _load_miniwob_intent_map(
    miniwob_run: Path, models: List[str],
) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """Back-compat alias delegating to :func:`_load_browser_intent_map`."""
    return _load_browser_intent_map(miniwob_run, models, source=MINIWOB_SOURCE)


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def _trim(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _patch_qa_row(
    row: Dict[str, Any], info: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Apply (op, sg, note) headline + hops_brief to a QA action_taking row."""
    op = info["operator"]
    sg = info["subgoal"]
    note = info.get("note") or ""
    intention_short = f"[{op}/{sg}] {_trim(note, 160)}"
    intention_full = f"[{op}/{sg}] {_trim(note, 400)}"
    changed = (
        row.get("intention_operator") != op
        or row.get("intention_subgoal") != sg
        or row.get("intention") != intention_short
    )
    row["intention_operator"] = op
    row["intention_subgoal"] = sg
    row["intention"] = intention_short
    row["intention_full"] = intention_full
    if info.get("intention_hops"):
        row["intention_hops"] = info["intention_hops"]
    return row, changed


def _patch_browser_row(
    row: Dict[str, Any],
    intent_map: Dict[Tuple[str, int, str], Dict[str, Any]],
    *,
    by_eid_step: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
    match_kind_counter: Optional[Counter] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Look up per-step (op, sg) for any browser row by (episode_id, step_idx).

    Works identically for miniwob and webshop because both sources share
    the same SFT row schema (``episode_id`` ends in ``__<source_model>``,
    ``step_idx`` is an int, etc.).

    Tracks where the matched info came from:
      * ``direct``    — same (eid, step, model)
      * ``fallback``  — same (eid, step) but a different model
      * ``unmatched`` — neither
    """
    eid_full = str(row.get("episode_id") or "")
    base_eid = eid_full
    model = ""
    if "__" in eid_full:
        base_eid, model = eid_full.rsplit("__", 1)
    # Normalise SFT-style ``model`` (long name) to labeled-rollout dir name
    # (short name) so the direct lookup hits the per-model entry.
    model_dir = SFT_MODEL_TO_LABEL_DIR.get(model, model)
    step_idx = int(row.get("step_idx") or 0)
    info = intent_map.get((base_eid, step_idx, model_dir))
    match_kind = "direct" if info is not None else None
    # Fallback: same episode rolled-out by a different model.  miniwob
    # episodes are deterministic per task seed, so the per-step intent is
    # similar enough across models to be a useful proxy when the SFT and
    # labeled-rollout snapshots used different model lineups.
    if info is None and by_eid_step is not None:
        info = by_eid_step.get((base_eid, step_idx))
        if info is not None:
            match_kind = "fallback"
    if match_kind_counter is not None:
        match_kind_counter[match_kind or "unmatched"] += 1
    if info is None:
        return row, False
    op = info["operator"]
    sg = info["subgoal"]
    note = info["note"]
    intention_short = f"[{op}/{sg}] {_trim(note, 160)}"
    intention_full = f"[{op}/{sg}] {_trim(note, 400)}"
    changed = (
        row.get("intention_operator") != op
        or row.get("intention_subgoal") != sg
        or row.get("intention") != intention_short
    )
    row["intention_operator"] = op
    row["intention_subgoal"] = sg
    row["intention"] = intention_short
    row["intention_full"] = intention_full
    return row, changed


def _patch_miniwob_row(
    row: Dict[str, Any],
    intent_map: Dict[Tuple[str, int, str], Dict[str, Any]],
    *,
    by_eid_step: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
    match_kind_counter: Optional[Counter] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Back-compat alias for :func:`_patch_browser_row`."""
    return _patch_browser_row(
        row, intent_map,
        by_eid_step=by_eid_step,
        match_kind_counter=match_kind_counter,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sft-dir", type=Path, required=True,
                   help="Path to labeling/decision_sft_jsonl/run_multimodal_<ts>/")
    p.add_argument("--multihop-run", type=Path, required=True,
                   help="Path to labeling/qa_multihop_out/run_<ts>/")
    p.add_argument("--miniwob-run", type=Path, required=True,
                   help="Path to labeling/qa_miniwob_labeled/run_<ts>/.  This "
                        "is a single labeled-run directory that may carry one "
                        "or both browser-source subtrees (``miniwob/`` and "
                        "``webshop/``); both are read when present.")
    p.add_argument("--webshop-run", type=Path, default=None,
                   help="Optional override for the webshop labeled-run dir.  "
                        "Defaults to --miniwob-run (which is the canonical "
                        "single-tree layout).  Use this only when you want "
                        "to source webshop labels from a separate timestamped "
                        "run than miniwob's.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="If set, write patched rows to <out-dir>/<bench>/action_taking.jsonl. "
                        "If --inplace also given, both writes happen.")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite the existing action_taking.jsonl files in --sft-dir. "
                        "Backups are saved to <bench>/action_taking.jsonl.bak.<ts>.")
    p.add_argument("--sources", type=str, nargs="+",
                   default=list(QA_SOURCES) + list(BROWSER_SOURCES))
    p.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.setLevel(logging.INFO)

    sft_dir: Path = args.sft_dir.resolve()
    if not sft_dir.is_dir():
        print(f"[relabel_qa_action_taking] missing sft-dir: {sft_dir}", file=sys.stderr)
        return 2
    if not args.inplace and args.out_dir is None:
        print("[relabel_qa_action_taking] one of --inplace or --out-dir required", file=sys.stderr)
        return 2

    qa_intent_map = _load_qa_intent_map(
        args.multihop_run.resolve(), list(args.sources), list(args.models)
    )
    logger.info("Loaded QA intent map: %d (source, sample_id) keys",
                 len(qa_intent_map))

    # Each browser source gets its own (eid, step, model) map and a
    # model-agnostic fallback derived from it.  We index by source so the
    # bench-loop below can pick the right map without re-walking the
    # labeled trees.
    browser_maps: Dict[str, Dict[Tuple[str, int, str], Dict[str, Any]]] = {}
    browser_fallbacks: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    for browser_src in BROWSER_SOURCES:
        if browser_src not in args.sources:
            continue
        if browser_src == WEBSHOP_SOURCE and args.webshop_run is not None:
            run_dir = args.webshop_run.resolve()
        else:
            run_dir = args.miniwob_run.resolve()
        intent_map = _load_browser_intent_map(
            run_dir, list(args.models), source=browser_src,
        )
        fallback: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for (eid, step, _model), info in intent_map.items():
            prev = fallback.get((eid, step))
            if prev is None or len(info.get("note", "")) > len(prev.get("note", "")):
                fallback[(eid, step)] = info
        browser_maps[browser_src] = intent_map
        browser_fallbacks[browser_src] = fallback
        logger.info("Loaded %s intent map: %d (eid, step, model) keys; "
                    "%d (eid, step) fallback keys",
                     browser_src, len(intent_map), len(fallback))
    # Aliases preserved for back-compat with anyone importing the module.
    miniwob_intent_map = browser_maps.get(MINIWOB_SOURCE, {})
    miniwob_by_eid_step = browser_fallbacks.get(MINIWOB_SOURCE, {})

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_per_bench: Dict[str, Dict[str, Any]] = {}

    for bench_dir in sorted(sft_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        if bench not in args.sources:
            continue
        in_path = bench_dir / "action_taking.jsonl"
        if not in_path.exists():
            continue

        rows = list(_iter_jsonl(in_path))
        n_total = len(rows)
        n_patched = 0
        n_unmatched = 0
        op_after: Counter = Counter()
        match_kind_counter: Counter = Counter()

        for row in rows:
            patched = False
            if bench in QA_SOURCES:
                # episode_id format from build_multimodal_decision_sft.py is
                # ``<bench>__<sample_id>__<source_model>``.  ``sample_id`` may
                # itself contain ``__`` (e.g. siv_bench paths), so strip the
                # known bench prefix and the trailing ``__<source_model>``.
                eid = str(row.get("episode_id") or "")
                src_model = str(row.get("source_model") or "")
                sid = eid
                if eid.startswith(f"{bench}__"):
                    sid = eid[len(bench) + 2:]
                if src_model and sid.endswith(f"__{src_model}"):
                    sid = sid[: -len(src_model) - 2]
                info = qa_intent_map.get((bench, sid))
                if info is not None:
                    row, changed = _patch_qa_row(row, info)
                    patched = True
                    if changed:
                        n_patched += 1
            elif bench in BROWSER_SOURCES:
                intent_map = browser_maps.get(bench, {})
                fallback = browser_fallbacks.get(bench, {})
                row, changed = _patch_browser_row(
                    row, intent_map,
                    by_eid_step=fallback,
                    match_kind_counter=match_kind_counter,
                )
                patched = changed
                if changed:
                    n_patched += 1
            if not patched:
                n_unmatched += 1
            op_after[(row.get("intention_operator", "?"),
                       row.get("intention_subgoal", "?"))] += 1

        summary_per_bench[bench] = {
            "n_total": n_total,
            "n_patched": n_patched,
            "n_unmatched": n_unmatched,
            "top_op_sg_after": op_after.most_common(8),
            "match_kind": dict(match_kind_counter) if match_kind_counter else None,
        }
        if match_kind_counter:
            logger.info("[%s] %d/%d patched (unmatched=%d) | match_kind=%s. top after: %s",
                         bench, n_patched, n_total, n_unmatched,
                         dict(match_kind_counter), op_after.most_common(5))
        else:
            logger.info("[%s] %d/%d patched (unmatched=%d). top after: %s",
                         bench, n_patched, n_total, n_unmatched,
                         op_after.most_common(5))

        # Write outputs.
        if args.inplace:
            backup = in_path.with_suffix(f".jsonl.bak.{ts}")
            if not backup.exists():
                in_path.rename(backup)
            _write_jsonl(in_path, rows)
        if args.out_dir is not None:
            out_path = args.out_dir.resolve() / bench / "action_taking.jsonl"
            _write_jsonl(out_path, rows)

    summary = {
        "sft_dir": str(sft_dir),
        "multihop_run": str(args.multihop_run),
        "miniwob_run": str(args.miniwob_run),
        "inplace": args.inplace,
        "out_dir": str(args.out_dir) if args.out_dir else None,
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "per_bench": summary_per_bench,
    }
    summary_path = sft_dir / f"_qa_action_taking_relabel_summary.{ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print("=" * 70)
    print(f"[relabel_qa_action_taking] DONE")
    for bench, stat in summary_per_bench.items():
        match_kind = stat.get("match_kind")
        line = (f"  {bench}: patched={stat['n_patched']}/{stat['n_total']}, "
                f"unmatched={stat['n_unmatched']}")
        if match_kind:
            line += f", match_kind={match_kind}"
        print(line)
        for (op, sg), n in stat["top_op_sg_after"][:5]:
            print(f"    {op:>10s}/{sg:<14s}  {n:>5d}")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
