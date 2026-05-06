#!/usr/bin/env python
"""Skill-query labeling for QA + MiniWob (Stage 3 of the QA pipeline).

This is the QA/MiniWob analogue of
``labeling/label_skill_actions_gpt54.py``.  The original driver only
understands GymV / env_wrappers episode JSONs; here we adapt the input
shape to:

* QA hops written by ``label_qa_multihop_gpt54.py``
  (``samples_with_hops.jsonl`` with ``hops: [{step, operator, subgoal,
  note, evidence, ...}]``).
* MiniWob ``rollouts.jsonl`` whose ``experiences[*]`` already carry
  ``intentions = "[OP/SG] note"``.

For every per-source bank (built by ``build_skillbank_qa_gpt54.py``)
we attach a ``skill_query`` block to each hop (QA) or experience
(miniwob) by calling :func:`labeling.label_skill_actions_gpt54.query_step`.
The slimming, fallback, and JSON-serialisation logic is identical to
the existing GymV pipeline so downstream SFT readers stay unchanged.

Output mirror layout (per source × model):

    labeling/skill_actions_qa_out/run_<ts>/<source>/<model>/<bucket>/<file>.jsonl

For QA:    bucket="qa",      file="samples_with_skill_query.jsonl"
For miniwob: bucket=<game>,  file="rollouts_with_skill_query.jsonl"

Plus a top-level ``_run_summary.json`` describing per-pair coverage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path / API key bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for p in (CODEBASE_ROOT, WORKSPACE_ROOT):
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

from labeling.label_skill_actions_gpt54 import (  # type: ignore
    DEFAULT_TOP_K,
    LoadedBank,
    SkillBankMVP,
    SkillQueryEngine,
    query_step,
)

# A single shared GPU-backed embedder, reused across every per-source
# engine.  Without this each ``SkillQueryEngine(bank)`` defaults to a
# CPU embedder (see ``skill_agents.query.SkillQueryEngine.__init__``)
# and a single dense-similarity query takes ~1.5-2s — utterly
# impractical for 40k+ unit/hop queries across 5 sources × 4 models.
# Forcing the underlying ``sentence_transformers`` model onto cuda:0
# brings per-query latency to <10 ms (skill embeddings are pre-built
# during ``_build_index`` once, then every query is one matmul).
_SHARED_GPU_EMBEDDER: Any = None


def _get_shared_gpu_embedder() -> Any:
    """Lazy-initialise a GPU-resident text embedder shared across engines.

    Falls back to whatever the engine constructs (cpu) when GPU init
    fails — we still want to *try* labeling rather than abort.
    """
    global _SHARED_GPU_EMBEDDER
    if _SHARED_GPU_EMBEDDER is not None:
        return _SHARED_GPU_EMBEDDER
    try:
        import os
        try:
            import torch  # noqa: F401  (force CUDA init early)
        except Exception:
            pass
        # Pin to the first visible GPU.  ``label_skill_actions_qa`` is
        # not a torch trainer — we have plenty of headroom on cuda:0.
        device = os.environ.get("LABEL_SKILL_QUERY_DEVICE", "cuda:0")
        from rag import get_text_embedder  # type: ignore
        _SHARED_GPU_EMBEDDER = get_text_embedder(device=device, shared=False)
        logger.info("loaded shared GPU embedder on device=%s", device)
    except Exception as exc:
        logger.warning(
            "GPU embedder init failed (%s) — engines will fall back to CPU.",
            exc,
        )
        _SHARED_GPU_EMBEDDER = None
    return _SHARED_GPU_EMBEDDER

logger = logging.getLogger("labeling.label_skill_actions_qa")

DEFAULT_WORKERS = 8
QA_SOURCES = ("video_holmes", "siv_bench", "tir_bench", "visual_toolbench")
MINIWOB_SOURCE = "miniwob"


# ---------------------------------------------------------------------------
# Bank loader (per-source; one bank per source covers all models/buckets)
# ---------------------------------------------------------------------------

def _load_bank_for_source(bank_run: Path, source: str) -> Optional[LoadedBank]:
    if SkillBankMVP is None or SkillQueryEngine is None:
        logger.error("skill bank library unavailable — cannot label.")
        return None
    bank_path = bank_run / source / "skill_bank.jsonl"
    if not bank_path.exists():
        logger.warning("[%s] no skill bank at %s", source, bank_path)
        return None
    try:
        bank = SkillBankMVP(path=str(bank_path))
        bank.load()
    except Exception as exc:
        logger.error("[%s] bank load failed: %s", source, exc)
        return None
    n = len(bank)
    if n == 0:
        logger.warning("[%s] bank loaded but empty", source)
        return None
    try:
        embedder = _get_shared_gpu_embedder()
        engine = (
            SkillQueryEngine(bank, embedder=embedder)
            if embedder is not None
            else SkillQueryEngine(bank)
        )
    except Exception as exc:
        logger.error("[%s] SkillQueryEngine init failed: %s", source, exc)
        return None
    logger.info("[%s] bank loaded: %d skills", source, n)
    return LoadedBank(bank=bank, engine=engine, n_skills=n, bank_path=bank_path)


# ---------------------------------------------------------------------------
# Pseudo-step adapters
# ---------------------------------------------------------------------------

def _hop_to_step(hop: Dict[str, Any], *, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one QA hop into a step-shaped dict that ``query_step`` expects.

    The bank engine reads ``intentions`` (for the ``[OP/SG] note`` text
    signal) and ``summary_state`` (for the predicate map).  We populate
    both deterministically from the hop record:

    * ``intentions``     ← ``"[OP/SG] note"``
    * ``summary_state``  ← built from question + evidence + tool_call so
      the engine has tokens to match against
      ``execution_hint.common_preconditions`` predicates.
    """
    op = str(hop.get("operator") or "COMMIT")
    sg = str(hop.get("subgoal") or "ANSWER")
    note = str(hop.get("note") or "")
    intent = f"[{op}/{sg}] {note}".strip()

    q = (sample.get("question") or sample.get("query") or "")[:200]
    evidence = str(hop.get("evidence") or "")
    tool = str(hop.get("tool_call") or "")
    correct = sample.get("correct")
    modality = sample.get("modality") or ""

    summary_parts: List[str] = []
    if q:
        summary_parts.append(f"question={q}")
    if evidence:
        summary_parts.append(f"evidence={evidence}")
    if tool:
        summary_parts.append(f"tool_invoked={tool}")
    if modality:
        summary_parts.append(f"modality={modality}")
    if correct is not None:
        summary_parts.append(f"correct={'true' if correct else 'false'}")
    summary_state = " | ".join(summary_parts)

    raw_step = hop.get("step", 0)
    try:
        hop_step_idx = int(raw_step) if raw_step is not None else 0
    except (TypeError, ValueError):
        hop_step_idx = 0
    return {
        "intentions": intent,
        "summary_state": summary_state,
        # carry ids for downstream join
        "step_idx": hop_step_idx,
    }


def _exp_to_step(exp: Dict[str, Any], *, episode: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a miniwob experience to ``query_step`` shape.

    The miniwob rollout already has ``intentions`` and a
    ``summary_state`` field — but ``summary_state`` is sometimes empty.
    Fall back to ``goal/task`` so the engine has something to retrieve
    against.
    """
    intent = (exp.get("intentions") or "").strip()
    if not intent:
        op = exp.get("intention_operator") or exp.get("intention_tag") or "COMMIT"
        sg = exp.get("intention_subgoal") or "EXECUTE"
        note = exp.get("intention_note") or ""
        intent = f"[{op}/{sg}] {note}".strip()

    summary_state = (exp.get("summary_state") or "").strip()
    if not summary_state:
        goal = (exp.get("goal") or episode.get("task") or episode.get("query") or "")
        action = exp.get("action") or exp.get("action_text") or ""
        parts: List[str] = []
        if goal:
            parts.append(f"goal={str(goal)[:160]}")
        if action:
            parts.append(f"action={str(action)[:80]}")
        summary_state = " | ".join(parts)

    # query_step looks at exp.get("step_idx") only for logging — pass through.
    # NB: ``idx`` is often the integer 0 (a valid step index that is falsy),
    # while ``step_id`` is a UUID string that cannot be cast to int. Use a
    # presence-based fallback chain and a safe int cast so neither breaks the
    # episode.
    raw_idx = exp.get("idx")
    if raw_idx is None:
        raw_idx = exp.get("step_idx")
    if raw_idx is None:
        raw_idx = exp.get("step_id")  # may be a UUID; safe-cast below
    try:
        step_idx = int(raw_idx) if raw_idx is not None else 0
    except (TypeError, ValueError):
        step_idx = 0
    return {
        "intentions": intent,
        "summary_state": summary_state,
        "step_idx": step_idx,
    }


# ---------------------------------------------------------------------------
# Per-pair processors
# ---------------------------------------------------------------------------

@dataclass
class PairStats:
    n_units: int = 0       # samples (QA) or episodes (miniwob)
    n_steps: int = 0       # hops or experiences
    n_with_skill: int = 0  # had a non-empty selected_skill_id
    n_failed: int = 0
    selected_counter: Counter = None  # type: ignore

    def __post_init__(self):
        if self.selected_counter is None:
            self.selected_counter = Counter()


def _process_qa_sample(
    sample: Dict[str, Any], loaded: LoadedBank, *, top_k: int,
) -> Tuple[Dict[str, Any], int, int, Counter, int]:
    """Attach skill_query to every hop in *sample*; return (sample, n_steps, n_with, sel_counter, n_failed)."""
    out = dict(sample)
    hops = list(out.get("hops") or [])
    n_with = 0
    n_failed = 0
    sel: Counter = Counter()

    enriched_hops: List[Dict[str, Any]] = []
    for hop in hops:
        pseudo = _hop_to_step(hop, sample=sample)
        try:
            skills_field, sq_field = query_step(pseudo, loaded, top_k=top_k)
        except Exception as exc:
            sq_field = {
                "candidates": [],
                "selected_skill_id": None,
                "empty": True,
                "error": f"{type(exc).__name__}: {exc}",
                "selection_method": "skill_query_engine_select_failed",
            }
            skills_field = {}
            n_failed += 1

        new_hop = dict(hop)
        new_hop["skill_query"] = sq_field
        if skills_field:
            new_hop["skills"] = skills_field
        sid = sq_field.get("selected_skill_id")
        if sid:
            n_with += 1
            sel[sid] += 1
        if sq_field.get("error"):
            n_failed += 1
        enriched_hops.append(new_hop)

    out["hops"] = enriched_hops
    out["n_hops_with_skill"] = n_with
    return out, len(hops), n_with, sel, n_failed


def _process_miniwob_episode(
    episode: Dict[str, Any], loaded: LoadedBank, *, top_k: int,
) -> Tuple[Dict[str, Any], int, int, Counter, int]:
    out = dict(episode)
    exps = list(out.get("experiences") or [])
    n_with = 0
    n_failed = 0
    sel: Counter = Counter()

    enriched_exps: List[Dict[str, Any]] = []
    for exp in exps:
        pseudo = _exp_to_step(exp, episode=episode)
        try:
            skills_field, sq_field = query_step(pseudo, loaded, top_k=top_k)
        except Exception as exc:
            sq_field = {
                "candidates": [],
                "selected_skill_id": None,
                "empty": True,
                "error": f"{type(exc).__name__}: {exc}",
                "selection_method": "skill_query_engine_select_failed",
            }
            skills_field = {}
            n_failed += 1

        new_exp = dict(exp)
        new_exp["skill_query"] = sq_field
        if skills_field:
            new_exp["skills"] = skills_field
        sid = sq_field.get("selected_skill_id")
        if sid:
            n_with += 1
            sel[sid] += 1
        if sq_field.get("error"):
            n_failed += 1
        enriched_exps.append(new_exp)

    out["experiences"] = enriched_exps
    out["n_steps_with_skill"] = n_with
    return out, len(exps), n_with, sel, n_failed


def _safe_iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def _process_qa_pair(
    *, multihop_run: Path, source: str, model: str,
    output_dir: Path, loaded: LoadedBank, top_k: int, workers: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    in_path = multihop_run / source / model / "samples_with_hops.jsonl"
    if not in_path.exists():
        return {"source": source, "model": model, "skipped": True,
                "reason": "input missing"}
    out_subdir = output_dir / source / model / "qa"
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / "samples_with_skill_query.jsonl"

    samples = list(_safe_iter_jsonl(in_path))
    if limit is not None:
        samples = samples[:limit]
    n_units = len(samples)

    stats = PairStats()
    stats.n_units = n_units
    t0 = time.time()
    with out_path.open("w") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_process_qa_sample, s, loaded, top_k=top_k): i
                    for i, s in enumerate(samples)}
            for fu in as_completed(futs):
                try:
                    out, n_steps, n_with, sel, n_failed = fu.result()
                except Exception as exc:
                    logger.warning("sample failed: %s", exc)
                    stats.n_failed += 1
                    continue
                stats.n_steps += n_steps
                stats.n_with_skill += n_with
                stats.n_failed += n_failed
                stats.selected_counter.update(sel)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    elapsed = time.time() - t0

    return {
        "source": source, "model": model, "bucket": "qa",
        "input_path": str(in_path),
        "output_path": str(out_path),
        "n_units": stats.n_units,
        "n_steps": stats.n_steps,
        "n_with_skill": stats.n_with_skill,
        "n_failed": stats.n_failed,
        "coverage": (stats.n_with_skill / stats.n_steps) if stats.n_steps else 0.0,
        "top_skills": dict(stats.selected_counter.most_common(8)),
        "elapsed_seconds": round(elapsed, 1),
    }


def _process_miniwob_pair(
    *, miniwob_run: Path, model: str, game_dir: Path,
    output_dir: Path, loaded: LoadedBank, top_k: int, workers: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    game = game_dir.name
    in_files = sorted(game_dir.glob("*.jsonl"))
    if not in_files:
        return {"source": MINIWOB_SOURCE, "model": model, "bucket": game,
                "skipped": True, "reason": "no rollouts"}
    out_subdir = output_dir / MINIWOB_SOURCE / model / game
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / "rollouts_with_skill_query.jsonl"

    episodes: List[Dict[str, Any]] = []
    for f in in_files:
        episodes.extend(_safe_iter_jsonl(f))
    if limit is not None:
        episodes = episodes[:limit]
    n_units = len(episodes)

    stats = PairStats()
    stats.n_units = n_units
    t0 = time.time()
    with out_path.open("w") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_process_miniwob_episode, ep, loaded, top_k=top_k): i
                    for i, ep in enumerate(episodes)}
            for fu in as_completed(futs):
                try:
                    out, n_steps, n_with, sel, n_failed = fu.result()
                except Exception as exc:
                    logger.warning("episode failed: %s", exc)
                    stats.n_failed += 1
                    continue
                stats.n_steps += n_steps
                stats.n_with_skill += n_with
                stats.n_failed += n_failed
                stats.selected_counter.update(sel)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    elapsed = time.time() - t0

    return {
        "source": MINIWOB_SOURCE, "model": model, "bucket": game,
        "input_files": [str(f) for f in in_files],
        "output_path": str(out_path),
        "n_units": stats.n_units,
        "n_steps": stats.n_steps,
        "n_with_skill": stats.n_with_skill,
        "n_failed": stats.n_failed,
        "coverage": (stats.n_with_skill / stats.n_steps) if stats.n_steps else 0.0,
        "top_skills": dict(stats.selected_counter.most_common(8)),
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

DEFAULT_MODELS = ("gpt-5.4", "claude", "gemini", "qwen")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-3 skill query labeling for QA + MiniWob.",
    )
    p.add_argument("--bank-run", type=Path, required=True,
                   help="Output dir of build_skillbank_qa_gpt54 "
                        "(labeling/skill_bank_qa/run_<ts>).")
    p.add_argument("--multihop-run", type=Path, default=None,
                   help="QA multihop run dir; required for QA sources.")
    p.add_argument("--miniwob-run", type=Path, default=None,
                   help="qa_miniwob_labeled run dir; required for miniwob.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: labeling/skill_actions_qa_out/run_<utc-ts>.")
    p.add_argument("--sources", type=str, nargs="+",
                   default=list(QA_SOURCES) + [MINIWOB_SOURCE])
    p.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.setLevel(logging.INFO)

    bank_run: Path = args.bank_run.resolve()
    if not bank_run.is_dir():
        print(f"[label_skill_actions_qa] bank run missing: {bank_run}",
              file=sys.stderr)
        return 2

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = (CODEBASE_ROOT / "labeling" / "skill_actions_qa_out" / f"run_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "bank_run": str(bank_run),
        "multihop_run": str(args.multihop_run.resolve()) if args.multihop_run else None,
        "miniwob_run": str(args.miniwob_run.resolve()) if args.miniwob_run else None,
        "output_dir": str(output_dir),
        "sources": list(args.sources),
        "models": list(args.models),
        "top_k": args.top_k,
        "workers": args.workers,
        "limit": args.limit,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "argv": sys.argv,
    }
    (output_dir / "_run_meta.json").write_text(json.dumps(run_meta, indent=2))

    # Pre-load all banks once.
    banks: Dict[str, LoadedBank] = {}
    for src in args.sources:
        b = _load_bank_for_source(bank_run, src)
        if b is not None:
            banks[src] = b
    if not banks:
        print("[label_skill_actions_qa] no banks loaded — nothing to do.",
              file=sys.stderr)
        return 2

    summaries: List[Dict[str, Any]] = []

    # QA sources first.
    qa_sources_to_run = [s for s in args.sources if s in QA_SOURCES and s in banks]
    if qa_sources_to_run:
        if args.multihop_run is None or not args.multihop_run.is_dir():
            logger.warning("--multihop-run not provided; skipping QA sources.")
        else:
            mh = args.multihop_run.resolve()
            for source in qa_sources_to_run:
                loaded = banks[source]
                for model in args.models:
                    try:
                        s = _process_qa_pair(
                            multihop_run=mh, source=source, model=model,
                            output_dir=output_dir, loaded=loaded,
                            top_k=args.top_k, workers=args.workers,
                            limit=args.limit,
                        )
                    except Exception as exc:
                        logger.error("[%s/%s] FAILED: %s", source, model, exc)
                        traceback.print_exc()
                        s = {"source": source, "model": model,
                             "error": f"{type(exc).__name__}: {exc}"}
                    summaries.append(s)
                    logger.info(
                        "[%s/%s] %d units, %d hops, %d w/ skill (%.1f%%), %.1fs",
                        s.get("source"), s.get("model"),
                        s.get("n_units", 0), s.get("n_steps", 0),
                        s.get("n_with_skill", 0), 100 * s.get("coverage", 0.0),
                        s.get("elapsed_seconds", 0.0),
                    )

    # MiniWob.
    if MINIWOB_SOURCE in args.sources and MINIWOB_SOURCE in banks:
        if args.miniwob_run is None or not args.miniwob_run.is_dir():
            logger.warning("--miniwob-run not provided; skipping miniwob.")
        else:
            mw = args.miniwob_run.resolve()
            mw_dir = mw / "miniwob"
            loaded = banks[MINIWOB_SOURCE]
            if mw_dir.is_dir():
                for model in args.models:
                    mdir = mw_dir / model
                    if not mdir.is_dir():
                        continue
                    for game_dir in sorted(mdir.iterdir()):
                        if not game_dir.is_dir():
                            continue
                        try:
                            s = _process_miniwob_pair(
                                miniwob_run=mw, model=model, game_dir=game_dir,
                                output_dir=output_dir, loaded=loaded,
                                top_k=args.top_k, workers=args.workers,
                                limit=args.limit,
                            )
                        except Exception as exc:
                            logger.error("[miniwob/%s/%s] FAILED: %s",
                                         model, game_dir.name, exc)
                            traceback.print_exc()
                            s = {"source": MINIWOB_SOURCE, "model": model,
                                 "bucket": game_dir.name,
                                 "error": f"{type(exc).__name__}: {exc}"}
                        summaries.append(s)
                        logger.info(
                            "[miniwob/%s/%s] %d eps, %d steps, %d w/ skill (%.1f%%), %.1fs",
                            s.get("model"), s.get("bucket"),
                            s.get("n_units", 0), s.get("n_steps", 0),
                            s.get("n_with_skill", 0), 100 * s.get("coverage", 0.0),
                            s.get("elapsed_seconds", 0.0),
                        )

    aggregate = {
        "run_meta": run_meta,
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "n_pairs": len(summaries),
        "n_units_total": sum(s.get("n_units", 0) for s in summaries),
        "n_steps_total": sum(s.get("n_steps", 0) for s in summaries),
        "n_with_skill_total": sum(s.get("n_with_skill", 0) for s in summaries),
        "per_pair": summaries,
    }
    (output_dir / "_run_summary.json").write_text(json.dumps(aggregate, indent=2))

    print()
    print("=" * 70)
    print(
        f"[label_skill_actions_qa] DONE — {len(summaries)} pairs, "
        f"{aggregate['n_steps_total']} steps, "
        f"{aggregate['n_with_skill_total']} w/ skill"
    )
    print(f"[label_skill_actions_qa] output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
