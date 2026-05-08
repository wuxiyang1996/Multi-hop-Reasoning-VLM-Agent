"""Build ``skill_selection.jsonl`` rows for QA + MiniWob from skill-query data.

This is the integration step that turns
``labeling/skill_actions_qa_out/run_<ts>/`` (the per-hop / per-step
skill_query labels) into per-benchmark ``skill_selection.jsonl`` files
that the SFT trainer's ``data_loader`` can ingest with no code changes
(format identical to ``labeling/build_decision_sft_jsonl.py``).

For each unit (a QA sample-hop or a miniwob experience-step) with
``skill_query.candidates`` of length ≥ 2 we emit one row of the form::

    {"prompt": "...numbered candidate skills...",
     "completion": "REASONING: ...\\nSKILL: <1-based index>",
     "intention":  "[OP/SG] note",
     "active_skill": "<selected_skill_id>",
     "candidates": [<skill_id>, ...],
     "selected_skill_id": "<id>",
     "game": "<bench_or_minigame>",
     "corpus": "visual_reasoning" | "browsergym",
     "episode_id": "<sample_id__model>" | "<episode_id__model>",
     "step_idx": <int>,
     "source_model": "gpt-5.4" | "claude" | ...}

Three modes of operation:

1. ``--out-root <dir>``               — write fresh per-source folders
   under ``<out-root>/<bench>/skill_selection.jsonl``.
2. ``--patch-sft-dir <existing run>`` — additionally append/replace the
   skill_selection.jsonl files under the existing SFT dataset run
   produced by ``scripts/build_multimodal_decision_sft.py``.  This
   lets us extend an already-built dataset without rebuilding from
   scratch.

Usage
~~~~~

    python -m scripts.build_qa_skill_selection_sft \\
        --skill-actions-run labeling/skill_actions_qa_out/run_<ts> \\
        --out-root          labeling/qa_skill_selection_sft/run_<ts> \\
        --patch-sft-dir     labeling/decision_sft_jsonl/run_multimodal_20260506_055105
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("scripts.build_qa_skill_selection_sft")

QA_SOURCES = ("video_holmes", "siv_bench", "tir_bench", "visual_toolbench")
MINIWOB_SOURCE = "miniwob"
WEBSHOP_SOURCE = "webshop"
BROWSER_SOURCES = (MINIWOB_SOURCE, WEBSHOP_SOURCE)
DEFAULT_MODELS = ("gpt-5.4", "claude", "gemini", "qwen")

SKILL_SELECTION_SYSTEM_PROMPT = (
    "You are a skill-selection module for a multimodal decision agent.  "
    "Given the current state and a numbered list of candidate strategies, "
    "pick the ONE strategy that best fits the situation.  Reply with a "
    "short reasoning line, then output 'SKILL: <number>' on a new line."
)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _qa_schema_text(sample: Dict[str, Any]) -> str:
    """Build a textual state description for a QA sample.

    Combines question, options block (when present), and any visual schema
    we have on the original record.  Mirrors the formatting used by the
    legacy ``_qa_sample_to_row`` in build_multimodal_decision_sft.py.
    """
    schema = (sample.get("schema") or "").strip()
    q = (sample.get("question") or sample.get("query") or "").strip()
    opts = (sample.get("options_block") or "").strip()
    has_inline = bool(re.search(r"(?:^|\n)\s*[A-F]\.\s+", q))
    parts: List[str] = []
    if schema:
        parts.append(schema)
    if has_inline or not opts:
        parts.append(f"<question>\n{q}")
    else:
        parts.append(f"<question>\n{q}\n\n{opts}")
    return "\n\n".join(parts).strip()


def _miniwob_schema_text(exp: Dict[str, Any], episode: Dict[str, Any]) -> str:
    """Compact state+goal text for any browsergym row (miniwob / webshop).

    For webshop the cold-start ``intentions`` is usually empty, so we
    additionally fall back to ``metadata.schema_canonical`` (the
    browsergym AXTree dump) — that is what the agent actually sees and
    is also what the downstream SFT prompt presents.
    """
    state = (exp.get("state") or exp.get("raw_state") or "").strip()
    if not state:
        state = (exp.get("summary_state") or "").strip()
    if not state:
        meta = exp.get("metadata") or {}
        state = (meta.get("schema_canonical") or meta.get("schema") or "").strip()
    goal = (exp.get("goal") or episode.get("task") or episode.get("query") or "").strip()
    parts: List[str] = []
    if state:
        parts.append(state)
    if goal:
        parts.append(f"<task>\n{goal}")
    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _build_skill_selection_row(
    *,
    schema_text: str,
    intention: str,
    note: str,
    candidates: List[str],
    selected: str,
    game: str,
    corpus: str,
    episode_id: str,
    step_idx: int,
    source_model: str,
    image: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Emit one ``skill_selection`` row matching the legacy schema."""
    if len(candidates) < 2 or selected not in candidates:
        return None
    if not schema_text:
        return None
    sel_idx = candidates.index(selected) + 1

    numbered = "\n".join(f"  {i + 1}. {sid}" for i, sid in enumerate(candidates))
    user = (
        f"Game state:\n{schema_text}\n\n"
        f"Available strategies (pick ONE by number):\n{numbered}\n\n"
        f"Choose the best strategy. Output REASONING then SKILL number."
    )
    prompt = SKILL_SELECTION_SYSTEM_PROMPT + "\n" + user

    why = note.strip() or "Best fit for current state."
    completion = f"REASONING: {why[:200]}\nSKILL: {sel_idx}"

    row: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "intention": intention,
        "active_skill": selected,
        "candidates": candidates,
        "selected_skill_id": selected,
        "game": game,
        "corpus": corpus,
        "episode_id": episode_id,
        "step_idx": step_idx,
        "source_model": source_model,
    }
    if image is not None:
        row["image"] = image
    if extras:
        for k, v in extras.items():
            if v is not None:
                row[k] = v
    return row


# ---------------------------------------------------------------------------
# QA source processor
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


def _process_qa_pair(
    *, source: str, model: str, in_path: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not in_path.exists():
        return rows
    for sample in _iter_jsonl(in_path):
        sample_id = str(
            sample.get("sample_id") or sample.get("task_id")
            or sample.get("question_id") or "qa"
        )
        schema = _qa_schema_text(sample)
        if not schema:
            continue
        episode_id = f"{source}__{sample_id}__{model}"
        # Image (video) — preserve the legacy attachment shape.
        image = None
        vmeta = sample.get("video_meta") or {}
        if vmeta.get("video_path"):
            image = {"path": vmeta["video_path"], "mime_type": "video/mp4"}

        for hop in (sample.get("hops") or []):
            sq = hop.get("skill_query") or {}
            cands = [c.get("skill_id") for c in (sq.get("candidates") or [])
                     if c.get("skill_id")]
            if len(cands) < 2:
                continue
            sel = (
                (hop.get("skills") or {}).get("skill_id")
                or sq.get("selected_skill_id")
                or cands[0]
            )
            op = str(hop.get("operator") or "COMMIT")
            sg = str(hop.get("subgoal") or "ANSWER")
            note = str(hop.get("note") or "")
            intention = f"[{op}/{sg}] {note}".strip()

            row = _build_skill_selection_row(
                schema_text=schema,
                intention=intention,
                note=note,
                candidates=cands,
                selected=sel,
                game=source,
                corpus="visual_reasoning",
                episode_id=episode_id,
                step_idx=int(hop.get("step", 0)),
                source_model=model,
                image=image,
                extras={
                    "sample_id": sample_id,
                    "evidence": hop.get("evidence"),
                    "tool_call": hop.get("tool_call"),
                    "qa_correct": sample.get("correct"),
                },
            )
            if row:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# MiniWob processor
# ---------------------------------------------------------------------------

def _process_browser_game(
    *, source: str, model: str, game_dir: Path,
) -> List[Dict[str, Any]]:
    """Skill-selection row builder for one BrowserGym (miniwob/webshop) game dir.

    The labeled-rollout shape is identical between sources, so the only
    source-specific bookkeeping is the ``corpus`` / ``source`` tag we set
    on each emitted row.  ``corpus='browsergym'`` is correct for both;
    we additionally stash ``browser_source`` in extras so downstream tools
    can split the two domains when desired.
    """
    rows: List[Dict[str, Any]] = []
    in_files = sorted(game_dir.glob("*.jsonl"))
    game = game_dir.name
    for f in in_files:
        for ep in _iter_jsonl(f):
            base_eid = str(ep.get("episode_id") or f.stem)
            episode_id = f"{base_eid}__{model}"
            for i, exp in enumerate(ep.get("experiences") or []):
                sq = exp.get("skill_query") or {}
                cands = [c.get("skill_id") for c in (sq.get("candidates") or [])
                         if c.get("skill_id")]
                if len(cands) < 2:
                    continue
                sel = (
                    (exp.get("skills") or {}).get("skill_id")
                    or sq.get("selected_skill_id")
                    or cands[0]
                )
                op = str(
                    exp.get("intention_operator")
                    or exp.get("intention_tag")
                    or "COMMIT"
                )
                sg = str(exp.get("intention_subgoal") or "EXECUTE")
                note = str(exp.get("intention_note") or "")
                intention = (
                    (exp.get("intentions") or "").strip()
                    or f"[{op}/{sg}] {note}".strip()
                )

                schema = _miniwob_schema_text(exp, ep)
                if not schema:
                    continue

                row = _build_skill_selection_row(
                    schema_text=schema,
                    intention=intention,
                    note=note,
                    candidates=cands,
                    selected=sel,
                    game=game,
                    corpus="browsergym",
                    episode_id=episode_id,
                    step_idx=int(exp.get("idx") or i),
                    source_model=model,
                    extras={
                        "outcome": ep.get("outcome"),
                        "action": exp.get("action") or exp.get("action_text"),
                        "browser_source": source,
                    },
                )
                if row:
                    rows.append(row)
    return rows


def _process_miniwob_game(
    *, model: str, game_dir: Path,
) -> List[Dict[str, Any]]:
    """Back-compat alias for the miniwob processor."""
    return _process_browser_game(
        source=MINIWOB_SOURCE, model=model, game_dir=game_dir,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skill-actions-run", type=Path, required=True,
                   help="Path to labeling/skill_actions_qa_out/run_<ts>/")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Where to write per-bench skill_selection.jsonl. "
                        "Default: labeling/qa_skill_selection_sft/run_<ts>")
    p.add_argument("--patch-sft-dir", type=Path, default=None,
                   help="Optional: also write the per-bench skill_selection.jsonl "
                        "directly under this existing SFT dataset root, "
                        "treating it as the canonical source for the trainer.")
    p.add_argument("--sources", type=str, nargs="+",
                   default=list(QA_SOURCES) + list(BROWSER_SOURCES))
    p.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--miniwob-merge", type=str, default="aggregate",
                   choices=("aggregate", "per_game"),
                   help="aggregate: one <source>/skill_selection.jsonl per "
                        "browser source, combining all per-task subdirs "
                        "(matches existing SFT layout for miniwob and "
                        "webshop). per_game: one per minigame/webshop task.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.setLevel(logging.INFO)

    sa_run: Path = args.skill_actions_run.resolve()
    if not sa_run.is_dir():
        print(f"[build_qa_skill_selection_sft] missing: {sa_run}", file=sys.stderr)
        return 2

    if args.out_root is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = REPO_ROOT / "labeling" / "qa_skill_selection_sft" / f"run_{ts}"
    else:
        out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows_by_bench: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_pair_summary: List[Dict[str, Any]] = []

    # 1) QA sources.
    for source in args.sources:
        if source not in QA_SOURCES:
            continue
        for model in args.models:
            in_path = sa_run / source / model / "qa" / "samples_with_skill_query.jsonl"
            rows = _process_qa_pair(source=source, model=model, in_path=in_path)
            per_pair_summary.append({
                "source": source, "model": model, "kind": "qa",
                "n_rows": len(rows), "input": str(in_path),
            })
            rows_by_bench[source].extend(rows)
            if rows:
                logger.info("[%s/%s] %d skill_selection rows", source, model, len(rows))

    # 2) Browser sources (miniwob, webshop).  Each writes to its own bench
    #    subdir; the trainer's data_loader treats them as independent benches.
    for browser_src in BROWSER_SOURCES:
        if browser_src not in args.sources:
            continue
        bs_root = sa_run / browser_src
        if not bs_root.is_dir():
            continue
        for model in args.models:
            mdir = bs_root / model
            if not mdir.is_dir():
                continue
            for game_dir in sorted(mdir.iterdir()):
                if not game_dir.is_dir():
                    continue
                rows = _process_browser_game(
                    source=browser_src, model=model, game_dir=game_dir,
                )
                per_pair_summary.append({
                    "source": browser_src, "model": model,
                    "kind": browser_src, "game": game_dir.name,
                    "n_rows": len(rows),
                })
                if args.miniwob_merge == "aggregate":
                    rows_by_bench[browser_src].extend(rows)
                else:
                    rows_by_bench[f"{browser_src}/{game_dir.name}"].extend(rows)
                if rows:
                    logger.info("[%s/%s/%s] %d rows",
                                 browser_src, model, game_dir.name, len(rows))

    # 3) Write per-bench skill_selection.jsonl.
    targets: List[Path] = [out_root]
    patch_dir: Optional[Path] = None
    if args.patch_sft_dir is not None:
        patch_dir = args.patch_sft_dir.resolve()
        if not patch_dir.exists():
            print(f"[build_qa_skill_selection_sft] --patch-sft-dir does not exist: "
                  f"{patch_dir}", file=sys.stderr)
            return 2
        targets.append(patch_dir)

    by_bench_count: Dict[str, int] = {}
    by_model_count: Counter = Counter()

    for bench, rows in rows_by_bench.items():
        if not rows:
            continue
        by_bench_count[bench] = len(rows)
        for r in rows:
            by_model_count[r.get("source_model", "unknown")] += 1
        for tgt in targets:
            bench_dir = tgt / bench
            bench_dir.mkdir(parents=True, exist_ok=True)
            out_path = bench_dir / "skill_selection.jsonl"
            with out_path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info("wrote %d rows -> %s", len(rows), out_path)

    summary = {
        "skill_actions_run": str(sa_run),
        "out_root": str(out_root),
        "patch_sft_dir": str(patch_dir) if patch_dir else None,
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "n_rows_total": sum(by_bench_count.values()),
        "by_bench": dict(sorted(by_bench_count.items(), key=lambda kv: -kv[1])),
        "by_source_model": dict(by_model_count.most_common()),
        "per_pair": per_pair_summary,
    }
    (out_root / "_summary.json").write_text(json.dumps(summary, indent=2))
    if patch_dir is not None:
        # Drop a sibling summary so it's traceable.
        (patch_dir / "_qa_skill_selection_patch_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

    print()
    print("=" * 70)
    print(f"[build_qa_skill_selection_sft] DONE — {summary['n_rows_total']} rows")
    print(f"  by bench: {summary['by_bench']}")
    print(f"  by model: {summary['by_source_model']}")
    print(f"  out:      {out_root}")
    if patch_dir is not None:
        print(f"  patched:  {patch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
