"""GPT-5.4-driven skill bank construction for EnvWrappers (4 games).

Why
---
The skill banks shipped with the original
``labeling/skill_bank_out/run_20260430_030637/env_wrappers/`` were
clustered without a curator pass and the raw output is dominated by
hash-only IDs (``skill-648266e9c6`` etc.) plus 1–3 named skills.  As a
result, the original ``decision_sft_jsonl/`` skill_selection rows for the
4 EnvWrappers ship with degenerate candidate sets:

  * candy_crush:        every step has a single candidate ``COMMIT/CLEAR``
  * super_mario:        every step has the same 2 candidates
  * twenty_forty_eight: every step has the same 3 candidates
  * tetris:             OK (6 named skills)

This script reuses the GPT-5.4 CONTRACT + CURATOR pipeline that produced
the QA banks (``labeling/build_skillbank_qa_gpt54.py``) but feeds it the
intention-labeled EnvWrapper rollouts produced by
``labeling/label_intentions_gpt54.py`` (in the dual-axis vocabulary).

Output layout matches what ``labeling/label_skill_actions_gpt54.py``
expects, so the downstream skill-query + SFT-emit chain is the same as
for gym_v::

    labeling/skill_bank_envwrappers/run_<ts>/env_wrappers/<game>/skill_bank.jsonl
    labeling/skill_bank_envwrappers/run_<ts>/env_wrappers/<game>/_summary.json

After this script you would:

    python labeling/label_skill_actions_gpt54.py \\
        --intentions-run labeling/intentions_out/run_dualaxis_20260429_224917 \\
        --bank-run       labeling/skill_bank_envwrappers/run_<ts> \\
        --corpus env_wrappers --all

    python labeling/build_decision_sft_jsonl.py \\
        --skill-actions-run labeling/skill_actions_out/run_<env_ts> \\
        --corpus env_wrappers --output-dir <staging>

Usage::

    python labeling/build_skillbank_envwrappers_gpt54.py \\
        --intentions-run labeling/intentions_out/run_dualaxis_20260429_224917 \\
        --output-dir     labeling/skill_bank_envwrappers/run_<ts> \\
        --games candy_crush super_mario tetris twenty_forty_eight \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
for p in [str(WORKSPACE_ROOT), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the LLM-driven CLUSTER/CONTRACT/CURATOR pipeline.
from labeling.build_skillbank_qa_gpt54 import (  # noqa: E402
    HopInstance,
    _process_source,
    DEFAULT_LABEL_MODEL,
    DEFAULT_WORKERS,
    DEFAULT_CURATOR_JACCARD,
)

logger = logging.getLogger("build_skillbank_envwrappers")

ENV_WRAPPER_GAMES: Tuple[str, ...] = (
    "candy_crush", "super_mario", "tetris", "twenty_forty_eight",
)

_INTENTION_TAG_RE = re.compile(r"\[\s*([A-Z_]+)\s*/\s*([A-Z_]+)\s*\]\s*(.*)")


# ---------------------------------------------------------------------------
# EnvWrapper episode iteration
# ---------------------------------------------------------------------------

def _iter_envwrapper_steps(
    intentions_run: Path,
    *,
    game: str,
    limit_episodes: Optional[int] = None,
) -> Iterable[HopInstance]:
    """Yield one HopInstance per step of every episode for ``game``.

    Parses (operator, subgoal, note) primarily from
    ``exp.intention_operator/subgoal/note`` fields; falls back to the
    ``[OP/SG] note`` prefix in ``exp.intentions`` when those are absent
    (the dual-axis labeler often leaves ``intention_operator=None`` even
    though the bracketed prefix is present).
    """
    game_dir = intentions_run / "env_wrappers" / game
    if not game_dir.is_dir():
        logger.warning("[%s] no labeled rollouts at %s", game, game_dir)
        return
    files = sorted(game_dir.glob("episode_*.json"))
    if limit_episodes is not None:
        files = files[:limit_episodes]

    for f in files:
        try:
            ep = json.loads(f.read_text())
        except Exception as exc:
            logger.warning("[%s] %s: %s", game, f.name, exc)
            continue
        eid = str(ep.get("episode_id") or f.stem)
        ctx = str(ep.get("task") or ep.get("query") or game)[:160]
        outcome = ep.get("outcome")
        for i, exp in enumerate(ep.get("experiences") or []):
            op = exp.get("intention_operator")
            sg = exp.get("intention_subgoal")
            note = exp.get("intention_note")

            if not op or not sg:
                m = _INTENTION_TAG_RE.match(str(exp.get("intentions") or "").strip())
                if m:
                    op = op or m.group(1)
                    sg = sg or m.group(2)
                    if not note:
                        note = m.group(3).strip()
            if not op or not sg:
                continue
            note = (note or "").strip()
            if not note:
                # Fallback to the full intentions line as note (richer than empty).
                note = str(exp.get("intentions") or "").strip()
            if not note:
                continue

            yield HopInstance(
                source=game,
                model=DEFAULT_LABEL_MODEL,
                sample_id=eid,
                bucket=game,
                step_idx=i,
                operator=str(op).upper(),
                subgoal=str(sg).upper(),
                note=note,
                evidence="step",
                tool_call="",
                action=str(exp.get("action") or "")[:160],
                context=ctx,
                correct=bool(outcome) if isinstance(outcome, bool) else None,
                reward=None,
            )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--intentions-run", type=Path,
                   default=REPO_ROOT / "labeling" / "intentions_out"
                                       / "run_dualaxis_20260429_224917",
                   help="Path to labeling/intentions_out/run_*/ produced by "
                        "labeling/label_intentions_gpt54.py.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write per-game banks. Default: "
                        "labeling/skill_bank_envwrappers/run_<ts>")
    p.add_argument("--games", nargs="+", default=list(ENV_WRAPPER_GAMES),
                   help="Subset of envwrapper games to process.")
    p.add_argument("--limit-episodes", type=int, default=None,
                   help="Process only the first N episodes per game (smoke).")
    p.add_argument("--label-model", type=str, default=DEFAULT_LABEL_MODEL,
                   help=f"GPT model for CONTRACT + CURATOR (default: "
                        f"{DEFAULT_LABEL_MODEL}).")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"CONTRACT thread pool size (default: {DEFAULT_WORKERS}).")
    p.add_argument("--curator-jaccard", type=float, default=DEFAULT_CURATOR_JACCARD,
                   help=f"Jaccard threshold for CURATOR merge candidates "
                        f"(default: {DEFAULT_CURATOR_JACCARD}).")
    p.add_argument("--skip-curator", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.setLevel(logging.INFO)

    intentions_run = args.intentions_run.resolve()
    if not intentions_run.is_dir():
        print(f"[build_skillbank_envwrappers] missing intentions-run: {intentions_run}",
              file=sys.stderr)
        return 2

    if args.output_dir is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_root = REPO_ROOT / "labeling" / "skill_bank_envwrappers" / f"run_{ts}"
    else:
        out_root = args.output_dir.resolve()
    # Mirror the gym_v/env_wrappers layout that label_skill_actions_gpt54.py
    # expects under --bank-run.
    env_dir = out_root / "env_wrappers"
    env_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for game in args.games:
        instances = list(_iter_envwrapper_steps(
            intentions_run, game=game, limit_episodes=args.limit_episodes,
        ))
        logger.info("[%s] gathered %d step instances", game, len(instances))
        summary = _process_source(
            source=game,
            instances=instances,
            output_dir=env_dir,
            label_model=args.label_model,
            workers=args.workers,
            curator_jaccard=args.curator_jaccard,
            skip_curator=args.skip_curator,
        )
        summary["game"] = game
        summaries.append(summary)

    overall = {
        "intentions_run": str(intentions_run),
        "output_dir": str(out_root),
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "label_model": args.label_model,
        "n_games": len(summaries),
        "n_skills_total": sum(s.get("n_skills_kept", 0) for s in summaries),
        "n_instances_total": sum(s.get("n_instances_total", 0) for s in summaries),
        "per_game": summaries,
    }
    (out_root / "_run_summary.json").write_text(json.dumps(overall, indent=2))

    print()
    print("=" * 70)
    print(f"[build_skillbank_envwrappers] DONE")
    for s in summaries:
        print(f"  {s.get('game'):<22s} | "
              f"instances={s.get('n_instances_total', 0):>6d} | "
              f"clusters={s.get('n_clusters_raw', 0):>3d} | "
              f"skills_kept={s.get('n_skills_kept', 0):>3d} | "
              f"errs={s.get('contract_errors', 0)}")
    print(f"  output: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
