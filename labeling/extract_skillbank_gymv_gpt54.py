#!/usr/bin/env python
"""Extract a per-env Skill Bank from gym-v cold-start rollouts (gpt-5.4 teacher).

Companion to :mod:`labeling.extract_skillbank_gpt54` for the gym-v
:class:`Temporal/<Title>-v0` envs. Runs the full
:mod:`skill_agents.SkillBankAgent` pipeline (SEGMENT / CONTRACT / CURATOR
LoRAs) against already-collected gym-v episodes — produces the cold-start
skill banks that seed the deferred Qwen3.5-9B Skill-Bank GRPO stage.

Why a gym-v variant
-------------------

The canonical extractor at
``skill_agents/extract_skillbank/extract_skillbank_grpo_gpt54.py`` expects
labeled episodes whose ``Experience`` objects already carry
``summary_state`` / ``intentions``. The env_wrappers labeler
(``labeling/label_episodes_gpt54.py``) populates those fields via the
game-aware extractors in ``decision_agents/agent_helper.py`` — but no
such extractors exist for the retro Genesis ROMs surfaced by gym-v. The
gym-v cold-start collector instead records a VLM ``<state>`` block at
``Experience.metadata.schema``; this driver maps that schema directly
onto the state fields and pairs it with the chosen action, so the
SkillBankAgent learns skills from ``(schema, action)`` pairs.

Pipeline reuse
--------------

Everything below is delegated verbatim to the canonical extractor:

* Stage 1+2 segmentation, per-episode (``SkillBankAgent.segment_episode``)
* Stage 3 contract learn / verify / refine (``run_contract_learning``)
* Stage 4 bank maintenance (split / merge / refine, CURATOR-filtered)
* Per-stage ``StageIORecord`` capture and a per-game ``stage_io_log.json``
* Per-episode skill-bank snapshot (``episode_snapshots/episode_<i>/``)
* Per-episode bank-management I/O (``per_episode_bank_management/``)
* All three LoRA targets' raw prompts/responses streamed to
  ``coldstart_io_all.jsonl`` via :mod:`skill_agents.coldstart_io`
  (CONTRACT + CURATOR fire on the API path; SEGMENT additionally writes
  ``teacher_io_coldstart.jsonl``).

Layout produced
---------------

::

    skill_bank_sft/<run_id>/
    ├── Temporal_Airstriker-v0/
    │   ├── skill_bank.jsonl                     # final per-env bank
    │   ├── skill_catalog.json
    │   ├── extraction_summary.json
    │   ├── stage_io_log.json                    # per-stage I/O
    │   ├── coldstart_io_all.jsonl               # CONTRACT + CURATOR
    │   ├── teacher_io_coldstart.jsonl           # SEGMENT
    │   ├── episode_snapshots/
    │   │   └── episode_<i>/
    │   │       ├── skill_bank.jsonl             # bank state after step i
    │   │       ├── stage_io_log.json
    │   │       └── llm_calls.json
    │   ├── per_episode_bank_management/
    │   │   └── episode_<i>/
    │   │       ├── bank_management_io.json
    │   │       └── skill_bank.jsonl
    │   └── reports/
    ├── ... (one per gym-v env)
    ├── _normalized_episodes/                    # cached schema-as-state copies
    │   └── Temporal_<env>/episode_<i>.json
    └── _run_meta.json

Usage
-----

::

    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # All envs in the gym-v cold-start run, gpt-5.4 teacher
    python labeling/extract_skillbank_gymv_gpt54.py \\
        --input_dir Cold-start-out-gymv/sft_gpt5p4_e20_s100_stream_20260429_080127 \\
        --output_dir skill_bank_sft \\
        --model gpt-5.4 -v

    # Or via the parallel shell launcher (one worker per env)
    bash labeling/run_extract_skillbank_gymv.sh

    # Quick test: 1 env, 1 episode
    python labeling/extract_skillbank_gymv_gpt54.py \\
        --input_dir Cold-start-out-gymv/<run> \\
        --envs Temporal_Airstriker-v0 \\
        --max_episodes 1 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — mirror sibling drivers
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for p in (CODEBASE_ROOT, WORKSPACE_ROOT):
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# Bootstrap api_keys.py from the workspace root (matches cold_start scripts).
try:
    import api_keys as _ak  # type: ignore
    if getattr(_ak, "openrouter_api_key", "") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = _ak.openrouter_api_key  # type: ignore
    if getattr(_ak, "openai_api_key", "") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _ak.openai_api_key  # type: ignore
except Exception:
    pass

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from skill_agents.extract_skillbank.extract_skillbank_grpo_gpt54 import (
    ExtractionCheckpoint,
    extract_skills_for_game,
    save_lora_adapters,
    setup_grpo_orchestrator,
    setup_local_model,
)

logger = logging.getLogger("labeling.extract_skillbank_gymv")

DEFAULT_MODEL = "gpt-5.4"

# ---------------------------------------------------------------------------
# Episode normalisation: schema → state / summary_state pair
# ---------------------------------------------------------------------------

_SCHEMA_KEYS = ("schema", "schema_canonical")
_SUMMARY_HARD_LIMIT = 600  # keep prompt budget tight for long schemas


def _pick_schema(metadata: Dict[str, Any]) -> str:
    """Return the best-available schema text for one step.

    Priority: VLM ``<state>`` block, then the deterministic canonical
    fallback the cold-start actor mirrored into ``schema_canonical``.
    Empty string if neither is present (the segmenter degrades
    gracefully — empty state contributes 0 fit score).
    """
    for k in _SCHEMA_KEYS:
        v = metadata.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _compact_summary_from_schema(schema: str, *, limit: int = _SUMMARY_HARD_LIMIT) -> str:
    """Single-line ``key=value | key=value`` view of a ``<state>`` block.

    The Stage 2 segmentation scorer reads ``Experience.summary_state`` for
    intention-fit + boundary-preference terms; an over-long blob bloats
    every preference-learning prompt. We strip the XML wrapper, drop the
    affordance / intentions / state_flags tail (least useful for skill
    discrimination) and join the remaining lines with ``|``.
    """
    if not schema:
        return ""
    body = schema.strip()
    if body.startswith("<state>"):
        body = body[len("<state>"):]
    if body.endswith("</state>"):
        body = body[: -len("</state>")]
    lines: List[str] = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s:
            continue
        # Drop the section markers + the tails that rarely change skill identity.
        low = s.lower()
        if low.startswith("<entities>") or low.startswith("</entities>"):
            continue
        if low.startswith("<affordances>") or low.startswith("</affordances>"):
            break  # rest of the schema is verbose enumeration
        lines.append(s)
        if sum(len(x) for x in lines) > limit:
            break
    summary = " | ".join(lines)
    return summary[:limit]


def _normalize_gymv_episode_dict(ep_data: Dict[str, Any]) -> Dict[str, Any]:
    """Re-map a gym-v cold-start episode so the bank pipeline sees
    ``(schema, action)`` pairs.

    Mutates a *copy* — the original on-disk JSON is untouched. For each
    experience we set:

    * ``state`` ← schema (the ``<state>`` block from gpt-5.4 vision)
    * ``summary_state`` ← compact ``key=value`` view of the same schema
    * ``summary`` ← falls back to ``summary_state`` (Stage 2 reads either)
    * ``next_state`` ← next step's schema (or empty for the last step)
    * ``raw_state`` / ``raw_next_state`` ← truncated original env text
      (kept for debugging — the bank pipeline never looks at them)

    Everything else (``action``, ``reward``, ``done``, ``intentions``,
    ``available_actions``, etc.) is preserved unchanged. ``intentions``
    already carries gpt-5.4's per-step action reasoning — the segmenter
    uses it as the LLM-teacher seed.
    """
    out = dict(ep_data)
    raw_exps = ep_data.get("experiences") or []
    n = len(raw_exps)
    if n == 0:
        out["experiences"] = []
        return out

    new_exps: List[Dict[str, Any]] = []
    for i, exp in enumerate(raw_exps):
        md = exp.get("metadata") or {}
        schema = _pick_schema(md)
        next_md = (raw_exps[i + 1].get("metadata") if i + 1 < n else {}) or {}
        next_schema = _pick_schema(next_md)
        summary_state = _compact_summary_from_schema(schema)

        new_exp = dict(exp)
        new_exp["state"] = schema
        new_exp["next_state"] = next_schema
        new_exp["summary_state"] = summary_state
        # Stage 2's intention-fit term reads ``summary`` if ``summary_state``
        # is empty; mirror to keep both populated.
        if not new_exp.get("summary"):
            new_exp["summary"] = summary_state
        new_exp["raw_state"] = (exp.get("raw_state") or "")[:1000]
        new_exp["raw_next_state"] = (exp.get("raw_next_state") or "")[:1000]
        new_exps.append(new_exp)

    out["experiences"] = new_exps
    # Tag the env_name / game_name explicitly so the SkillBankAgent's
    # phase detector falls through to the generic temporal-thirds rule
    # (we have no game-specific extractor for retro Genesis ROMs).
    out.setdefault("env_name", "gym_v")
    out.setdefault(
        "game_name", ep_data.get("game_name") or "gym_v"
    )
    return out


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------

def _find_envs(input_dir: Path) -> Dict[str, List[Path]]:
    """Discover ``Temporal_*`` env subfolders and their sealed episode JSONs."""
    out: Dict[str, List[Path]] = {}
    if not input_dir.exists():
        return out
    for env_dir in sorted(input_dir.iterdir()):
        if not env_dir.is_dir():
            continue
        if not env_dir.name.startswith("Temporal_"):
            continue  # ignore _logs / latest / _run_meta.json etc
        eps = sorted(
            f for f in env_dir.glob("episode_[0-9]*.json")
            if f.name != "episode_buffer.json"
        )
        if eps:
            out[env_dir.name] = eps
    return out


def _load_normalized_episodes(
    env_name: str,
    files: List[Path],
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load + normalise episode JSONs; optionally cache the rewritten copies."""
    out: List[Dict[str, Any]] = []
    cache_subdir: Optional[Path] = None
    if cache_dir is not None:
        cache_subdir = cache_dir / env_name
        cache_subdir.mkdir(parents=True, exist_ok=True)

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            logger.warning("[%s] failed to load %s: %s", env_name, fp.name, exc)
            continue
        norm = _normalize_gymv_episode_dict(raw)
        if cache_subdir is not None:
            try:
                with open(cache_subdir / fp.name, "w", encoding="utf-8") as f:
                    json.dump(norm, f, indent=2, ensure_ascii=False, default=str)
            except Exception:
                pass
        out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Per-env driver
# ---------------------------------------------------------------------------

def run_one_env(
    env_name: str,
    episode_files: List[Path],
    output_dir: Path,
    *,
    model: str,
    max_episodes: Optional[int],
    verbose: bool,
    cache_dir: Optional[Path],
    checkpoint: Optional[ExtractionCheckpoint] = None,
) -> Dict[str, Any]:
    """Run the SkillBankAgent pipeline on one gym-v env."""
    print()
    print(f"==============================================================")
    print(f"  ENV: {env_name}  ({len(episode_files)} sealed episode(s))")
    print(f"==============================================================")

    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]
    episodes_data = _load_normalized_episodes(env_name, episode_files, cache_dir)
    if not episodes_data:
        print(f"  [SKIP] {env_name}: no usable episodes")
        return {"env": env_name, "skipped": True, "n_episodes": 0}

    env_out = output_dir / env_name
    env_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    summary: Dict[str, Any] = {
        "env": env_name,
        "n_episodes": len(episodes_data),
        "model": model,
        "started_at": datetime.now().isoformat(),
    }

    catalog: Dict[str, Dict[str, Any]] = {}
    try:
        agent, catalog, sub_episodes, io_log = extract_skills_for_game(
            episodes_data=episodes_data,
            game_name=env_name,
            output_dir=env_out,
            model=model,
            verbose=verbose,
            resegment=False,
            checkpoint=checkpoint,
            resume_from_episode=0,
        )
        summary["n_skills"] = len(agent.skill_ids)
        summary["skill_ids"] = list(agent.skill_ids)
        summary["n_sub_episodes"] = len(sub_episodes)
        summary["error"] = None
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [ERROR] {env_name}: {summary['error']}")
        if verbose:
            traceback.print_exc()

    summary["elapsed_seconds"] = round(time.time() - t0, 2)
    summary["ended_at"] = datetime.now().isoformat()

    # ── Unified per-env outputs ─────────────────────────────────────────
    # Both pipelines (env_wrappers + gym-v) must write skill_catalog.json
    # and extraction_summary.json with the same set of canonical keys so
    # the cross-corpus aggregator can read them uniformly. ``corpus`` and
    # ``source_name`` make every catalog file self-describing.
    canonical_catalog = {
        "corpus": "gym_v",
        "source_name": env_name,
        "model": model,
        "pipeline": "skill_agents",
        "timestamp": datetime.now().isoformat(),
        "n_skills": len(catalog),
        "skills": list(catalog.values()),
    }
    try:
        with open(env_out / "skill_catalog.json", "w", encoding="utf-8") as f:
            json.dump(canonical_catalog, f, indent=2, ensure_ascii=False, default=str)
    except Exception as exc:
        if verbose:
            print(f"  [WARN] {env_name}: failed to write skill_catalog.json: {exc}")

    canonical_extraction_summary = {
        "corpus": "gym_v",
        "source_name": env_name,
        "model": model,
        "pipeline": "skill_agents",
        "timestamp": datetime.now().isoformat(),
        "input_dir": str((episode_files[0].parent) if episode_files else env_out),
        "episodes_processed": summary["n_episodes"],
        "skills_extracted": summary.get("n_skills", 0),
        "sub_episodes": summary.get("n_sub_episodes", 0),
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    try:
        with open(env_out / "extraction_summary.json", "w", encoding="utf-8") as f:
            json.dump(canonical_extraction_summary, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass

    # Driver-internal summary (debug / progress) — kept distinct from the
    # canonical files above.
    try:
        with open(env_out / "_env_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass

    print(
        f"  -> {env_name}: skills={summary.get('n_skills', '?')}  "
        f"sub_episodes={summary.get('n_sub_episodes', '?')}  "
        f"elapsed={summary['elapsed_seconds']}s"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-env Skill Banks from gym-v cold-start rollouts using "
            "the SkillBankAgent pipeline (SEGMENT / CONTRACT / CURATOR LoRAs) "
            "with gpt-5.4 as the SFT teacher."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to a cold-start gym-v run dir (contains "
             "Temporal_<env>/episode_NNN.json subfolders).",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(CODEBASE_ROOT / "skill_bank_sft"),
        help="Output directory (default: <repo>/skill_bank_sft).",
    )
    parser.add_argument(
        "--envs", type=str, nargs="+", default=None,
        help="Restrict to a subset of env folder names (Temporal_<env>-v0).",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"SFT teacher model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Cap episodes per env (default: all).",
    )
    parser.add_argument(
        "--cache_normalized", action="store_true",
        help="Cache the schema-as-state-rewritten episodes under "
             "<output_dir>/_normalized_episodes/ for inspection.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint (skips envs that already finished).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Just enumerate envs / episodes, do not run the pipeline.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Per-step prints from the bank pipeline.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s %(name)s: %(message)s")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  labeling/extract_skillbank_gymv: Skill Bank cold-start (gpt-5.4)")
    print("=" * 62)
    print(f"  Input run dir : {input_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Model         : {args.model}")
    print(f"  Max episodes  : {args.max_episodes if args.max_episodes is not None else '<all>'}")
    print(f"  Cache norm    : {args.cache_normalized}")
    print(f"  Resume        : {args.resume}")
    print()

    discovered = _find_envs(input_dir)
    if not discovered:
        print(f"  [ERROR] No Temporal_* env subfolders under {input_dir}")
        return 2

    if args.envs:
        keep = set(args.envs)
        discovered = {k: v for k, v in discovered.items() if k in keep}
        if not discovered:
            print(f"  [ERROR] None of --envs {args.envs} found in input dir.")
            return 2

    print(f"  Found {len(discovered)} env(s):")
    for env_name, files in discovered.items():
        print(f"    {env_name:34s} {len(files):2d} sealed episode(s)")
    print()

    if args.dry_run:
        print("  [DRY RUN] Stopping before pipeline launch.")
        return 0

    cache_dir = (output_dir / "_normalized_episodes") if args.cache_normalized else None
    checkpoint = ExtractionCheckpoint(output_dir) if args.resume else None

    run_meta: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "max_episodes": args.max_episodes,
        "started_at": datetime.now().isoformat(),
        "envs": [],
    }

    t0 = time.time()
    for env_name, files in discovered.items():
        if checkpoint is not None and checkpoint.is_game_complete(env_name):
            print(f"  [RESUME] {env_name}: already complete, skipping")
            continue
        env_summary = run_one_env(
            env_name=env_name,
            episode_files=files,
            output_dir=output_dir,
            model=args.model,
            max_episodes=args.max_episodes,
            verbose=args.verbose,
            cache_dir=cache_dir,
            checkpoint=checkpoint,
        )
        run_meta["envs"].append(env_summary)
        if checkpoint is not None and env_summary.get("error") is None:
            checkpoint.mark_game_complete(env_name)

        # Persist run-level meta after every env so an interrupt is recoverable.
        run_meta["elapsed_seconds"] = round(time.time() - t0, 2)
        try:
            with open(output_dir / "_run_meta.json", "w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

    run_meta["ended_at"] = datetime.now().isoformat()
    run_meta["elapsed_seconds"] = round(time.time() - t0, 2)
    with open(output_dir / "_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False, default=str)

    # ── Cross-env unified index ─────────────────────────────────────────
    # Always produce <output_dir>/_unified/ so downstream consumers
    # (RAG, cross-corpus retrieval, gate stack inputs) can read one
    # canonical index regardless of how many envs were run in this
    # invocation. The dispatcher shell calls the same aggregator after
    # parallel workers complete; running it here covers the
    # single-process path too.
    try:
        from labeling.unify_skill_index import unify_roots as _unify_roots
        unify_summary = _unify_roots(
            roots=[output_dir],
            output_dir=output_dir,
            pipeline="skill_agents",
            verbose=args.verbose,
        )
        print(f"  Unified index : {unify_summary['skill_index_path']}")
        print(f"                  ({unify_summary['n_skills']} skills "
              f"across {unify_summary['n_sources']} source(s))")
    except Exception as exc:
        print(f"  [WARN] unified index aggregation failed: {exc}")
        if args.verbose:
            traceback.print_exc()

    n_ok = sum(1 for s in run_meta["envs"] if not s.get("error") and not s.get("skipped"))
    n_err = sum(1 for s in run_meta["envs"] if s.get("error"))
    total_skills = sum(s.get("n_skills", 0) for s in run_meta["envs"])

    print()
    print("=" * 62)
    print("  Skill Bank cold-start labeling — DONE")
    print("=" * 62)
    print(f"  Envs OK       : {n_ok}/{len(run_meta['envs'])}  (errors={n_err})")
    print(f"  Total skills  : {total_skills}")
    print(f"  Elapsed       : {run_meta['elapsed_seconds']}s")
    print(f"  Run meta      : {output_dir/'_run_meta.json'}")
    print("=" * 62)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
