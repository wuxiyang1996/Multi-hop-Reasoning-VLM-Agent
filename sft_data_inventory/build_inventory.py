#!/usr/bin/env python
"""Build / refresh the SFT data inventory under ``sft_data_inventory/``.

This script walks the canonical data sources scattered across the repo
and consolidates them under ``sft_data_inventory/{games,non_game}/<task>/``
via **symlinks** (no data is copied).  Each task gets a ``MANIFEST.json``
recording row counts, skill-bank status, and per-teacher rollout coverage,
and a top-level ``INVENTORY.json`` rolls everything up.

Run::

    python sft_data_inventory/build_inventory.py            # build / refresh
    python sft_data_inventory/build_inventory.py --dry-run  # show plan only

Idempotent — re-running re-points symlinks at whatever the latest
``run_*`` directory is for each source.

Scope
-----
Games (12)
  * 8 Gym-V Temporal_*-v0 (all four teachers: gpt-5.4 / claude / gemini /
    qwen3-vl-235b)
  * 4 env_wrappers (tetris / super_mario / candy_crush / twenty_forty_eight)
    — only gpt-5.4 teacher

Non-game (6)
  * miniwob — 4 teachers, full pipeline (rollouts + bank + SFT)
  * webshop — 4 teachers rollouts only; no skill bank, no SFT yet
  * video_holmes / siv_bench / tir_bench / visual_toolbench — 4 teachers
    QA samples; bank exists; skill_selection bank-aware, action_taking
    still uses synthetic IDs (see notes in MANIFEST.json)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Source paths (latest production runs as of 2026-05-07)
# ---------------------------------------------------------------------------
GYMV_BANK_RUN       = REPO / "labeling/skill_bank_out/run_repair_20260510_051643/gym_v"
# ^^^ Repaired bank: backfills GPT-5.4 contract predicates (preconditions /
# postconditions / example_predicates / eff_add / eff_del / failure_modes)
# onto the legacy event-mining bank at run_20260430_030637/gym_v.  See
# scripts/repair_gymv_contracts_gpt54.py.  All other fields (protocol,
# sub_episodes, n_instances, report) are preserved verbatim.
ENVWR_BANK_RUN      = REPO / "labeling/skill_bank_envwrappers/run_20260506_201030/env_wrappers"
QA_BANK_RUN         = REPO / "labeling/skill_bank_qa/run_20260506_184439"
WEBSHOP_BANK_RUN    = REPO / "labeling/skill_bank_qa/run_webshop_20260510_044000"

# Layer C — modality-agnostic procedural templates for every skill.
# One ``template_bank.jsonl`` per task, structured as
# ``<TEMPLATE_RUN>/<cohort>/<task>/template_bank.jsonl`` where cohort ∈
# {gymv_game, env_wr_game, web, vr_image, vr_video}.  See
# ``scripts/lift_skill_templates_gpt54.py``.
TEMPLATE_RUN        = REPO / "labeling/skill_templates/run_20260510_053121"

GYMV_FRONTIER_SFT   = REPO / "labeling/frontier_distill_jsonl/run_20260506_065830_skill_enriched"
ENVWR_SFT           = REPO / "labeling/decision_sft_jsonl/run_envwrapper_native_20260507_025519"
NON_GAME_SFT        = ENVWR_SFT  # multimodal benches share this run

GYMV_GPT54_ROLLOUTS = REPO / "labeling/skill_actions_out/run_20260430_064325/gym_v"
GYMV_FRONTIER_RLT   = REPO / "labeling/skill_actions_out/run_frontier_20260506_062027"  # /<model>/gymv/<game>
ENVWR_GPT54_RLT     = REPO / "labeling/skill_actions_out/run_20260430_064325/env_wrappers"

# Webshop pipeline outputs (built 2026-05-10 by the user-driven extraction run)
WEBSHOP_LABELED_RUN     = REPO / "labeling/qa_miniwob_labeled/run_webshop_20260510_043615/webshop"
WEBSHOP_SKILL_QUERY_RUN = REPO / "labeling/skill_actions_qa_out/run_webshop_20260510_044300/webshop"

OR_TX               = REPO / "openrouter-transfer-baselines-out/2026-05-01_08-06-44"
COLD_START_BG       = REPO / "Cold-start-out-browsergym"
COLD_START_VR_IMG   = REPO / "Cold-start-out-visual-reasoning"
COLD_START_VR_VID   = REPO / "Cold-start-out-visual-reasoning-video"

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
GYMV_GAMES = [
    "Temporal_Airstriker-v0",
    "Temporal_AlteredBeast-v0",
    "Temporal_Columns-v0",
    "Temporal_DynamiteHeaddy-v0",
    "Temporal_SpaceHarrierII-v0",
    "Temporal_StreetsOfRage2-v0",
    "Temporal_Strider-v0",
    "Temporal_ThunderForceIII-v0",
]
ENVWR_GAMES = ["tetris", "super_mario", "candy_crush", "twenty_forty_eight"]
NON_GAME_QA = ["video_holmes", "siv_bench", "tir_bench", "visual_toolbench"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def link(src: Path, dst: Path, *, dry_run: bool = False) -> bool:
    """Create / refresh a symlink ``dst -> src``.  Returns True if src exists.

    Resolves to an absolute path so the link works from anywhere.
    """
    if not src.exists():
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            try:
                dst.unlink()
            except IsADirectoryError:
                # directory symlink would fail unlink in some setups
                import shutil
                shutil.rmtree(dst)
        dst.symlink_to(src)
    return True


def count_lines(path: Path) -> int:
    """Cheap line count for a JSONL file (returns 0 on missing/error)."""
    if not path.exists():
        return 0
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def count_episodes(rollout_dir: Path) -> int:
    """Count ``episode_*.json`` in a rollout dir (or webshop ``webshop.N`` subdirs)."""
    if not rollout_dir.exists():
        return 0
    n = len(list(rollout_dir.glob("episode_*.json")))
    if n:
        return n
    # webshop / browsergym layout: per-task subdirs containing one episode_000.json
    return sum(1 for d in rollout_dir.iterdir() if d.is_dir() and (d / "episode_000.json").exists())


def short(p: Path) -> str:
    """Render a path relative to the repo root, prefixed with ``$REPO/``."""
    try:
        return f"$REPO/{p.relative_to(REPO).as_posix()}"
    except ValueError:
        return p.as_posix()


def link_template_bank(
    *, task: str, cohort: str, out_dir: Path, dry_run: bool,
) -> Optional[Dict[str, Any]]:
    """Symlink the lifted procedural-template bank for one task.

    Returns a manifest fragment (``{source, n_skills, ...}``) when the
    template bank exists, else ``None``.
    """
    src = TEMPLATE_RUN / cohort / task / "template_bank.jsonl"
    if not link(src, out_dir / "template_bank.jsonl", dry_run=dry_run):
        return None
    return {
        "source": short(src),
        "n_skills": count_lines(src),
        "kind": "lifted-procedural-template (Layer C)",
        "controlled_vocab": ["PERCEIVE", "RECALL", "COMPARE", "FILTER",
                              "DECIDE", "COMMIT", "VERIFY", "RECOVER"],
        "cohort": cohort,
    }


# ---------------------------------------------------------------------------
# Per-task builders
# ---------------------------------------------------------------------------
def build_gymv_game(game: str, out_root: Path, *, dry_run: bool) -> Dict[str, Any]:
    out_dir = out_root / "games" / game
    manifest: Dict[str, Any] = {
        "task": game, "category": "game", "corpus": "gym_v",
        "teachers": ["gpt-5.4", "claude-4.6", "gemini-3.1-pro", "qwen3-vl-235b"],
    }

    # Skill bank (gpt-5.4 extracted)
    bank_src = GYMV_BANK_RUN / game / "skill_bank.jsonl"
    if link(bank_src, out_dir / "skill_bank.jsonl", dry_run=dry_run):
        manifest["skill_bank"] = {
            "source": short(bank_src),
            "n_skills": count_lines(bank_src),
            "kind": "bank-extracted (OPERATOR/SUBGOAL)",
        }

    # Template bank (Layer C — lifted modality-agnostic procedural template)
    tb = link_template_bank(task=game, cohort="gymv_game", out_dir=out_dir, dry_run=dry_run)
    if tb is not None:
        manifest["template_bank"] = tb

    # SFT (frontier distill, skill-enriched)
    at = GYMV_FRONTIER_SFT / game / "action_taking.jsonl"
    ss = GYMV_FRONTIER_SFT / game / "skill_selection.jsonl"
    sft = {}
    if link(at, out_dir / "sft/action_taking.jsonl", dry_run=dry_run):
        sft["action_taking"] = {"source": short(at), "n_rows": count_lines(at)}
    if link(ss, out_dir / "sft/skill_selection.jsonl", dry_run=dry_run):
        sft["skill_selection"] = {"source": short(ss), "n_rows": count_lines(ss)}
    if sft:
        sft["run"] = short(GYMV_FRONTIER_SFT)
        sft["active_skill_kind"] = "bank-derived (OPERATOR/SUBGOAL)"
        manifest["sft"] = sft

    # Rollouts (per teacher)
    rollouts: Dict[str, Any] = {}
    gpt = GYMV_GPT54_ROLLOUTS / game
    if link(gpt, out_dir / "rollouts/gpt54", dry_run=dry_run):
        rollouts["gpt-5.4"] = {"source": short(gpt), "n_episodes": count_episodes(gpt),
                               "labeled": True, "has_skill_query": True}
    for m in ("claude", "gemini", "qwen"):
        src = GYMV_FRONTIER_RLT / m / "gymv" / game
        if link(src, out_dir / "rollouts" / m, dry_run=dry_run):
            rollouts[m] = {"source": short(src), "n_episodes": count_episodes(src),
                           "labeled": True, "has_skill_query": True}
    manifest["rollouts"] = rollouts

    if not dry_run:
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_envwr_game(game: str, out_root: Path, *, dry_run: bool) -> Dict[str, Any]:
    out_dir = out_root / "games" / game
    manifest: Dict[str, Any] = {
        "task": game, "category": "game", "corpus": "env_wrappers",
        "teachers": ["gpt-5.4"],
        "notes": "Only gpt-5.4 rollouts available — no claude/gemini/qwen run for env_wrappers.",
    }

    bank_src = ENVWR_BANK_RUN / game / "skill_bank.jsonl"
    if link(bank_src, out_dir / "skill_bank.jsonl", dry_run=dry_run):
        manifest["skill_bank"] = {
            "source": short(bank_src),
            "n_skills": count_lines(bank_src),
            "kind": "bank-extracted (OPERATOR/SUBGOAL)",
        }

    tb = link_template_bank(task=game, cohort="env_wr_game", out_dir=out_dir, dry_run=dry_run)
    if tb is not None:
        manifest["template_bank"] = tb

    at = ENVWR_SFT / game / "action_taking.jsonl"
    ss = ENVWR_SFT / game / "skill_selection.jsonl"
    sft = {}
    if link(at, out_dir / "sft/action_taking.jsonl", dry_run=dry_run):
        sft["action_taking"] = {"source": short(at), "n_rows": count_lines(at)}
    if link(ss, out_dir / "sft/skill_selection.jsonl", dry_run=dry_run):
        sft["skill_selection"] = {"source": short(ss), "n_rows": count_lines(ss)}
    if sft:
        sft["run"] = short(ENVWR_SFT)
        sft["active_skill_kind"] = "bank-derived (OPERATOR/SUBGOAL)"
        manifest["sft"] = sft

    gpt = ENVWR_GPT54_RLT / game
    if link(gpt, out_dir / "rollouts/gpt54", dry_run=dry_run):
        manifest["rollouts"] = {"gpt-5.4": {
            "source": short(gpt), "n_episodes": count_episodes(gpt),
            "labeled": True, "has_skill_query": True,
        }}

    if not dry_run:
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_miniwob(out_root: Path, *, dry_run: bool) -> Dict[str, Any]:
    out_dir = out_root / "non_game" / "miniwob"
    manifest: Dict[str, Any] = {
        "task": "miniwob", "category": "non_game", "modality": "web (browsergym)",
        "teachers": ["gpt-5.4", "claude-4.6", "gemini-3.1-pro", "qwen3-vl-235b"],
    }

    bank_src = QA_BANK_RUN / "miniwob" / "skill_bank.jsonl"
    if link(bank_src, out_dir / "skill_bank.jsonl", dry_run=dry_run):
        manifest["skill_bank"] = {
            "source": short(bank_src),
            "n_skills": count_lines(bank_src),
            "kind": "bank-extracted (OPERATOR/SUBGOAL)",
        }

    tb = link_template_bank(task="miniwob", cohort="web", out_dir=out_dir, dry_run=dry_run)
    if tb is not None:
        manifest["template_bank"] = tb

    at = NON_GAME_SFT / "miniwob" / "action_taking.jsonl"
    ss = NON_GAME_SFT / "miniwob" / "skill_selection.jsonl"
    sft = {}
    if link(at, out_dir / "sft/action_taking.jsonl", dry_run=dry_run):
        sft["action_taking"] = {
            "source": short(at), "n_rows": count_lines(at),
            "active_skill_kind": "synthetic (web/<VERB>) — NOT bank-derived",
        }
    if link(ss, out_dir / "sft/skill_selection.jsonl", dry_run=dry_run):
        sft["skill_selection"] = {
            "source": short(ss), "n_rows": count_lines(ss),
            "active_skill_kind": "bank-derived (OPERATOR/SUBGOAL)",
        }
    if sft:
        sft["run"] = short(NON_GAME_SFT)
        sft["mismatch_warning"] = (
            "action_taking and skill_selection use DIFFERENT active_skill schemes "
            "for the same sample (synthetic vs bank-derived); see MANIFEST notes."
        )
        manifest["sft"] = sft

    rollouts: Dict[str, Any] = {}
    if link(COLD_START_BG, out_dir / "rollouts/gpt54", dry_run=dry_run):
        n = sum(1 for d in COLD_START_BG.iterdir()
                if d.is_dir() and d.name.startswith("miniwob.") and (d / "episode_000.json").exists())
        rollouts["gpt-5.4"] = {"source": short(COLD_START_BG),
                               "n_task_dirs": n, "task_glob": "miniwob.*"}
    for m in ("claude", "gemini", "qwen"):
        src = OR_TX / m / "browsergym"
        if link(src, out_dir / "rollouts" / m, dry_run=dry_run):
            n = sum(1 for d in src.iterdir()
                    if d.is_dir() and d.name.startswith("miniwob.") and (d / "episode_000.json").exists())
            rollouts[m] = {"source": short(src), "n_task_dirs": n,
                           "task_glob": "miniwob.*"}
    manifest["rollouts"] = rollouts

    if not dry_run:
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_webshop(out_root: Path, *, dry_run: bool) -> Dict[str, Any]:
    out_dir = out_root / "non_game" / "webshop"
    manifest: Dict[str, Any] = {
        "task": "webshop", "category": "non_game", "modality": "web (browsergym)",
        "teachers": ["gpt-5.4", "claude-4.6", "gemini-3.1-pro", "qwen3-vl-235b"],
        "status": "ROLLOUTS + BANK + SKILL_QUERY (no decision-SFT JSONL yet)",
        "notes": [
            "Skill bank extracted 2026-05-10 from 200 episodes / 2538 steps via "
            "the standard 3-stage pipeline (label_qa_miniwob_intentions -> "
            "build_skillbank_qa_gpt54 -> label_skill_actions_qa_gpt54).",
            "scripts/build_multimodal_decision_sft.py still needs to be run to "
            "emit action_taking.jsonl / skill_selection.jsonl for webshop.",
        ],
    }

    bank_src = WEBSHOP_BANK_RUN / "webshop" / "skill_bank.jsonl"
    if link(bank_src, out_dir / "skill_bank.jsonl", dry_run=dry_run):
        manifest["skill_bank"] = {
            "source": short(bank_src),
            "n_skills": count_lines(bank_src),
            "kind": "bank-extracted (OPERATOR/SUBGOAL)",
        }

    tb = link_template_bank(task="webshop", cohort="web", out_dir=out_dir, dry_run=dry_run)
    if tb is not None:
        manifest["template_bank"] = tb

    rollouts: Dict[str, Any] = {}
    for tag, model in [("low", "gpt-5.4"), ("claude", "claude-4.6"),
                       ("gemini", "gemini-3.1-pro"), ("qwen", "qwen3-vl-235b")]:
        src = COLD_START_BG / f"webshop_50task_{tag}"
        link_name = "gpt54" if tag == "low" else tag
        if link(src, out_dir / "rollouts" / link_name, dry_run=dry_run):
            n = sum(1 for d in src.iterdir()
                    if d.is_dir() and d.name.startswith("webshop.") and (d / "episode_000.json").exists())
            rewards: List[float] = []
            for sub in src.iterdir():
                if sub.is_dir() and sub.name.startswith("webshop."):
                    rs = sub / "rollout_summary.json"
                    if rs.exists():
                        try:
                            rewards.append(float(json.loads(rs.read_text()).get("mean_reward") or 0))
                        except Exception:
                            pass
            entry: Dict[str, Any] = {
                "source": short(src), "n_task_dirs": n, "task_glob": "webshop.*",
                "mean_reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
                "n_above_half": sum(1 for r in rewards if r >= 0.5),
            }
            # Stage-1 labeled rollouts (intentions in OPERATOR/SUBGOAL form).
            tag_for_label = "gpt-5.4" if tag == "low" else tag
            labeled_dir = WEBSHOP_LABELED_RUN / tag_for_label
            if labeled_dir.is_dir():
                if link(labeled_dir, out_dir / "labeled" / link_name,
                        dry_run=dry_run):
                    n_labeled = sum(1 for d in labeled_dir.iterdir()
                                    if d.is_dir() and d.name.startswith("webshop."))
                    entry["labeled_intentions"] = {
                        "source": short(labeled_dir),
                        "n_task_dirs": n_labeled,
                    }
            # Stage-3 skill-query labeled rollouts.
            sq_dir = WEBSHOP_SKILL_QUERY_RUN / tag_for_label
            if sq_dir.is_dir():
                if link(sq_dir, out_dir / "skill_query" / link_name,
                        dry_run=dry_run):
                    n_sq = sum(1 for d in sq_dir.iterdir()
                               if d.is_dir() and d.name.startswith("webshop."))
                    entry["skill_query"] = {
                        "source": short(sq_dir),
                        "n_task_dirs": n_sq,
                    }
            rollouts[model] = entry
    manifest["rollouts"] = rollouts

    # Top-level pipeline summary so callers don't need to chase paths.
    manifest["pipeline"] = {
        "stage1_labeled_intentions": short(WEBSHOP_LABELED_RUN.parent),
        "stage2_skill_bank":         short(WEBSHOP_BANK_RUN),
        "stage3_skill_query":        short(WEBSHOP_SKILL_QUERY_RUN.parent),
    }

    if not dry_run:
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_qa_bench(bench: str, out_root: Path, *, dry_run: bool) -> Dict[str, Any]:
    """video_holmes / siv_bench / tir_bench / visual_toolbench."""
    out_dir = out_root / "non_game" / bench
    is_video = bench in ("video_holmes", "siv_bench")
    modality = "video QA" if is_video else "image QA"
    manifest: Dict[str, Any] = {
        "task": bench, "category": "non_game", "modality": modality,
        "teachers": ["gpt-5.4", "claude-4.6", "gemini-3.1-pro", "qwen3-vl-235b"],
    }

    bank_src = QA_BANK_RUN / bench / "skill_bank.jsonl"
    if link(bank_src, out_dir / "skill_bank.jsonl", dry_run=dry_run):
        manifest["skill_bank"] = {
            "source": short(bank_src),
            "n_skills": count_lines(bank_src),
            "kind": "bank-extracted (OPERATOR/SUBGOAL)",
        }

    tb_cohort = "vr_video" if is_video else "vr_image"
    tb = link_template_bank(task=bench, cohort=tb_cohort, out_dir=out_dir, dry_run=dry_run)
    if tb is not None:
        manifest["template_bank"] = tb

    at = NON_GAME_SFT / bench / "action_taking.jsonl"
    ss = NON_GAME_SFT / bench / "skill_selection.jsonl"
    sft = {}
    if link(at, out_dir / "sft/action_taking.jsonl", dry_run=dry_run):
        synth_prefix = {
            "video_holmes": "video_qa/<DIM>",
            "siv_bench": "social_qa/<DIM>",
            "tir_bench": "image_qa/<DIM> or tir_bench/UNKNOWN",
            "visual_toolbench": "visual_toolbench/OPEN_QA",
        }[bench]
        sft["action_taking"] = {
            "source": short(at), "n_rows": count_lines(at),
            "active_skill_kind": f"synthetic ({synth_prefix}) — NOT bank-derived",
        }
    if link(ss, out_dir / "sft/skill_selection.jsonl", dry_run=dry_run):
        sft["skill_selection"] = {
            "source": short(ss), "n_rows": count_lines(ss),
            "active_skill_kind": "bank-derived (OPERATOR/SUBGOAL)",
        }
    if sft:
        sft["run"] = short(NON_GAME_SFT)
        sft["mismatch_warning"] = (
            "action_taking uses synthetic dimension/qtype IDs; "
            "skill_selection uses bank-derived OPERATOR/SUBGOAL IDs. "
            "Same sample → DIFFERENT active_skill in the two files."
        )
        manifest["sft"] = sft

    # Rollouts (samples.jsonl, one per teacher)
    cold_start_root = COLD_START_VR_VID if is_video else COLD_START_VR_IMG
    rollouts: Dict[str, Any] = {}
    src = cold_start_root / bench / "samples.jsonl"
    if link(src, out_dir / "rollouts/gpt54.samples.jsonl", dry_run=dry_run):
        rollouts["gpt-5.4"] = {"source": short(src), "n_samples": count_lines(src)}

    transfer_kind = "vr_video" if is_video else "vr_image"
    for m in ("claude", "gemini", "qwen"):
        s = OR_TX / m / transfer_kind / bench / "samples.jsonl"
        if link(s, out_dir / "rollouts" / f"{m}.samples.jsonl", dry_run=dry_run):
            rollouts[m] = {"source": short(s), "n_samples": count_lines(s)}
    manifest["rollouts"] = rollouts

    if not dry_run:
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Plan only; create no files.")
    ap.add_argument("--out-dir", type=Path, default=ROOT,
                    help="Where to put the inventory (default: this script's dir).")
    args = ap.parse_args()

    out_root = args.out_dir.resolve()
    print(f"[inventory] repo     = {REPO}")
    print(f"[inventory] out_root = {out_root}{' (dry)' if args.dry_run else ''}")
    print()

    games_manifests: List[Dict[str, Any]] = []
    nongame_manifests: List[Dict[str, Any]] = []

    def _row(prefix: str, m: Dict[str, Any]) -> None:
        sft = m.get("sft", {})
        bank_n = m.get("skill_bank", {}).get("n_skills", "-")
        tmpl_n = m.get("template_bank", {}).get("n_skills", "-")
        print(f"  {prefix:<16} {m['task']:<35} bank={bank_n} tmpl={tmpl_n} "
              f"sft={sft.get('action_taking',{}).get('n_rows','-')} "
              f"+{sft.get('skill_selection',{}).get('n_rows','-')} "
              f"teachers={len(m.get('rollouts',{}))}")

    for g in GYMV_GAMES:
        m = build_gymv_game(g, out_root, dry_run=args.dry_run)
        games_manifests.append(m)
        _row("game/gym_v", m)

    for g in ENVWR_GAMES:
        m = build_envwr_game(g, out_root, dry_run=args.dry_run)
        games_manifests.append(m)
        _row("game/env_wr", m)

    nongame_manifests.append(build_miniwob(out_root, dry_run=args.dry_run))
    nongame_manifests.append(build_webshop(out_root, dry_run=args.dry_run))
    for b in NON_GAME_QA:
        nongame_manifests.append(build_qa_bench(b, out_root, dry_run=args.dry_run))

    for m in nongame_manifests:
        _row("non_game", m)

    inventory: Dict[str, Any] = {
        "repo_root": str(REPO),
        "scope": {
            "games": {
                "gym_v": GYMV_GAMES,
                "env_wrappers": ENVWR_GAMES,
                "n_games": len(GYMV_GAMES) + len(ENVWR_GAMES),
            },
            "non_game": {
                "browsergym": ["miniwob", "webshop"],
                "visual_reasoning": NON_GAME_QA,
                "n_tasks": 2 + len(NON_GAME_QA),
            },
        },
        "totals": {
            "n_skill_banks": sum(1 for m in games_manifests + nongame_manifests if m.get("skill_bank")),
            "n_template_banks": sum(1 for m in games_manifests + nongame_manifests if m.get("template_bank")),
            "n_lifted_templates": sum(
                m.get("template_bank", {}).get("n_skills", 0)
                for m in games_manifests + nongame_manifests
            ),
            "n_action_taking_rows": sum(
                m.get("sft", {}).get("action_taking", {}).get("n_rows", 0)
                for m in games_manifests + nongame_manifests
            ),
            "n_skill_selection_rows": sum(
                m.get("sft", {}).get("skill_selection", {}).get("n_rows", 0)
                for m in games_manifests + nongame_manifests
            ),
        },
        "games": games_manifests,
        "non_game": nongame_manifests,
    }

    if not args.dry_run:
        (out_root / "INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n")
        print(f"\n[inventory] wrote {out_root}/INVENTORY.json")
    print(f"[inventory] totals: "
          f"banks={inventory['totals']['n_skill_banks']}  "
          f"template_banks={inventory['totals']['n_template_banks']}  "
          f"lifted_templates={inventory['totals']['n_lifted_templates']}  "
          f"action_taking_rows={inventory['totals']['n_action_taking_rows']}  "
          f"skill_selection_rows={inventory['totals']['n_skill_selection_rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
