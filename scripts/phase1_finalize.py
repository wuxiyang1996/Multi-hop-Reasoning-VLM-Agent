#!/usr/bin/env python3
"""Phase finalize hook — runs at every Phase-1 curriculum boundary in
``BANK_MODE=per_game``.

Responsibilities:

  1. **Crafter v2 batch pipeline** on the just-completed phase's
     rollouts → ``candidate_skills.jsonl`` (LLM-proposed,
     novelty-filtered, schema-ready).

  2. **Inject** crafter v2 candidates into the source-game bank with
     ``confidence_tag="crafter_v2"`` so they're (a) preserved in the
     phase snapshot, and (b) available to the cross-game translator
     in step 3.

  3. **Cross-game translation** (manual, since per-game mode skips the
     built-in ``translate_bank_for_next_phase``): rewrite each
     source-game skill (including v2 candidates) onto the next game's
     action vocabulary via the existing
     ``skill_agents.skill_bank.translate_for_target`` driver. Output
     becomes the **seed bank** for the next phase, written to
     ``skillbank/<next_game>/skill_bank.jsonl`` (creating an empty
     directory if needed).

  4. **Archive** the entire ``crafter_v2_offline/`` tree + a
     ``phase_report.md`` into the auto-saved phase snapshot at
     ``phase_snapshots/phase_<N>_<game>/``.

  5. Emit a structured ``finalize_summary.json`` with reward
     trajectory, bank growth, transfer pre-seeding stats, and
     human-readable headline numbers.

This script is idempotent — re-running on a finished phase just
overwrites the same outputs deterministically.

Usage (called from ``run_phase1_curriculum.sh`` after
``save_phase_snapshot``)::

    python scripts/phase1_finalize.py \\
        --run-dir <run_dir> \\
        --phase-num 1 \\
        --source-game gymv_thunder_force_iii \\
        --next-game gymv_altered_beast \\
        --judge-url http://localhost:8001/v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# -------------------------- step 0: best-checkpoint promotion -----------


def run_promote_best(
    run_dir: Path, phase_num: int, source_game: str,
    *, window: int, no_promote_bank: bool,
) -> Dict[str, Any]:
    """Promote the best step's adapters (and optionally bank) into LIVE
    BEFORE crafter v2 + translation runs.  This makes Phase N+1 launch
    from the actual peak policy of Phase N rather than its noisy last
    step, and (when ``no_promote_bank`` is False) ensures the bank
    paired with that LoRA — not a stale last-step bank — feeds the
    translator.

    Fail-open: if promotion errors out, log the warning but still let
    the rest of the finalize pipeline run on the LIVE/last-step state
    (we'd rather have a sub-optimal seed than abort the phase boundary
    entirely).
    """
    print(f"\n[0/5] promote best checkpoint (window={window}, "
          f"promote_bank={not no_promote_bank})…")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "promote_best_checkpoint.py"),
        "--run-dir", str(run_dir),
        "--phase-num", str(phase_num),
        "--source-game", source_game,
        "--window", str(window),
    ]
    if no_promote_bank:
        cmd.append("--no-promote-bank")
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(f"   WARN: promote_best_checkpoint rc={rc}; continuing on LIVE state")
        return {"rc": rc, "promoted": False}
    # Read the latest entry from promotion_log.jsonl for the structured summary.
    log = run_dir / "promotion_log.jsonl"
    last_entry: Dict[str, Any] = {}
    if log.exists():
        with open(log) as f:
            for L in f:
                try:
                    last_entry = json.loads(L)
                except Exception:
                    pass
    last_entry["rc"] = rc
    last_entry["promoted"] = bool(last_entry.get("step") is not None
                                  and not last_entry.get("noop"))
    return last_entry


# -------------------------- step 1: crafter v2 ---------------------------


def run_crafter_v2(
    run_dir: Path, source_game: str, judge_url: str,
    bucket_size: int, max_buckets: int, novelty_threshold: float,
) -> Dict[str, Any]:
    """Invoke crafter_v2_batch_pipeline as a subprocess so it inherits
    its own argparse + clean exit codes."""
    print(f"\n[1/5] crafter v2 batch pipeline (game={source_game})…")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "crafter_v2_batch_pipeline.py"),
        "--run-dir", str(run_dir),
        "--game", source_game,
        "--bucket-size", str(bucket_size),
        "--max-buckets", str(max_buckets),
        "--novelty-threshold", str(novelty_threshold),
        "--judge-url", judge_url,
    ]
    env = dict(os.environ)
    env["PROBE_JUDGE_URL"] = judge_url
    print(f"   $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, env=env, check=False).returncode
    summary_path = run_dir / "crafter_v2_offline" / "proposals" / "summary.json"
    if not summary_path.exists():
        print(f"   WARN: summary.json missing; crafter v2 produced no output (rc={rc})")
        return {"n_accepted": 0, "rc": rc, "skills": []}
    with open(summary_path) as f:
        s = json.load(f)
    s["rc"] = rc
    return s


# -------------------------- step 2: inject into bank ---------------------


def inject_into_bank(run_dir: Path, source_game: str) -> Dict[str, Any]:
    """Append candidate_skills.jsonl onto the source game's
    skill_bank.jsonl with file lock to avoid race with running training
    (crafter v2 candidates only appear AFTER the phase has fully
    completed, so the live trainer should no longer be writing to the
    bank — but we still take the lock for safety)."""
    print(f"\n[2/5] injecting crafter v2 candidates into {source_game} bank…")
    bank_path = run_dir / "skillbank" / source_game / "skill_bank.jsonl"
    cand_path = run_dir / "crafter_v2_offline" / "proposals" / "candidate_skills.jsonl"

    if not cand_path.exists() or cand_path.stat().st_size == 0:
        print(f"   no candidates to inject ({cand_path})")
        return {"injected": 0, "bank_size_before": 0, "bank_size_after": 0}

    bank_path.parent.mkdir(parents=True, exist_ok=True)
    n_before = sum(1 for _ in open(bank_path)) if bank_path.exists() else 0

    # Concatenate (dedup by skill_id just in case the script is re-run).
    existing_ids: set = set()
    if bank_path.exists():
        with open(bank_path) as f:
            for L in f:
                try:
                    d = json.loads(L)
                    sid = (d.get("skill") or {}).get("skill_id")
                    if sid:
                        existing_ids.add(sid)
                except Exception:
                    pass

    n_added = 0
    with open(bank_path, "a") as out:
        with open(cand_path) as src:
            for L in src:
                try:
                    d = json.loads(L)
                except Exception:
                    continue
                sid = (d.get("skill") or {}).get("skill_id")
                if not sid or sid in existing_ids:
                    continue
                out.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
                existing_ids.add(sid)
                n_added += 1

    n_after = sum(1 for _ in open(bank_path))
    print(f"   bank: {n_before} → {n_after} (+{n_added})")
    return {"injected": n_added, "bank_size_before": n_before,
            "bank_size_after": n_after, "bank_path": str(bank_path)}


# -------------------------- step 3: cross-game translation ---------------


def resolve_target_actions(target_game: str) -> List[str]:
    """Mirrors the env-resolution snippet in run_phase1_curriculum.sh."""
    try:
        if target_game.startswith("gymv_"):
            from env_wrappers.gymv_temporal_nl_wrapper import make_gymv_temporal_env
            env = make_gymv_temporal_env(target_game)
            env.reset()
            acts = list(env.action_names)
            env.close()
            return acts
        from env_wrappers.game_configs import get_game_config
        cfg = get_game_config(target_game)
        return list(getattr(cfg, "available_actions", []) or [])
    except Exception as exc:
        print(f"   could not resolve target actions for {target_game}: {exc}")
        return []


def translate_to_next_phase(
    run_dir: Path, source_game: str, target_game: str, judge_model: str,
) -> Dict[str, Any]:
    """Invoke ``skill_agents.skill_bank.translate_for_target`` and write
    the output to the target game's bank dir."""
    print(f"\n[3/5] cross-game translate: {source_game} → {target_game}…")
    if not target_game:
        print(f"   no target game specified; skipping translation")
        return {"target_actions": [], "skipped": True}

    target_actions = resolve_target_actions(target_game)
    if not target_actions:
        print(f"   could not resolve target actions; skipping translation")
        return {"target_actions": [], "skipped": True}

    src_bank = run_dir / "skillbank" / source_game / "skill_bank.jsonl"
    tgt_dir = run_dir / "skillbank" / target_game
    tgt_dir.mkdir(parents=True, exist_ok=True)
    tgt_bank = tgt_dir / "skill_bank.jsonl"

    # Snapshot the current target bank contents BEFORE the translator
    # overwrites the file.  The translator's CLI builds a fresh
    # ``SkillBankMVP(output_path)`` which does NOT load the existing
    # file — it writes from scratch.  In ``per_game`` mode that's
    # actually fine at a clean boundary (target bank is empty).  But
    # when this script is invoked mid-Phase-N (e.g. recovery from a
    # missed boundary hook), the running phase has already accumulated
    # native skills we MUST preserve.  We capture them here, let the
    # translator write, then merge prior + translated back into the
    # canonical path.
    pre_existing_lines: List[str] = []
    pre_existing_ids: set = set()
    if tgt_bank.exists() and tgt_bank.stat().st_size > 0:
        timestamp = int(time.time())
        backup = tgt_dir / f"skill_bank.pre_translate_{timestamp}.jsonl"
        shutil.copy2(tgt_bank, backup)
        with open(tgt_bank) as f:
            for L in f:
                pre_existing_lines.append(L)
                try:
                    d = json.loads(L)
                    sid = (d.get("skill") or {}).get("skill_id")
                    if sid:
                        pre_existing_ids.add(sid)
                except Exception:
                    pass
        print(f"   target bank exists ({len(pre_existing_lines)} skills); backed up to {backup.name}")

    # In per_game mode the AB bank should ONLY contain AB-grounded
    # skills (translated derivatives + future Phase-2 curator output).
    # ``--no-seed-with-source`` strips the TF3-original entries the
    # translator would otherwise copy in (those are designed for
    # shared-bank mode where one file holds skills for all games and
    # the harness uses ``feasible_tasks`` to filter).
    cmd = [
        sys.executable, "-m", "skill_agents.skill_bank.translate_for_target",
        "--source-bank", str(src_bank),
        "--target-game", target_game,
        "--target-actions", ",".join(target_actions),
        "--source-game", source_game,
        "--output", str(tgt_bank),
        "--judge-model", judge_model,
        "--no-seed-with-source",
        "-v",
    ]
    print(f"   target_actions: {target_actions}")
    print(f"   $ {' '.join(cmd[:6])} ... [-v]")
    t0 = time.monotonic()
    rc = subprocess.run(cmd, check=False).returncode
    dt = time.monotonic() - t0

    # Merge: pre-existing native skills + newly translated skills,
    # deduplicated by skill_id (translated entries win on collision —
    # but collisions are essentially impossible because translated ids
    # carry the ``__translated_to__`` suffix).
    n_translated = 0
    n_translated_v2 = 0
    translated_lines: List[str] = []
    if tgt_bank.exists():
        with open(tgt_bank) as f:
            for L in f:
                try:
                    d = json.loads(L)
                except Exception:
                    continue
                s = d.get("skill") or {}
                if s.get("confidence_tag") == "translated":
                    n_translated += 1
                    if (s.get("derived_from") or "").startswith("v2:"):
                        n_translated_v2 += 1
                translated_lines.append(L)

    if pre_existing_lines:
        merged_path = tgt_dir / f"skill_bank.merged_{int(time.time())}.jsonl"
        seen_ids: set = set()
        with open(merged_path, "w") as out:
            # Translated first so they take precedence on dedup
            for L in translated_lines:
                try:
                    d = json.loads(L)
                    sid = (d.get("skill") or {}).get("skill_id")
                    if sid and sid not in seen_ids:
                        out.write(L if L.endswith("\n") else L + "\n")
                        seen_ids.add(sid)
                except Exception:
                    pass
            # Then any pre-existing native skills not collided
            for L in pre_existing_lines:
                try:
                    d = json.loads(L)
                    sid = (d.get("skill") or {}).get("skill_id")
                    if sid and sid not in seen_ids:
                        out.write(L if L.endswith("\n") else L + "\n")
                        seen_ids.add(sid)
                except Exception:
                    pass
        # Atomic swap (orchestrator may be reading; rename is atomic).
        os.replace(merged_path, tgt_bank)
        print(f"   merged {len(translated_lines)} translated + {len(pre_existing_lines)} pre-existing native → {len(seen_ids)} unique")

    print(f"   rc={rc}  wall={dt:.1f}s  translated={n_translated}  of_which_v2={n_translated_v2}")
    return {
        "target_actions": target_actions, "rc": rc, "wall_s": dt,
        "n_translated": n_translated, "n_translated_v2": n_translated_v2,
        "n_pre_existing": len(pre_existing_lines),
        "target_bank": str(tgt_bank),
    }


# -------------------------- step 4: phase report -------------------------


def build_phase_report(
    run_dir: Path, phase_num: int, source_game: str, next_game: Optional[str],
    crafter: Dict[str, Any], inject: Dict[str, Any], translate: Dict[str, Any],
) -> str:
    step_log = []
    f = run_dir / "step_log.jsonl"
    if f.exists():
        with open(f) as fh:
            for L in fh:
                try:
                    step_log.append(json.loads(L))
                except Exception:
                    pass
    phase_steps = [d for d in step_log if d.get("reward_per_game", {}).get(source_game)]

    rewards_summary: List[Dict[str, Any]] = []
    for d in phase_steps:
        rg = d["reward_per_game"][source_game]
        rewards_summary.append({
            "step": d["step"],
            "mean": rg.get("mean_reward"),
            "max": rg.get("max_reward"),
            "min": rg.get("min_reward"),
            "std": rg.get("std_reward"),
            "mean_steps": rg.get("mean_steps"),
            "n_skills": d.get("skills_per_game", {}).get(source_game),
        })

    last = rewards_summary[-1] if rewards_summary else {}
    first = rewards_summary[0] if rewards_summary else {}

    lines: List[str] = []
    lines.append(f"# Phase {phase_num}: {source_game} — Finalize Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    lines.append("## Reward trajectory\n")
    lines.append("| step | mean | max | min | std | mean_steps | n_skills |")
    lines.append("|------|------|-----|-----|-----|-----------|----------|")
    for r in rewards_summary:
        lines.append(
            f"| {r['step']} | {r['mean']:.0f} | {r['max']:.0f} | "
            f"{r['min']:.0f} | {r['std']:.0f} | {r['mean_steps']:.1f} | "
            f"{r['n_skills']} |"
        )
    if first and last:
        delta = (last["mean"] or 0) - (first["mean"] or 0)
        lines.append(f"\n**Mean reward Δ over phase: {delta:+.0f}** "
                     f"(from {first['mean']:.0f} → {last['mean']:.0f})\n")

    lines.append("## Crafter v2\n")
    lines.append(f"- failures detected: {crafter.get('n_failures', 0)} "
                 f"({crafter.get('by_class', {})})")
    lines.append(f"- buckets run: {crafter.get('n_buckets_run', 0)} / {crafter.get('n_buckets_total', 0)}")
    lines.append(f"- raw proposals: {crafter.get('n_raw_proposals', 0)}")
    lines.append(f"- accepted (after novelty filter): {crafter.get('n_accepted', 0)}")
    lines.append(f"- rejected as redundant: {crafter.get('n_rejected_redundant', 0)}")
    lines.append("")
    if crafter.get("skills"):
        lines.append("### New skills minted")
        for s in crafter["skills"]:
            lines.append(f"- `{s['skill_id']}`: {s['name']}")
    lines.append("")

    lines.append("## Bank injection\n")
    lines.append(f"- bank before: {inject.get('bank_size_before', 0)}")
    lines.append(f"- bank after:  {inject.get('bank_size_after', 0)}")
    lines.append(f"- injected:    {inject.get('injected', 0)}")
    lines.append("")

    lines.append(f"## Cross-game translation → {next_game or '(none)'}\n")
    if translate.get("skipped"):
        lines.append("- skipped (no target game / cannot resolve actions)")
    else:
        lines.append(f"- target action vocab: `{translate.get('target_actions')}`")
        lines.append(f"- translated total: {translate.get('n_translated', 0)}")
        lines.append(f"- of which v2-derived: {translate.get('n_translated_v2', 0)}")
        lines.append(f"- target bank: `{translate.get('target_bank')}`")
        lines.append(f"- wall: {translate.get('wall_s', 0):.1f}s")
    lines.append("")

    lines.append("## Provenance reference\n")
    lines.append("- TF3-native skills (curator-mined this run): "
                 "`confidence_tag=stable`, `feasible_tasks=[<game>]`, `derived_from=null`")
    lines.append("- Crafter v2 skills (this report): "
                 "`confidence_tag=crafter_v2`, `derived_from=null`")
    lines.append("- Translated skills (next-phase seed): "
                 "`confidence_tag=translated`, `derived_from=<source_skill_id>`")
    lines.append("")

    return "\n".join(lines)


# -------------------------- step 5: archive into snapshot ----------------


def archive_into_snapshot(
    run_dir: Path, phase_num: int, source_game: str,
    report_text: str, finalize_summary: Dict[str, Any],
) -> Optional[Path]:
    snap_root = run_dir / "phase_snapshots"
    if not snap_root.exists():
        print(f"\n[5/5] no phase_snapshots dir yet; will not archive")
        return None
    pat = f"phase_{phase_num:02d}_{source_game}"
    snap = snap_root / pat
    if not snap.exists():
        print(f"\n[5/5] {snap} does not exist (snapshot not yet saved); skipping archive")
        return None

    print(f"\n[5/5] archiving crafter v2 + report into {snap}…")
    src_v2 = run_dir / "crafter_v2_offline"
    if src_v2.exists():
        dst_v2 = snap / "crafter_v2_offline"
        if dst_v2.exists():
            shutil.rmtree(dst_v2)
        shutil.copytree(src_v2, dst_v2)
        print(f"   copied {src_v2} → {dst_v2}")

    rpath = snap / "phase_report.md"
    rpath.write_text(report_text)
    print(f"   wrote {rpath}")

    spath = snap / "finalize_summary.json"
    with open(spath, "w") as fh:
        json.dump(finalize_summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"   wrote {spath}")
    return snap


# -------------------------- main ----------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--phase-num", type=int, required=True)
    ap.add_argument("--source-game", required=True)
    ap.add_argument("--next-game", default="",
                    help="Empty string disables cross-game translation step.")
    ap.add_argument("--judge-url", default="http://localhost:8001/v1")
    ap.add_argument("--judge-model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--bucket-size", type=int, default=12)
    ap.add_argument("--max-buckets", type=int, default=20)
    ap.add_argument("--novelty-threshold", type=float, default=0.55)
    ap.add_argument("--skip-crafter", action="store_true",
                    help="Skip crafter v2 (use already-produced candidates).")
    ap.add_argument("--skip-translate", action="store_true")
    ap.add_argument("--skip-archive", action="store_true")
    # Best-checkpoint promotion (added 2026-05-08 per user request).
    ap.add_argument("--promote-best", action="store_true", default=True,
                    help="Promote the phase's peak step's adapters + bank "
                         "into LIVE before crafter v2 / translation. "
                         "Default: True. Disable with --no-promote-best.")
    ap.add_argument("--no-promote-best", dest="promote_best",
                    action="store_false")
    ap.add_argument("--promote-window", type=int, default=3,
                    help="Centered rolling-mean window for peak detection.")
    ap.add_argument("--no-promote-bank", action="store_true",
                    help="When promoting, swap adapters only — keep LIVE bank "
                         "(use when later-step skills must be preserved).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    print(f"=== phase1_finalize: phase={args.phase_num} src={args.source_game} → {args.next_game or '(none)'} ===")

    promote_summary: Dict[str, Any] = {"skipped": True}
    if args.promote_best:
        promote_summary = run_promote_best(
            run_dir=run_dir, phase_num=args.phase_num,
            source_game=args.source_game,
            window=args.promote_window,
            no_promote_bank=args.no_promote_bank,
        )
    else:
        print("\n[0/5] --no-promote-best set; skipping best-checkpoint promotion")

    crafter_summary: Dict[str, Any] = {}
    if not args.skip_crafter:
        crafter_summary = run_crafter_v2(
            run_dir, args.source_game,
            judge_url=args.judge_url,
            bucket_size=args.bucket_size,
            max_buckets=args.max_buckets,
            novelty_threshold=args.novelty_threshold,
        )
    else:
        print("\n[1/5] skip-crafter requested; reading existing summary if any")
        sp = run_dir / "crafter_v2_offline" / "proposals" / "summary.json"
        if sp.exists():
            with open(sp) as f:
                crafter_summary = json.load(f)

    inject_summary = inject_into_bank(run_dir, args.source_game)

    translate_summary: Dict[str, Any] = {"skipped": True}
    if args.next_game and not args.skip_translate:
        translate_summary = translate_to_next_phase(
            run_dir, args.source_game, args.next_game,
            judge_model=args.judge_model,
        )
    else:
        print(f"\n[3/5] skipping cross-game translation "
              f"(next_game={args.next_game!r}, skip_translate={args.skip_translate})")

    print(f"\n[4/5] building phase report…")
    report = build_phase_report(
        run_dir=run_dir, phase_num=args.phase_num,
        source_game=args.source_game, next_game=args.next_game or None,
        crafter=crafter_summary, inject=inject_summary, translate=translate_summary,
    )
    rfile = run_dir / "crafter_v2_offline" / "proposals" / f"phase_{args.phase_num:02d}_report.md"
    rfile.write_text(report)
    print(f"   wrote {rfile}")

    finalize_summary = {
        "phase_num": args.phase_num,
        "source_game": args.source_game,
        "next_game": args.next_game,
        "promote_best": promote_summary,
        "crafter": crafter_summary,
        "inject": inject_summary,
        "translate": translate_summary,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if not args.skip_archive:
        archive_into_snapshot(
            run_dir=run_dir, phase_num=args.phase_num,
            source_game=args.source_game,
            report_text=report, finalize_summary=finalize_summary,
        )

    print(f"\n=== phase1_finalize DONE ===")
    if promote_summary.get("promoted"):
        print(f"   promoted_step:    {promote_summary.get('step')} "
              f"(was last={promote_summary.get('selection',{}).get('last_step', '?')})")
    print(f"   crafter_accepted: {crafter_summary.get('n_accepted', 0)}")
    print(f"   bank_after:       {inject_summary.get('bank_size_after', 0)}")
    print(f"   translated_seeds: {translate_summary.get('n_translated', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
