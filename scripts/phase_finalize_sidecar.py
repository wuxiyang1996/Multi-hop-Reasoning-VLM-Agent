#!/usr/bin/env python3
"""Phase boundary sidecar daemon.

Watches ``<run_dir>/phase_snapshots/`` for new ``phase_NN_<game>/``
directories and runs ``phase1_finalize.py`` automatically when one
appears with a complete ``phase_meta.json``.  This makes the
phase-finalize hook robust against bash's "script-loaded-once-at-start"
caching: even if ``run_phase1_curriculum.sh`` is already running with
an out-of-date in-memory parse of itself, this sidecar runs in a
separate process and reads the current versions of all the python
scripts on every invocation.

State persistence
-----------------
* ``<run_dir>/.phase_finalize_sidecar_state.json`` records which phases
  have been finalized (by ``phase_NN_<game>`` name).  Re-launching the
  sidecar is idempotent — already-handled snapshots are skipped.

* When the sidecar starts up, any ``phase_snapshots/phase_NN_<game>/``
  that already contains ``finalize_summary.json`` is auto-marked as
  handled (so a manually-recovered phase like Phase 1 in the
  2026-05-08 incident isn't double-finalized).

Curriculum resolution
---------------------
The next-game lookup is done by parsing the ``PHASES=( ... )`` array
out of ``scripts/run_phase1_curriculum.sh`` so the sidecar stays in
sync with whatever curriculum the user actually ran.

Usage::

    python scripts/phase_finalize_sidecar.py \\
        --run-dir runs/Qwen3.5-9B_<ts> \\
        --poll-seconds 60 \\
        --judge-url http://localhost:8001/v1

Run as a backgrounded process::

    nohup setsid python scripts/phase_finalize_sidecar.py \\
        --run-dir runs/<run> --log /tmp/sidecar.log \\
        > /tmp/sidecar.out 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

STATE_FILENAME = ".phase_finalize_sidecar_state.json"


def parse_curriculum(script_path: Path) -> List[Tuple[int, str, str]]:
    """Extract ``[(phase_num, game_slug, display), ...]`` from
    ``run_phase1_curriculum.sh``'s ``PHASES=()`` array."""
    if not script_path.is_file():
        return []
    text = script_path.read_text()
    m = re.search(r"PHASES=\(\s*(.*?)\s*\)", text, flags=re.DOTALL)
    if not m:
        return []
    out: List[Tuple[int, str, str]] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # "1:gymv_thunder_force_iii:Thunder Force III"
        sm = re.match(r'"\s*(\d+)\s*:\s*([^:]+)\s*:\s*([^"]+)\s*"', line)
        if sm:
            out.append((int(sm.group(1)), sm.group(2).strip(), sm.group(3).strip()))
    return out


def load_state(state_path: Path) -> Dict:
    if state_path.is_file():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            return {}
    return {}


def save_state(state_path: Path, state: Dict) -> None:
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    os.replace(tmp, state_path)


def list_complete_snapshots(snap_root: Path) -> List[Path]:
    """Return phase_NN_<game> dirs that have a phase_meta.json."""
    if not snap_root.is_dir():
        return []
    out: List[Path] = []
    for d in sorted(snap_root.iterdir()):
        if not d.is_dir():
            continue
        if not re.match(r"phase_\d{2,}_", d.name):
            continue
        if (d / "phase_meta.json").is_file():
            out.append(d)
    return out


def parse_phase_dir_name(name: str) -> Optional[Tuple[int, str]]:
    m = re.match(r"phase_(\d+)_(.+?)(?:_FAILED)?$", name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def run_finalize(
    *, run_dir: Path, phase_num: int, source_game: str, next_game: str,
    judge_url: str, judge_model: str,
    bucket_size: int, max_buckets: int, novelty_threshold: float,
    promote_best: bool, promote_window: int, no_promote_bank: bool,
    log_fp,
) -> int:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "phase1_finalize.py"),
        "--run-dir", str(run_dir),
        "--phase-num", str(phase_num),
        "--source-game", source_game,
        "--next-game", next_game,
        "--judge-url", judge_url,
        "--judge-model", judge_model,
        "--bucket-size", str(bucket_size),
        "--max-buckets", str(max_buckets),
        "--novelty-threshold", str(novelty_threshold),
        "--promote-window", str(promote_window),
    ]
    if not promote_best:
        cmd.append("--no-promote-best")
    if no_promote_bank:
        cmd.append("--no-promote-bank")
    log_fp.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] running: {' '.join(cmd)}\n")
    log_fp.flush()
    env = dict(os.environ)
    env["PROBE_JUDGE_URL"] = judge_url
    rc = subprocess.run(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT,
                        check=False).returncode
    log_fp.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] phase1_finalize exit={rc}\n")
    log_fp.flush()
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--curriculum-script", default=str(ROOT / "scripts" / "run_phase1_curriculum.sh"))
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--judge-url", default="http://localhost:8001/v1")
    ap.add_argument("--judge-model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--bucket-size", type=int, default=12)
    ap.add_argument("--max-buckets", type=int, default=25)
    ap.add_argument("--novelty-threshold", type=float, default=0.55)
    # Best-checkpoint promotion (forwarded to phase1_finalize.py).
    ap.add_argument("--promote-best", action="store_true", default=True,
                    help="Promote phase peak step's adapters + bank to LIVE "
                         "before crafter v2 / translation. Default: True.")
    ap.add_argument("--no-promote-best", dest="promote_best",
                    action="store_false")
    ap.add_argument("--promote-window", type=int, default=3)
    ap.add_argument("--no-promote-bank", action="store_true",
                    help="Promote adapters only — keep LIVE bank intact.")
    ap.add_argument("--log", default="",
                    help="Log file path; empty → stdout/stderr")
    ap.add_argument("--max-iterations", type=int, default=0,
                    help="Exit after N polls (0 = run forever)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    snap_root = run_dir / "phase_snapshots"
    state_path = run_dir / STATE_FILENAME

    if args.log:
        log_fp = open(args.log, "a", buffering=1)
    else:
        log_fp = sys.stdout

    def log(msg: str) -> None:
        log_fp.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        log_fp.flush()

    log(f"sidecar starting; run_dir={run_dir}  poll={args.poll_seconds}s")

    curriculum = parse_curriculum(Path(args.curriculum_script))
    if not curriculum:
        log(f"ERROR: could not parse curriculum from {args.curriculum_script}; exiting")
        return 1
    log(f"curriculum: {curriculum}")

    next_game_for: Dict[str, str] = {}
    by_num: Dict[int, Tuple[str, str]] = {}
    for i, (num, game, display) in enumerate(curriculum):
        by_num[num] = (game, display)
        if i + 1 < len(curriculum):
            next_game_for[game] = curriculum[i + 1][1]
        else:
            next_game_for[game] = ""

    state = load_state(state_path)
    state.setdefault("handled", [])

    # Auto-mark already-finalized snapshots so we don't re-run them.
    initial_seen = list_complete_snapshots(snap_root)
    for snap in initial_seen:
        if (snap / "finalize_summary.json").is_file() and snap.name not in state["handled"]:
            log(f"already-finalized: marking {snap.name} as handled (has finalize_summary.json)")
            state["handled"].append(snap.name)
    save_state(state_path, state)

    iteration = 0
    shutdown = False

    def _stop(*_):
        nonlocal shutdown
        log("received signal; will exit after current poll")
        shutdown = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while not shutdown:
        iteration += 1
        try:
            snaps = list_complete_snapshots(snap_root)
            for snap in snaps:
                if snap.name in state["handled"]:
                    continue
                parsed = parse_phase_dir_name(snap.name)
                if not parsed:
                    log(f"skip: cannot parse phase from {snap.name}")
                    state["handled"].append(snap.name)
                    save_state(state_path, state)
                    continue
                phase_num, game = parsed
                next_game = next_game_for.get(game, "")
                log(f"NEW SNAPSHOT detected: {snap.name} (phase={phase_num}, "
                    f"source={game}, next={next_game or 'none'}); running finalize…")
                rc = run_finalize(
                    run_dir=run_dir, phase_num=phase_num,
                    source_game=game, next_game=next_game,
                    judge_url=args.judge_url, judge_model=args.judge_model,
                    bucket_size=args.bucket_size, max_buckets=args.max_buckets,
                    novelty_threshold=args.novelty_threshold,
                    promote_best=args.promote_best,
                    promote_window=args.promote_window,
                    no_promote_bank=args.no_promote_bank,
                    log_fp=log_fp,
                )
                state["handled"].append(snap.name)
                state.setdefault("history", []).append({
                    "snap": snap.name, "rc": rc,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                save_state(state_path, state)
                if rc != 0:
                    log(f"WARNING: finalize for {snap.name} returned rc={rc}")
        except Exception as exc:                                    # noqa: BLE001
            log(f"poll iteration {iteration} raised: {exc!r}")

        if args.max_iterations and iteration >= args.max_iterations:
            log(f"max-iterations ({args.max_iterations}) reached; exiting")
            break

        for _ in range(args.poll_seconds):
            if shutdown:
                break
            time.sleep(1)

    log("sidecar exiting cleanly")
    if args.log:
        log_fp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
