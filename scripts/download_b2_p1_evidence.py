#!/usr/bin/env python3
"""P1: download verifiable-transfer raw evidence from B2.

Downloads what exists remotely:
  - SFT_Data/gymv_games/          (full GymV episode outputs)
  - SFT_Data/env_wrapper_games/   (Tetris/2048/Mario/CandyCrush episodes)
  - Game-AI-Agent labeling skill-labeled episodes (stand-in for empty
    labeling/skill_actions_out + intentions_out)
  - runs/*/grpo_data, rewards, transfer/intention/step_progress logs

Note: Multi-hop labeling/skill_actions_out and intentions_out are empty
on both emnlp2026 and emnlp2026-2.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from b2sdk.v2 import InMemoryAccountInfo, B2Api

KEY_ID = os.environ.get("B2_KEY_ID")
APP_KEY = os.environ.get("B2_APPLICATION_KEY")
if not KEY_ID or not APP_KEY:
    raise SystemExit("Set B2_KEY_ID and B2_APPLICATION_KEY in the environment.")

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "b2_p1_evidence_download.log"
MANIFEST = REPO / "b2_p1_evidence_manifest.tsv"
WORKERS = 24

# (bucket, remote_prefix, local_prefix, filter_fn|None)
# filter_fn(remote_name, size) -> bool
SKILL_LABELED_GAMES = {
    "tetris",
    "twenty_forty_eight",
    "super_mario",
    "candy_crush",
    "sokoban",
}

RUNS_EMNLP2 = [
    "candy_crush_coevo_v4_20260519_093912",
    "gymv_columns_coevo_v4_20260519_072135",
    "gymv_streets_of_rage_2_coevo_v5_20260520_010806",
    "gymv_strider_coevo_v5_20260519_184613",
    "gymv_thunder_force_iii_coevo_v10_asym_warmstart",
    "tetris_coevo_v4_20260520_063432",
]
RUNS_EMNLP = [
    "gymv_airstriker_stage2_v9_20260521_180740",
    "gymv_altered_beast_stage2_v9_20260521_214938",
    "gymv_dynamite_headdy_stage2_20260520_094617",
    "gymv_space_harrier_ii_stage2_20260520_094617",
]

EVIDENCE_PREFIXES = (
    "grpo_data/",
    "rewards/",
    "transfer_log/",
    "intention_log/",
    "step_progress_log/",
    "reward_shaping_log/",
)


def skill_labeled_filter(remote: str, _size: int) -> bool:
    # .../gpt54_skill_labeled/<game>/...
    parts = remote.split("/")
    try:
        i = parts.index("gpt54_skill_labeled")
        game = parts[i + 1]
    except (ValueError, IndexError):
        return False
    return game in SKILL_LABELED_GAMES


def run_evidence_filter(remote: str, _size: int) -> bool:
    # keep only evidence subtrees under the run
    # remote = .../runs/<run>/<rel>
    marker = "/runs/"
    if marker not in remote:
        return False
    rel = remote.split(marker, 1)[1]
    # <run_name>/<sub...>
    if "/" not in rel:
        return False
    sub = rel.split("/", 1)[1]
    return sub.startswith(EVIDENCE_PREFIXES)


JOBS = [
    ("emnlp2026-2", "workspace/SFT_Data/gymv_games/", "SFT_Data/gymv_games/", None),
    ("emnlp2026-2", "workspace/SFT_Data/env_wrapper_games/", "SFT_Data/env_wrapper_games/", None),
    (
        "emnlp2026-2",
        "workspace/Game-AI-Agent/labeling/output/gpt54_skill_labeled/",
        "labeling/gpt54_skill_labeled/",  # stand-in for empty skill_actions_out
        skill_labeled_filter,
    ),
    (
        "emnlp2026-2",
        "workspace/Game-AI-Agent/labeling/output/gpt54_skillbank/",
        "labeling/gpt54_skillbank/",
        None,
    ),
]

for run in RUNS_EMNLP2:
    JOBS.append(
        (
            "emnlp2026-2",
            f"workspace/Multi-hop-Reasoning-VLM-Agent/runs/{run}/",
            f"runs/{run}/",
            run_evidence_filter,
        )
    )
for run in RUNS_EMNLP:
    JOBS.append(
        (
            "emnlp2026",
            f"Multi-hop-Reasoning-VLM-Agent/runs/{run}/",
            f"runs/{run}/",
            run_evidence_filter,
        )
    )

_lock = threading.Lock()
done = skipped = failed = 0
done_bytes = 0
failures: list[tuple[str, str]] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _lock:
        with open(LOG, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)


def hr(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def authorize() -> B2Api:
    info = InMemoryAccountInfo()
    api = B2Api(info)
    api.authorize_account("production", KEY_ID, APP_KEY)
    return api


def list_job(api: B2Api, bucket_name: str, remote_prefix: str, filt):
    bucket = api.get_bucket_by_name(bucket_name)
    out = []
    for fv, _ in bucket.ls(folder_to_list=remote_prefix, recursive=True):
        if filt is not None and not filt(fv.file_name, fv.size):
            continue
        out.append((fv.file_name, fv.size))
    return out


def download_one(api: B2Api, bucket_name: str, remote: str, size: int, dest: Path):
    global done, skipped, failed, done_bytes
    if dest.exists():
        try:
            if dest.stat().st_size == size:
                with _lock:
                    skipped += 1
                    done += 1
                    done_bytes += size
                return ("skip", remote, None)
        except OSError:
            pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, 5):
        try:
            bucket = api.get_bucket_by_name(bucket_name)
            bucket.download_file_by_name(remote).save_to(str(tmp))
            os.replace(tmp, dest)
            with _lock:
                done += 1
                done_bytes += size
            return ("ok", remote, None)
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt, 30))
    with _lock:
        failed += 1
        failures.append((remote, str(last_err)))
    return ("fail", remote, str(last_err))


def main() -> int:
    LOG.write_text("")
    log(f"Repo: {REPO}")
    api = authorize()
    log("Authorized")

    # Record empty official paths for audit
    gap_note = REPO / "b2_p1_evidence_GAPS.txt"
    gap_note.write_text(
        "\n".join(
            [
                "OFFICIAL PATHS EMPTY ON B2 (emnlp2026 + emnlp2026-2):",
                "  Multi-hop-Reasoning-VLM-Agent/labeling/skill_actions_out/",
                "  Multi-hop-Reasoning-VLM-Agent/labeling/intentions_out/",
                "",
                "SUBSTITUTES DOWNLOADED:",
                "  labeling/gpt54_skill_labeled/{tetris,twenty_forty_eight,...}/",
                "    -> per-step intentions + skills + available_actions + state/next_state",
                "  SFT_Data/env_wrapper_games/ -> full Tetris/2048 episode rollouts",
                "  SFT_Data/gymv_games/ -> full GymV episode rollouts",
                "  runs/*/grpo_data|rewards|transfer_log|... -> coevo step evidence",
                "",
            ]
        )
    )

    plan = []
    with open(MANIFEST, "w") as mf:
        mf.write("bucket\tremote\tlocal\tsize\n")
        for bucket_name, remote_prefix, local_prefix, filt in JOBS:
            files = list_job(api, bucket_name, remote_prefix, filt)
            log(f"Plan {bucket_name}/{remote_prefix} -> {len(files)} files, {hr(sum(s for _, s in files))}")
            for remote, size in files:
                rel = remote[len(remote_prefix) :]
                dest = REPO / local_prefix / rel
                plan.append((bucket_name, remote, size, dest))
                mf.write(f"{bucket_name}\t{remote}\t{dest.relative_to(REPO)}\t{size}\n")

    total_files = len(plan)
    total_bytes = sum(s for _, _, s, _ in plan)
    log(f"TOTAL: {total_files} files, {hr(total_bytes)}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(download_one, api, b, r, s, d) for b, r, s, d in plan]
        for i, fut in enumerate(as_completed(futs), 1):
            status, name, err = fut.result()
            if status == "fail":
                log(f"FAIL {name}: {err}")
            if i % 50 == 0 or i == total_files:
                elapsed = max(time.time() - t0, 1)
                log(
                    f"PROGRESS {done}/{total_files} skipped={skipped} failed={failed} "
                    f"{hr(done_bytes)}/{hr(total_bytes)} rate={hr(done_bytes/elapsed)}/s"
                )

    log("=" * 60)
    log(
        f"DONE done={done}/{total_files} skipped={skipped} failed={failed} "
        f"bytes={hr(done_bytes)} elapsed={(time.time()-t0)/60:.1f} min"
    )
    if failures:
        fail_path = REPO / "b2_p1_evidence_failures.tsv"
        with open(fail_path, "w") as f:
            for n, e in failures:
                f.write(f"{n}\t{e}\n")
        log(f"Failures: {fail_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
