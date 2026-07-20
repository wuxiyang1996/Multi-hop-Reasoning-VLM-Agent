#!/usr/bin/env python3
"""Download game-running checkpoints + skill extracts from B2 emnlp buckets.

Sources:
  - emnlp2026-2: latest coevo runs + SFT skill_banks + sft_per_game_v3 finals
  - emnlp2026:   stage2 runs for games not in -2

Local layout (under repo root):
  runs/<run_name>/...
  SFT_Data/skill_banks/<game>/skill_bank.jsonl
  runs/sft_per_game_v3/<game>/...
  b2_artifacts_manifest.tsv
"""

from __future__ import annotations

import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from b2sdk.v2 import InMemoryAccountInfo, B2Api
from b2sdk.v2.exception import B2Error

KEY_ID = os.environ.get("B2_KEY_ID")
APP_KEY = os.environ.get("B2_APPLICATION_KEY")
if not KEY_ID or not APP_KEY:
    raise SystemExit("Set B2_KEY_ID and B2_APPLICATION_KEY in the environment.")

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO  # store directly in the github-main checkout
LOG = REPO / "b2_artifacts_download.log"
MANIFEST = REPO / "b2_artifacts_manifest.tsv"
WORKERS = 16

# (bucket, remote_prefix, local_relpath_under_ROOT, kind)
# kind: coevo_slim | skill_banks | sft_finals
JOBS = [
    # --- latest coevo per game (emnlp2026-2) ---
    ("emnlp2026-2", "workspace/Multi-hop-Reasoning-VLM-Agent/runs/candy_crush_coevo_v4_20260519_093912/", "runs/candy_crush_coevo_v4_20260519_093912/", "coevo_slim"),
    ("emnlp2026-2", "workspace/Multi-hop-Reasoning-VLM-Agent/runs/gymv_columns_coevo_v4_20260519_072135/", "runs/gymv_columns_coevo_v4_20260519_072135/", "coevo_slim"),
    ("emnlp2026-2", "workspace/Multi-hop-Reasoning-VLM-Agent/runs/gymv_streets_of_rage_2_coevo_v5_20260520_010806/", "runs/gymv_streets_of_rage_2_coevo_v5_20260520_010806/", "coevo_slim"),
    ("emnlp2026-2", "workspace/Multi-hop-Reasoning-VLM-Agent/runs/gymv_strider_coevo_v5_20260519_184613/", "runs/gymv_strider_coevo_v5_20260519_184613/", "coevo_slim"),
    ("emnlp2026-2", "workspace/Multi-hop-Reasoning-VLM-Agent/runs/gymv_thunder_force_iii_coevo_v10_asym_warmstart/", "runs/gymv_thunder_force_iii_coevo_v10_asym_warmstart/", "coevo_slim"),
    ("emnlp2026-2", "workspace/Multi-hop-Reasoning-VLM-Agent/runs/tetris_coevo_v4_20260520_063432/", "runs/tetris_coevo_v4_20260520_063432/", "coevo_slim"),
    # --- stage2 games only on emnlp2026 ---
    ("emnlp2026", "Multi-hop-Reasoning-VLM-Agent/runs/gymv_airstriker_stage2_v9_20260521_180740/", "runs/gymv_airstriker_stage2_v9_20260521_180740/", "coevo_slim"),
    ("emnlp2026", "Multi-hop-Reasoning-VLM-Agent/runs/gymv_altered_beast_stage2_v9_20260521_214938/", "runs/gymv_altered_beast_stage2_v9_20260521_214938/", "coevo_slim"),
    ("emnlp2026", "Multi-hop-Reasoning-VLM-Agent/runs/gymv_dynamite_headdy_stage2_20260520_094617/", "runs/gymv_dynamite_headdy_stage2_20260520_094617/", "coevo_slim"),
    ("emnlp2026", "Multi-hop-Reasoning-VLM-Agent/runs/gymv_space_harrier_ii_stage2_20260520_094617/", "runs/gymv_space_harrier_ii_stage2_20260520_094617/", "coevo_slim"),
    # --- curated skill extracts ---
    ("emnlp2026-2", "workspace/SFT_Data/skill_banks/", "SFT_Data/skill_banks/", "skill_banks"),
    # --- per-game SFT final adapters (not hf_trainer intermediates) ---
    ("emnlp2026-2", "workspace/Multi-hop-Reasoning-VLM-Agent/runs/sft_per_game_v3/", "runs/sft_per_game_v3/", "sft_finals"),
]

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


def select_files(api: B2Api, bucket_name: str, remote_prefix: str, kind: str):
    """Return list of (remote_name, size) to download for this job."""
    bucket = api.get_bucket_by_name(bucket_name)
    all_files = [(fv.file_name, fv.size) for fv, _ in bucket.ls(folder_to_list=remote_prefix, recursive=True)]

    if kind == "skill_banks":
        # skip .pre_layerc_bak backups
        return [(n, s) for n, s in all_files if not n.endswith(".pre_layerc_bak")]

    if kind == "sft_finals":
        out = []
        for n, s in all_files:
            rel = n[len(remote_prefix) :]
            if "hf_trainer" in rel or "optimizer" in rel:
                continue
            base = rel.rsplit("/", 1)[-1]
            if base in ("adapter_model.safetensors", "adapter_config.json", "README.md") or rel.endswith(
                ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
            ):
                out.append((n, s))
            # also keep small meta next to final adapter dirs
            elif base in ("training_args.bin", "trainer_state.json"):
                continue
        # Prefer only final adapter dirs: <game>/<role>/<game>__<role>/...
        # The pattern above already excludes hf_trainer.
        return out

    if kind == "coevo_slim":
        steps = set()
        for n, _ in all_files:
            m = re.search(r"checkpoints/step_(\d+)/", n)
            if m:
                steps.add(int(m.group(1)))
        real = sorted(s for s in steps if s < 90000)
        latest = real[-1] if real else None
        out = []
        for n, s in all_files:
            rel = n[len(remote_prefix) :]
            if "optimizer" in rel:
                continue
            if rel.startswith("lora_adapters/") or rel.startswith("skillbank/"):
                out.append((n, s))
            elif rel in ("config.json", "step_log.jsonl", "coevolution.log", "launch.log"):
                out.append((n, s))
            elif latest is not None and rel.startswith(f"checkpoints/step_{latest:04d}/"):
                out.append((n, s))
        return out

    raise ValueError(kind)


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
            downloaded = bucket.download_file_by_name(remote)
            downloaded.save_to(str(tmp))
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
    log(f"Repo root: {ROOT}")
    api = authorize()
    log("Authorized B2")

    plan: list[tuple[str, str, int, Path]] = []  # bucket, remote, size, dest
    with open(MANIFEST, "w") as mf:
        mf.write("bucket\tremote\tlocal\tsize\tkind\n")
        for bucket_name, remote_prefix, local_prefix, kind in JOBS:
            files = select_files(api, bucket_name, remote_prefix, kind)
            log(f"Plan {kind}: {bucket_name}/{remote_prefix} -> {len(files)} files, {hr(sum(s for _, s in files))}")
            for remote, size in files:
                rel = remote[len(remote_prefix) :]
                dest = ROOT / local_prefix / rel
                plan.append((bucket_name, remote, size, dest))
                mf.write(f"{bucket_name}\t{remote}\t{dest.relative_to(ROOT)}\t{size}\t{kind}\n")

    total_files = len(plan)
    total_bytes = sum(s for _, _, s, _ in plan)
    log(f"TOTAL: {total_files} files, {hr(total_bytes)}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [
            pool.submit(download_one, api, b, r, s, d) for b, r, s, d in plan
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            status, name, err = fut.result()
            if status == "fail":
                log(f"FAIL {name}: {err}")
            if i % 25 == 0 or i == total_files:
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
        fail_path = REPO / "b2_artifacts_failures.tsv"
        with open(fail_path, "w") as f:
            for n, e in failures:
                f.write(f"{n}\t{e}\n")
        log(f"Failures: {fail_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
