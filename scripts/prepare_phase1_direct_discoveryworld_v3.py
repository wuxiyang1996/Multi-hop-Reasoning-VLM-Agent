#!/usr/bin/env python3
"""Collect six V3 target-only episodes once, in isolated parallel shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    file_sha256,
    read_object,
    validate_self_hash,
)


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    return int(completed.returncode)


def _collect_one(
    *, index: int, task_id: str, config_path: Path, keys: Path, shard_root: Path,
) -> dict[str, Any]:
    output_dir = shard_root / f"task_{index}"
    episode_path = output_dir / f"{task_id}.json"
    code = _run(
        [
            sys.executable,
            str(REPO / "scripts/run_discoveryworld_target_only_v1.py"),
            "--config", str(config_path),
            "--keys", str(keys),
            "--output-dir", str(output_dir),
            "--role", "formal_reserve",
            "--task-index", str(index),
        ],
        shard_root / "logs" / f"{task_id}.log",
    )
    episode = read_object(episode_path) if episode_path.is_file() else {}
    return {
        "index": index,
        "task_id": task_id,
        "exit_code": code,
        "episode_path": episode_path,
        "status": episode.get("status"),
        "episode_sha256": episode.get("episode_sha256"),
        "runtime_hashes": episode.get("runtime_hashes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase1_direct_prospective_v3/discoveryworld_manifest.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "runs/phase1_direct_prospective_v3/discoveryworld",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    validate_self_hash(manifest, "manifest_sha256")
    for relative, expected in manifest["runtime_file_sha256"].items():
        if file_sha256(REPO / relative) != expected:
            raise RuntimeError(f"frozen V3 runtime changed: {relative}")
    receipt_path = args.output_root / "preparation_receipt.json"
    if receipt_path.is_file():
        receipt = read_object(receipt_path)
        validate_self_hash(receipt, "preparation_receipt_sha256")
        if receipt.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("incompatible V3 preparation receipt")
        print(json.dumps(receipt, indent=2))
        return 0

    config_path = REPO / str(manifest["target_config"])
    protocol_path = REPO / str(manifest["protocol"])
    tasks = [dict(row["target_task"]) for row in manifest["cells"]]
    shard_root = args.output_root / "target_only_shards"
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(tasks)))) as pool:
        for index, task in enumerate(tasks):
            futures.append(pool.submit(
                _collect_one,
                index=index, task_id=str(task["task_id"]),
                config_path=config_path, keys=args.keys, shard_root=shard_root,
            ))
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: int(row["index"]))
    failed = [
        row for row in results
        if row["exit_code"] != 0 or row["status"] != "TARGET_ONLY_EPISODE_COMPLETE"
    ]
    if failed:
        raise RuntimeError(f"V3 target-only shard failed; reserve is consumed: {failed}")
    runtime_hashes = {
        stable_hash(row["runtime_hashes"]) for row in results
    }
    if len(runtime_hashes) != 1:
        raise RuntimeError("V3 target-only shards used different frozen runtimes")

    target_dir = args.output_root / "target_only"
    target_dir.mkdir(parents=True, exist_ok=True)
    for row in results:
        destination = target_dir / f"{row['task_id']}.json"
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite merged V3 episode: {destination}")
        shutil.copyfile(row["episode_path"], destination)
    summary_code = _run(
        [
            sys.executable,
            str(REPO / "scripts/run_discoveryworld_target_only_v1.py"),
            "--config", str(config_path),
            "--keys", str(args.keys),
            "--output-dir", str(target_dir),
            "--role", "formal_reserve",
        ],
        args.output_root / "target_only_merge.log",
    )
    summary_path = target_dir / "summary.json"
    if summary_code != 0 or not summary_path.is_file():
        raise RuntimeError("V3 merged target-only summary failed")

    fork_dir = args.output_root / "frozen_forks"
    freeze_code = _run(
        [
            sys.executable,
            str(REPO / "scripts/freeze_discoveryworld_qualification_forks_v1.py"),
            "--protocol", str(protocol_path),
            "--baseline-dir", str(target_dir),
            "--output-dir", str(fork_dir),
        ],
        args.output_root / "fork_freeze.log",
    )
    fork_receipt_path = fork_dir / "fork_freeze_receipt.json"
    if freeze_code != 0 or not fork_receipt_path.is_file():
        raise RuntimeError("V3 fork freeze failed")
    fork_receipt = read_object(fork_receipt_path)
    generated = list(fork_receipt.get("generated_configs") or ())
    if len(generated) != len(tasks):
        raise RuntimeError(
            f"V3 requires six eligible forks, generated {len(generated)}"
        )

    rows = []
    for result in results:
        episode_path = target_dir / f"{result['task_id']}.json"
        rows.append({
            "task_id": result["task_id"],
            "target_process_count": 1,
            "episode_sha256": result["episode_sha256"],
            "episode_file_sha256": file_sha256(episode_path),
        })
    body = {
        "schema_version": "phase1-direct-discoveryworld-preparation-v3",
        "status": "SIX_TARGETS_COLLECTED_ONCE_AND_SIX_FORKS_FROZEN",
        "manifest_sha256": manifest["manifest_sha256"],
        "parallel_workers": max(1, min(args.workers, len(tasks))),
        "target_summary_file_sha256": file_sha256(summary_path),
        "fork_freeze_receipt_file_sha256": file_sha256(fork_receipt_path),
        "tasks": rows,
    }
    receipt = body | {"preparation_receipt_sha256": stable_hash(body)}
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
