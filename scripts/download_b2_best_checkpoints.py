#!/usr/bin/env python3
"""Select and download the official best per-game checkpoints from B2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from b2sdk.v2 import B2Api, InMemoryAccountInfo


REPO = Path(__file__).resolve().parents[1]
FORMAL_SIX_GAMES = (
    "candy_crush",
    "gymv_columns",
    "gymv_streets_of_rage_2",
    "gymv_strider",
    "gymv_thunder_force_iii",
    "tetris",
)
STAGE2_GAMES = (
    "gymv_space_harrier_ii",
    "gymv_airstriker",
    "gymv_altered_beast",
    "gymv_dynamite_headdy",
    "twenty_forty_eight",
    "super_mario",
)
# These two predate the step_99800 alias convention.  Their run-level
# ``best/`` directories contain the highest-reward *saved* full checkpoint
# (not necessarily the highest unsaved rollout step) and carry metadata.
LEGACY_STAGE2_BEST = {
    "twenty_forty_eight": {
        "bucket": "emnlp2026-2",
        "run": "Qwen3-8B_2048_20260322_071227",
        "checkpoint_prefix": (
            "workspace/Game-AI-Agent/runs/"
            "Qwen3-8B_2048_20260322_071227/best/"
        ),
    },
    "super_mario": {
        "bucket": "emnlp2026-2",
        "run": "Qwen3-8B_super_mario_20260323_030839",
        "checkpoint_prefix": (
            "workspace/Game-AI-Agent/runs/"
            "Qwen3-8B_super_mario_20260323_030839/best/"
        ),
    },
}
BUCKET_ROOTS = (
    ("emnlp2026-2", "workspace/Multi-hop-Reasoning-VLM-Agent/runs/", 1),
    ("emnlp2026", "Multi-hop-Reasoning-VLM-Agent/runs/", 0),
)
ADAPTERS = {
    "decision": ("skill_selection", "action_taking"),
    "skillbank": ("segment", "contract", "curator"),
}


def _credentials(keys_file: Path) -> tuple[str, str]:
    key_id = os.environ.get("B2_KEY_ID", "").strip()
    app_key = os.environ.get("B2_APPLICATION_KEY", "").strip()
    if key_id and app_key:
        return key_id, app_key
    spec = importlib.util.spec_from_file_location("_local_b2_keys", keys_file)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import credentials file: {keys_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key_id = str(getattr(module, "B2_KEY_ID", "") or "").strip()
    app_key = str(getattr(module, "B2_APPLICATION_KEY", "") or "").strip()
    if not key_id or not app_key:
        raise SystemExit("B2 credentials unavailable")
    return key_id, app_key


def _api(keys_file: Path) -> B2Api:
    key_id, app_key = _credentials(keys_file)
    api = B2Api(InMemoryAccountInfo())
    api.authorize_account("production", key_id, app_key)
    return api


def _download_json(bucket, remote: str) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="b2-best-meta-") as stream:
        bucket.download_file_by_name(remote).save_to(stream.name)
        return json.loads(Path(stream.name).read_text(encoding="utf-8"))


def _version(run: str) -> int:
    matches = re.findall(r"(?:^|_)v(\d+)(?:_|$)", run)
    return int(matches[-1]) if matches else 0


def select(api: B2Api, games: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {game: [] for game in games}
    suffix = "/checkpoints/step_99800/metadata.json"
    for bucket_name, root, bucket_priority in BUCKET_ROOTS:
        bucket = api.get_bucket_by_name(bucket_name)
        for file_version, _ in bucket.ls(folder_to_list=root, recursive=True):
            remote = file_version.file_name
            if not remote.endswith(suffix):
                continue
            metadata = _download_json(bucket, remote)
            game = str(metadata.get("game") or "")
            if game not in candidates or metadata.get("best") is not True:
                continue
            run = remote[len(root):].split("/", 1)[0]
            checkpoint_prefix = remote[: -len("metadata.json")]
            candidates[game].append({
                "bucket": bucket_name,
                "bucket_priority": bucket_priority,
                "root": root,
                "run": run,
                "run_version": _version(run),
                "checkpoint_prefix": checkpoint_prefix,
                "metadata": metadata,
                "selection_metric": "official_step_99800_best_alias_mean_reward",
            })
    selected: dict[str, dict[str, Any]] = {}
    for game, rows in candidates.items():
        if not rows:
            legacy = LEGACY_STAGE2_BEST.get(game)
            if legacy is None:
                raise SystemExit(f"no official best checkpoint found for {game}")
            bucket = api.get_bucket_by_name(legacy["bucket"])
            prefix = legacy["checkpoint_prefix"]
            metadata = _download_json(bucket, prefix + "metadata.json")
            metadata_game = next(iter(metadata.get("skills_per_game") or {}), "")
            if metadata_game != game:
                raise SystemExit(
                    f"legacy best metadata game mismatch: {game} != {metadata_game}"
                )
            rows.append({
                **legacy,
                "bucket_priority": 1,
                "root": "workspace/Game-AI-Agent/runs/",
                "run_version": _version(legacy["run"]),
                "metadata": metadata,
                "selection_metric": "legacy_run_best_highest_saved_checkpoint_reward",
            })
        selected[game] = max(rows, key=lambda row: (
            float(row["metadata"].get("mean_reward", float("-inf"))),
            int(row["run_version"]),
            float(row["metadata"].get("timestamp", 0.0)),
            int(row["bucket_priority"]),
        ))
    return selected


def _required(game: str) -> set[str]:
    files = {"metadata.json", f"banks/{game}/skill_bank.jsonl"}
    for group, names in ADAPTERS.items():
        for name in names:
            files.add(f"adapters/{group}/{name}/adapter_config.json")
            files.add(f"adapters/{group}/{name}/adapter_model.safetensors")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(api: B2Api, selected: dict[str, dict[str, Any]], output: Path,
             workers: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    plan: list[dict[str, Any]] = []
    for game, choice in selected.items():
        bucket = api.get_bucket_by_name(choice["bucket"])
        prefix = choice["checkpoint_prefix"]
        objects = []
        for file_version, _ in bucket.ls(folder_to_list=prefix, recursive=True):
            rel = file_version.file_name[len(prefix):]
            objects.append((file_version, rel))
        present = {rel for _, rel in objects}
        missing = sorted(_required(game) - present)
        if missing:
            raise SystemExit(f"incomplete best checkpoint for {game}: {missing}")
        for file_version, rel in objects:
            plan.append({
                "game": game,
                "bucket": choice["bucket"],
                "remote": file_version.file_name,
                "relative": rel,
                "size": int(file_version.size),
                "b2_content_sha1": str(file_version.content_sha1 or ""),
                "local": output / game / rel,
            })

    def fetch(item: dict[str, Any]) -> dict[str, Any]:
        path: Path = item["local"]
        path.parent.mkdir(parents=True, exist_ok=True)
        status = "downloaded"
        if path.is_file() and path.stat().st_size == item["size"]:
            status = "reused_same_size"
        else:
            tmp = path.with_suffix(path.suffix + ".part")
            api.get_bucket_by_name(item["bucket"]).download_file_by_name(
                item["remote"]
            ).save_to(str(tmp))
            if tmp.stat().st_size != item["size"]:
                raise RuntimeError(f"size mismatch after download: {item['remote']}")
            os.replace(tmp, path)
        return {
            "game": item["game"],
            "bucket": item["bucket"],
            "remote": item["remote"],
            "local": str(path.relative_to(output)),
            "size": item["size"],
            "b2_content_sha1": item["b2_content_sha1"],
            "local_sha256": _sha256(path),
            "status": status,
        }

    files = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in plan]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            files.append(result)
            print(f"[{index}/{len(plan)}] {result['game']}/{Path(result['local']).name}", flush=True)

    selection_receipt = {}
    for game, choice in selected.items():
        selection_receipt[game] = {
            "bucket": choice["bucket"],
            "run": choice["run"],
            "checkpoint_prefix": choice["checkpoint_prefix"],
            "selection_metric": choice["selection_metric"],
            "mean_reward": choice["metadata"]["mean_reward"],
            "original_step": choice["metadata"].get(
                "original_step", choice["metadata"].get("step")
            ),
            "metadata": choice["metadata"],
        }
    manifest = {
        "schema_version": 1,
        "selection_scope": list(selected),
        "selection": selection_receipt,
        "files": sorted(files, key=lambda row: row["local"]),
        "total_files": len(files),
        "total_bytes": sum(row["size"] for row in files),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keys-file", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / "runs/b2_best_checkpoints",
    )
    parser.add_argument(
        "--roster", choices=("formal-six", "stage2"), default="formal-six",
        help="Named game roster to select (default: formal-six).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    api = _api(args.keys_file)
    games = FORMAL_SIX_GAMES if args.roster == "formal-six" else STAGE2_GAMES
    selected = select(api, games)
    print(json.dumps({game: {
        "bucket": row["bucket"],
        "run": row["run"],
        "mean_reward": row["metadata"]["mean_reward"],
        "original_step": row["metadata"].get(
            "original_step", row["metadata"].get("step")
        ),
    } for game, row in selected.items()}, indent=2, sort_keys=True))
    if args.select_only:
        return 0
    manifest = download(api, selected, args.output, max(1, args.workers))
    print(json.dumps({
        "output": str(args.output),
        "total_files": manifest["total_files"],
        "total_bytes": manifest["total_bytes"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
