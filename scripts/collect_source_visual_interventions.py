#!/usr/bin/env python3
"""Freeze and collect visual intervention receipts from source rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.visual_intervention_receipts import (
    build_visual_intervention_plan,
    collect_plan_split,
    load_runtime_env_factory,
    runtime_file_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def prepare(config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")
    plan = build_visual_intervention_plan(
        _resolve(config["evidence_dir"]),
        game=str(config["game"]),
        snapshots_per_episode=int(config["snapshots_per_episode"]),
        minimum_prefix_steps=int(config["minimum_prefix_steps"]),
        maximum_prefix_steps=int(config["maximum_prefix_steps"]),
        max_episode_steps=int(config["max_episode_steps"]),
        config_receipt={
            "path": str(config_path.resolve()),
            "contents": config,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "plan": str(output.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "split_counts": plan["split_counts"],
    }, indent=2, sort_keys=True))


def collect(
    plan_path: Path,
    output_dir: Path,
    runtime_root: Path,
    split: str,
    workers: int,
    snapshot_limit: int | None,
) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = collect_plan_split(
        plan,
        split=split,
        output_dir=output_dir,
        env_factory=load_runtime_env_factory(runtime_root),
        workers=workers,
        snapshot_limit=snapshot_limit,
        runtime_receipt=runtime_file_receipt(runtime_root),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest["all_interventions_observed"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--plan", required=True, type=Path)
    collect_parser.add_argument("--output-dir", required=True, type=Path)
    collect_parser.add_argument("--runtime-root", required=True, type=Path)
    collect_parser.add_argument(
        "--split", choices=("discovery", "qualification", "held_out"),
        required=True,
    )
    collect_parser.add_argument("--workers", type=int, default=4)
    collect_parser.add_argument("--snapshot-limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config, args.output)
    else:
        collect(
            args.plan,
            args.output_dir,
            args.runtime_root,
            args.split,
            args.workers,
            args.snapshot_limit,
        )


if __name__ == "__main__":
    main()
