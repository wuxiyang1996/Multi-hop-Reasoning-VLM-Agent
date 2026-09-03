#!/usr/bin/env python3
"""Freeze and source-qualify the Sokoban preconditioned-commit skill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motif_transfer.sokoban_commit_skill import (
    build_fresh_confirmation_plan,
    build_plan,
    fit_discovery_artifact,
    qualify_artifact,
)


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("prepare")
    plan_parser.add_argument("--source-dir", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--snapshots-per-episode", type=int, default=2)
    plan_parser.add_argument("--maximum-source-step", type=int, default=120)

    fresh_parser = subparsers.add_parser("prepare-fresh")
    fresh_parser.add_argument("--output", type=Path, required=True)
    fresh_parser.add_argument("--seed-base", type=int, default=94001)
    fresh_parser.add_argument("--episodes", type=int, default=18)
    fresh_parser.add_argument("--snapshots-per-episode", type=int, default=4)
    fresh_parser.add_argument("--width", type=int, default=8)
    fresh_parser.add_argument("--height", type=int, default=8)
    fresh_parser.add_argument("--box-count", type=int, default=2)
    fresh_parser.add_argument("--reverse-pulls", type=int, default=4)
    fresh_parser.add_argument("--interior-wall-count", type=int, default=2)
    fresh_parser.add_argument("--maximum-solver-nodes", type=int, default=100_000)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--plan", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--maximum-solver-nodes", type=int, default=100_000)
    fit_parser.add_argument("--ridge-alpha", type=float, default=0.5)
    fit_parser.add_argument("--minimum-examples-per-option", type=int, default=6)

    qualify_parser = subparsers.add_parser("qualify")
    qualify_parser.add_argument("--plan", type=Path, required=True)
    qualify_parser.add_argument("--artifact", type=Path, required=True)
    qualify_parser.add_argument("--output", type=Path, required=True)
    qualify_parser.add_argument(
        "--split", choices=("qualification", "held_out"), required=True,
    )
    qualify_parser.add_argument("--maximum-solver-nodes", type=int, default=100_000)
    qualify_parser.add_argument("--minimum-eligible-snapshots", type=int, default=12)
    qualify_parser.add_argument("--minimum-examples-per-option", type=int, default=6)
    qualify_parser.add_argument("--minimum-accuracy", type=float, default=0.60)

    args = parser.parse_args()
    if args.command == "prepare":
        payload = build_plan(
            args.source_dir,
            snapshots_per_episode=args.snapshots_per_episode,
            maximum_source_step=args.maximum_source_step,
        )
    elif args.command == "prepare-fresh":
        payload = build_fresh_confirmation_plan(
            seeds=tuple(range(args.seed_base, args.seed_base + args.episodes)),
            snapshots_per_episode=args.snapshots_per_episode,
            width=args.width,
            height=args.height,
            box_count=args.box_count,
            reverse_pulls=args.reverse_pulls,
            interior_wall_count=args.interior_wall_count,
            maximum_solver_nodes=args.maximum_solver_nodes,
        )
    elif args.command == "fit":
        payload = fit_discovery_artifact(
            _read(args.plan),
            maximum_solver_nodes=args.maximum_solver_nodes,
            ridge_alpha=args.ridge_alpha,
            minimum_examples_per_option=args.minimum_examples_per_option,
        )
    else:
        payload = qualify_artifact(
            _read(args.plan), _read(args.artifact), split=args.split,
            maximum_solver_nodes=args.maximum_solver_nodes,
            minimum_eligible_snapshots=args.minimum_eligible_snapshots,
            minimum_examples_per_option=args.minimum_examples_per_option,
            minimum_accuracy=args.minimum_accuracy,
        )
    _write_new(args.output, payload)
    summary = {
        key: payload[key] for key in (
            "plan_sha256", "artifact_sha256", "report_sha256", "split_counts",
            "source_gate_passed", "next_step", "condition_metrics", "gates",
        ) if key in payload
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.command == "qualify" and not payload["source_gate_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
