#!/usr/bin/env python3
"""Execute fresh ALFWorld cells with lineage-specific source automata."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_search_automaton_v16 import (  # noqa: E402
    AUTHENTIC,
    CONDITIONS,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    file_sha256,
    make_cell_execution_receipt,
    read_object,
    validate_manifest,
    validate_self_hash,
)
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
)
from scripts.run_alfworld_search_automaton_v16 import _run_episode  # noqa: E402


ALF_CONFIG = Path(
    "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1/"
    "configs/alfworld_base_config.yaml"
)
ALF_DATA = Path(
    "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main/"
    ".cache/alfworld_data"
)


def _cell(manifest: dict, game: str) -> dict:
    matches = [
        row for row in manifest["cells"]
        if row["source_game"] == game and row["target_domain"] == "alfworld"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one ALFWorld cell for {game}")
    return dict(matches[0])


def _run_cell(
    *, manifest: dict, game: str, output_root: Path, max_steps: int
) -> dict:
    cell = _cell(manifest, game)
    output_dir = output_root / game
    output_path = output_dir / "report.json"
    if output_path.is_file():
        existing = read_object(output_path)
        receipt = existing.get("cell_execution_receipt")
        if receipt:
            validate_self_hash(receipt, "cell_receipt_sha256")
            if receipt.get("manifest_sha256") == manifest["manifest_sha256"]:
                return existing
        raise RuntimeError(f"refusing incompatible ALFWorld resume: {output_path}")

    source_path = REPO / str(cell["source_artifact"])
    source = SourceSearchAutomaton(
        read_object(source_path),
        expected_sha256=str(cell["source_artifact_sha256"]),
    )
    target_path = REPO / str(manifest["targets"]["alfworld"]["target_grounder"])
    target_artifact = read_object(target_path)
    if not target_artifact.get("target_grounder_gate", {}).get("passed"):
        raise RuntimeError("ALFWorld target-native neural grounder gate failed")

    task_id = str(cell["target_task_id"])
    episodes = {}
    runtime_error = None
    try:
        for condition in CONDITIONS:
            environment = ALFWorldTextBatchEnvironment(
                config_path=str(ALF_CONFIG),
                data_path=str(ALF_DATA),
                split="eval_out_of_distribution",
                seed=88601,
                game_ids=[task_id],
                max_steps=max_steps,
            )
            try:
                episode = _run_episode(
                    environment=environment,
                    condition=condition,
                    source=source,
                    target_grounder=target_artifact["target_grounder"],
                    max_steps=max_steps,
                )
            finally:
                environment.close()
            episodes[condition] = episode
            print(json.dumps({
                "cell_id": cell["cell_id"],
                "condition": condition,
                "success": episode["official_success"],
                "steps": episode["steps"],
                "source_decisions": episode["source_decisions"],
            }), flush=True)
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"

    authentic = episodes.get(AUTHENTIC, {})
    source_trace = list(authentic.get("source_trace") or ())
    initial_hashes = [
        str(row["records"][0]["before_state_sha256"])
        for row in episodes.values() if row.get("records")
    ]
    receipt = make_cell_execution_receipt(
        manifest_sha256=str(manifest["manifest_sha256"]),
        cell=cell,
        source_artifact_sha256=source.artifact_sha256,
        conditions_executed=[
            condition for condition in CONDITIONS if condition in episodes
        ],
        expected_conditions=CONDITIONS,
        target_initial_state_hashes=initial_hashes,
        authentic_source_decisions=source_trace,
        target_native_grounding_used=True,
        target_reset_or_sample_open_count=1,
        outcome_was_reused=False,
        runtime_error=runtime_error,
    )
    body = {
        "schema_version": "phase1-direct-alfworld-cell-v1",
        "status": receipt["status"],
        "claim_boundary": manifest["claim_boundary"],
        "cell": cell,
        "source_artifact_file_sha256": file_sha256(source_path),
        "target_grounder_file_sha256": file_sha256(target_path),
        "runtime_file_sha256": file_sha256(Path(__file__)),
        "episodes": episodes,
        "authentic_source_action_counts": dict(sorted(Counter(
            row["source_action"] for row in source_trace if row.get("admitted")
        ).items())),
        "cell_execution_receipt": receipt,
    }
    report = body | {"report_sha256": stable_hash(body)}
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase1_direct_prospective_v1/manifest.json",
    )
    parser.add_argument("--source-game", choices=SOURCE_GAMES, action="append")
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "runs/phase1_direct_prospective_v1/alfworld",
    )
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    validate_manifest(manifest, repo=REPO)
    games = tuple(args.source_game or SOURCE_GAMES)
    reports = [
        _run_cell(
            manifest=manifest, game=game, output_root=args.output_root,
            max_steps=args.max_steps,
        )
        for game in games
    ]
    summary = {
        "domain": "alfworld",
        "passed": sum(
            report["cell_execution_receipt"]["status"]
            == "DIRECT_PROSPECTIVE_CELL_PASSED"
            for report in reports
        ),
        "attempted": len(reports),
        "cells": [report["cell"]["cell_id"] for report in reports],
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] == summary["attempted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
