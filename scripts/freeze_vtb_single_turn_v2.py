#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.frozen_split import freeze_one_shot_split
from motif_transfer.vtb_evaluator import OFFICIAL_COMMIT, OFFICIAL_REPOSITORY


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a corrected VTB single-turn diagnostic split.")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "configs/vtb_single_turn_manifest_v2.json")
    args = parser.parse_args()
    import duckdb

    rows = duckdb.connect().execute(
        "SELECT row_number() OVER () - 1 AS row_index, id, turncase, num_turns FROM read_parquet(?)",
        [str(args.parquet)],
    ).fetchall()
    single = [f"row:{index}" for index, _task_id, turncase, num_turns in rows
              if str(turncase).lower() == "single-turn" and int(num_turns) == 1]
    # row:865 was already executed before the multi-turn bug was found. It is
    # declared, not silently discarded. It is outside the single-turn pool in
    # any case, but recording it makes the selection history auditable.
    observed_before_v2 = ["row:865"]
    available = [value for value in single if value not in observed_before_v2]
    split = freeze_one_shot_split(
        available, available, namespace="motif-transfer:visual_toolbench:single-turn:v2"
    )
    split["smoke_test_ids"] = split["test_ids"][:2]
    payload = {
        "schema_version": 2,
        "cell": "visual_toolbench_single_turn",
        "claim_scope": "VTB single-turn subset only; never report as full VisualToolBench",
        "frozen_before_v2_target_runs": True,
        "selection_filter": "turncase == single-turn AND num_turns == 1",
        "selection_used_prompt_answer_rubric_or_reward": False,
        "observed_before_freeze": observed_before_v2,
        "observed_exclusion_reason": "prior infrastructure smoke before multi-turn mismatch was discovered",
        "dataset_sha256": _sha(args.parquet),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_tool_call_cap": 20,
        "single_turn_pool_size": len(single),
        **split,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "single_turn_pool_size": len(single),
        "adaptation_id": split["adaptation_id"],
        "smoke_test_ids": split["smoke_test_ids"],
        "test_pool_size": split["test_pool_size"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
