#!/usr/bin/env python3
"""Calibrate broad-oracle operator authorization on consumed train reserves."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.agqa_broad_oracle_executor import execute_broad_public_plan  # noqa: E402
from motif_transfer.agqa_oracle_query_mdp import (  # noqa: E402
    AGQAOracleQueryBackend, AGQAOracleToolBudget, load_agqa_id_to_text,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.official_video_event_graph import load_builtin_only_pickle, sha256_file  # noqa: E402

STSG = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/AGQA_train_stsgs.pkl")
ONTOLOGY = STSG.parent / "ENG.txt"
RUNS = (
    REPO / "runs/agqa2_oracle_query_mdp_v2_fresh",
    REPO / "runs/agqa2_oracle_query_mdp_v3_transfer",
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text()); body = dict(value)
    claimed = body.pop("artifact_sha256")
    if stable_hash(body) != claimed:
        raise ValueError(f"artifact hash mismatch: {path}")
    return value


def _normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    return re.sub(r"^(?:a|an|the)\s+", "", text)


def main() -> None:
    ontology = load_agqa_id_to_text(ONTOLOGY)
    corpus = load_builtin_only_pickle(STSG)
    stsg_sha = sha256_file(STSG)
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"parsed": 0, "committed": 0, "correct": 0}
    )
    total = 0
    for run in RUNS:
        public = _read(run / "public_cohort.json")
        evaluator = _read(run / "evaluator_only.json")
        gold = {row["task_id"]: row["answer"] for row in evaluator["rows"]}
        for row in public["rows"]:
            total += 1
            plan = parse_public_question_plan(row["question"])
            if plan is None:
                continue
            bucket = stats[plan.comparison]
            bucket["parsed"] += 1
            video_id = str(row["video_id"])
            result = execute_broad_public_plan(
                plan,
                AGQAOracleQueryBackend(
                    video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                    graph_sha256=stable_hash([stsg_sha, video_id]),
                    budget=AGQAOracleToolBudget(3),
                ),
            )
            if result.prediction is None:
                continue
            bucket["committed"] += 1
            bucket["correct"] += (
                _normalize(result.prediction) == _normalize(gold[row["task_id"]])
            )
    minimum_commits = 20
    minimum_accuracy = 0.95
    metrics = {}
    authorized = []
    for comparison, row in sorted(stats.items()):
        accuracy = row["correct"] / row["committed"] if row["committed"] else 0.0
        passed = row["committed"] >= minimum_commits and accuracy >= minimum_accuracy
        metrics[comparison] = row | {"conditional_accuracy": accuracy, "authorized": passed}
        if passed:
            authorized.append(comparison)
    body = {
        "schema_version": "agqa2-broad-oracle-authorization-v1",
        "status": "FROZEN_FROM_CONSUMED_TRAIN_DEVELOPMENT",
        "development_tasks": total,
        "development_runs": [str(path.relative_to(REPO)) for path in RUNS],
        "minimum_commits": minimum_commits,
        "minimum_conditional_accuracy": minimum_accuracy,
        "authorized_comparisons": authorized,
        "metrics": metrics,
        "runtime_answer_program_or_sg_grounding_read": False,
        "official_train_stsg_sha256": stsg_sha,
        "official_ontology_sha256": sha256_file(ONTOLOGY),
        "claim_boundary": (
            "Target-native operator qualification on consumed train development "
            "reserves; target outcomes select only an operator allowlist, never a "
            "source program or per-example decision."
        ),
    }
    artifact = body | {"artifact_sha256": stable_hash(body)}
    output = REPO / "configs/agqa2_broad_oracle_authorization_v1.json"
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
