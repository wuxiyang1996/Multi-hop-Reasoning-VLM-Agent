#!/usr/bin/env python3
"""Retrospective official-test replay with frozen V61 actor predictions.

The 1,769 V61 runtime receipts provide answer-blind public actor predictions.
This script freezes matched oracle-query policy arms using the official test
STSG before it opens the consumed V61 evaluator report.  Because V61 outcomes
were previously observed and informed later development, this is a diagnostic,
not a new confirmatory test result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import io
import json
from pathlib import Path
import re
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from motif_transfer.agqa_oracle_query_mdp import (  # noqa: E402
    AGQAOracleQueryBackend, AGQAOracleToolBudget,
    compose_localized_with_generic, execute_temporal_object_query,
    load_agqa_id_to_text,
)
from motif_transfer.agqa_active_frame_grounder import (  # noqa: E402
    parse_public_question_plan,
)
from motif_transfer.agqa_broad_oracle_executor import (  # noqa: E402
    execute_broad_public_plan,
)
from motif_transfer.agqa_query_object_source_specific import (  # noqa: E402
    exact_one_sided_pvalue,
)
from motif_transfer.agqa_temporal_localized_query import (  # noqa: E402
    parse_temporal_localized_object_question,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.official_video_event_graph import (  # noqa: E402
    load_builtin_only_pickle, sha256_file,
)
from motif_transfer.source_goal_acquisition_induction import (  # noqa: E402
    validate_goal_acquisition_program,
)
from scripts.freeze_agqa2_oracle_query_cohort_v1 import (  # noqa: E402
    iter_top_level_object,
)

ARCHIVE = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/AGQA_balanced.zip")
MEMBER = "AGQA_balanced/test_balanced.txt"
STSG = Path("/fs/gamma-projects/vlm-robot/datasets/AGQA2-official/scene_graphs/AGQA_scene_graphs/AGQA_test_stsgs.pkl")
ONTOLOGY = STSG.parent / "ENG.txt"
RUN = REPO / "runs/agqa2_official_test_oracle_transfer_v4_consumed"
V61 = REPO / "runs/agqa2_full_distribution_v61_formal"
MANIFEST = REPO / "configs/agqa2_full_distribution_v61_formal_manifest.json"
SOURCE = REPO / "runs/sokoban_goal_acquisition_v1/artifact.json"
AUTHORIZATION = REPO / "configs/agqa2_broad_oracle_authorization_v1.json"


def _verified(path: Path, hash_field: str) -> dict:
    value = json.loads(path.read_text()); body = dict(value)
    claimed = body.pop(hash_field)
    if stable_hash(body) != claimed:
        raise ValueError(f"artifact hash mismatch: {path}")
    return value


def _normalize(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    for prefix in ("the answer is ", "it is ", "they were ", "they are "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return {"true": "yes", "false": "no"}.get(text, text)


def _matches(prediction: object, gold: object) -> bool:
    predicted, expected = _normalize(prediction), _normalize(gold)
    if expected in {"yes", "no", "before", "after"}:
        return bool(predicted) and predicted.split(maxsplit=1)[0] == expected
    return predicted == expected


def _questions(task_ids: set[str]) -> dict[str, dict]:
    output = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(MEMBER) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8")
            for task_id, row in iter_top_level_object(text):
                if task_id in task_ids:
                    output[task_id] = {
                        "task_id": task_id, "video_id": str(row["video_id"]),
                        "question": str(row["question"]),
                        "question_sha256": stable_hash(str(row["question"])),
                    }
                    if len(output) == len(task_ids):
                        break
    if set(output) != task_ids:
        raise ValueError("official test questions missing V61 task IDs")
    return output


def _metric(rows: list[dict], arm: str) -> dict:
    correct = sum(row["correct"][arm] for row in rows)
    return {"correct": correct, "accuracy": correct / len(rows)}


def _paired(rows: list[dict], first: str, second: str) -> dict:
    wins = sum(r["correct"][first] and not r["correct"][second] for r in rows)
    losses = sum(not r["correct"][first] and r["correct"][second] for r in rows)
    return {
        "wins": wins, "losses": losses, "ties": len(rows) - wins - losses,
        "net_wins": wins - losses,
        "exact_one_sided_pvalue": exact_one_sided_pvalue(
            source_wins=wins, source_losses=losses,
        ),
    }


def main() -> None:
    manifest = _verified(MANIFEST, "manifest_sha256")
    task_ids = {str(row["task_id"]) for row in manifest["samples"]}
    questions = _questions(task_ids)
    direct = {}
    receipt_hashes = {}
    for task_id in sorted(task_ids):
        receipt = _verified(V61 / "runtime_receipts" / f"{task_id}.json", "runtime_receipt_sha256")
        if receipt["runtime_answer_read"] or receipt["runtime_functional_program_read"]:
            raise ValueError("V61 runtime receipt crossed evaluator authority")
        direct[task_id] = str(receipt["direct_response"])
        receipt_hashes[task_id] = receipt["runtime_receipt_sha256"]
    source = json.loads(SOURCE.read_text())
    validate_goal_acquisition_program(source)
    cardinality = int(source["program"]["relation_binding_cardinality"]["value"])
    if cardinality != 1:
        raise ValueError("source program does not induce unique binding")
    authorization = _verified(AUTHORIZATION, "artifact_sha256")
    authorized_broad = set(authorization["authorized_comparisons"])
    ontology = load_agqa_id_to_text(ONTOLOGY)
    corpus = load_builtin_only_pickle(STSG)
    stsg_sha = sha256_file(STSG)
    permutation = {"BEFORE": "AFTER", "AFTER": "BEFORE", "WHILE": "BEFORE", "BETWEEN": "WHILE"}
    runtime_rows = []
    for task_id in sorted(task_ids):
        row = questions[task_id]
        plan = parse_temporal_localized_object_question(row["question"])
        broad_plan = parse_public_question_plan(row["question"])
        localized = generic = permuted = None
        broad_candidate = None
        local_calls = generic_calls = permuted_calls = broad_calls = 0
        video_id = row["video_id"]
        graph_hash = stable_hash([stsg_sha, video_id])
        broad_authorized = (
            broad_plan is not None and broad_plan.comparison in authorized_broad
        )
        if broad_authorized:
            broad_backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            broad_result = execute_broad_public_plan(broad_plan, broad_backend)
            broad_candidate = broad_result.prediction
            broad_calls = len(broad_result.receipts)
        if plan is not None:
            local_backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            local_result = execute_temporal_object_query(plan, local_backend)
            localized = local_result.prediction
            local_calls = len(local_result.receipts)

            generic_backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            generic_backend.locate_action(plan.anchor_a)
            generic_result = generic_backend.query_relation(
                plan.relation, frames=generic_backend.all_frame_numbers(),
            )
            generic_calls = len(generic_backend.receipts)
            if len(generic_result["objects"]) == cardinality:
                generic = str(generic_result["objects"][0])

            permuted_plan = replace(
                plan, temporal_operator=permutation[plan.temporal_operator],
                anchor_b="" if plan.temporal_operator == "BETWEEN" else plan.anchor_b,
            )
            permuted_backend = AGQAOracleQueryBackend(
                video_id=video_id, graph=corpus[video_id], id_to_text=ontology,
                graph_sha256=graph_hash, budget=AGQAOracleToolBudget(3),
            )
            permuted_result = execute_temporal_object_query(permuted_plan, permuted_backend)
            permuted = permuted_result.prediction
            permuted_calls = len(permuted_result.receipts)
        base = direct[task_id]
        # The generic target stack selects the grammar-matched executor before
        # any outcome is visible: localized-query questions use an unlocalized
        # relation query; all other supported questions use the broad executor.
        target_candidate = generic if plan is not None else broad_candidate
        target_calls = generic_calls if plan is not None else broad_calls
        source_policy = compose_localized_with_generic(localized, target_candidate, base)
        permuted_policy = compose_localized_with_generic(permuted, target_candidate, base)
        generic_policy = compose_localized_with_generic(None, target_candidate, base)
        source_calls = local_calls + (target_calls if localized is None else 0)
        permuted_total = permuted_calls + (target_calls if permuted is None else 0)
        runtime_rows.append({
            "task_id": task_id, "video_id": row["video_id"],
            "question_sha256": row["question_sha256"],
            "v61_runtime_receipt_sha256": receipt_hashes[task_id],
            "parse_applicable": plan is not None or broad_plan is not None,
            "localized_parse_applicable": plan is not None,
            "broad_parse_applicable": broad_plan is not None,
            "broad_authorized": broad_authorized,
            "broad_comparison": broad_plan.comparison if broad_plan is not None else None,
            "localized_candidate": localized, "generic_candidate": generic,
            "broad_candidate": broad_candidate,
            "predictions": {
                "frozen_actor": base,
                "generic_scaffold": generic_policy.prediction,
                "source_permuted": permuted_policy.prediction,
                "source_induced": source_policy.prediction,
                "isomorphic_target_controller": source_policy.prediction,
            },
            "routes": {
                "generic_scaffold": generic_policy.route,
                "source_permuted": permuted_policy.route,
                "source_induced": source_policy.route,
            },
            "tool_calls": {
                "frozen_actor": 0, "generic_scaffold": target_calls,
                "source_permuted": permuted_total,
                "source_induced": source_calls,
                "isomorphic_target_controller": source_calls,
            },
            "matched_max_tool_calls": 5,
            "runtime_answer_read": False,
            "runtime_functional_program_read": False,
            "runtime_sg_grounding_read": False,
        })
    if len(runtime_rows) != 1769 or any(
        max(row["tool_calls"].values()) > 5 for row in runtime_rows
    ):
        raise ValueError("official-test runtime cardinality/budget failure")
    runtime_body = {
        "schema_version": "agqa2-official-test-oracle-runtime-v4-consumed",
        "role": "PREDICTIONS_FROZEN_BEFORE_CONSUMED_EVALUATOR_OPEN",
        "v61_manifest_sha256": manifest["manifest_sha256"],
        "source_artifact_sha256": source["artifact_sha256"],
        "broad_authorization_sha256": authorization["artifact_sha256"],
        "official_test_stsg_sha256": stsg_sha,
        "rows": runtime_rows,
    }
    runtime = runtime_body | {"artifact_sha256": stable_hash(runtime_body)}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "runtime_predictions.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")

    # Evaluator authority starts here. V61 is already consumed historically.
    evaluator = _verified(V61 / "base_report.json", "report_sha256")
    gold = {str(row["task_id"]): row["gold_answer_evaluator_only"] for row in evaluator["rows"]}
    evaluated = []
    for row in runtime_rows:
        evaluated.append(row | {"correct": {
            arm: _matches(prediction, gold[row["task_id"]])
            for arm, prediction in row["predictions"].items()
        }})
    arms = tuple(runtime_rows[0]["predictions"])
    metrics = {arm: _metric(evaluated, arm) for arm in arms}
    if metrics["frozen_actor"]["correct"] != evaluator["source_vs_direct"]["direct_correct"]:
        raise ValueError("frozen actor score differs from sealed V61 evaluator")
    paired = {
        arm: _paired(evaluated, "source_induced", arm)
        for arm in arms if arm != "source_induced"
    }
    broad_by_comparison = {}
    for comparison in sorted({
        row["broad_comparison"] for row in evaluated
        if row["broad_comparison"] is not None
    }):
        subset = [row for row in evaluated if row["broad_comparison"] == comparison]
        broad_by_comparison[comparison] = {
            "rows": len(subset),
            "broad_candidate_rows": sum(row["broad_candidate"] is not None for row in subset),
            "source_vs_actor": _paired(subset, "source_induced", "frozen_actor"),
            "generic_vs_actor": _paired(subset, "generic_scaffold", "frozen_actor"),
        }
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in evaluated:
        by_video[row["video_id"]].append(row)
    cluster_deltas = []
    for video_rows in by_video.values():
        cluster_deltas.append(
            sum(row["correct"]["source_induced"] for row in video_rows)
            - sum(row["correct"]["frozen_actor"] for row in video_rows)
        )
    body = {
        "schema_version": "agqa2-official-test-oracle-transfer-v4-consumed",
        "status": "PASSED_DIAGNOSTIC",
        "claim_boundary": (
            "Retrospective official-test diagnostic on previously consumed V61 identities. "
            "Uses official answer-blind test STSG and frozen Qwen3-VL-32B actor receipts; "
            "not comparable to raw-video AGQA SOTA and not a fresh confirmatory result."
        ),
        "tasks": len(evaluated), "unique_videos": len(by_video),
        "public_compiler_applicable": sum(row["parse_applicable"] for row in evaluated),
        "localized_compiler_applicable": sum(row["localized_parse_applicable"] for row in evaluated),
        "broad_compiler_applicable": sum(row["broad_parse_applicable"] for row in evaluated),
        "metrics": metrics, "source_paired": paired,
        "broad_by_comparison": broad_by_comparison,
        "source_vs_actor_video_clusters": {
            "positive": sum(value > 0 for value in cluster_deltas),
            "negative": sum(value < 0 for value in cluster_deltas),
            "tied": sum(value == 0 for value in cluster_deltas),
        },
        "authority": {
            "runtime_answer_program_or_sg_grounding_reads": 0,
            "predictions_frozen_before_evaluator_open": True,
            "official_test_outcomes_historically_consumed": True,
            "same_official_stsg_backend_for_all_harness_arms": True,
            "matched_max_tool_calls": 5,
        },
        "lineage": {
            "runtime_artifact_sha256": runtime["artifact_sha256"],
            "v61_base_report_sha256": evaluator["report_sha256"],
            "v61_manifest_sha256": manifest["manifest_sha256"],
            "source_artifact_sha256": source["artifact_sha256"],
            "broad_authorization_sha256": authorization["artifact_sha256"],
            "official_test_stsg_sha256": stsg_sha,
            "official_ontology_sha256": sha256_file(ONTOLOGY),
        },
    }
    report = body | {"report_sha256": stable_hash(body)}
    output = REPO / "docs/results/agqa2_official_test_oracle_transfer_v4_consumed.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
