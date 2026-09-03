#!/usr/bin/env python3
"""Freeze an unchanged-grounder 30-row AGQA V14 replication."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.agqa_program_transfer import (  # noqa: E402
    RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE, profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v12_reserve import _eligible  # noqa: E402


ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
NONCE = "agqa2-v14-unchanged-v13-grounder-30-row-replication"
CANDIDATES_PER_ROUTE = 12
EVALUATED_PER_ROUTE = 10


def _program_answer_space_matches(plan, program: str) -> bool:
    root = program.strip().split("(", 1)[0]
    expected = {
        "EXISTS": "Exists",
        "CHOOSE_OBJECT": "Choose",
        "BEFORE_AFTER": "Compare",
        "SELECT_LONGER": "Choose",
        "SELECT_SHORTER": "Choose",
        "VERIFY_A_LONGER": "Equals",
        "VERIFY_A_SHORTER": "Equals",
    }
    return expected.get(plan.comparison) == root


def _select(development: dict, excluded: set[str]) -> dict:
    candidates: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(development["archive_path"]) as bundle, bundle.open(
        development["entry"], "r"
    ) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if not video_id or video_id in excluded:
                    continue
                question = str(row.get("question", ""))
                plan = parse_public_question_plan(question)
                if plan is None or not _eligible(plan):
                    continue
                program = str(row.get("program", ""))
                route = profile_program(task_id=task_id, program=program).route_kind
                if (
                    route != plan.obligation_kind
                    or route not in ROUTES
                    or not _program_answer_space_matches(plan, program)
                ):
                    continue
                candidates[route].append({
                    "task_id": task_id,
                    "video_id": video_id,
                    "oracle_route": route,
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                    "public_parser_plan_sha256": plan.plan_sha256,
                    "applicability_rule": (
                        "V14_V13_ATOMIC_TYPED_ARITY_PLUS_PROGRAM_ROOT_ANSWER_"
                        "SPACE_COMPATIBILITY"
                    ),
                    "rank_sha256": stable_hash(f"{NONCE}:{task_id}"),
                })
    for route in ROUTES:
        candidates[route].sort(key=lambda row: row["rank_sha256"])
    root = Path(development["video_root"])
    selected: list[dict] = []
    used: set[str] = set()
    for route in sorted(
        ROUTES,
        key=lambda value: len({row["video_id"] for row in candidates[value]}),
    ):
        for row in candidates[route]:
            if row["video_id"] in used:
                continue
            path = root / f"{row['video_id']}.mp4"
            selected.append(row | {
                "video_path": str(path),
                "video_present_at_selection": path.is_file(),
            })
            used.add(row["video_id"])
            if sum(x["oracle_route"] == route for x in selected) == CANDIDATES_PER_ROUTE:
                break
    counts = {
        route: sum(row["oracle_route"] == route for row in selected)
        for route in ROUTES
    }
    if any(value != CANDIDATES_PER_ROUTE for value in counts.values()):
        raise RuntimeError(f"insufficient V14 candidates: {counts}")
    selected.sort(key=lambda row: (ROUTES.index(row["oracle_route"]), row["rank_sha256"]))
    core = {
        "schema_version": "agqa2-active-grounding-selection-v14",
        "status": "FROZEN_V14_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V14_CALLS",
        "split": "reserve",
        "claim_boundary": (
            "UNCHANGED_V13_GROUNDER;36_NEW_VIDEO_DISJOINT_CANDIDATES;30_ROW_"
            "OUTCOME_BLIND_REPLICATION;NOT_UNTOUCHED_METADATA"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "EXCLUDE_V13_SELECTION_AND_ALL_ITS_PRIOR_EXCLUSIONS;REQUIRE_ATOMIC_"
            "TYPED_ARITY_AND_PROGRAM_ROOT_ANSWER_SPACE_COMPATIBILITY;TWELVE_"
            "FIXED_HASH_CANDIDATES_PER_ROUTE;NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "archive_path": development["archive_path"],
        "archive_sha256": development["archive_sha256"],
        "entry": development["entry"],
        "video_root": development["video_root"],
        "per_route_candidates": CANDIDATES_PER_ROUTE,
        "per_route_evaluated": EVALUATED_PER_ROUTE,
        "route_counts": counts,
        "samples": selected,
        "sample_count": len(selected),
        "unique_video_count": len(used),
        "excluded_exposed_video_ids": sorted(excluded),
        "parent_v13_selection_sha256": (
            "7aa6a7ab951798b36871a3ed9b714e0f380f20b19c28b7bfbe399e21780c89a0"
        ),
        "answer_read_during_freeze": False,
        "scene_graph_grounding_read_during_freeze": False,
        "direct_response_read_during_freeze": False,
        "functional_program_root_read_for_answer_space_only": True,
        "prior_v14_raw_video_exposure": False,
        "raw_video_archive": {
            "url": (
                "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/"
                "charades/Charades_v1_480.zip"
            ),
            "archive_prefix": "Charades_v1_480/",
        },
    }
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: dict) -> dict:
    samples = []
    for row in selection["samples"]:
        path = Path(row["video_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        samples.append(dict(row) | {
            "video_sha256": _sha256(path), "video_bytes": path.stat().st_size,
        })
    core = {
        key: deepcopy(value) for key, value in selection.items()
        if key not in {"manifest_sha256", "raw_video_archive"}
    }
    core.update({
        "schema_version": "agqa2-active-grounding-manifest-v14",
        "status": "FROZEN_V14_RAW_VIDEO_UNSEEN_BEFORE_V14_NEURAL_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": sum(
            not row["video_present_at_selection"] for row in samples
        ),
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v14_replication"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V14 replication is already consumed")
    v13_result_path = REPO_ROOT / "docs/results/agqa2_active_grounding_v13_reserve_result.json"
    v13_result = json.loads(v13_result_path.read_text())
    if not v13_result.get("grounder_qualified"):
        raise ValueError("V13 reserve did not qualify")
    v13_selection = _verified_json(
        REPO_ROOT / "configs/agqa2_active_grounding_v13_reserve_selection.json",
        "manifest_sha256",
    )
    excluded = set(v13_selection["excluded_exposed_video_ids"]) | {
        str(row["video_id"]) for row in v13_selection["samples"]
    }
    development = _verified_json(
        REPO_ROOT / "configs/agqa2_active_grounding_v13_development_manifest.json",
        "manifest_sha256",
    )
    selection_path = REPO_ROOT / "configs/agqa2_active_grounding_v14_replication_selection.json"
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path.is_file() else _select(development, excluded)
    )
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    missing = [
        row["video_id"] for row in selection["samples"]
        if not Path(row["video_path"]).is_file()
    ]
    if missing:
        print(json.dumps({
            "status": selection["status"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "selected_video_ids": [row["video_id"] for row in selection["samples"]],
            "missing_video_ids": missing,
            "next": "download frozen videos and rerun",
        }, indent=2))
        return

    manifest = _seal(selection)
    manifest_path = REPO_ROOT / "configs/agqa2_active_grounding_v14_replication_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    base_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v13_reserve.json"
    ).read_text())
    runtime_selection = deepcopy(base_config["runtime_selection"])
    runtime_selection.update({
        "candidate_count": 36,
        "per_predicted_route": EVALUATED_PER_ROUTE,
    })
    gates = deepcopy(base_config["qualification_gates"])
    gates.update({
        "required_valid_runtime_rows": 30,
        "minimum_route_correct": 30,
        "minimum_decisive_executions": 20,
        "minimum_typed_vs_direct_wins": 2,
        "maximum_typed_vs_direct_losses": 0,
        "required_source_permuted_abstentions": 30,
        "required_target_written_equivalent_matches": 30,
        "maximum_reported_provider_cost_usd": 0.30,
    })
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v14-replication",
        "status": "FROZEN_BEFORE_ANY_V14_REPLICATION_NEURAL_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "unchanged_v13_grounder_sha256": v13_result["grounder_sha256"],
        "v13_result_file_sha256": _sha256(v13_result_path),
        "selection": str(selection_path.relative_to(REPO_ROOT)),
        "selection_sha256": selection["manifest_sha256"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": manifest["manifest_sha256"],
        "runtime_selection": runtime_selection,
        "execution_calibration": deepcopy(base_config["execution_calibration"]),
        "acquisition": deepcopy(base_config["acquisition"]),
        "replication_gates": gates,
        "cost_scaling": {
            "v13_candidate_count": 12,
            "v13_cost_usd": v13_result["reported_provider_cost_usd"],
            "v14_candidate_count": 36,
            "linear_projection_usd": 3 * v13_result["reported_provider_cost_usd"],
            "frozen_cap_usd": 0.30,
        },
        "failure_policy": {
            "replication": "RUN_ONCE_ON_FROZEN_V14_POOL;NO_POST_RESULT_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v14_replication_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config = deepcopy(base_config)
    config.update({
        "schema_version": "agqa2-active-grounding-replication-config-v14",
        "status": "FROZEN_V14_UNCHANGED_V13_GROUNDER_REPLICATION",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": preregistration["status"],
        "development_qualification_report": str(v13_result_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(v13_result_path),
        "runtime_selection": runtime_selection,
        "qualification_gates": gates,
        "report_version": "V14",
    })
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v14_replication.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    if config["grounder"] != base_config["grounder"]:
        raise AssertionError("V14 grounder lineage changed from V13")
    print(json.dumps({
        "status": preregistration["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_count": manifest["sample_count"],
        "evaluated_count": 3 * EVALUATED_PER_ROUTE,
        "excluded_video_count": len(excluded),
        "config_file_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
