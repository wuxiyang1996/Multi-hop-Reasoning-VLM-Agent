#!/usr/bin/env python3
"""Freeze a V12 reserve with relation-operator arity closure."""

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
from scripts.freeze_agqa2_active_grounding_v6 import target_observability_applicable  # noqa: E402


ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
NONCE = "agqa2-v12-typed-arity-closed-fresh-confirmation"


def _eligible(plan) -> bool:
    if plan.obligation_kind != RELATION_ROUTE:
        return target_observability_applicable(plan)
    if plan.comparison == "CHOOSE_OBJECT":
        return True
    if plan.comparison != "EXISTS" or not target_observability_applicable(plan):
        return False
    operand = plan.operand_a.casefold()
    return not any(marker in operand for marker in (
        " both ", " but not ", " neither ", " either ", " and ",
    ))


def _select(development: dict, exposed: list[dict], extra_excluded: set[str]) -> dict:
    excluded = extra_excluded | {
        str(row["video_id"]) for manifest in exposed for row in manifest["samples"]
    }
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
                if route != plan.obligation_kind or route not in ROUTES:
                    continue
                candidates[route].append({
                    "task_id": task_id,
                    "video_id": video_id,
                    "oracle_route": route,
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                    "public_parser_plan_sha256": plan.plan_sha256,
                    "applicability_rule": "V12_SOURCE_OPERATOR_TYPED_ARITY_CLOSURE_V1",
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
            if sum(x["oracle_route"] == route for x in selected) == 4:
                break
    counts = {
        route: sum(row["oracle_route"] == route for row in selected)
        for route in ROUTES
    }
    if any(value != 4 for value in counts.values()):
        raise RuntimeError(f"insufficient V12 candidates: {counts}")
    selected.sort(key=lambda row: (ROUTES.index(row["oracle_route"]), row["rank_sha256"]))
    core = {
        "schema_version": "agqa2-active-grounding-selection-v12",
        "status": "FROZEN_V12_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V12_CALLS",
        "split": "reserve",
        "claim_boundary": (
            "V11_DEVELOPMENT_QUALIFIED;V11_RESERVE_ABORTED_BEFORE_CALL;12_NEW_"
            "VIDEO_DISJOINT_TYPED_ARITY_CLOSED_CANDIDATES;NOT_UNTOUCHED_METADATA"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_PRIOR_MANIFEST_AND_ABORTED_V11_SELECTION_VIDEOS;REQUIRE_"
            "PUBLIC_PLAN_SOURCE_OPERATOR_ARITY_CLOSURE;FOUR_FIXED_HASH_RANKED_"
            "CANDIDATES_PER_ROUTE;NO_ANSWER_SCENE_GRAPH_OR_DIRECT_READ"
        ),
        "archive_path": development["archive_path"],
        "archive_sha256": development["archive_sha256"],
        "entry": development["entry"],
        "video_root": development["video_root"],
        "per_route_candidates": 4,
        "per_route_evaluated": 3,
        "route_counts": counts,
        "samples": selected,
        "sample_count": 12,
        "unique_video_count": len(used),
        "excluded_exposed_video_ids": sorted(excluded),
        "parent_manifest_sha256": sorted(x["manifest_sha256"] for x in exposed),
        "aborted_v11_selection_sha256": (
            "79cc37bd78b1acbbb49f5057ea0c0be918a67829b66da3472e715ee07633ac2f"
        ),
        "answer_read_during_freeze": False,
        "scene_graph_grounding_read_during_freeze": False,
        "direct_response_read_during_freeze": False,
        "prior_v12_raw_video_exposure": False,
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
        "schema_version": "agqa2-active-grounding-manifest-v12",
        "status": "FROZEN_V12_RAW_VIDEO_UNSEEN_BEFORE_V12_NEURAL_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": sum(
            not row["video_present_at_selection"] for row in samples
        ),
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v12_reserve"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V12 reserve is already consumed")
    dev_report_path = REPO_ROOT / "runs/agqa2_active_grounding_v11_development/report.json"
    dev_report = json.loads(dev_report_path.read_text())
    if not dev_report.get("grounder_qualified"):
        raise ValueError("V11 development did not qualify")
    summary_path = REPO_ROOT / "docs/results/agqa2_active_grounding_v11_development_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["grounder_sha256"] != dev_report["grounder_sha256"]:
        raise ValueError("V11 summary grounder mismatch")

    manifest_paths = sorted(
        path for path in (REPO_ROOT / "configs").glob(
            "agqa2_active_grounding_v*_manifest.json"
        )
        if "v12_reserve" not in path.name
    )
    exposed = [_verified_json(path, "manifest_sha256") for path in manifest_paths]
    v11_selection = _verified_json(
        REPO_ROOT / "configs/agqa2_active_grounding_v11_reserve_selection.json",
        "manifest_sha256",
    )
    extra_excluded = {str(row["video_id"]) for row in v11_selection["samples"]}
    development = _verified_json(
        REPO_ROOT / "configs/agqa2_active_grounding_v11_development_manifest.json",
        "manifest_sha256",
    )
    selection_path = REPO_ROOT / "configs/agqa2_active_grounding_v12_reserve_selection.json"
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path.is_file()
        else _select(development, exposed, extra_excluded)
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
            "selected_task_ids": [row["task_id"] for row in selection["samples"]],
            "selected_video_ids": [row["video_id"] for row in selection["samples"]],
            "missing_video_ids": missing,
            "next": "download frozen videos and rerun",
        }, indent=2))
        return

    reserve = _seal(selection)
    reserve_path = REPO_ROOT / "configs/agqa2_active_grounding_v12_reserve_manifest.json"
    reserve_path.write_text(json.dumps(reserve, indent=2, sort_keys=True) + "\n")
    development_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v11_development.json"
    ).read_text())
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v12-reserve",
        "status": "FROZEN_BEFORE_ANY_V12_RESERVE_NEURAL_CALL",
        "claim_boundary": reserve["claim_boundary"],
        "development_qualification_summary": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(summary_path),
        "reserve_selection": str(selection_path.relative_to(REPO_ROOT)),
        "reserve_selection_sha256": selection["manifest_sha256"],
        "reserve_manifest": str(reserve_path.relative_to(REPO_ROOT)),
        "reserve_manifest_sha256": reserve["manifest_sha256"],
        "applicability_rule": "V12_SOURCE_OPERATOR_TYPED_ARITY_CLOSURE_V1",
        "execution_calibration": deepcopy(development_config["execution_calibration"]),
        "runtime_selection": deepcopy(development_config["runtime_selection"]),
        "acquisition": deepcopy(development_config["acquisition"]),
        "reserve_gates": deepcopy(development_config["qualification_gates"]),
        "failure_policy": {
            "reserve": "RUN_ONCE_ON_FROZEN_V12_POOL;NO_POST_RESERVE_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v12_reserve_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config = deepcopy(development_config)
    config.update({
        "schema_version": "agqa2-active-grounding-reserve-config-v12",
        "status": "FROZEN_V12_TYPED_ARITY_CLOSED_RESERVE",
        "split": "reserve",
        "claim_boundary": reserve["claim_boundary"],
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "manifest": str(reserve_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(reserve_path),
        "expected_manifest_status": reserve["status"],
        "expected_preregistration_status": preregistration["status"],
        "development_qualification_report": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V12",
    })
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v12_reserve.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "reserve_manifest_sha256": reserve["manifest_sha256"],
        "candidate_video_ids": [row["video_id"] for row in reserve["samples"]],
        "excluded_video_count": len(selection["excluded_exposed_video_ids"]),
        "config_file_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
