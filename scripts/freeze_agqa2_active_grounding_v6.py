#!/usr/bin/env python3
"""Freeze AGQA V6 with a V5-induced target-observability applicability gate."""

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

from motif_transfer.agqa_active_frame_grounder import (  # noqa: E402
    AGQAQueryPlan,
    parse_public_question_plan,
)
from motif_transfer.agqa_program_transfer import (  # noqa: E402
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
    profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import _iter_top_level_object  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402


ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
NON_GROUNDED_REFERENCES = (
    "thing they", "object they", "anything", "the first action",
    "the second action", "the first event", "the second event",
)
LOW_BOUNDARY_VISIBILITY = ("smiling at", "laughing at")
RESERVE_NONCE = "agqa2-v6-v5-induced-observability-gate-fresh-reserve"


def target_observability_applicable(plan: AGQAQueryPlan) -> bool:
    operands = f"{plan.operand_a} {plan.operand_b}".casefold()
    if any(marker in operands for marker in NON_GROUNDED_REFERENCES):
        return False
    if plan.obligation_kind == RELATION_ROUTE:
        return plan.comparison == "EXISTS"
    if plan.obligation_kind == TEMPORAL_SINGLE_ROUTE:
        return not any(marker in operands for marker in LOW_BOUNDARY_VISIBILITY)
    return plan.obligation_kind == TEMPORAL_PAIR_ROUTE


def _select(*, development: dict, exposed: list[dict]) -> dict:
    excluded = {
        str(row["video_id"])
        for manifest in exposed
        for row in manifest["samples"]
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
                if plan is None or not target_observability_applicable(plan):
                    continue
                program = str(row.get("program", ""))
                route = profile_program(task_id=task_id, program=program).route_kind
                if route not in ROUTES or route != plan.obligation_kind:
                    continue
                candidates[route].append({
                    "task_id": task_id,
                    "video_id": video_id,
                    "oracle_route": route,
                    "public_parser_plan_sha256": plan.plan_sha256,
                    "applicability_rule": "TARGET_OBSERVABILITY_V1",
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                    "rank_sha256": stable_hash(f"{RESERVE_NONCE}:{task_id}"),
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
            if sum(x["oracle_route"] == route for x in selected) == 3:
                break
    counts = {
        route: sum(row["oracle_route"] == route for row in selected)
        for route in ROUTES
    }
    if any(value != 3 for value in counts.values()):
        raise RuntimeError(f"insufficient V6 applicable candidates: {counts}")
    selected.sort(key=lambda row: (
        ROUTES.index(row["oracle_route"]), row["rank_sha256"],
    ))
    core = {
        "schema_version": "agqa2-active-grounding-selection-v6",
        "status": "FROZEN_V6_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V6_CALLS",
        "split": "reserve",
        "claim_boundary": (
            "V5_INDUCED_TARGET_OBSERVABLE_EXPLICIT_OPERAND_SUBSET;RAW_VIDEO_"
            "UNSEEN_FROM_ALL_V3_V4_V5_CALLS;ADAPTATION_BASED_NOT_UNTOUCHED"
        ),
        "selection_nonce": RESERVE_NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_V3_V4_V5_EXPOSED_VIDEOS;REQUIRE_EXPLICIT_PUBLIC_"
            "OPERANDS_AND_TARGET_OBSERVABILITY_V1;SHA256_RANK_WITHIN_ROUTE;"
            "NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "applicability_rule": {
            "relation": "FIXED_OBJECT_EXISTS_ONLY",
            "temporal_pair": "EXPLICIT_NON_INDIRECT_OPERANDS",
            "temporal_single": "EXPLICIT_OPERANDS_EXCLUDING_SMILE_LAUGH_BOUNDARIES",
            "forbidden_indirect_markers": list(NON_GROUNDED_REFERENCES),
        },
        "archive_path": development["archive_path"],
        "archive_sha256": development["archive_sha256"],
        "entry": development["entry"],
        "video_root": development["video_root"],
        "per_route": 3,
        "route_counts": counts,
        "samples": selected,
        "sample_count": 9,
        "unique_video_count": len(used),
        "excluded_exposed_video_ids": sorted(excluded),
        "parent_manifest_sha256": [row["manifest_sha256"] for row in exposed],
        "answer_read_during_freeze": False,
        "scene_graph_grounding_read_during_freeze": False,
        "functional_program_visible_to_grounder": False,
        "prior_v6_raw_video_exposure": False,
        "raw_video_archive": {
            "url": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
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
            "video_sha256": _sha256(path),
            "video_bytes": path.stat().st_size,
        })
    core = {
        key: deepcopy(value)
        for key, value in selection.items()
        if key not in {"manifest_sha256", "raw_video_archive"}
    }
    core.update({
        "schema_version": "agqa2-active-grounding-manifest-v6",
        "status": "FROZEN_V6_RAW_VIDEO_UNSEEN_BEFORE_V6_NEURAL_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": sum(
            not row["video_present_at_selection"] for row in samples
        ),
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    reserve_run = REPO_ROOT / "runs/agqa2_active_grounding_v6_reserve"
    if reserve_run.exists() and any(reserve_run.rglob("*.json")):
        raise RuntimeError("V6 reserve already has receipts; refusing to refreeze")
    v5_result_path = REPO_ROOT / "docs/results/agqa2_active_grounding_v5_reserve_result.json"
    v5_result = json.loads(v5_result_path.read_text())
    if v5_result["status"] != "AGQA2_ACTIVE_GROUNDER_V5_RESERVE_NOT_QUALIFIED":
        raise ValueError("V6 requires the preserved V5 adaptation result")
    manifest_paths = [
        REPO_ROOT / "configs/agqa2_active_grounding_v3_development_manifest.json",
        REPO_ROOT / "configs/agqa2_active_grounding_v3_reserve_manifest.json",
        REPO_ROOT / "configs/agqa2_active_grounding_v4_reserve_manifest.json",
        REPO_ROOT / "configs/agqa2_active_grounding_v5_reserve_manifest.json",
    ]
    exposed = [_verified_json(path, "manifest_sha256") for path in manifest_paths]
    selection_path = REPO_ROOT / "configs/agqa2_active_grounding_v6_reserve_selection.json"
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path.is_file()
        else _select(development=exposed[0], exposed=exposed)
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

    reserve = _seal(selection)
    reserve_path = REPO_ROOT / "configs/agqa2_active_grounding_v6_reserve_manifest.json"
    reserve_path.write_text(json.dumps(reserve, indent=2, sort_keys=True) + "\n")
    base_config_path = REPO_ROOT / "configs/agqa2_active_grounding_v5_development.json"
    config = json.loads(base_config_path.read_text())
    gates = deepcopy(config["qualification_gates"])
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v6",
        "status": "FROZEN_BEFORE_ANY_V6_NEURAL_CALL",
        "claim_boundary": selection["claim_boundary"],
        "development_manifest": str(manifest_paths[0].relative_to(REPO_ROOT)),
        "reserve_selection": str(selection_path.relative_to(REPO_ROOT)),
        "reserve_selection_sha256": selection["manifest_sha256"],
        "reserve_manifest": str(reserve_path.relative_to(REPO_ROOT)),
        "v5_adaptation_result": str(v5_result_path.relative_to(REPO_ROOT)),
        "v5_adaptation_result_file_sha256": _sha256(v5_result_path),
        "applicability_rule": selection["applicability_rule"],
        "development_gates": gates,
        "reserve_gates": deepcopy(gates),
        "failure_policy": {
            "development": "MUST_REQUALIFY_WITHOUT_LOWERING_GATES",
            "reserve": "RUN_ONCE_ON_V6_ONLY_RAW_VIDEO_RESERVE;NO_POST_RESERVE_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v6_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "schema_version": "agqa2-active-grounding-development-config-v6",
        "status": "FROZEN_V6_TARGET_OBSERVABILITY_DEVELOPMENT_CANDIDATE",
        "split": "development",
        "claim_boundary": selection["claim_boundary"],
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "manifest": str(manifest_paths[0].relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_paths[0]),
        "qualification_gates": gates,
        "expected_preregistration_status": preregistration["status"],
        "query_parser_mode": "DETERMINISTIC_EXPLICIT_OPERAND_GRAMMAR_V1",
        "applicability_mode": "TARGET_OBSERVABILITY_V1",
        "report_version": "V6",
    })
    config["grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["grounder"]["module"]
    )
    config["grounder"]["collector_sha256"] = _sha256(
        REPO_ROOT / config["grounder"]["collector"]
    )
    config["local_object_grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["local_object_grounder"]["module"]
    )
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v6_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "reserve_manifest_sha256": reserve["manifest_sha256"],
        "reserve_video_ids": [row["video_id"] for row in reserve["samples"]],
        "excluded_video_count": len(selection["excluded_exposed_video_ids"]),
        "development_config_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
