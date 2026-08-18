#!/usr/bin/env python3
"""Freeze AGQA V7 outcome-blind neural-grounding applicability selection."""

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
from scripts.freeze_agqa2_active_grounding_v6 import (  # noqa: E402
    target_observability_applicable,
)


ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
NONCE = "agqa2-v7-outcome-blind-typed-evidence-pool-selection"


def _select(development: dict, exposed: list[dict]) -> dict:
    excluded = {
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
                if plan is None or not target_observability_applicable(plan):
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
                    "rank_sha256": stable_hash(f"{NONCE}:{task_id}"),
                })
    for route in ROUTES:
        candidates[route].sort(key=lambda row: row["rank_sha256"])
    root = Path(development["video_root"])
    selected, used = [], set()
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
    counts = {route: sum(x["oracle_route"] == route for x in selected) for route in ROUTES}
    if any(value != 4 for value in counts.values()):
        raise RuntimeError(f"insufficient V7 candidates: {counts}")
    selected.sort(key=lambda row: (ROUTES.index(row["oracle_route"]), row["rank_sha256"]))
    core = {
        "schema_version": "agqa2-active-grounding-selection-v7",
        "status": "FROZEN_V7_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V7_CALLS",
        "split": "reserve",
        "claim_boundary": (
            "V6_ADAPTED_RECURRENT_CONSENSUS;12_FRESH_CANDIDATES;OUTCOME_BLIND_"
            "TYPED_EVIDENCE_SELECTS_3_PER_SOURCE_TYPE_BEFORE_GOLD;NOT_UNTOUCHED"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_V3_TO_V6_EXPOSED_VIDEOS;FOUR_SHA256_RANKED_TARGET_"
            "OBSERVABLE_CANDIDATES_PER_ROUTE;RUNTIME_TYPED_EVIDENCE_SELECTS_"
            "THREE_PER_PREDICTED_ROUTE;NO_DIRECT_OR_GOLD_IN_RUNTIME_SELECTION"
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
        "parent_manifest_sha256": [row["manifest_sha256"] for row in exposed],
        "answer_read_during_freeze": False,
        "scene_graph_grounding_read_during_freeze": False,
        "prior_v7_raw_video_exposure": False,
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
            "video_sha256": _sha256(path), "video_bytes": path.stat().st_size,
        })
    core = {
        key: deepcopy(value) for key, value in selection.items()
        if key not in {"manifest_sha256", "raw_video_archive"}
    }
    core.update({
        "schema_version": "agqa2-active-grounding-manifest-v7",
        "status": "FROZEN_V7_RAW_VIDEO_UNSEEN_BEFORE_V7_NEURAL_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": sum(not row["video_present_at_selection"] for row in samples),
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    if (REPO_ROOT / "runs/agqa2_active_grounding_v7_reserve").exists():
        if any((REPO_ROOT / "runs/agqa2_active_grounding_v7_reserve").rglob("*.json")):
            raise RuntimeError("V7 reserve already consumed; refusing to refreeze")
    v6_result_path = REPO_ROOT / "docs/results/agqa2_active_grounding_v6_reserve_result.json"
    if json.loads(v6_result_path.read_text())["status"] != (
        "AGQA2_ACTIVE_GROUNDER_V6_RESERVE_NOT_QUALIFIED"
    ):
        raise ValueError("V7 requires preserved V6 adaptation evidence")
    manifest_paths = [
        REPO_ROOT / f"configs/agqa2_active_grounding_{version}_{split}_manifest.json"
        for version, split in (
            ("v3", "development"), ("v3", "reserve"), ("v4", "reserve"),
            ("v5", "reserve"), ("v6", "reserve"),
        )
    ]
    exposed = [_verified_json(path, "manifest_sha256") for path in manifest_paths]
    selection_path = REPO_ROOT / "configs/agqa2_active_grounding_v7_reserve_selection.json"
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path.is_file() else _select(exposed[0], exposed)
    )
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    missing = [row["video_id"] for row in selection["samples"] if not Path(row["video_path"]).is_file()]
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
    reserve_path = REPO_ROOT / "configs/agqa2_active_grounding_v7_reserve_manifest.json"
    reserve_path.write_text(json.dumps(reserve, indent=2, sort_keys=True) + "\n")
    base_config_path = REPO_ROOT / "configs/agqa2_active_grounding_v6_development.json"
    config = json.loads(base_config_path.read_text())
    gates = deepcopy(config["qualification_gates"])
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v7",
        "status": "FROZEN_BEFORE_ANY_V7_NEURAL_CALL",
        "claim_boundary": selection["claim_boundary"],
        "development_manifest": str(manifest_paths[0].relative_to(REPO_ROOT)),
        "reserve_selection": str(selection_path.relative_to(REPO_ROOT)),
        "reserve_selection_sha256": selection["manifest_sha256"],
        "reserve_manifest": str(reserve_path.relative_to(REPO_ROOT)),
        "v6_adaptation_result": str(v6_result_path.relative_to(REPO_ROOT)),
        "v6_adaptation_result_file_sha256": _sha256(v6_result_path),
        "runtime_selection": {
            "mode": "OUTCOME_BLIND_TYPED_EVIDENCE_RANK_V1",
            "candidate_count": 12,
            "per_predicted_route": 3,
            "forbidden": ["direct_response", "answer", "program", "scene_graph"],
        },
        "development_gates": gates,
        "reserve_gates": deepcopy(gates),
        "failure_policy": {
            "development": "MUST_REQUALIFY_WITHOUT_LOWERING_GATES",
            "reserve": "RUN_ONCE_ON_V7_FRESH_POOL;NO_POST_RESERVE_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v7_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "schema_version": "agqa2-active-grounding-development-config-v7",
        "status": "FROZEN_V7_RECURRENT_CONSENSUS_DEVELOPMENT_CANDIDATE",
        "split": "development",
        "claim_boundary": selection["claim_boundary"],
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "manifest": str(manifest_paths[0].relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_paths[0]),
        "qualification_gates": gates,
        "expected_preregistration_status": preregistration["status"],
        "query_parser_mode": "DETERMINISTIC_EXPLICIT_OPERAND_GRAMMAR_V1",
        "applicability_mode": "TARGET_OBSERVABILITY_PLUS_TYPED_EVIDENCE_V1",
        "runtime_selection": preregistration["runtime_selection"],
        "report_version": "V7",
    })
    config["grounder"].update({
        "module_sha256": _sha256(REPO_ROOT / config["grounder"]["module"]),
        "collector_sha256": _sha256(REPO_ROOT / config["grounder"]["collector"]),
        "executor": "src/motif_transfer/agqa_frame_grounder.py",
        "executor_sha256": _sha256(REPO_ROOT / "src/motif_transfer/agqa_frame_grounder.py"),
    })
    config["local_object_grounder"]["module_sha256"] = _sha256(
        REPO_ROOT / config["local_object_grounder"]["module"]
    )
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v7_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "reserve_manifest_sha256": reserve["manifest_sha256"],
        "candidate_video_ids": [row["video_id"] for row in reserve["samples"]],
        "excluded_video_count": len(selection["excluded_exposed_video_ids"]),
        "development_config_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
