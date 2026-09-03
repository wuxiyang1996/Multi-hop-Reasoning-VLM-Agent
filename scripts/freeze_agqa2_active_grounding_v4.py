#!/usr/bin/env python3
"""Freeze AGQA V4 after V3 runtime-only failures and before V4 calls.

V4 changes only generic inflection validation, malformed transport-envelope
retrying, and semantic lineage coverage.  No V3 reserve answer or scene graph
is read.  The V4 reserve excludes every V3 development and reserve raw video.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_program_transfer import (  # noqa: E402
    RELATION_ROUTE,
    TEMPORAL_PAIR_ROUTE,
    TEMPORAL_SINGLE_ROUTE,
    profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object,
)


ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
RESERVE_NONCE = "agqa2-active-grounding-v4-after-v3-runtime-failure"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_json(path: Path, hash_field: str) -> dict:
    payload = json.loads(path.read_text())
    body = dict(payload)
    claimed = body.pop(hash_field)
    if stable_hash(body) != claimed:
        raise ValueError(f"content hash mismatch: {path}")
    return payload


def _select_reserve(
    *, development: dict, v3_reserve: dict, per_route: int,
) -> dict:
    excluded = {
        str(row["video_id"])
        for manifest in (development, v3_reserve)
        for row in manifest["samples"]
    }
    video_root = Path(development["video_root"])
    candidates: dict[str, list[dict]] = defaultdict(list)
    archive = Path(development["archive_path"])
    with zipfile.ZipFile(archive) as bundle, bundle.open(
        development["entry"], "r"
    ) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if not video_id or video_id in excluded:
                    continue
                question = str(row.get("question", ""))
                program = str(row.get("program", ""))
                route = profile_program(task_id=task_id, program=program).route_kind
                if route not in ROUTES:
                    continue
                candidates[route].append({
                    "task_id": task_id,
                    "video_id": video_id,
                    "oracle_route": route,
                    "question_sha256": stable_hash(question),
                    "program_sha256": stable_hash(program),
                    "rank_sha256": stable_hash(f"{RESERVE_NONCE}:{task_id}"),
                })
    for route in ROUTES:
        candidates[route].sort(key=lambda row: row["rank_sha256"])
    selected: list[dict] = []
    used: set[str] = set()
    route_order = sorted(
        ROUTES,
        key=lambda route: len({row["video_id"] for row in candidates[route]}),
    )
    for route in route_order:
        for row in candidates[route]:
            if row["video_id"] in used:
                continue
            video_path = video_root / f"{row['video_id']}.mp4"
            selected.append(row | {
                "video_path": str(video_path),
                "video_present_at_selection": video_path.is_file(),
            })
            used.add(row["video_id"])
            if sum(x["oracle_route"] == route for x in selected) == per_route:
                break
    counts = {
        route: sum(row["oracle_route"] == route for row in selected)
        for route in ROUTES
    }
    if any(value != per_route for value in counts.values()):
        raise RuntimeError(f"insufficient V4 reserve videos: {counts}")
    selected.sort(key=lambda row: (
        ROUTES.index(row["oracle_route"]), row["rank_sha256"],
    ))
    core = {
        "schema_version": "agqa2-active-grounding-selection-v4",
        "status": "FROZEN_V4_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V4_CALLS",
        "split": "reserve",
        "claim_boundary": (
            "V4_RAW_VIDEO_UNSEEN_FROM_ALL_V3_DEVELOPMENT_AND_RESERVE_CALLS;"
            "AGQA_TEST_METADATA_PREVIOUSLY_SCANNED;NOT_UNTOUCHED_BENCHMARK"
        ),
        "selection_nonce": RESERVE_NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_V3_DEVELOPMENT_AND_RESERVE_VIDEO_IDS;SHA256_RANK_"
            "WITHIN_ORACLE_ROUTE;GLOBAL_VIDEO_DISJOINT;NO_ANSWER_OR_SCENE_GRAPH_READ"
        ),
        "archive_path": development["archive_path"],
        "archive_sha256": development["archive_sha256"],
        "entry": development["entry"],
        "video_root": development["video_root"],
        "available_video_count_at_selection": len(list(video_root.glob("*.mp4"))),
        "per_route": per_route,
        "route_counts": counts,
        "samples": selected,
        "sample_count": len(selected),
        "unique_video_count": len(used),
        "excluded_v3_video_ids": sorted(excluded),
        "v3_development_manifest_sha256": development["manifest_sha256"],
        "v3_reserve_manifest_sha256": v3_reserve["manifest_sha256"],
        "answer_read_during_freeze": False,
        "scene_graph_grounding_read_during_freeze": False,
        "functional_program_visible_to_grounder": False,
        "prior_v4_raw_video_exposure": False,
        "raw_video_archive": {
            "url": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
            "archive_prefix": "Charades_v1_480/"
        },
    }
    return core | {"manifest_sha256": stable_hash(core)}


def _seal_selection(selection: dict) -> dict:
    samples = []
    for row in selection["samples"]:
        video_path = Path(row["video_path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        samples.append(dict(row) | {
            "video_sha256": _sha256(video_path),
            "video_bytes": video_path.stat().st_size,
        })
    core = {
        key: deepcopy(value)
        for key, value in selection.items()
        if key not in {"manifest_sha256", "raw_video_archive"}
    }
    core.update({
        "schema_version": "agqa2-active-grounding-manifest-v4",
        "status": "FROZEN_V4_RAW_VIDEO_UNSEEN_BEFORE_V4_NEURAL_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": sum(
            not row["video_present_at_selection"]
            for row in samples
        ),
    })
    return core | {"manifest_sha256": stable_hash(core)}


def freeze(
    *, v3_development_manifest_path: Path, v3_reserve_manifest_path: Path,
    v3_development_config_path: Path, v3_failure_path: Path,
    selection_path: Path | None = None,
) -> tuple[dict, dict | None, dict | None, dict | None]:
    development = _verified_json(v3_development_manifest_path, "manifest_sha256")
    v3_reserve = _verified_json(v3_reserve_manifest_path, "manifest_sha256")
    failure = json.loads(v3_failure_path.read_text())
    if failure.get("official_answers_read") is not False:
        raise ValueError("V4 may not adapt after V3 reserve answers were read")
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path is not None and selection_path.is_file()
        else _select_reserve(
            development=development, v3_reserve=v3_reserve, per_route=3,
        )
    )
    if any(not Path(row["video_path"]).is_file() for row in selection["samples"]):
        return selection, None, None, None
    reserve = _seal_selection(selection)
    v3_config = json.loads(v3_development_config_path.read_text())
    gates = deepcopy(v3_config["qualification_gates"])
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v4",
        "status": "FROZEN_BEFORE_ANY_V4_NEURAL_CALL",
        "claim_boundary": (
            "V3_RUNTIME_FAILURE_ONLY_REPAIR;DEVELOPMENT_REQUALIFICATION_THEN_"
            "SINGLE_FRESH_RAW_VIDEO_UNSEEN_RESERVE;NO_UNTOUCHED_BENCHMARK_OR_"
            "SOURCE_PROVENANCE_CLAIM"
        ),
        "development_manifest": str(v3_development_manifest_path.relative_to(REPO_ROOT)),
        "reserve_manifest": "configs/agqa2_active_grounding_v4_reserve_manifest.json",
        "reserve_selection": "configs/agqa2_active_grounding_v4_reserve_selection.json",
        "reserve_selection_sha256": selection["manifest_sha256"],
        "v3_failure_evidence": str(v3_failure_path.relative_to(REPO_ROOT)),
        "v3_failure_evidence_file_sha256": _sha256(v3_failure_path),
        "repairs": [
            "MORPHOLOGICAL_EQUIVALENCE_IN_QUERY_RELATION_VALIDATION",
            "IDENTICAL_REQUEST_RETRY_FOR_NULL_PROVIDER_CHOICES",
            "COLLECTOR_HASH_INCLUDED_IN_GROUNDER_SEMANTIC_LINEAGE",
        ],
        "development_gates": gates,
        "reserve_gates": deepcopy(gates),
        "failure_policy": {
            "development": "MUST_REQUALIFY_WITHOUT_LOWERING_GATES",
            "reserve": "RUN_ONCE_ON_V4_ONLY_RAW_VIDEO_RESERVE;NO_POST_RESERVE_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    config = deepcopy(v3_config)
    config.update({
        "schema_version": "agqa2-active-grounding-development-config-v4",
        "status": "FROZEN_V4_RUNTIME_REPAIR_DEVELOPMENT_CANDIDATE",
        "split": "development",
        "claim_boundary": preregistration["claim_boundary"],
        "preregistration": "configs/agqa2_active_grounding_v4_preregistration.json",
        "manifest": str(v3_development_manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(v3_development_manifest_path),
        "qualification_gates": deepcopy(gates),
        "expected_preregistration_status": preregistration["status"],
        "report_version": "V4",
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
    return selection, reserve, preregistration, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-route", type=int, default=3)
    args = parser.parse_args()
    if args.per_route != 3:
        raise ValueError("V4 preregistration fixes three samples per route")
    output_root = REPO_ROOT / "runs/agqa2_active_grounding_v4_reserve"
    if output_root.exists() and any(output_root.rglob("*.json")):
        raise RuntimeError("V4 reserve already has receipts; refusing to refreeze")
    selection_path = REPO_ROOT / "configs/agqa2_active_grounding_v4_reserve_selection.json"
    selection, reserve, preregistration, config = freeze(
        v3_development_manifest_path=REPO_ROOT / "configs/agqa2_active_grounding_v3_development_manifest.json",
        v3_reserve_manifest_path=REPO_ROOT / "configs/agqa2_active_grounding_v3_reserve_manifest.json",
        v3_development_config_path=REPO_ROOT / "configs/agqa2_active_grounding_v3_development.json",
        v3_failure_path=REPO_ROOT / "docs/results/agqa2_active_grounding_v3_reserve_failure.json",
        selection_path=selection_path,
    )
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    missing = [
        row["video_id"] for row in selection["samples"]
        if not Path(row["video_path"]).is_file()
    ]
    if reserve is None or preregistration is None or config is None:
        print(json.dumps({
            "status": selection["status"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "selected_video_ids": [row["video_id"] for row in selection["samples"]],
            "missing_video_ids": missing,
            "next": "download selected official Charades videos, then rerun this freezer",
        }, indent=2))
        return
    outputs = (
        (REPO_ROOT / "configs/agqa2_active_grounding_v4_reserve_manifest.json", reserve),
        (REPO_ROOT / "configs/agqa2_active_grounding_v4_preregistration.json", preregistration),
    )
    for path, payload in outputs:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    prereg_path = outputs[1][0]
    config["preregistration_file_sha256"] = _sha256(prereg_path)
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v4_development.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "reserve_manifest_sha256": reserve["manifest_sha256"],
        "reserve_video_ids": [row["video_id"] for row in reserve["samples"]],
        "excluded_v3_video_count": len(reserve["excluded_v3_video_ids"]),
        "development_config_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
