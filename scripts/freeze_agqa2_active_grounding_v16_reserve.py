#!/usr/bin/env python3
"""Freeze a fresh 30-row AGQA V16 selective-transfer replication."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.agqa_active_frame_grounder import parse_public_question_plan  # noqa: E402
from motif_transfer.agqa_program_transfer import (  # noqa: E402
    RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE, profile_program,
)
from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.audit_agqa2_program_transfer_v1 import (  # noqa: E402
    _iter_top_level_object, _load_sources,
)
from scripts.collect_agqa2_active_grounding_v3 import (  # noqa: E402
    _evaluation_protocol_core, _grounder_semantic_core,
)
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v12_reserve import _eligible  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v14_replication import (  # noqa: E402
    _program_answer_space_matches,
)


ROUTES = (RELATION_ROUTE, TEMPORAL_PAIR_ROUTE, TEMPORAL_SINGLE_ROUTE)
NONCE = "agqa2-v16-selective-override-fresh-30-row-replication"
CANDIDATES_PER_ROUTE = 12
EVALUATED_PER_ROUTE = 10


def _provider_cache_flags(value: Any) -> Iterator[bool]:
    if isinstance(value, Mapping):
        usage = value.get("usage")
        if (
            "cache_reused" in value
            and isinstance(usage, Mapping)
            and not usage.get("local_non_provider_call", False)
        ):
            yield bool(value["cache_reused"])
        for child in value.values():
            yield from _provider_cache_flags(child)
    elif isinstance(value, list):
        for child in value:
            yield from _provider_cache_flags(child)


def _development_summary() -> tuple[Path, dict[str, Any]]:
    report_path = REPO_ROOT / "runs/agqa2_active_grounding_v16_development/report.json"
    report = json.loads(report_path.read_text())
    if not report.get("grounder_qualified"):
        raise ValueError("V16 development did not qualify")
    receipt_paths = sorted((report_path.parent / "runtime_receipts").glob("*.json"))
    provider_flags: list[bool] = []
    for path in receipt_paths:
        receipt = json.loads(path.read_text())
        if not receipt.get("direct_cache_reused"):
            raise ValueError(f"V16 development direct call was not replayed: {path}")
        provider_flags.extend(_provider_cache_flags(receipt))
    if not provider_flags or not all(provider_flags):
        raise ValueError("V16 development made a non-replayed provider call")
    fields = (
        "status", "grounder_qualified", "grounder_sha256",
        "evaluation_protocol_sha256", "metrics", "controls",
        "qualification_gates", "reported_provider_cost_usd", "report_sha256",
    )
    core = {key: deepcopy(report[key]) for key in fields}
    core.update({
        "schema_version": "agqa2-active-grounding-v16-development-summary",
        "development_report_file_sha256": _sha256(report_path),
        "accepted_provider_receipts_replayed_from_v15": len(provider_flags),
        "new_provider_calls_during_v16_requalification": 0,
        "runtime_receipt_count": len(receipt_paths),
    })
    summary = core | {"summary_sha256": stable_hash(core)}
    path = REPO_ROOT / "docs/results/agqa2_active_grounding_v16_development_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path, summary


def _configured_video_ids() -> set[str]:
    ids: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if value.get("video_id") is not None:
                ids.add(str(value["video_id"]))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for path in sorted((REPO_ROOT / "configs").glob("*.json")):
        if "agqa2_active_grounding_v16_reserve" in path.name:
            continue
        try:
            walk(json.loads(path.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return ids


def _select(development: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root = Path(development["video_root"])
    with zipfile.ZipFile(development["archive_path"]) as bundle, bundle.open(
        development["entry"], "r"
    ) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for task_id, row in _iter_top_level_object(text):
                video_id = str(row.get("video_id", ""))
                if (
                    not video_id
                    or video_id in excluded
                    or (root / f"{video_id}.mp4").is_file()
                ):
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
                        "V16_ATOMIC_TYPED_ARITY_PLUS_PROGRAM_ROOT_ANSWER_"
                        "SPACE_COMPATIBILITY"
                    ),
                    "rank_sha256": stable_hash(f"{NONCE}:{task_id}"),
                })
    for route in ROUTES:
        candidates[route].sort(key=lambda row: row["rank_sha256"])
    selected: list[dict[str, Any]] = []
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
                "video_present_at_selection": False,
            })
            used.add(row["video_id"])
            if sum(x["oracle_route"] == route for x in selected) == CANDIDATES_PER_ROUTE:
                break
    counts = {
        route: sum(row["oracle_route"] == route for row in selected)
        for route in ROUTES
    }
    if any(value != CANDIDATES_PER_ROUTE for value in counts.values()):
        raise RuntimeError(f"insufficient V16 candidates: {counts}")
    selected.sort(key=lambda row: (ROUTES.index(row["oracle_route"]), row["rank_sha256"]))
    core = {
        "schema_version": "agqa2-active-grounding-selection-v16",
        "status": "FROZEN_V16_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V16_CALLS",
        "split": "reserve",
        "claim_boundary": (
            "V16_SELECTIVE_OVERRIDE_GROUNDER;36_NEW_CROSS_EXPERIMENT_VIDEO_"
            "DISJOINT_CANDIDATES;30_ROW_OUTCOME_BLIND_REPLICATION;"
            "NOT_UNTOUCHED_METADATA"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_VIDEO_IDS_REFERENCED_BY_PRIOR_CONFIGS_AND_ALL_MP4S_"
            "PRESENT_IN_SHARED_CHARADES_ROOT;REQUIRE_ATOMIC_TYPED_ARITY_AND_"
            "PROGRAM_ROOT_ANSWER_SPACE_COMPATIBILITY;TWELVE_FIXED_HASH_"
            "CANDIDATES_PER_ROUTE;NO_ANSWER_OR_SCENE_GRAPH_READ"
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
        "excluded_prior_config_or_present_video_count": len(excluded),
        "excluded_video_ids_sha256": stable_hash(sorted(excluded)),
        "answer_read_during_freeze": False,
        "scene_graph_grounding_read_during_freeze": False,
        "direct_response_read_during_freeze": False,
        "functional_program_root_read_for_answer_space_only": True,
        "prior_v16_neural_grounder_exposure": False,
        "raw_video_archive": {
            "url": (
                "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/"
                "charades/Charades_v1_480.zip"
            ),
            "archive_prefix": "Charades_v1_480/",
        },
    }
    return core | {"manifest_sha256": stable_hash(core)}


def _seal(selection: dict[str, Any]) -> dict[str, Any]:
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
        "schema_version": "agqa2-active-grounding-manifest-v16-reserve",
        "status": "FROZEN_V16_RAW_VIDEO_UNSEEN_BY_NEURAL_GROUNDER_BEFORE_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": len(samples),
        "local_integrity_decode_probe_completed": True,
        "prior_neural_grounder_or_model_video_exposure": False,
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v16_reserve"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V16 reserve is already consumed")
    summary_path, development = _development_summary()
    development_manifest = _verified_json(
        REPO_ROOT / "configs/agqa2_active_grounding_v16_development_manifest.json",
        "manifest_sha256",
    )
    excluded = _configured_video_ids()
    excluded.update(path.stem for path in Path(development_manifest["video_root"]).glob("*.mp4"))
    selection_path = REPO_ROOT / "configs/agqa2_active_grounding_v16_reserve_selection.json"
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path.is_file() else _select(development_manifest, excluded)
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
            "candidate_count": selection["sample_count"],
            "selected_video_ids": [row["video_id"] for row in selection["samples"]],
            "missing_video_ids": missing,
            "next": "download exact frozen videos and rerun",
        }, indent=2))
        return

    manifest = _seal(selection)
    manifest_path = REPO_ROOT / "configs/agqa2_active_grounding_v16_reserve_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    dev_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v16_development.json"
    ).read_text())
    config = deepcopy(dev_config)
    config.update({
        "schema_version": "agqa2-active-grounding-reserve-config-v16",
        "status": "FROZEN_V16_SELECTIVE_OVERRIDE_RESERVE",
        "split": "reserve",
        "claim_boundary": manifest["claim_boundary"],
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(manifest_path),
        "expected_manifest_status": manifest["status"],
        "expected_preregistration_status": "FROZEN_BEFORE_ANY_V16_RESERVE_NEURAL_CALL",
        "development_qualification_report": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_file_sha256": _sha256(summary_path),
        "report_version": "V16",
    })
    sources, _ = _load_sources(config)
    expected_grounder_sha256 = stable_hash(_grounder_semantic_core(config, sources))
    expected_evaluation_sha256 = stable_hash(_evaluation_protocol_core(config))
    if expected_grounder_sha256 != development["grounder_sha256"]:
        raise AssertionError("V16 reserve grounder differs from development")
    if expected_evaluation_sha256 != development["evaluation_protocol_sha256"]:
        raise AssertionError("V16 reserve evaluation protocol differs from development")
    receipt_path = REPO_ROOT / "runs/agqa2_active_grounding_v16_download/receipt.json"
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v16-reserve",
        "status": "FROZEN_BEFORE_ANY_V16_RESERVE_NEURAL_CALL",
        "claim_boundary": manifest["claim_boundary"],
        "qualified_grounder_sha256": expected_grounder_sha256,
        "evaluation_protocol_sha256": expected_evaluation_sha256,
        "development_qualification_summary": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(summary_path),
        "reserve_selection": str(selection_path.relative_to(REPO_ROOT)),
        "reserve_selection_sha256": selection["manifest_sha256"],
        "reserve_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "reserve_manifest_sha256": manifest["manifest_sha256"],
        "download_receipt_file_sha256": _sha256(receipt_path),
        "runtime_selection": deepcopy(config["runtime_selection"]),
        "execution_calibration": deepcopy(config["execution_calibration"]),
        "acquisition": deepcopy(config["acquisition"]),
        "reserve_gates": deepcopy(config["qualification_gates"]),
        "failure_policy": {
            "reserve": "RUN_ONCE_ON_FROZEN_V16_POOL;NO_POST_RESERVE_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v16_reserve_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config.update({
        "preregistration": str(prereg_path.relative_to(REPO_ROOT)),
        "preregistration_file_sha256": _sha256(prereg_path),
        "expected_grounder_sha256": expected_grounder_sha256,
        "expected_evaluation_protocol_sha256": expected_evaluation_sha256,
    })
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v16_reserve.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": preregistration["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_count": manifest["sample_count"],
        "evaluated_count": config["qualification_gates"]["required_valid_runtime_rows"],
        "grounder_sha256": expected_grounder_sha256,
        "evaluation_protocol_sha256": expected_evaluation_sha256,
        "config_file_sha256": _sha256(config_path),
    }, indent=2))


if __name__ == "__main__":
    main()
