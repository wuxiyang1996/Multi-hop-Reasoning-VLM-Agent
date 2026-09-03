#!/usr/bin/env python3
"""Freeze V13 confirmation after OR-to-CHOOSE parser requalification."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from motif_transfer.contracts import stable_hash  # noqa: E402
from scripts.freeze_agqa2_active_grounding_v4 import _sha256, _verified_json  # noqa: E402
import scripts.freeze_agqa2_active_grounding_v12_reserve as v12  # noqa: E402


NONCE = "agqa2-v13-or-to-choose-typed-arity-fresh-confirmation"


def _selection(development: dict, exposed: list[dict], excluded: set[str]) -> dict:
    v12.NONCE = NONCE
    inherited = v12._select(development, exposed, excluded)
    core = dict(inherited)
    core.pop("manifest_sha256")
    core.update({
        "schema_version": "agqa2-active-grounding-selection-v13",
        "status": "FROZEN_V13_SELECTION_BEFORE_VIDEO_DOWNLOAD_OR_V13_CALLS",
        "claim_boundary": (
            "V13_PARSER_REQUALIFIED;V11_V12_RESERVES_ABORTED_BEFORE_CALL;12_"
            "NEW_VIDEO_DISJOINT_TYPED_ARITY_CLOSED_CANDIDATES"
        ),
        "selection_nonce": NONCE,
        "selection_rule": (
            "EXCLUDE_ALL_PRIOR_MANIFEST_AND_ABORTED_V11_V12_SELECTION_VIDEOS;"
            "REQUIRE_RELATION_INDEPENDENT_OR_TO_CHOOSE_AND_SOURCE_OPERATOR_ARITY_"
            "CLOSURE;FOUR_FIXED_HASH_RANKED_CANDIDATES_PER_ROUTE;NO_OUTCOMES"
        ),
        "aborted_v11_selection_sha256": (
            "79cc37bd78b1acbbb49f5057ea0c0be918a67829b66da3472e715ee07633ac2f"
        ),
        "aborted_v12_selection_sha256": (
            "ad57aa3ec04b79f5b7e59b116d4eeb74461d812042f915773dd5083ba477dcb4"
        ),
        "prior_v12_raw_video_exposure": False,
        "prior_v13_raw_video_exposure": False,
    })
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
        "schema_version": "agqa2-active-grounding-manifest-v13",
        "status": "FROZEN_V13_RAW_VIDEO_UNSEEN_BEFORE_V13_NEURAL_CALLS",
        "samples": samples,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "new_video_downloads": sum(
            not row["video_present_at_selection"] for row in samples
        ),
    })
    return core | {"manifest_sha256": stable_hash(core)}


def main() -> None:
    run_root = REPO_ROOT / "runs/agqa2_active_grounding_v13_reserve"
    if run_root.exists() and any(run_root.rglob("*.json")):
        raise RuntimeError("V13 reserve is already consumed")
    dev_report_path = REPO_ROOT / "runs/agqa2_active_grounding_v13_development/report.json"
    dev_report = json.loads(dev_report_path.read_text())
    if not dev_report.get("grounder_qualified"):
        raise ValueError("V13 development did not qualify")
    summary_core = {
        key: deepcopy(dev_report[key]) for key in (
            "status", "grounder_qualified", "grounder_sha256", "metrics",
            "controls", "qualification_gates", "reported_provider_cost_usd",
            "report_sha256",
        )
    }
    summary_core["schema_version"] = "agqa2-active-grounding-v13-development-summary"
    summary = summary_core | {"summary_sha256": stable_hash(summary_core)}
    summary_path = REPO_ROOT / "docs/results/agqa2_active_grounding_v13_development_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    manifest_paths = sorted(
        path for path in (REPO_ROOT / "configs").glob(
            "agqa2_active_grounding_v*_manifest.json"
        )
        if "v13_reserve" not in path.name
    )
    exposed = [_verified_json(path, "manifest_sha256") for path in manifest_paths]
    excluded = set()
    for version in ("v11", "v12"):
        prior = _verified_json(
            REPO_ROOT / f"configs/agqa2_active_grounding_{version}_reserve_selection.json",
            "manifest_sha256",
        )
        excluded.update(str(row["video_id"]) for row in prior["samples"])
    development = _verified_json(
        REPO_ROOT / "configs/agqa2_active_grounding_v13_development_manifest.json",
        "manifest_sha256",
    )
    selection_path = REPO_ROOT / "configs/agqa2_active_grounding_v13_reserve_selection.json"
    selection = (
        _verified_json(selection_path, "manifest_sha256")
        if selection_path.is_file()
        else _selection(development, exposed, excluded)
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
    reserve_path = REPO_ROOT / "configs/agqa2_active_grounding_v13_reserve_manifest.json"
    reserve_path.write_text(json.dumps(reserve, indent=2, sort_keys=True) + "\n")
    development_config = json.loads((
        REPO_ROOT / "configs/agqa2_active_grounding_v13_development.json"
    ).read_text())
    preregistration = {
        "schema_version": "agqa2-active-grounding-preregistration-v13-reserve",
        "status": "FROZEN_BEFORE_ANY_V13_RESERVE_NEURAL_CALL",
        "claim_boundary": reserve["claim_boundary"],
        "development_qualification_summary": str(summary_path.relative_to(REPO_ROOT)),
        "development_qualification_summary_file_sha256": _sha256(summary_path),
        "reserve_selection": str(selection_path.relative_to(REPO_ROOT)),
        "reserve_selection_sha256": selection["manifest_sha256"],
        "reserve_manifest": str(reserve_path.relative_to(REPO_ROOT)),
        "reserve_manifest_sha256": reserve["manifest_sha256"],
        "applicability_rule": "V13_OR_TO_CHOOSE_PLUS_SOURCE_OPERATOR_ARITY_CLOSURE",
        "execution_calibration": deepcopy(development_config["execution_calibration"]),
        "runtime_selection": deepcopy(development_config["runtime_selection"]),
        "acquisition": deepcopy(development_config["acquisition"]),
        "reserve_gates": deepcopy(development_config["qualification_gates"]),
        "failure_policy": {
            "reserve": "RUN_ONCE_ON_FROZEN_V13_POOL;NO_POST_RESERVE_TUNING",
            "qualification": "FAIL_CLOSED_TO_MATCHED_DIRECT_BASELINE",
        },
    }
    prereg_path = REPO_ROOT / "configs/agqa2_active_grounding_v13_reserve_preregistration.json"
    prereg_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True) + "\n")
    config = deepcopy(development_config)
    config.update({
        "schema_version": "agqa2-active-grounding-reserve-config-v13",
        "status": "FROZEN_V13_OR_TO_CHOOSE_TYPED_ARITY_RESERVE",
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
        "report_version": "V13",
    })
    config_path = REPO_ROOT / "configs/agqa2_active_grounding_v13_reserve.json"
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
