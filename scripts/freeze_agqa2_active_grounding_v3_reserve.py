#!/usr/bin/env python3
"""Freeze the qualified AGQA V3 grounder before its one reserve run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def freeze_reserve(
    *, development_config_path: Path, development_report_path: Path,
    summary_path: Path, reserve_config_path: Path,
) -> tuple[dict, dict]:
    development = json.loads(development_config_path.read_text())
    report = json.loads(development_report_path.read_text())
    report_body = dict(report)
    report_sha256 = report_body.pop("report_sha256")
    if stable_hash(report_body) != report_sha256:
        raise ValueError("AGQA V3 development report content hash mismatch")
    if report.get("split") != "development":
        raise ValueError("AGQA V3 qualification dependency is not development")
    if not report.get("grounder_qualified"):
        raise ValueError("AGQA V3 development grounder is not qualified")
    if not all(report.get("qualification_gates", {}).values()):
        raise ValueError("AGQA V3 development did not pass every gate")
    version_match = re.search(r"v(\d+)$", development["schema_version"])
    version = f"v{version_match.group(1)}" if version_match else "v3"

    summary_core = {
        "schema_version": f"agqa2-active-grounding-development-qualification-{version}",
        "status": report["status"],
        "split": "development",
        "grounder_sha256": report["grounder_sha256"],
        "grounder_qualified": True,
        "report_path": _portable_path(development_report_path),
        "report_file_sha256": _sha256(development_report_path),
        "report_sha256": report_sha256,
        "manifest_sha256": report["manifest_sha256"],
        "metrics": report["metrics"],
        "controls": report["controls"],
        "qualification_gates": report["qualification_gates"],
        "provider_calls": report["provider_calls"],
        "reported_provider_cost_usd": report["reported_provider_cost_usd"],
        "reserve_was_read_or_called_before_this_freeze": False,
    }
    summary = summary_core | {"summary_sha256": stable_hash(summary_core)}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    preregistration = json.loads(
        (REPO_ROOT / development["preregistration"]).read_text()
    )
    reserve_manifest_path = REPO_ROOT / preregistration["reserve_manifest"]
    reserve = deepcopy(development)
    reserve.update({
        "schema_version": f"agqa2-active-grounding-reserve-config-{version}",
        "status": (
            f"FROZEN_{version.upper()}_AFTER_DEVELOPMENT_QUALIFICATION_"
            "BEFORE_RESERVE_CALLS"
        ),
        "split": "reserve",
        "claim_boundary": (
            f"SINGLE_RUN_RAW_VIDEO_UNSEEN_VIDEO_DISJOINT_{version.upper()}_RESERVE;"
            "AGQA_TEST_METADATA_PREVIOUSLY_SCANNED;NO_UNTOUCHED_BENCHMARK_"
            "OR_SOURCE_PROVENANCE_CLAIM"
        ),
        "manifest": str(reserve_manifest_path.relative_to(REPO_ROOT)),
        "manifest_file_sha256": _sha256(reserve_manifest_path),
        "qualification_gates": preregistration["reserve_gates"],
        "development_qualification_report": _portable_path(summary_path),
        "development_qualification_file_sha256": _sha256(summary_path),
    })
    reserve_manifest = json.loads(reserve_manifest_path.read_text())
    reserve["expected_manifest_status"] = reserve_manifest["status"]
    reserve_config_path.parent.mkdir(parents=True, exist_ok=True)
    reserve_config_path.write_text(
        json.dumps(reserve, indent=2, sort_keys=True) + "\n"
    )
    return summary, reserve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-config", type=Path,
        default=REPO_ROOT / "configs/agqa2_active_grounding_v3_development.json",
    )
    parser.add_argument(
        "--development-report", type=Path,
        default=REPO_ROOT / "runs/agqa2_active_grounding_v3_development/report.json",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=REPO_ROOT / "docs/results/agqa2_active_grounding_v3_development_summary.json",
    )
    parser.add_argument(
        "--reserve-config", type=Path,
        default=REPO_ROOT / "configs/agqa2_active_grounding_v3_reserve.json",
    )
    args = parser.parse_args()
    version_match = re.search(r"v(\d+)", args.reserve_config.stem)
    version = f"v{version_match.group(1)}" if version_match else "v3"
    reserve_cache = (
        REPO_ROOT / f"runs/agqa2_active_grounding_{version}_reserve/call_cache"
    )
    if reserve_cache.exists() and any(reserve_cache.rglob("*.json")):
        raise RuntimeError("reserve call cache is nonempty; refusing to refreeze")
    summary, reserve = freeze_reserve(
        development_config_path=args.development_config.resolve(),
        development_report_path=args.development_report.resolve(),
        summary_path=args.summary.resolve(),
        reserve_config_path=args.reserve_config.resolve(),
    )
    print(json.dumps({
        "development_report_sha256": summary["report_sha256"],
        "development_summary_sha256": summary["summary_sha256"],
        "grounder_sha256": summary["grounder_sha256"],
        "reserve_config_sha256": _sha256(args.reserve_config.resolve()),
        "reserve_manifest": reserve["manifest"],
        "reserve_status": reserve["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
