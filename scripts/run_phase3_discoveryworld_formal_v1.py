#!/usr/bin/env python3
"""Execute and aggregate the frozen DiscoveryWorld Phase-3 five-arm reserve."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
DISCOVERYWORLD = REPO.parent / "discoveryworld-official"
if DISCOVERYWORLD.is_dir():
    sys.path.insert(0, str(DISCOVERYWORLD))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.phase3_discoveryworld_formal import (  # noqa: E402
    analyze_formal_results,
)
from motif_transfer.phase3_discoveryworld_transfer import (  # noqa: E402
    Phase3DiscoveryWorldPortfolioSelector,
    call_phase3_binder,
    call_phase3_grounder,
    extract_phase3_acquisition_evidence,
)
import scripts.run_discoveryworld_commit_recovery_v1 as base  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value); claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def run_one(
    *, manifest_path: Path, fork_row: Mapping[str, Any], keys: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest = _read(manifest_path)
    config_path = REPO / str(fork_row["fork_config"])
    if _file_sha256(config_path) != fork_row["fork_config_file_sha256"]:
        raise ValueError("frozen fork config file hash mismatch")
    config = _read(config_path)
    _self_hash(config, "config_sha256")
    _self_hash(config["fork_receipt"], "fork_receipt_sha256")
    if config.get("status") != "FORMAL_RESERVE_FROZEN_STRUCTURAL_FORK":
        raise ValueError("fork config is not frozen for formal reserve")
    if config.get("formal_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("fork/formal manifest mismatch")
    task_id = str(fork_row["task_id"])
    output_path = output_root / task_id / "matched_result.json"
    if output_path.is_file():
        result = _read(output_path); _self_hash(result, "result_sha256")
        return result
    reference = _read(REPO / str(config["reference_episode"]))
    evidence = extract_phase3_acquisition_evidence(
        reference, int(config["fork_after_episode_step"]),
    )
    artifacts = []
    for row in manifest["source_artifacts"]:
        path = REPO / str(row["path"])
        if _file_sha256(path) != row["file_sha256"]:
            raise ValueError("source artifact file hash mismatch")
        artifact = _read(path); _self_hash(artifact, "artifact_sha256")
        artifacts.append(artifact)
    selector = Phase3DiscoveryWorldPortfolioSelector(source_artifacts=artifacts)

    def binder(backend, observation, *, memory, hypotheses, attempts):
        return call_phase3_binder(
            backend, observation, memory=memory, hypotheses=hypotheses,
            attempts=attempts, acquisition_evidence=evidence,
        )

    old_grounder, old_binder, old_selector = (
        base.call_grounder, base.call_binder, base.select_candidate,
    )
    base.call_grounder = call_phase3_grounder
    base.call_binder = binder
    base.select_candidate = selector.select
    old_argv = sys.argv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.argv = [
            str(REPO / "scripts/run_discoveryworld_commit_recovery_v1.py"),
            "--config", str(config_path), "--keys", str(keys),
            "--output", str(output_path),
        ]
        base.main()
    finally:
        sys.argv = old_argv
        base.call_grounder, base.call_binder, base.select_candidate = (
            old_grounder, old_binder, old_selector,
        )
    result = _read(output_path); _self_hash(result, "result_sha256")
    return result


def _worker(manifest, row, keys, output):
    try:
        result = run_one(
            manifest_path=Path(manifest), fork_row=row, keys=Path(keys),
            output_root=Path(output),
        )
        arm_errors = {
            name: arm.get("runtime_error")
            for name, arm in (result.get("conditions") or {}).items()
            if name != "target_only_recorded" and arm.get("runtime_error")
        }
        return {"task_id": row["task_id"], "error": (
            f"matched arm errors: {arm_errors}" if arm_errors else None
        )}
    except BaseException as exc:
        return {"task_id": row["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fork-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-dir", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    manifest = _read(args.manifest); _self_hash(manifest, "manifest_sha256")
    forks = _read(args.fork_manifest); _self_hash(forks, "fork_manifest_sha256")
    if forks.get("formal_manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("fork/formal manifest mismatch")
    for path, expected in manifest["runtime_file_sha256"].items():
        if _file_sha256(REPO / path) != expected:
            raise SystemExit(f"frozen runtime changed: {path}")
    selected = set(args.task_id or ())
    rows = [
        row for row in forks["tasks"]
        if not selected or row["task_id"] in selected
    ]
    if selected and len(rows) != len(selected):
        raise SystemExit("unknown task ID")
    workers = min(
        len(rows), args.workers or int(manifest["matched_runtime"]["task_workers"])
    )
    progress = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _worker, str(args.manifest), dict(row), str(args.keys),
                str(args.output_dir),
            ): row["task_id"] for row in rows
        }
        for future in as_completed(futures):
            row = future.result(); progress.append(row)
            print(json.dumps(row), flush=True)
    progress.sort(key=lambda row: row["task_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "progress.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8",
    )

    all_results = []
    for task in manifest["tasks"]:
        path = args.output_dir / task["task_id"] / "matched_result.json"
        if not path.is_file():
            continue
        result = _read(path); _self_hash(result, "result_sha256")
        all_results.append(result)
    if len(all_results) == manifest["task_count"]:
        acquisition = _read(args.acquisition_dir / "summary.json")
        _self_hash(acquisition, "summary_sha256")
        report = analyze_formal_results(
            manifest=manifest, acquisition_summary=acquisition,
            results=all_results,
        )
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps({
            "status": report["status"],
            "program_aligned_successes": report["program_aligned_successes"],
            "gates": report["gates"],
            "report_sha256": report["report_sha256"],
        }, indent=2))
    if any(row["error"] for row in progress):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
