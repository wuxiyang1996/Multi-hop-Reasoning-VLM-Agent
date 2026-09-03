#!/usr/bin/env python3
"""Execute six fresh WebShop cells under their frozen source lineages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    file_sha256,
    make_cell_execution_receipt,
    read_object,
    validate_manifest,
    validate_self_hash,
)
from motif_transfer.webshop_search_automaton_v16 import (  # noqa: E402
    AUTHENTIC,
    CONDITIONS,
)


def _cell(manifest: dict, game: str) -> dict:
    rows = [
        row for row in manifest["cells"]
        if row["source_game"] == game and row["target_domain"] == "webshop"
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one WebShop cell for {game}")
    return dict(rows[0])


def _wait_server(process: subprocess.Popen, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"WebShop server exited with {process.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:3000/", timeout=3):
                return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError(f"WebShop server did not become ready: {last_error}")


def _run_cell(manifest: dict, game: str, output_root: Path) -> dict:
    cell = _cell(manifest, game)
    output_dir = output_root / game
    wrapper_report = output_dir / "direct_report.json"
    if wrapper_report.is_file():
        existing = read_object(wrapper_report)
        receipt = existing.get("cell_execution_receipt")
        if receipt:
            validate_self_hash(receipt, "cell_receipt_sha256")
            if receipt.get("manifest_sha256") == manifest["manifest_sha256"]:
                return existing
        raise RuntimeError(f"refusing incompatible WebShop resume: {wrapper_report}")

    source_path = REPO / str(cell["source_artifact"])
    goal_manifest = REPO / str(manifest["targets"]["webshop"]["goal_manifest"])
    protocol = REPO / str(cell["domain_protocol"])
    qualification = (
        REPO / "runs/webshop_search_automaton_v16_development_"
        "gpt41mini_anytime/report.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO / "scripts/run_webshop_search_automaton_v16.py"),
        "--task-ids", str(cell["target_task_id"]),
        "--role", "formal_reserve",
        "--goal-manifest", str(goal_manifest),
        "--source-artifact", str(source_path),
        "--output-dir", str(output_dir),
        "--qualification-report", str(qualification),
        "--formal-protocol", str(protocol),
        "--cache-seed", str(output_dir / "no_historical_cache_seed.json"),
        "--model", "qwen/qwen3.5-35b-a3b",
        "--maximum-output-tokens", "3200",
        "--maximum-steps", "12",
        "--run-id", f"phase1-direct-{game}-webshop",
    ]
    log_path = output_dir / "execution.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    report_path = output_dir / "report.json"
    runtime_error = None
    if not report_path.is_file():
        runtime_error = f"underlying runner exit={completed.returncode}; no report"
        underlying = {}
    else:
        underlying = read_object(report_path)
        # Exit 2 means the deliberately underpowered one-task efficacy gate did
        # not pass.  Direct operational gates are evaluated independently.
        failures = []
        task_id = str(cell["target_task_id"])
        for condition in CONDITIONS:
            path = output_dir / f"{task_id}.{condition}.json"
            if not path.is_file():
                failures.append(f"missing {path.name}")
                continue
            row = read_object(path)
            if row.get("failure") is not None:
                failures.append(f"{condition}: {row['failure']}")
        if failures:
            runtime_error = "; ".join(failures)

    condition_rows = {}
    task_id = str(cell["target_task_id"])
    for condition in CONDITIONS:
        path = output_dir / f"{task_id}.{condition}.json"
        if path.is_file():
            condition_rows[condition] = read_object(path)
    authentic_trace = list(
        condition_rows.get(AUTHENTIC, {}).get("v16_controller", {}).get(
            "source_trace", ()
        )
    )
    initial_hashes = [
        str(condition_rows[condition]["initial_state_hash"])
        for condition in CONDITIONS if condition in condition_rows
    ]
    receipt = make_cell_execution_receipt(
        manifest_sha256=str(manifest["manifest_sha256"]),
        cell=cell,
        source_artifact_sha256=str(cell["source_artifact_sha256"]),
        conditions_executed=[
            condition for condition in CONDITIONS if condition in condition_rows
        ],
        expected_conditions=CONDITIONS,
        target_initial_state_hashes=initial_hashes,
        authentic_source_decisions=authentic_trace,
        target_native_grounding_used=bool(condition_rows),
        target_reset_or_sample_open_count=1,
        outcome_was_reused=False,
        runtime_error=runtime_error,
    )
    body = {
        "schema_version": "phase1-direct-webshop-cell-v1",
        "status": receipt["status"],
        "claim_boundary": manifest["claim_boundary"],
        "cell": cell,
        "underlying_runner_exit_code": completed.returncode,
        "underlying_one_task_efficacy_status": underlying.get("status"),
        "underlying_report_file_sha256": (
            file_sha256(report_path) if report_path.is_file() else None
        ),
        "execution_log_file_sha256": file_sha256(log_path),
        "cell_execution_receipt": receipt,
    }
    report = body | {"report_sha256": stable_hash(body)}
    wrapper_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase1_direct_prospective_v1/manifest.json",
    )
    parser.add_argument("--source-game", choices=SOURCE_GAMES, action="append")
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "runs/phase1_direct_prospective_v1/webshop",
    )
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    validate_manifest(manifest, repo=REPO)
    games = tuple(args.source_game or SOURCE_GAMES)
    server_log_path = args.output_root / "server.log"
    args.output_root.mkdir(parents=True, exist_ok=True)
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            [
                sys.executable,
                str(REPO / "scripts/run_webshop_direct_server_v1.py"),
                "--goal-seed",
                str(manifest["targets"]["webshop"]["goal_seed"]),
            ],
            cwd=REPO,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_server(server)
            reports = [
                _run_cell(manifest, game, args.output_root) for game in games
            ]
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=20)
    passed = sum(
        report["cell_execution_receipt"]["status"]
        == "DIRECT_PROSPECTIVE_CELL_PASSED"
        for report in reports
    )
    print(json.dumps({
        "domain": "webshop", "passed": passed, "attempted": len(reports)
    }, indent=2))
    return 0 if passed == len(reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
