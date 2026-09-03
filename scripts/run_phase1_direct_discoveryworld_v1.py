#!/usr/bin/env python3
"""Execute direct online source routing on fresh DiscoveryWorld forks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.direct_prospective_matrix_v1 import (  # noqa: E402
    SOURCE_GAMES,
    file_sha256,
    make_cell_execution_receipt,
    read_object,
    validate_manifest,
    validate_self_hash,
)
from motif_transfer.discoveryworld_search_automaton_v16 import (  # noqa: E402
    AUTHENTIC,
)
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
    bind_native_action,
    ground_target_event,
)
from motif_transfer.sokoban_search_automaton_v16 import (  # noqa: E402
    COMMIT,
    EXPLORE,
)


def _cell(manifest: dict, game: str) -> dict:
    rows = [
        row for row in manifest["cells"]
        if row["source_game"] == game
        and row["target_domain"] == "discoveryworld"
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one DiscoveryWorld cell for {game}")
    return dict(rows[0])


def _run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    return int(completed.returncode)


def _prepare_target_forks(
    *, manifest: dict, keys: Path, output_root: Path
) -> tuple[Path, Path]:
    target_dir = output_root / "target_only"
    fork_dir = output_root / "frozen_forks"
    target_summary = target_dir / "summary.json"
    target_config = REPO / str(
        manifest["targets"]["discoveryworld"]["target_config"]
    )
    if not target_summary.is_file():
        code = _run_command(
            [
                sys.executable,
                str(REPO / "scripts/run_discoveryworld_target_only_v1.py"),
                "--config", str(target_config),
                "--keys", str(keys),
                "--output-dir", str(target_dir),
                "--role", "formal_reserve",
            ],
            output_root / "target_only.log",
        )
        if code != 0 or not target_summary.is_file():
            raise RuntimeError(f"DiscoveryWorld target-only collection failed: exit={code}")
    if not (fork_dir / "fork_freeze_receipt.json").is_file():
        protocol = REPO / str(
            manifest["targets"]["discoveryworld"]["protocol"]
        )
        code = _run_command(
            [
                sys.executable,
                str(REPO / "scripts/freeze_discoveryworld_qualification_forks_v1.py"),
                "--protocol", str(protocol),
                "--baseline-dir", str(target_dir),
                "--output-dir", str(fork_dir),
            ],
            output_root / "fork_freeze.log",
        )
        if code != 0 or not (fork_dir / "fork_freeze_receipt.json").is_file():
            raise RuntimeError(f"DiscoveryWorld fork freeze failed: exit={code}")
    return target_dir, fork_dir


def _run_matched_with_online_source(
    *,
    config_path: Path,
    keys: Path,
    output_path: Path,
    source: SourceSearchAutomaton,
    task_id: str,
) -> tuple[dict, list[dict[str, Any]], str | None]:
    """Patch only the call boundary so source routing occurs before env.step."""

    import scripts.run_discoveryworld_commit_recovery_v1 as runner

    original_select = runner.select_candidate
    original_realize = runner.realize_localized_spatial_position
    pending: dict[str, Any] = {}
    routes: list[dict[str, Any]] = []

    def select_wrapper(condition, candidates, observation, **kwargs):
        selected, selection = original_select(
            condition, candidates, observation, **kwargs
        )
        pending.clear()
        pending.update({
            "condition": str(condition),
            "selected": selected,
            "selection": selection,
            "observation": observation,
        })
        return selected, selection

    def realize_wrapper(selected, observation, target_binding, **kwargs):
        realized_action, realization = original_realize(
            selected, observation, target_binding, **kwargs
        )
        if pending.get("condition") == AUTHENTIC:
            selection = pending["selection"]
            role = str(selected.target_role)
            if role == "COMMIT":
                if not bool(selection.positive_commit_effect_witnessed):
                    raise RuntimeError(
                        "fail closed: authentic DiscoveryWorld proposed an "
                        "unverified COMMIT"
                    )
                event_name = "VERIFIED"
                abstract_action = COMMIT
                evidence_kind = "target_positive_commit_effect_witness"
            elif role == "POSITION":
                event_name = "UNBOUND"
                abstract_action = EXPLORE
                evidence_kind = "target_neural_untried_position_candidate"
            else:
                raise RuntimeError(f"unsupported target role: {role}")
            event = ground_target_event(
                domain="discoveryworld",
                episode_id=task_id,
                decision_index=len(routes),
                untried_candidate_available=event_name == "UNBOUND",
                active_candidate_refuted=False,
                terminal_commit_verified=event_name == "VERIFIED",
                evidence_kind=evidence_kind,
                evidence_payload={
                    "selection_receipt_sha256": selection.receipt_sha256,
                    "selected_candidate_sha256": selected.candidate_sha256,
                    "target_policy_state_sha256": observation.policy_state_sha256,
                    "target_realization_receipt_sha256": realization[
                        "receipt_sha256"
                    ],
                },
                grounding_confidence=1.0,
            )
            if event is None:
                raise RuntimeError("DiscoveryWorld target event abstained")
            binding = bind_native_action(
                event,
                abstract_action=abstract_action,
                native_action_id=stable_hash(realized_action),
                native_action=realized_action,
                grounding_confidence=1.0,
            )
            routed = source.route(event, {abstract_action: binding})
            if not routed.admitted or routed.native_action != realized_action:
                raise RuntimeError("source automaton rejected target-native realization")
            routes.append(asdict(routed))
        return realized_action, realization

    runner.select_candidate = select_wrapper
    runner.realize_localized_spatial_position = realize_wrapper
    runtime_error = None
    try:
        old_argv = sys.argv
        sys.argv = [
            str(REPO / "scripts/run_discoveryworld_commit_recovery_v1.py"),
            "--config", str(config_path),
            "--keys", str(keys),
            "--output", str(output_path),
        ]
        try:
            runner.main()
        finally:
            sys.argv = old_argv
    except BaseException as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        runner.select_candidate = original_select
        runner.realize_localized_spatial_position = original_realize
    result = read_object(output_path) if output_path.is_file() else {}
    return result, routes, runtime_error


def _failed_report(
    *, manifest: dict, cell: dict, source: SourceSearchAutomaton,
    output_path: Path, error: str,
) -> dict:
    receipt = make_cell_execution_receipt(
        manifest_sha256=str(manifest["manifest_sha256"]),
        cell=cell,
        source_artifact_sha256=source.artifact_sha256,
        conditions_executed=[],
        expected_conditions=manifest["conditions"]["discoveryworld"],
        target_initial_state_hashes=[],
        authentic_source_decisions=[],
        target_native_grounding_used=False,
        target_reset_or_sample_open_count=1,
        outcome_was_reused=False,
        runtime_error=error,
    )
    body = {
        "schema_version": "phase1-direct-discoveryworld-cell-v1",
        "status": receipt["status"],
        "cell": cell,
        "cell_execution_receipt": receipt,
    }
    report = body | {"report_sha256": stable_hash(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _run_cell(
    *, manifest: dict, game: str, keys: Path, output_root: Path, fork_dir: Path
) -> dict:
    cell = _cell(manifest, game)
    cell_dir = output_root / "cells" / game
    report_path = cell_dir / "direct_report.json"
    if report_path.is_file():
        existing = read_object(report_path)
        receipt = existing.get("cell_execution_receipt")
        if receipt:
            validate_self_hash(receipt, "cell_receipt_sha256")
            if receipt.get("manifest_sha256") == manifest["manifest_sha256"]:
                return existing
        raise RuntimeError(f"refusing incompatible DiscoveryWorld resume: {report_path}")
    source_path = REPO / str(cell["source_artifact"])
    source = SourceSearchAutomaton(
        read_object(source_path),
        expected_sha256=str(cell["source_artifact_sha256"]),
    )
    config_path = fork_dir / f"{cell['target_task_id']}.json"
    if not config_path.is_file():
        return _failed_report(
            manifest=manifest, cell=cell, source=source,
            output_path=report_path,
            error="PREDECLARED_TARGET_COMMIT_FORK_NOT_ELIGIBLE",
        )
    matched_path = cell_dir / "matched_result.json"
    cell_dir.mkdir(parents=True, exist_ok=True)
    result, routes, runtime_error = _run_matched_with_online_source(
        config_path=config_path,
        keys=keys,
        output_path=matched_path,
        source=source,
        task_id=str(cell["target_task_id"]),
    )
    expected_conditions = tuple(manifest["conditions"]["discoveryworld"])
    executed = [name for name in expected_conditions if name in result.get("conditions", {})]
    initial_hashes = [
        str(result["conditions"][name]["matched_fork_policy_state_sha256"])
        for name in executed
    ]
    if result and not result.get("all_matched_forks"):
        runtime_error = runtime_error or "underlying matched fork gate failed"
    receipt = make_cell_execution_receipt(
        manifest_sha256=str(manifest["manifest_sha256"]),
        cell=cell,
        source_artifact_sha256=source.artifact_sha256,
        conditions_executed=executed,
        expected_conditions=expected_conditions,
        target_initial_state_hashes=initial_hashes,
        authentic_source_decisions=routes,
        target_native_grounding_used=bool(result.get("target_binding")),
        target_reset_or_sample_open_count=1,
        outcome_was_reused=False,
        runtime_error=runtime_error,
    )
    body = {
        "schema_version": "phase1-direct-discoveryworld-cell-v1",
        "status": receipt["status"],
        "claim_boundary": manifest["claim_boundary"],
        "cell": cell,
        "source_artifact_file_sha256": file_sha256(source_path),
        "fork_config_file_sha256": file_sha256(config_path),
        "matched_result_file_sha256": (
            file_sha256(matched_path) if matched_path.is_file() else None
        ),
        "online_route_receipts": routes,
        "cell_execution_receipt": receipt,
    }
    report = body | {"report_sha256": stable_hash(body)}
    report_path.write_text(
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
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument("--source-game", choices=SOURCE_GAMES, action="append")
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "runs/phase1_direct_prospective_v1/discoveryworld",
    )
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    validate_manifest(manifest, repo=REPO)
    _, fork_dir = _prepare_target_forks(
        manifest=manifest, keys=args.keys, output_root=args.output_root
    )
    games = tuple(args.source_game or SOURCE_GAMES)
    reports = [
        _run_cell(
            manifest=manifest, game=game, keys=args.keys,
            output_root=args.output_root, fork_dir=fork_dir,
        )
        for game in games
    ]
    passed = sum(
        report["cell_execution_receipt"]["status"]
        == "DIRECT_PROSPECTIVE_CELL_PASSED"
        for report in reports
    )
    print(json.dumps({
        "domain": "discoveryworld", "passed": passed,
        "attempted": len(reports),
    }, indent=2))
    return 0 if passed == len(reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
