#!/usr/bin/env python3
"""Run the one-shot 32-goal, six-source Phase-2 WebShop utility matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from typing import Any, Mapping
from urllib import request


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.phase2_webshop_utility_v1 import (  # noqa: E402
    PASSED_STATUS,
    build_report,
    file_sha256,
    validate_manifest,
    validate_self_hash,
)
from motif_transfer.search_automaton_transfer_v16 import (  # noqa: E402
    SourceSearchAutomaton,
)
from motif_transfer.webshop_constraint_coverage_v14 import (  # noqa: E402
    audit_receipt_commits,
)
from motif_transfer.webshop_neural_symbolic_v9 import (  # noqa: E402
    TargetOutcomeMLP,
)
from motif_transfer.webshop_search_automaton_v16 import (  # noqa: E402
    AUTHENTIC,
    CEILING,
    CONDITIONS,
    LEDGER_BLIND,
    WebShopSearchAutomatonController,
)
import scripts.run_webshop_neural_symbolic_v9 as v9_runner  # noqa: E402
from scripts.run_webshop_search_automaton_v16 import (  # noqa: E402
    _candidate_augmenter,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _wait_server(process: subprocess.Popen[Any], timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"WebShop server exited with {process.returncode}")
        try:
            with request.urlopen("http://127.0.0.1:3000/", timeout=3):
                return
        except Exception as exc:  # pragma: no cover - live server timing
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError(f"WebShop server did not become ready: {last_error}")


def _cache_usage(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"unique_provider_completions": 0}
    cache = _read(path)
    entries = list((cache.get("entries") or {}).values())
    output: dict[str, Any] = {"unique_provider_completions": len(entries)}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        values = [
            row.get("usage", {}).get(key)
            for row in entries
            if isinstance(row.get("usage", {}).get(key), (int, float))
        ]
        if values:
            output[key] = sum(values)
    output["decision_cache_file_sha256"] = file_sha256(path)
    return output


def _receipt_path(output_dir: Path, index: int, condition: str) -> Path:
    return output_dir / "receipts" / f"cell_{index:02d}.{condition}.json"


def _started_path(output_dir: Path, index: int, condition: str) -> Path:
    return output_dir / "started" / f"cell_{index:02d}.{condition}.json"


def _validate_existing_receipt(
    receipt: Mapping[str, Any], *, manifest: Mapping[str, Any], task: Mapping[str, Any],
    condition: str,
) -> None:
    validate_self_hash(receipt, "receipt_sha256")
    expected = {
        "manifest_sha256": manifest["manifest_sha256"],
        "target_identity": task["target_identity"],
        "task_id": task["task_id"],
        "condition": condition,
        "source_game": task["source_game"],
        "source_artifact_sha256": task["source_artifact_sha256"],
    }
    mismatches = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items() if receipt.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible Phase-2 receipt: {mismatches}")


def _run(manifest: dict[str, Any], *, keys: Path, wrapper_root: Path, output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "report.json"
    if report_path.is_file():
        report = _read(report_path)
        validate_self_hash(report, "report_sha256")
        if report.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("refusing incompatible completed Phase-2 report")
        return report

    values = runpy.run_path(str(keys))
    api_key = values.get("OPENROUTER_API_KEY") or values.get("openrouter_api_key")
    if not api_key:
        raise SystemExit("OpenRouter API key is missing")
    os.environ["PHASE2_WEBSHOP_OPENROUTER_KEY"] = str(api_key)
    parameters = dict(manifest["parameters"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipts").mkdir(parents=True, exist_ok=True)
    (output_dir / "started").mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "decision_cache.json"
    backend = MemoizedCompletionBackend(
        OpenAICompatibleBackend(
            str(parameters["base_url"]),
            {"decision": str(parameters["model"])},
            api_key_env="PHASE2_WEBSHOP_OPENROUTER_KEY",
            json_mode=True,
            temperature=0,
            timeout_seconds=int(parameters["timeout_seconds"]),
            request_overrides={
                "max_tokens": int(parameters["maximum_output_tokens"]),
            },
        ),
        cache_path=cache_path,
    )
    target_artifact_path = REPO / str(manifest["target_grounder"])
    target_artifact = _read(target_artifact_path)
    if not target_artifact.get("preflight_passed"):
        raise RuntimeError("frozen target neural grounder did not pass preflight")
    grounder = TargetOutcomeMLP.from_dict(target_artifact["grounder"])
    sources = {
        str(game): SourceSearchAutomaton(_read(REPO / str(row["artifact"])))
        for game, row in manifest["sources"].items()
    }
    receipts: list[dict[str, Any]] = []
    for index, task in enumerate(manifest["tasks"]):
        source_game = str(task["source_game"])
        source = sources[source_game]
        goal_options = dict(task.get("goal", {}).get("goal_options", {}))
        for condition in CONDITIONS:
            destination = _receipt_path(output_dir, index, condition)
            started = _started_path(output_dir, index, condition)
            if destination.is_file():
                receipt = _read(destination)
                _validate_existing_receipt(
                    receipt, manifest=manifest, task=task, condition=condition,
                )
                receipts.append(receipt)
                continue
            if started.exists():
                raise RuntimeError(
                    "formal cell was started but has no receipt; refusing outcome-aware rerun: "
                    f"{task['target_identity']} {condition}"
                )
            marker_body = {
                "schema_version": "phase2-webshop-cell-start-v1",
                "manifest_sha256": manifest["manifest_sha256"],
                "target_identity": task["target_identity"],
                "condition": condition,
                "one_shot_no_rerun": True,
            }
            marker = marker_body | {"marker_sha256": stable_hash(marker_body)}
            started.write_text(
                json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            controller = WebShopSearchAutomatonController(
                condition=condition,
                source=source,
                episode_id=str(task["target_identity"]),
                goal_options=goal_options,
                maximum_steps=int(parameters["maximum_steps"]),
            )
            v9_runner._decision_candidates = _candidate_augmenter(goal_options)
            v9_runner.choose_transfer_action = controller
            receipt = v9_runner._run_condition(
                task_id=str(task["task_id"]),
                condition=condition,
                backend=backend,
                grounder=grounder,
                source_models={"artifact": {}},
                source_policy={"uncertainty_scale": 0.0, "decision_margin": 0.0},
                expected_goal=str(task["instruction_text"]),
                wrapper_root=wrapper_root,
                session_namespace=(
                    f"phase2-ws-{manifest['manifest_sha256'][:10]}-c{index:02d}-"
                    f"{condition.replace('_', '-')}"
                ),
                number_of_goals=int(manifest["number_of_registered_tasks_required"]),
                maximum_steps=int(parameters["maximum_steps"]),
                candidate_count=int(parameters["candidate_count"]),
                schema_retries=int(parameters["schema_retries"]),
            )
            receipt["manifest_sha256"] = manifest["manifest_sha256"]
            receipt["target_identity"] = task["target_identity"]
            receipt["goal_sha256"] = task["goal_sha256"]
            receipt["source_game"] = source_game
            receipt["source_artifact_sha256"] = source.artifact_sha256
            receipt["v16_controller"] = controller.as_dict()
            receipt["commit_audit"] = audit_receipt_commits(receipt)
            receipt["unsafe_commits"] = [
                row for row in receipt["commit_audit"]
                if goal_options
                and not row["authorized"]
                and not bool(receipt["steps"][int(row["step"])]["source_abstained"])
            ] if condition in {AUTHENTIC, LEDGER_BLIND, CEILING} else []
            receipt["historical_target_outcome_reused"] = False
            receipt["target_reset_or_sample_open_count"] = 1
            receipt["runtime_hashes"] = {
                "runner": file_sha256(Path(__file__)),
                "controller": file_sha256(
                    REPO / "src/motif_transfer/webshop_search_automaton_v16.py"
                ),
                "utility_contract": file_sha256(
                    REPO / "src/motif_transfer/phase2_webshop_utility_v1.py"
                ),
                "source_artifact": file_sha256(REPO / str(task["source_artifact"])),
                "target_grounder": file_sha256(target_artifact_path),
            }
            receipt["receipt_sha256"] = stable_hash(receipt)
            destination.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            receipts.append(receipt)
            print(json.dumps({
                "cell": index,
                "target_identity": task["target_identity"],
                "source_game": source_game,
                "condition": condition,
                "strict": receipt["strict_success"],
                "reward": receipt["official_reward"],
                "steps": receipt["step_count"],
                "source_actions": controller.as_dict()["source_action_counts"],
                "failure": receipt["failure"],
            }), flush=True)
    report = build_report(
        manifest, receipts, cache_usage=_cache_usage(cache_path),
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "configs/phase2_webshop_utility_v1/manifest.json",
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/keys.py"),
    )
    parser.add_argument(
        "--wrapper-root", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "runs/phase2_webshop_utility_v1",
    )
    args = parser.parse_args()
    manifest = _read(args.manifest)
    validate_manifest(manifest, repo=REPO)
    server_log_path = args.output_dir / "server.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with server_log_path.open("a", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            [
                sys.executable,
                str(REPO / "scripts/run_webshop_direct_server_v1.py"),
                "--goal-seed", str(manifest["goal_seed"]),
            ],
            cwd=REPO, stdout=server_log, stderr=subprocess.STDOUT,
        )
        try:
            _wait_server(server)
            report = _run(
                manifest, keys=args.keys, wrapper_root=args.wrapper_root,
                output_dir=args.output_dir,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:  # pragma: no cover - live server cleanup
                server.kill()
                server.wait(timeout=20)
    print(json.dumps({
        "status": report["status"],
        "strict_successes": report["summaries"][AUTHENTIC]["strict_successes"],
        "raw_strict_successes": report["summaries"]["raw_target_only"]["strict_successes"],
        "gates_passed": sum(bool(value) for value in report["gates"].values()),
        "gates_required": len(report["gates"]),
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if report["status"] == PASSED_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
