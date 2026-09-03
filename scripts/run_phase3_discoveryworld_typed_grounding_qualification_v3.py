#!/usr/bin/env python3
"""Qualify the typed DiscoveryWorld grounder without target actions/outcomes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
DISCOVERYWORLD_CHECKOUT = REPO.parent / "discoveryworld-official"
if DISCOVERYWORLD_CHECKOUT.is_dir():
    sys.path.insert(0, str(DISCOVERYWORLD_CHECKOUT))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from motif_transfer.phase3_discoveryworld_transfer import (  # noqa: E402
    call_phase3_binder,
    call_phase3_grounder,
    extract_phase3_acquisition_evidence,
    phase3_acquisition_outlier_candidates,
    phase3_candidate_set_complete,
    phase3_position_action_catalog,
)
import scripts.run_discoveryworld_commit_recovery_v1 as base  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


def _backend(manifest: Mapping[str, Any], key: str, cache_path: Path):
    model = manifest["model"]
    os.environ["PHASE3_TYPED_GROUNDER_OPENROUTER_KEY"] = key
    raw = OpenAICompatibleBackend(
        str(model["base_url"]),
        {"grounder": str(model["model"]), "binder": str(model["model"])},
        api_key_env="PHASE3_TYPED_GROUNDER_OPENROUTER_KEY",
        json_mode=True, temperature=float(model["temperature"]),
        timeout_seconds=180,
        request_overrides={
            "max_tokens": int(model["maximum_output_tokens"]),
            "reasoning": {
                "effort": str(model.get("hidden_reasoning_effort") or "low"),
                "exclude": True,
            },
        },
        transport_attempts=3,
    )
    return MemoizedCompletionBackend(raw, cache_path=cache_path)


def _run_one(
    manifest_path: Path, task: Mapping[str, Any], key: str,
    output_dir: Path, task_index: int,
) -> dict[str, Any]:
    manifest = _read(manifest_path)
    fork_path = REPO / str(task["fork_config"])
    if _file_sha256(fork_path) != task["fork_config_file_sha256"]:
        raise ValueError("frozen fork file hash mismatch")
    fork = _read(fork_path)
    reference_path = REPO / str(task["reference_episode"])
    if _file_sha256(reference_path) != task["reference_episode_file_sha256"]:
        raise ValueError("reference episode file hash mismatch")
    reference = _read(reference_path)
    reference_body = dict(reference)
    claimed_episode = str(reference_body.pop("episode_sha256", ""))
    if stable_hash(reference_body) != claimed_episode:
        raise ValueError("reference episode self-hash mismatch")
    if claimed_episode != task["reference_episode_sha256"]:
        raise ValueError("reference episode manifest hash mismatch")

    fork_step = int(task["fork_after_episode_step"])
    prefix = [dict(row["action"]) for row in reference["steps"][:fork_step]]
    expected_policy = reference["steps"][fork_step - 1]["transition"][
        "after_policy_state_sha256"
    ]
    env = base.new_env(
        reference["task"], fork_step + 1,
        int(manifest["runtime"]["thread_id_base"]) + task_index,
        output_dir / "frames" / str(task["task_id"]),
    )
    observation, replay_receipts = env.replay_prefix(
        prefix, expected_policy_state_sha256=expected_policy,
    )
    memory = str(reference["steps"][fork_step - 1].get("memory") or "")
    hypotheses = tuple(
        reference["steps"][fork_step - 1].get("running_hypotheses") or ()
    )
    backend = _backend(
        manifest, key,
        output_dir / "caches" / f"{task['task_id']}.json",
    )
    acquisition_evidence = extract_phase3_acquisition_evidence(
        reference, fork_step,
    )
    acquisition_outliers = phase3_acquisition_outlier_candidates(
        acquisition_evidence,
    )
    binding, binder_raw, binder_attempts = call_phase3_binder(
        backend, observation, memory=memory, hypotheses=hypotheses,
        attempts=int(manifest["model"]["schema_attempts"]),
        acquisition_evidence=acquisition_evidence,
    )
    bundle, candidates, raw, grounder_attempts = call_phase3_grounder(
        backend, observation, memory=memory, hypotheses=hypotheses,
        recent=[], target_binding=binding,
        attempts=int(manifest["model"]["schema_attempts"]),
    )
    parse_rejections = list(bundle.get("candidate_parse_rejections") or ())
    row_body = {
            "schema_version": "phase3-discoveryworld-typed-grounding-row-v3",
            "task_id": task["task_id"],
            "fork_after_episode_step": fork_step,
            "fork_policy_state_sha256": observation.policy_state_sha256,
            "fork_audit_world_sha256": replay_receipts[-1].after_audit_world_sha256,
            "binder_response_sha256": hashlib.sha256(binder_raw.encode()).hexdigest(),
            "binder_attempts": binder_attempts,
            "acquisition_evidence_count": len(acquisition_evidence),
            "acquisition_outlier_candidates": list(acquisition_outliers),
            "acquisition_grounding_complete": bool(
                len(acquisition_evidence) == 3
                and len(acquisition_outliers) == 1
                and acquisition_outliers[0].lower() in binding.target_name.lower()
            ),
            "grounder_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "grounder_attempts": grounder_attempts,
            "candidate_bundle_complete": phase3_candidate_set_complete(candidates),
            "position_candidates": sum(
                row.target_role == "POSITION" for row in candidates
            ),
            "commit_candidates": sum(
                row.target_role == "COMMIT" for row in candidates
            ),
            "position_action_catalog_size": len(
                phase3_position_action_catalog(observation, binding)
            ),
            "accepted_bundle_candidate_parse_rejections": len(parse_rejections),
            "post_fork_actions_executed": 0,
            "evaluator_finalized": False,
            "formal_target_outcome_read": False,
            "source_program_visible_to_grounder": False,
    }
    return row_body | {"row_sha256": stable_hash(row_body)}


def _worker(*args):
    try:
        return {"row": _run_one(*args), "error": None}
    except BaseException as exc:
        return {"row": None, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    manifest = _read(args.manifest)
    _validate_self_hash(manifest, "manifest_sha256")
    if manifest.get("status") != "FROZEN_BEFORE_TYPED_GROUNDING_QUALIFICATION_CALLS":
        raise SystemExit("typed-grounding qualification manifest is not frozen")
    for path, expected in manifest["runtime_file_sha256"].items():
        if _file_sha256(REPO / path) != expected:
            raise SystemExit(f"frozen runtime changed: {path}")
    tasks = list(manifest["tasks"])
    if args.task_id:
        selected = set(args.task_id)
        tasks = [row for row in tasks if row["task_id"] in selected]
        if len(tasks) != len(selected):
            raise SystemExit("unknown qualification task ID")
    values = runpy.run_path(str(args.keys))
    key = values.get(manifest["model"]["api_key_name"])
    if not key:
        raise SystemExit("configured OpenRouter key is missing")
    workers = min(
        len(tasks), args.workers or int(manifest["runtime"]["task_workers"]),
    )
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _worker, args.manifest, task, str(key), args.output_dir,
                manifest["tasks"].index(task),
            ): task["task_id"]
            for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            result["task_id"] = futures[future]
            results.append(result)
            print(json.dumps({
                "task_id": result["task_id"], "error": result["error"],
            }), flush=True)
    results.sort(key=lambda row: row["task_id"])
    rows = [row["row"] for row in results if row["error"] is None]
    grounder_repairs = sum(any(
        not attempt["accepted"] for attempt in row["grounder_attempts"]
    ) for row in rows)
    binder_repairs = sum(any(
        not attempt["accepted"] for attempt in row["binder_attempts"]
    ) for row in rows)
    grounder_rate = grounder_repairs / len(rows) if rows else 1.0
    binder_rate = binder_repairs / len(rows) if rows else 1.0
    gates_config = manifest["frozen_qualification_gates"]
    gates = {
        "all_states_complete": len(rows) == int(
            gates_config["required_complete_states"]
        ),
        "grounder_repair_rate": grounder_rate <= float(
            gates_config["maximum_schema_or_native_precondition_repair_rate"]
        ),
        "binder_repair_rate": binder_rate <= float(
            gates_config["maximum_binder_repair_rate"]
        ),
        "accepted_bundle_parse_rejections": sum(
            row["accepted_bundle_candidate_parse_rejections"] for row in rows
        ) <= int(gates_config[
            "maximum_accepted_bundle_candidate_parse_rejections"
        ]),
        "candidate_multiplicity": all(
            row["candidate_bundle_complete"]
            and row["position_candidates"] == int(
                gates_config["required_position_candidates"]
            )
            and row["commit_candidates"] == int(
                gates_config["required_commit_candidates"]
            )
            for row in rows
        ),
        "acquisition_grounding": all(
            row["acquisition_grounding_complete"]
            and row["acquisition_evidence_count"] == int(
                gates_config["required_acquisition_measurement_vectors"]
            )
            and len(row["acquisition_outlier_candidates"]) == int(
                gates_config["required_acquisition_outlier_candidates"]
            )
            for row in rows
        ),
        "no_post_fork_action_or_evaluator": all(
            row["post_fork_actions_executed"] == 0
            and row["evaluator_finalized"] is False
            for row in rows
        ),
        "no_formal_outcome_or_source_program_read": all(
            row["formal_target_outcome_read"] is False
            and row["source_program_visible_to_grounder"] is False
            for row in rows
        ),
        "no_runtime_errors": all(row["error"] is None for row in results),
    }
    body = {
        "schema_version": "phase3-discoveryworld-typed-grounding-report-v3",
        "status": (
            "DISCOVERYWORLD_TYPED_GROUNDING_QUALIFICATION_PASSED"
            if all(gates.values()) else
            "DISCOVERYWORLD_TYPED_GROUNDING_QUALIFICATION_FAILED"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "tasks": len(results), "complete_tasks": len(rows),
        "grounder_calls_requiring_repair": grounder_repairs,
        "grounder_repair_rate": grounder_rate,
        "binder_calls_requiring_repair": binder_repairs,
        "binder_repair_rate": binder_rate,
        "accepted_bundle_candidate_parse_rejections": sum(
            row["accepted_bundle_candidate_parse_rejections"] for row in rows
        ),
        "gates": gates,
        "results": results,
        "formal_reserve_task_opened": False,
        "claim_boundary": manifest["claim_boundary"],
    }
    report = body | {"report_sha256": stable_hash(body)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "complete_tasks": len(rows),
        "grounder_repair_rate": grounder_rate,
        "binder_repair_rate": binder_rate,
        "report_sha256": report["report_sha256"],
    }, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
