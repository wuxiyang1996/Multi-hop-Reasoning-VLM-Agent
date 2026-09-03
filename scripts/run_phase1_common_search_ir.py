#!/usr/bin/env python3
"""Prepare, collect, and analyze Phase-1 common search-IR source evidence."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import json
import os
from pathlib import Path
import sys

from motif_transfer.phase1_common_search_ir import (
    analyze_common_search_ir,
    build_discovery_option_template_artifact,
    build_discovery_primitive_template_artifact,
    build_observed_prefix_plan,
    collect_option_forks,
    read_jsonl,
    validate_option_template_artifact,
    write_jsonl,
)
from motif_transfer.contracts import stable_hash


REPO = Path(__file__).resolve().parents[1]


def _load_adapter(source_script: Path):
    spec = importlib.util.spec_from_file_location(
        "phase1_common_search_runtime", source_script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source runtime: {source_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._SourceReplayAdapter


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def prepare(config: dict, config_path: Path, output: Path) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")
    if config.get("gymv_python"):
        os.environ["GYMV_PYTHON"] = str(
            Path(config["gymv_python"]).resolve()
        )
    adapter = _load_adapter(Path(config["source_runtime_script"]).resolve())
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            plan = build_observed_prefix_plan(
                adapter,
                game=str(config["game"]),
                seeds=[int(seed) for seed in config["seeds"]],
                namespace=str(config["namespace"]),
                max_steps=int(config["max_steps"]),
                rollout_steps=int(config["rollout_steps"]),
                snapshots_per_episode=int(config["snapshots_per_episode"]),
                actions_per_snapshot=int(config["candidate_count"]),
                minimum_step=int(config["minimum_step"]),
                runtime_receipt={
                    "source_runtime_script": str(
                        Path(config["source_runtime_script"]).resolve()
                    ),
                    "config_path": str(config_path.resolve()),
                    "gymv_python": os.environ.get("GYMV_PYTHON"),
                },
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "game": config["game"],
        "plan": str(output.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "snapshots": len(plan["snapshots"]),
    }, indent=2, sort_keys=True))


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def _validate_resume_rows(
    rows: list[dict],
    *,
    config: dict,
    plan: dict,
    option_templates,
) -> set[str]:
    snapshots = {
        str(row["snapshot_id"]): row for row in plan.get("snapshots") or []
    }
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        body = dict(row)
        claimed = body.pop("row_sha256", None)
        if claimed != stable_hash(body):
            raise SystemExit("resume row hash mismatch")
        snapshot_id = str(row.get("snapshot_id"))
        if snapshot_id not in snapshots:
            raise SystemExit("resume rows contain an unknown snapshot")
        if row.get("game") != config["game"]:
            raise SystemExit("resume row game mismatch")
        if int(row.get("horizon", -1)) != int(config["horizon"]):
            raise SystemExit("resume row horizon mismatch")
        grouped.setdefault(snapshot_id, []).append(row)
    completed: set[str] = set()
    repeats = int(config["repeats"])
    for snapshot_id, snapshot_rows in grouped.items():
        snapshot = snapshots[snapshot_id]
        candidate_count = (
            len(option_templates)
            if option_templates is not None
            else len(snapshot["selected_actions"])
        )
        expected_pairs = {
            (candidate_rank, repeat_index)
            for candidate_rank in range(candidate_count)
            for repeat_index in range(repeats)
        }
        observed_pairs = {
            (int(row["candidate_rank"]), int(row["repeat_index"]))
            for row in snapshot_rows
        }
        if len(snapshot_rows) != len(expected_pairs) or observed_pairs != expected_pairs:
            raise SystemExit(
                f"resume snapshot is not an atomic complete group: {snapshot_id}"
            )
        completed.add(snapshot_id)
    return completed


def collect(
    config: dict,
    plan_path: Path,
    output: Path,
    workers: int,
    *,
    incremental: bool,
) -> None:
    if output.exists() and not incremental:
        raise SystemExit(f"refusing to overwrite: {output}")
    if config.get("gymv_python"):
        os.environ["GYMV_PYTHON"] = str(
            Path(config["gymv_python"]).resolve()
        )
    adapter = _load_adapter(Path(config["source_runtime_script"]).resolve())
    plan = _read(plan_path)
    option_templates = None
    if config.get("option_template_artifact"):
        artifact = _read(Path(config["option_template_artifact"]))
        option_templates = validate_option_template_artifact(
            artifact,
            game=str(config["game"]),
            horizon=int(config["horizon"]),
        )
    common_arguments = {
        "adapter_class": adapter,
        "game": str(config["game"]),
        "horizon": int(config["horizon"]),
        "repeats": int(config["repeats"]),
        "namespace": str(config["namespace"]),
        "workers": workers,
        "continuation_mode": str(config.get("continuation_mode", "common")),
        "option_templates": option_templates,
    }
    if not incremental:
        rows = collect_option_forks(plan, **common_arguments)
        write_jsonl(output, rows)
    else:
        rows = read_jsonl(output) if output.exists() else []
        completed = _validate_resume_rows(
            rows,
            config=config,
            plan=plan,
            option_templates=option_templates,
        )
        audit_path = output.with_suffix(output.suffix + ".collection-audit.json")
        audit = _read(audit_path) if audit_path.exists() else {
            "schema_version": "phase1-incremental-collection-audit-v1",
            "game": str(config["game"]),
            "plan_sha256": str(plan["plan_sha256"]),
            "retry_authority": (
                "WHOLE_SNAPSHOT_ONLY_AFTER_INTERVENTION_FAILED;OUTCOME_UNREAD"
            ),
            "snapshots": {},
        }
        if audit.get("plan_sha256") != plan.get("plan_sha256"):
            raise SystemExit("collection-audit plan hash mismatch")
        maximum_attempts = int(
            config.get("maximum_infrastructure_attempts_per_snapshot", 1)
        )
        if maximum_attempts < 1:
            raise SystemExit("maximum infrastructure attempts must be positive")
        for snapshot in plan.get("snapshots") or []:
            snapshot_id = str(snapshot["snapshot_id"])
            if snapshot_id in completed:
                continue
            attempts = []
            snapshot_rows = []
            for attempt_index in range(maximum_attempts):
                snapshot_rows = collect_option_forks(
                    plan,
                    snapshot_ids={snapshot_id},
                    **common_arguments,
                )
                status_counts: dict[str, int] = {}
                for row in snapshot_rows:
                    status = str(row["status"])
                    status_counts[status] = status_counts.get(status, 0) + 1
                attempts.append({
                    "attempt_index": attempt_index,
                    "status_counts": status_counts,
                    "row_sha256s": [row["row_sha256"] for row in snapshot_rows],
                    "errors": [
                        str(row["error"])
                        for row in snapshot_rows if row.get("error")
                    ],
                })
                if status_counts.get("INTERVENTION_FAILED", 0) == 0:
                    break
            audit["snapshots"][snapshot_id] = {
                "attempts": attempts,
                "accepted_attempt_index": len(attempts) - 1,
                "retry_decision_read_outcome": False,
            }
            rows.extend(snapshot_rows)
            _write_json_atomic(audit_path, audit)
            _write_jsonl_atomic(output, rows)
            print(json.dumps({
                "checkpoint_snapshot_id": snapshot_id,
                "completed_snapshots": len(audit["snapshots"]),
                "attempts": len(attempts),
                "rows": len(rows),
            }, sort_keys=True), flush=True)
    status_counts = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    print(json.dumps({
        "game": config["game"],
        "output": str(output.resolve()),
        "rows": len(rows),
        "status_counts": status_counts,
    }, indent=2, sort_keys=True))


def analyze(config: dict, rows_path: Path, output: Path) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")
    expected = config.get("expected_policy_sha256")
    report = analyze_common_search_ir(
        read_jsonl(rows_path),
        primary_horizon=int(config["horizon"]),
        source_gate_requirements=config["source_gate"],
        minimum_eligible_fraction_each_split=float(
            config["minimum_eligible_fraction_each_split"]
        ),
        expected_policy_sha256=str(expected) if expected else None,
        maximum_intervention_failed_rows=(
            int(config["maximum_intervention_failed_rows"])
            if "maximum_intervention_failed_rows" in config else None
        ),
    )
    report.update({
        "game": str(config["game"]),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": __import__("hashlib").sha256(rows_path.read_bytes()).hexdigest(),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["source_gate_passed"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare-templates", "prepare", "collect", "analyze")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()
    config = _read(args.config)
    if args.command == "prepare-templates":
        template_strategy = str(
            config.get("template_strategy", "discovery_execution_controls")
        )
        builders = {
            "discovery_execution_controls": (
                build_discovery_option_template_artifact
            ),
            "discovery_native_primitives": (
                build_discovery_primitive_template_artifact
            ),
        }
        if template_strategy not in builders:
            raise SystemExit(f"unsupported template_strategy: {template_strategy}")
        artifact = builders[template_strategy](
            config["template_source_evidence"],
            game=str(config["game"]),
            horizon=int(config["horizon"]),
        )
        if config.get("horizon_authority") == (
            "maximum discovery-only maximal skill-execution length"
        ) and int(artifact.get("maximum_discovery_execution_length", -1)) != int(
            config["horizon"]
        ):
            raise SystemExit(
                "configured horizon does not match discovery execution authority"
            )
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "artifact": str(args.output.resolve()),
            "artifact_sha256": artifact["artifact_sha256"],
            "game": artifact["game"],
            "template_strategy": template_strategy,
            "template_count": len(artifact["templates"]),
            "selected_skill_id": (
                artifact.get("selected_discovery_execution") or {}
            ).get("skill_id"),
        }, indent=2, sort_keys=True))
    elif args.command == "prepare":
        prepare(config, args.config, args.output)
    elif args.command == "collect":
        if args.plan is None:
            parser.error("collect requires --plan")
        collect(
            config,
            args.plan,
            args.output,
            args.workers,
            incremental=args.incremental,
        )
    else:
        if args.rows is None:
            parser.error("analyze requires --rows")
        analyze(config, args.rows, args.output)


if __name__ == "__main__":
    main()
