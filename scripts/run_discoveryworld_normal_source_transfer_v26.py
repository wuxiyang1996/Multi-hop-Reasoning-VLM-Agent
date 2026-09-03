#!/usr/bin/env python3
"""Run frozen online Normal transfer and matched destructive controls."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
DISCOVERYWORLD = REPO.parent / "discoveryworld-official"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
if DISCOVERYWORLD.is_dir():
    sys.path.insert(0, str(DISCOVERYWORLD))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.discoveryworld_env import DETERMINISM_PROTOCOL  # noqa: E402
from motif_transfer.discoveryworld_normal_transfer import (  # noqa: E402
    SourceProgramMonitor,
    predict_grounding,
)
from scripts.run_discoveryworld_proteomics_normal_v24 import (  # noqa: E402
    ProteomicsSurveyBackend,
)
from scripts.run_discoveryworld_target_only_v1 import (  # noqa: E402
    file_sha256,
    run_episode,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    claimed = body.pop(field, None)
    if not claimed or claimed != stable_hash(body):
        raise ValueError(f"invalid {field}")


class GuardedSurveyBackend:
    """Shared target survey with only the symbolic monitor varied."""

    def __init__(self, grounder: Mapping[str, Any], condition: str) -> None:
        self.survey = ProteomicsSurveyBackend()
        self.grounder = dict(grounder)
        self.monitor = SourceProgramMonitor(condition)
        self.condition = condition
        self.decisions: list[dict[str, Any]] = []
        self.denied = False
        self.last_usage: dict[str, Any] = {
            "provider": "FROZEN_TARGET_GROUNDER_PLUS_SOURCE_MONITOR",
            "cost": 0.0,
        }

    def complete(self, model: str, system_prompt: str, user_prompt: Any) -> str:
        candidate_raw = self.survey.complete(model, system_prompt, user_prompt)
        candidate = json.loads(candidate_raw)
        facts = dict(user_prompt["target_native_facts"])
        step = {
            "action": {
                key: value for key, value in candidate.items()
                if key in {"action", "arg1", "arg2"}
            },
            "before_target_native_facts": facts,
            "memory": candidate.get("memory", "{}"),
        }
        grounded_role, probabilities = predict_grounding(self.grounder, step)
        allowed, monitor_reason = self.monitor.authorize(grounded_role)
        self.decisions.append({
            "candidate_action": step["action"],
            "grounded_role": grounded_role,
            "grounding_probabilities": probabilities,
            "allowed": allowed,
            "monitor_reason": monitor_reason,
            "monitor_phase_after": self.monitor.phase,
        })
        if allowed:
            candidate["reason"] = f"{candidate.get('reason', '')} [{monitor_reason}]"
            return json.dumps(candidate)
        self.denied = True
        return json.dumps({
            "action": "DISCOVERY_FEED_GET_UPDATES",
            "memory": candidate.get("memory", "{}"),
            "running_hypotheses": candidate.get("running_hypotheses", []),
            "expected_effect": "Fail closed after symbolic monitor abstention.",
            "reason": monitor_reason,
        })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _read(args.config)
    if config.get("status") != "FROZEN_BEFORE_V26_FORMAL_RESET_OR_OUTCOME":
        raise SystemExit("V26 formal config is not frozen")
    runpy.run_path(str(args.keys))
    qualification_path = REPO / config["lineage"]["qualification_report"]
    grounder_path = REPO / config["lineage"]["neural_grounder"]
    source_path = REPO / config["lineage"]["source_program"]
    qualification = _read(qualification_path)
    grounder = _read(grounder_path)
    source = _read(source_path)
    _self_hash(qualification, "report_sha256")
    _self_hash(grounder, "grounder_sha256")
    _self_hash(source, "artifact_sha256")
    if qualification.get("all_qualification_gates_passed") is not True:
        raise SystemExit("V26 qualification did not authorize formal")
    expected = config["lineage"]["expected_hashes"]
    actual = {
        "qualification_report_sha256": qualification["report_sha256"],
        "neural_grounder_sha256": grounder["grounder_sha256"],
        "source_program_sha256": source["artifact_sha256"],
        "runner_file_sha256": file_sha256(Path(__file__)),
        "grounding_module_file_sha256": file_sha256(REPO / "src/motif_transfer/discoveryworld_normal_transfer.py"),
    }
    if actual != expected:
        raise SystemExit(f"frozen lineage mismatch: expected={expected}, actual={actual}")

    conditions = list(map(str, config["conditions"]))
    tasks = list(config["tasks"])
    all_rows = []
    runtime_hashes = {
        "config": file_sha256(args.config),
        **actual,
        "base_runner": file_sha256(REPO / "scripts/run_discoveryworld_target_only_v1.py"),
        "environment_wrapper": file_sha256(REPO / "src/motif_transfer/discoveryworld_env.py"),
        "official_environment_commit": config["official_environment_commit"],
    }
    for condition_index, condition in enumerate(conditions):
        condition_dir = args.output_dir / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        for task_index, task in enumerate(tasks):
            backend = GuardedSurveyBackend(grounder, condition)
            # Official DiscoveryWorld is very verbose; preserve only the
            # auditable receipt, not non-semantic sprite/history chatter.
            with contextlib.redirect_stdout(io.StringIO()):
                receipt = run_episode(
                    task=task,
                    config=config,
                    backend=backend,
                    output_dir=condition_dir,
                    runtime_hashes=runtime_hashes | {"condition": condition},
                    thread_id=126000 + condition_index * 1000 + task_index,
                )
            monitor = {
                "condition": condition,
                "decisions": backend.decisions,
                "authorized_actions": backend.monitor.authorized,
                "abstentions": backend.monitor.abstentions,
                "final_phase": backend.monitor.phase,
                "denied_any_candidate": backend.denied,
            }
            monitor["monitor_sha256"] = stable_hash(monitor)
            monitor_path = condition_dir / f"{receipt['task_id']}.monitor.json"
            monitor_path.write_text(json.dumps(monitor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            row = {
                "condition": condition,
                "task_id": receipt["task_id"],
                "official_success": bool(receipt["evaluation"]["official_success"]),
                "steps": len(receipt["steps"]),
                "episode_sha256": receipt["episode_sha256"],
                "monitor_sha256": monitor["monitor_sha256"],
                "abstentions": monitor["abstentions"],
                "final_phase": monitor["final_phase"],
                "policy_runtime_saw_oracle_scorecard": receipt["policy_runtime_saw_oracle_scorecard"],
            }
            all_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    by_condition = {}
    for condition in conditions:
        rows = [row for row in all_rows if row["condition"] == condition]
        by_condition[condition] = {
            "tasks": len(rows),
            "official_successes": sum(row["official_success"] for row in rows),
            "success_rate": sum(row["official_success"] for row in rows) / len(rows),
            "tasks_with_abstention": sum(row["abstentions"] > 0 for row in rows),
            "all_zero_oracle": all(not row["policy_runtime_saw_oracle_scorecard"] for row in rows),
        }
    gates = {
        "authentic_full_coverage": by_condition["authentic_source"]["official_successes"] == len(tasks),
        "authentic_nonnegative_vs_neural_only": by_condition["authentic_source"]["official_successes"] >= by_condition["neural_only"]["official_successes"],
        "authentic_strictly_beats_source_permuted": by_condition["authentic_source"]["official_successes"] > by_condition["source_permuted"]["official_successes"],
        "permuted_fails_closed_every_task": by_condition["source_permuted"]["tasks_with_abstention"] == len(tasks),
        "all_conditions_zero_oracle": all(row["all_zero_oracle"] for row in by_condition.values()),
        "matched_task_counts": all(row["tasks"] == len(tasks) for row in by_condition.values()),
    }
    passed = all(gates.values())
    body = {
        "schema_version": "discoveryworld-normal-source-transfer-v26-formal-report",
        "status": "DISCOVERYWORLD_NORMAL_SOURCE_TRANSFER_VALIDATED" if passed else "DISCOVERYWORLD_NORMAL_SOURCE_TRANSFER_FAILED",
        "claim_boundary": (
            "Prospective fresh-seed online validation of nonnegative authentic source monitoring and destructive source-program control separation. "
            "Neural-only shares the same source-blind target survey and measures incremental success headroom; equality means mechanism transfer without a success-rate gain."
        ),
        "tasks_per_condition": len(tasks),
        "conditions": by_condition,
        "gates": gates,
        "all_formal_gates_passed": passed,
        "rows": all_rows,
        "runtime_hashes": runtime_hashes,
        "determinism_protocol": DETERMINISM_PROTOCOL,
    }
    report = body | {"report_sha256": stable_hash(body)}
    path = args.output_dir / "formal_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "conditions": by_condition,
        "failed_gates": sorted(key for key, value in gates.items() if not value),
        "report_sha256": report["report_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
