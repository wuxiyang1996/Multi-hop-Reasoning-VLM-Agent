#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.alfworld_env import ALFWorldTextBatchEnvironment  # noqa: E402
from motif_transfer.alfworld_neural_grounder import (  # noqa: E402
    action_role,
    choose_grounded_action,
    deserialize_value_ensemble,
    score_native_actions,
    target_symbolic_features,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


CONDITIONS = (
    "target_only",
    "authentic_source_plus_target",
    "shuffled_source_plus_target",
    "source_marginal_plus_target",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config["target"]
    artifact_path = (REPO / target["artifact"]).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact["status"] != "QUALIFICATION_AUTHORIZED":
        raise SystemExit("candidate artifact did not authorize qualification")
    manifest_path = (REPO / target["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_name = str(target["evaluation_manifest_split"])
    if split_name != "qualification":
        raise SystemExit("development runner may read qualification only")
    task_ids = manifest["cells"]["alfworld_valid_unseen"]["splits"][split_name]
    grounder = artifact["target_grounder"]
    source_models = {
        condition: deserialize_value_ensemble(artifact["source"]["models"][condition])
        for condition in CONDITIONS
        if condition != "target_only"
    }
    episodes: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        environment = ALFWorldTextBatchEnvironment(
            config_path=str(target["alfworld_config"]),
            data_path=str(target["alfworld_data"]),
            split=str(target["evaluation_split"]),
            seed=int(target["seed"]),
            game_ids=tuple(map(str, task_ids)),
            max_steps=int(target["evaluation_max_steps"]),
        )
        try:
            for task_index, task_id in enumerate(task_ids):
                observation = environment.reset()
                history: list[str] = []
                records = []
                for step in range(int(target["evaluation_max_steps"])):
                    native = list(observation.native_actions)
                    scores = score_native_actions(
                        goal=str(observation.state.get("task_goal", "")),
                        observation=str(observation.state.get("observation", "")),
                        native_actions=native,
                        step=step,
                        action_history=history,
                        artifact=grounder,
                    )
                    if not scores:
                        scores = {native[0]: 1.0}
                    symbolic = target_symbolic_features(
                        actions=native,
                        scores=scores,
                        step=step,
                        max_steps=int(target["evaluation_max_steps"]),
                        action_history=history,
                    )
                    decision = choose_grounded_action(
                        actions=native,
                        grounder_scores=scores,
                        symbolic_features=symbolic,
                        source_model=source_models.get(condition),
                        uncertainty_scale=float(config["policy"]["uncertainty_scale"]),
                        decision_margin=float(config["policy"]["decision_margin"]),
                    )
                    selected = str(decision["action"])
                    after, reward = environment.step(selected)
                    records.append({
                        "step": step,
                        "action": selected,
                        "action_role": action_role(selected),
                        "fallback_action": decision["fallback_action"],
                        "fallback_role": action_role(decision["fallback_action"]),
                        "source_admitted": decision["source_admitted"],
                        "changed_action": decision["changed_action"],
                        "changed_role": decision["changed_role"],
                        "diagnostic": decision["diagnostic"],
                        "reward": float(reward),
                        "official_success_after": bool(after.official_success),
                        "receipt_sha256": stable_hash({
                            "task_id": task_id,
                            "condition": condition,
                            "step": step,
                            "before": dict(observation.state),
                            "native": observation.native_actions,
                            "decision": decision,
                            "after": dict(after.state),
                            "reward": reward,
                            "success": after.official_success,
                        }),
                    })
                    history.append(selected)
                    observation = after
                    if after.terminal or after.official_success:
                        break
                success = bool(records and records[-1]["official_success_after"])
                episodes[condition].append({
                    "task_index": task_index,
                    "task_id": task_id,
                    "official_success": success,
                    "steps": len(records),
                    "source_admissions": sum(row["source_admitted"] for row in records),
                    "changed_actions": sum(row["changed_action"] for row in records),
                    "changed_roles": sum(row["changed_role"] for row in records),
                    "diagnostics": dict(Counter(row["diagnostic"] for row in records)),
                    "records": records,
                })
                print(json.dumps({
                    "condition": condition,
                    "task_index": task_index,
                    "steps": len(records),
                    "success": success,
                }), flush=True)
        finally:
            environment.close()
    summaries = {}
    for condition, rows in episodes.items():
        total_steps = sum(row["steps"] for row in rows)
        summaries[condition] = {
            "tasks": len(rows),
            "successes": sum(row["official_success"] for row in rows),
            "success_rate": sum(row["official_success"] for row in rows) / len(rows),
            "mean_steps": sum(row["steps"] for row in rows) / len(rows),
            "source_admission_rate": sum(row["source_admissions"] for row in rows) / total_steps,
            "changed_action_rate": sum(row["changed_actions"] for row in rows) / total_steps,
            "changed_role_rate": sum(row["changed_roles"] for row in rows) / total_steps,
        }
    authentic = summaries["authentic_source_plus_target"]
    controls = [summaries[name] for name in CONDITIONS if name != "authentic_source_plus_target"]
    nontrivial = authentic["changed_role_rate"] >= float(
        config["policy"]["minimum_authentic_intervention_rate"]
    )
    superiority = all(authentic["successes"] > row["successes"] for row in controls)
    candidate_passed = nontrivial and superiority
    report = {
        "schema_version": "multisource-alfworld-qualification-v1",
        "status": "QUALIFICATION_CANDIDATE_PASSED" if candidate_passed else "QUALIFICATION_CANDIDATE_FAILED",
        "claim_boundary": config["claim_boundary"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_split": split_name,
        "heldout_read": False,
        "conditions": list(CONDITIONS),
        "summaries": summaries,
        "nontriviality_gate": {
            "metric": "authentic changed TEST/COMMIT role rate",
            "observed": authentic["changed_role_rate"],
            "minimum": float(config["policy"]["minimum_authentic_intervention_rate"]),
            "passed": nontrivial,
        },
        "qualification_superiority_gate": {
            "metric": "authentic successes strictly greater than every control",
            "passed": superiority,
        },
        "episodes": episodes,
        "cross_domain_transfer_supported": False,
    }
    output = (REPO / target["qualification_report"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "summaries": summaries,
        "nontriviality_gate": report["nontriviality_gate"],
        "qualification_superiority_gate": report["qualification_superiority_gate"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0 if candidate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
