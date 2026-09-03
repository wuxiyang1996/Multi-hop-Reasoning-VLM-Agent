#!/usr/bin/env python3
"""Run frozen valid-unseen ALFWorld qualification or formal transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_alfworld_goal_relation_macro_v3 as v3  # noqa: E402
from motif_transfer.alfworld_goal_relation_macro_v5 import (  # noqa: E402
    choose_goal_relation_action,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_self_hash(payload: dict, field: str) -> None:
    body = dict(payload)
    claimed = str(body.pop(field, ""))
    if not claimed or stable_hash(body) != claimed:
        raise SystemExit(f"invalid {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_self_hash(config, "config_sha256")
    role = str(config.get("role"))
    if role not in {"qualification", "formal"}:
        raise SystemExit("V6 role must be qualification or formal")
    experiment_version = str(config.get("experiment_version"))
    version_tags = {
        "VALID_UNSEEN_FAIL_CLOSED_V6": "v6",
        "COMPILED_VALID_UNSEEN_FAIL_CLOSED_V7": "v7",
        "PLANNER_COMPILED_VALID_UNSEEN_FAIL_CLOSED_V8": "v8",
    }
    if experiment_version not in version_tags:
        raise SystemExit("not a frozen valid-unseen relation-macro config")
    version_tag = version_tags[experiment_version]
    dependencies = {
        "v6_runner_file_sha256": Path(__file__).resolve(),
        "v5_target_runtime_file_sha256": (
            REPO / "src/motif_transfer/alfworld_goal_relation_macro_v5.py"
        ),
    }
    for field, path in dependencies.items():
        if _sha256(path) != config.get(field):
            raise SystemExit(f"frozen V6 dependency changed: {path}")
    for task_id, expected in config.get("generated_game_file_sha256", {}).items():
        game = (
            Path(config["alfworld_data"]) / "json_2.1.1" / "valid_unseen"
            / task_id
        )
        if _sha256(game) != expected:
            raise SystemExit(f"frozen generated game changed: {game}")

    if role == "formal":
        qualification_path = REPO / config["qualification_report"]
        if not qualification_path.is_file():
            raise SystemExit("formal reserve sealed: qualification report missing")
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        _validate_self_hash(qualification, "report_sha256")
        if (
            qualification.get("status") != "UNTOUCHED_QUALIFICATION_GATE_PASSED"
            or qualification.get("config_sha256")
            != config.get("qualification_config_sha256")
            or not all(qualification.get("gates", {}).values())
        ):
            raise SystemExit("formal reserve sealed: qualification gate failed")

    v3.choose_goal_relation_action = choose_goal_relation_action
    real_environment = v3.ALFWorldTextBatchEnvironment

    def frozen_split_environment(**kwargs):
        return real_environment(**(kwargs | {"split": config["alfworld_split"]}))

    v3.ALFWorldTextBatchEnvironment = frozen_split_environment
    sys.argv = [str(Path(v3.__file__).resolve()), "--config", str(args.config)]
    result = v3.main()
    output = REPO / config["output"]
    report = json.loads(output.read_text(encoding="utf-8"))
    report.pop("report_sha256", None)
    passed = all(report["gates"].values())
    report |= {
        "schema_version": (
            f"alfworld-goal-relation-macro-{role}-{version_tag}"
        ),
        "status": (
            "UNTOUCHED_QUALIFICATION_GATE_PASSED"
            if role == "qualification" and passed
            else "UNTOUCHED_QUALIFICATION_GATE_FAILED"
            if role == "qualification"
            else "FRESH_FORMAL_ALFWORLD_TRANSFER_VALIDATED"
            if passed
            else "FRESH_FORMAL_ALFWORLD_TRANSFER_FAILED"
        ),
        "role": role,
        "config_sha256": str(config["config_sha256"]),
        "experiment_version": str(config["experiment_version"]),
        "claim_boundary": str(config["claim_boundary"]),
        "historical_identity_audit_sha256": str(
            config["historical_identity_audit_sha256"]
        ),
    }
    report["report_sha256"] = stable_hash(report)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "v6_status": report["status"],
        "v6_report_sha256": report["report_sha256"],
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
