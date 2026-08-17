#!/usr/bin/env python3
"""Run effect-gated V4 through the frozen V3 evaluation implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_alfworld_goal_relation_macro_v3 as v3  # noqa: E402
from motif_transfer.alfworld_goal_relation_macro_v4 import (  # noqa: E402
    choose_goal_relation_action,
)
from motif_transfer.contracts import stable_hash  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config_path = (
        Path(sys.argv[sys.argv.index("--config") + 1])
        if "--config" in sys.argv
        else REPO / "configs/alfworld_goal_relation_macro_v4_development.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_version") != "EFFECT_GATED_RECURRENCE_V4":
        raise SystemExit("not an effect-gated V4 config")
    dependencies = {
        "v4_runner_file_sha256": Path(__file__).resolve(),
        "v4_target_runtime_file_sha256": (
            REPO / "src/motif_transfer/alfworld_goal_relation_macro_v4.py"
        ),
    }
    for field, path in dependencies.items():
        if _sha256(path) != config.get(field):
            raise SystemExit(f"frozen V4 dependency changed: {path}")

    # The V3 runner owns the matched environment/evaluation implementation.
    # Replacing only this callable makes the applicability change auditable.
    v3.choose_goal_relation_action = choose_goal_relation_action
    sys.argv = [str(Path(v3.__file__).resolve()), "--config", str(config_path)]
    result = v3.main()

    output = REPO / config["output"]
    report = json.loads(output.read_text(encoding="utf-8"))
    report.pop("report_sha256", None)
    report["schema_version"] = "alfworld-goal-relation-macro-development-v4"
    report["experiment_version"] = str(config["experiment_version"])
    report["applicability_boundary"] = (
        "SOURCE SELF_LOOP ADMITTED ONLY AFTER TARGET_NATIVE OBSERVATION OF "
        "THE FIRST ENTITY_GOAL_RELATION EFFECT"
    )
    report["report_sha256"] = stable_hash(report)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "v4_status": report["status"],
        "v4_report_sha256": report["report_sha256"],
    }, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
