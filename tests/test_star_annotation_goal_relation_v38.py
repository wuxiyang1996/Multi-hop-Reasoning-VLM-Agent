from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def test_consumed_star_preflight_replays_and_stops_without_provider_cost(
    tmp_path: Path,
):
    output = tmp_path / "star-v38.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/run_star_annotation_goal_relation_v38.py"),
            "--config",
            str(REPO / "configs/star_annotation_goal_relation_v38_development.json"),
            "--output", str(output),
        ],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    metrics = report["condition_metrics_vs_neural"]
    assert report["status"] == "STAR_V38_NOT_QUALIFIED_STOP"
    assert metrics["neural_only_uniform_direct"]["correct"] == 60
    assert metrics["authentic_source_semantics_counterfactual"]["correct"] == 63
    assert metrics["target_native_relation_rule"]["correct"] == 63
    assert metrics["uniform_direct_proof_oracle_ceiling"]["correct"] == 71
    assert metrics["four_policy_oracle_ceiling"]["correct"] == 83
    assert metrics["official_star_symbolic_executor_ceiling"]["correct"] == 128
    assert report["unified_runtime"]["executor_calls"] == 0
    assert report["cost"]["incremental_external_provider_calls"] == 0
    assert report["cost"]["incremental_external_provider_cost_usd"] == 0.0
    assert result.returncode == 0
