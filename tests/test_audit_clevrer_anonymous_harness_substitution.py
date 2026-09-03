from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from motif_transfer.anonymous_video_harness import compile_anonymous_source_controller


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/audit_clevrer_anonymous_harness_substitution_v1.py"
    spec = importlib.util.spec_from_file_location("clevrer_anonymous_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_frozen_clevrer_predictions_are_exact_under_anonymous_controller() -> None:
    controller = compile_anonymous_source_controller(
        root=ROOT,
        lineage_directory=Path("runs/phase3_source_induction_v1_development/lineages"),
    )
    predictions = json.loads(
        (ROOT / "runs/clevrer_full_raw_video_v2/five_arm_predictions.json").read_text()
    )
    result = _module().audit(controller, predictions)
    assert result["status"] == "CLEVRER_ANONYMOUS_HARNESS_SUBSTITUTION_VERIFIED"
    assert result["anonymous_commits"] == 800
    assert result["anonymous_fallbacks"] == 800
    assert not result["mismatches"]
