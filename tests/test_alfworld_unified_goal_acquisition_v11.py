from pathlib import Path
import hashlib
import json

from motif_transfer.alfworld_unified_goal_acquisition_v11 import (
    ROUTE_ID,
    build_unified_authorization,
)
from motif_transfer.unified_transfer_runtime import (
    PairedCalibration,
    TransferVerdict,
)


REPO = Path(__file__).resolve().parents[1]


def _read(relative):
    return json.loads((REPO / relative).read_text())


def _sha(relative):
    return hashlib.sha256((REPO / relative).read_bytes()).hexdigest()


def _build(**calibration):
    return build_unified_authorization(
        task_id="fresh/alfworld/task",
        acquisition_artifact=_read(
            "runs/sokoban_goal_acquisition_v1/artifact.json"
        ),
        acquisition_confirmation=_read(
            "runs/sokoban_goal_acquisition_v1/fresh_confirmation_report.json"
        ),
        target_grounder_sha256=_sha(
            "runs/procedural_game_alfworld_v1_development/"
            "frozen_candidate_artifact.json"
        ),
        target_executor_sha256=_sha(
            "src/motif_transfer/alfworld_unified_goal_acquisition_v11.py"
        ),
        evidence_report_sha256=_read(
            "runs/alfworld_goal_acquisition_v10_development/analysis_report.json"
        )["analysis_report_sha256"],
        inducer_artifact_sha256=_sha(
            "src/motif_transfer/source_goal_acquisition_induction.py"
        ),
        **calibration,
    )


def test_v10_calibration_authorizes_new_exact_unified_route():
    context = _build()
    assert context.phase7.verdict == TransferVerdict.SELECT_SKILL
    assert context.phase7.route_id == ROUTE_ID
    assert context.phase7.selected_program_sha256 == (
        "7ff3e950f3eebaf75cca015df88ab6a01f2f364fe246102984a3cb4ee095f0d7"
    )
    assert context.phase7.target_action_emitted is False
    assert context.utility.utility_lower_bound > 0.5
    assert context.utility.authenticity_lower_bound > 0.5


def test_uncalibrated_goal_acquisition_route_fails_closed():
    try:
        _build(
            utility_vs_neural=PairedCalibration(2, 0, 22),
            authenticity_vs_source_permuted=PairedCalibration(1, 1, 22),
        )
    except ValueError as exc:
        assert "abstained" in str(exc)
    else:
        raise AssertionError("uncalibrated ALFWorld route was authorized")
