import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def test_consumed_two_video_reports_fit_shared_grounding_contract(tmp_path):
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/audit_two_video_shared_grounding_v1.py"),
            "--output", str(output),
        ],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    report = json.loads(output.read_text())
    assert report["status"] == "TWO_VIDEO_SHARED_GROUNDING_COMPATIBILITY_PASSED"
    assert all(report["gates"].values())
    by_benchmark = {
        row["benchmark"]: row
        for row in report["matched_measurement"]["summaries"]
    }
    assert by_benchmark["clevrer"]["tasks"] == 360
    assert by_benchmark["clevrer"]["source_induced_correct"] == 252
    assert by_benchmark["clevrer"]["neural_only_correct"] == 236
    assert by_benchmark["agqa2"]["tasks"] == 900
    assert by_benchmark["agqa2"]["source_induced_correct"] == 290
    assert by_benchmark["agqa2"]["neural_only_correct"] == 249
    assert by_benchmark["agqa2"]["arm_correct"]["generic_scaffold"] == 290
    assert by_benchmark["agqa2"]["paired_comparisons"][
        "source_vs_generic_scaffold"
    ] == {"wins": 0, "losses": 0, "ties": 900, "exact_two_sided_p": 1.0}
    assert report["fresh_evidence"] is False
