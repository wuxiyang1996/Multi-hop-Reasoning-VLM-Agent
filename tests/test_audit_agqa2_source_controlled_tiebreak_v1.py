import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]


def test_consumed_agqa_receipts_reexecute_source_recurrence(tmp_path):
    output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/audit_agqa2_source_controlled_tiebreak_v1.py"),
            "--output", str(output),
        ],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    report = json.loads(output.read_text())
    assert report["status"] == (
        "AGQA2_SOURCE_CONTROLLED_TIEBREAK_RETROSPECTIVE_SUPPORTED_FRESH_REQUIRED"
    )
    assert all(report["gates"].values())
    development, qualification, formal = report["audits"]
    assert (development["rows"], development["tiebreak_rows"]) == (40, 25)
    assert qualification["source_vs_generic"] == {
        "wins": 3, "losses": 0, "ties": 237, "exact_two_sided_p": 0.25,
    }
    assert formal["source_vs_generic"] == {
        "wins": 16, "losses": 2, "ties": 882,
        "exact_two_sided_p": 0.001312255859375,
    }
    assert report["fresh_evidence"] is False
