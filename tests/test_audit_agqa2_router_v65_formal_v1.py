import json
from pathlib import Path

from motif_transfer.contracts import stable_hash


ROOT = Path(__file__).resolve().parents[1]


def test_sealed_router_v65_formal_audit_is_not_misreported_as_passed():
    path = ROOT / "docs/results/agqa2_router_v65_formal_v1_audit.json"
    report = json.loads(path.read_text())
    body = dict(report)
    claimed = body.pop("audit_sha256")
    assert stable_hash(body) == claimed
    assert report["status"] == (
        "V1_CONFIRMATORY_FAILED_AND_NO_SECOND_UNTOUCHED_COMPATIBLE_COHORT"
    )
    assert report["formal_result"]["arm_correct"] == {
        "neural_only": 45,
        "source_induced": 53,
        "source_permuted": 45,
        "target_written_equivalent": 53,
    }
    assert report["formal_result"]["source_vs_neural_only"] == {
        "wins": 14, "losses": 6, "ties": 60, "net_gain": 8,
        "one_sided_exact_binomial_pvalue": 0.057659149169921875,
    }
    assert report["gates"]["success_gain_gate_passed"] is False
    assert report["gates"]["negative_transfer_gate_passed"] is False
    assert report["remaining_inventory"][
        "currently_unseen_router_qualified_exists_tasks"
    ] == 0
