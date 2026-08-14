from scripts.audit_natural_video_cost_boundary_v38 import _all_costs


def test_all_costs_counts_each_usage_record_once() -> None:
    payload = [
        {"usage": {"direct": {"cost": 0.1}, "proof": {"cost": 0.2}}},
        {"usage": {"direct": {"cost": 0.3}}},
    ]
    assert _all_costs(payload) == [0.1, 0.2, 0.3]
