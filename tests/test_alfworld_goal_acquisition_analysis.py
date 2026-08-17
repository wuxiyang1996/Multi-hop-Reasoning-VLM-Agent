from __future__ import annotations

from scripts.analyze_alfworld_goal_acquisition_v10 import wrong_handle_counts


def test_wrong_handle_gate_counts_only_source_active_relations() -> None:
    report = {
        "episodes": {
            "authentic_source_goal_relation_macro": [{
                "records": [
                    {
                        "target_effect_receipt": "RELATE_NO_PROGRESS",
                        "completed_count_before": 0,
                    },
                    {
                        "target_effect_receipt": "IGNORE",
                        "completed_count_before": 1,
                    },
                    {
                        "target_effect_receipt": "RELATE_NO_PROGRESS",
                        "completed_count_before": 1,
                    },
                ],
            }],
        },
    }
    assert wrong_handle_counts(report) == (2, 1)
