from scripts.collect_agqa_anchor_localizations_v2 import _artifact_status


def test_anchor_merge_reuses_phase_safe_collector_status() -> None:
    assert _artifact_status(True).startswith("CONSUMED_DEVELOPMENT")
    assert _artifact_status(False).endswith("FROZEN_BEFORE_TARGET_OUTCOME")
