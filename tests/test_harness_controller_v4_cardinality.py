from pathlib import Path

from scripts.freeze_harness_controller_v4_target_reserve import freeze, _sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v4_source_dataset_is_source_only_and_covers_arity_grid():
    import json

    manifest = json.loads((
        REPO_ROOT / "runs/harness_controller_sft_v4_cardinality/manifest.json"
    ).read_text())
    assert manifest["target_data_used"] is False
    assert manifest["summary"]["target_data_used"] is False
    assert all(manifest["gates"].values())
    assert manifest["cardinality_equivariance"]["grid"] == list(range(2, 13))
    for split in ("train", "validation", "source_held_out"):
        counts = manifest["sft_summary"]["candidate_count_by_split"][split]
        assert set(range(2, 13)) <= {int(value) for value in counts}


def test_fresh_target_reserve_is_disjoint_and_reproducible(tmp_path):
    config = REPO_ROOT / "configs/harness_controller_v4_fresh_target_reserve.json"
    first = freeze(config, tmp_path / "first")
    second = freeze(config, tmp_path / "second")
    assert first["status"] == "FROZEN_PROSPECTIVE_TO_V4_BEFORE_WEIGHT_UPDATES"
    assert all(first["gates"].values())
    assert first["reserve"]["rows"] == 1000
    assert first["reserve"]["pairs"] == 500
    assert first["summary"]["candidate_count_counts_audit_after_selection"][2] > 0
    assert _sha256(tmp_path / "first/reserve.jsonl") == _sha256(
        tmp_path / "second/reserve.jsonl"
    )
