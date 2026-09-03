from pathlib import Path

from scripts.build_harness_controller_scientific_v4_eval import build, _sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scientific_v4_freeze_is_disjoint_complete_and_reproducible(tmp_path):
    config = REPO_ROOT / "configs/harness_controller_scientific_v4.json"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = build(config, first_dir)
    second = build(config, second_dir)

    assert first["status"] == "FROZEN_TARGET_IR_ZERO_SHOT_EVALUATION_READY"
    assert all(first["gates"].values())
    assert first["summary"]["source_target_prompt_intersection"] == 0
    assert first["summary"]["source_target_example_id_intersection"] == 0
    assert first["evaluation_file"]["rows"] == 1500
    assert set(first["summary"]["group_counts"]) == {
        "webshop", "alfworld", "discoveryworld", "tirbench",
        "video/clevrer", "video/agqa2",
    }
    assert _sha256(first_dir / "zero_shot_eval.jsonl") == _sha256(
        second_dir / "zero_shot_eval.jsonl"
    )
    assert first["evaluation_file"]["sha256"] == second["evaluation_file"]["sha256"]
