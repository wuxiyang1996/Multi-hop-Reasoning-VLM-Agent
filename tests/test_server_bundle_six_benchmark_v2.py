import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_portable_bundle_declares_every_indirect_dependency_root() -> None:
    config = json.loads((
        REPO / "configs/server_bundle_six_benchmark_v2.json"
    ).read_text(encoding="utf-8"))
    roots = set(config["dependency_roots"])
    assert {
        "runs/phase3_source_function_v4_reserve",
        "runs/harness_9b_six_benchmark_substitution_v1",
        "runs/webshop_structural_transfer_v21_formal",
        "runs/discoveryworld_structural_transfer_v1_matched",
        "runs/tir_maze_structural_transfer_v3",
        "runs/alfworld_unified_goal_acquisition_v13_formal",
        "runs/clevrer_unified_goal_relation_v15_reserve",
        "runs/agqa2_full_distribution_v62",
        "runs/harness_controller_sft_v4_cardinality",
        "runs/harness_controller_scientific_v4_zero_shot",
    } <= roots
    assert all((REPO / path).exists() for path in roots)


def test_clean_room_contract_requires_all_three_data_archives() -> None:
    config = json.loads((
        REPO / "configs/server_bundle_six_benchmark_v2.json"
    ).read_text(encoding="utf-8"))
    assert config["clean_room_archives"] == [
        "10-harness-source-only-core.tar.zst",
        "11-target-adapted-baseline.tar.zst",
        "13-six-benchmark-portable-dependencies.tar.zst",
    ]
    assert config["expected_pytest_passes"] == 22
