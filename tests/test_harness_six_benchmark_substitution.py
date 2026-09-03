from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import pytest

from scripts.build_harness_multi_ir_selector_sft_v1 import _load_contracts
from scripts.freeze_harness_9b_six_benchmark_substitution_v1 import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    REPO,
    freeze,
)
from scripts.audit_harness_9b_six_benchmark_action_equivalence_v1 import (
    _agqa_receipts,
    _alfworld_receipts,
    _clevrer_receipts,
    _discoveryworld_receipts,
    _rows,
    _tir_receipts,
    _webshop_receipts,
)
from scripts import audit_harness_9b_six_benchmark_action_equivalence_v1 as action_audit
from scripts.build_harness_9b_six_benchmark_paper_report_v1 import (
    _write_if_absent_or_identical,
    formal_success_rows,
)
from motif_transfer.portable_paths import resolve_repo_artifact


def _artifact(value: str) -> Path:
    return resolve_repo_artifact(value, REPO)


def test_optional_temporal_source_extends_catalog_without_changing_v1() -> None:
    v1 = json.loads((
        REPO / "configs/harness_multi_ir_selector_sft_v1.json"
    ).read_text(encoding="utf-8"))
    v2 = json.loads((
        REPO / "configs/harness_multi_ir_selector_sft_v2.json"
    ).read_text(encoding="utf-8"))
    v1_contracts, _, _ = _load_contracts(v1)
    v2_contracts, _, _ = _load_contracts(v2)
    assert len(v1_contracts) == 6
    assert len(v2_contracts) == 7
    assert {row.ir_kind for row in v2_contracts} - {
        row.ir_kind for row in v1_contracts
    } == {"SPARSE_TEMPORAL_EFFECT_FUNCTION"}


def test_six_benchmark_freeze_uses_all_preoutcome_task_ids(tmp_path: Path) -> None:
    # The production freezer must stop being callable once the five-schema
    # adapter exists: otherwise a later task selection could masquerade as a
    # preregistration.  In a post-training checkout, assert that lifecycle
    # guard and audit the already-frozen artifact instead of bypassing it.
    adapter_root = REPO / "runs/harness_controller_qwen35_9b_mixed_v2"
    if adapter_root.exists():
        with pytest.raises(ValueError, match="five_schema_adapter_not_yet_created"):
            freeze(DEFAULT_CONFIG, tmp_path / "substitution")
        manifest = json.loads(
            (DEFAULT_OUTPUT / "preregistration.json").read_text(encoding="utf-8")
        )
    else:
        manifest = freeze(DEFAULT_CONFIG, tmp_path / "substitution")
    assert all(manifest["gates"].values())
    assert manifest["native_replay_index"]["group_counts"] == {
        "agqa2": 900,
        "alfworld": 24,
        "clevrer": 360,
        "discoveryworld": 12,
        "tirbench": 18,
        "webshop": 32,
    }
    assert manifest["route_selector_replay"]["group_counts"]["agqa2"] == 1800
    rows = [
        json.loads(line)
        for line in _artifact(manifest["route_selector_replay"]["path"])
        .read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 2246
    assert len({row["prompt"] for row in rows}) == len(rows)
    assert all(
        json.loads(row["completion"])["decision"] == "SELECT_SKILL"
        for row in rows
    )


def test_fresh_presentation_reserve_is_disjoint_and_preupdate_frozen() -> None:
    parent_root = REPO / "runs/harness_9b_six_benchmark_substitution_v1"
    fresh_root = REPO / "runs/harness_9b_six_benchmark_substitution_fresh_v2"
    parent = json.loads(
        (parent_root / "preregistration.json").read_text(encoding="utf-8")
    )
    fresh = json.loads(
        (fresh_root / "preregistration.json").read_text(encoding="utf-8")
    )
    assert fresh["status"] == (
        "FROZEN_FRESH_PRESENTATION_BEFORE_SOURCE_ONLY_PERMUTATION_UPDATE"
    )
    assert all(fresh["gates"].values())
    assert fresh["route_selector_replay"]["rows"] == 2246
    assert fresh["native_replay_index"]["tasks"] == 1346
    parent_rows = _rows(_artifact(parent["route_selector_replay"]["path"]))
    fresh_rows = _rows(_artifact(fresh["route_selector_replay"]["path"]))
    assert {row["prompt"] for row in parent_rows}.isdisjoint(
        {row["prompt"] for row in fresh_rows}
    )
    assert {row["example_id"] for row in parent_rows}.isdisjoint(
        {row["example_id"] for row in fresh_rows}
    )
    for key in ("route_selector_replay", "native_replay_index"):
        path = _artifact(fresh[key]["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fresh[key]["sha256"]


def test_source_only_permutation_closure_is_balanced_and_target_free() -> None:
    root = REPO / "runs/harness_multi_ir_permutation_closure_sft_v3"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN_SOURCE_ONLY_MULTI_IR_SELECTOR_SUPERVISION"
    assert all(manifest["gates"].values())
    assert manifest["target_data_used"] is False
    assert manifest["target_outcome_used_for_controller_labels"] is False
    for split, counts in manifest["closure_counts"].items():
        assert counts["decisions"] == {
            "ABSTAIN": counts["rows"] // 2,
            "SELECT_SKILL": counts["rows"] // 2,
        }
        path = root / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[
            "files"
        ][split]["sha256"]
    serialized_train = (root / "train.jsonl").read_text(encoding="utf-8").lower()
    for token in (
        "webshop", "alfworld", "discoveryworld", "tirbench", "clevrer",
        "agqa2", "native_action", "formal_task_id",
    ):
        assert token not in serialized_train


def test_all_frozen_native_receipts_are_content_addressed() -> None:
    index = _rows(
        REPO
        / "runs/harness_9b_six_benchmark_substitution_v1/native_replay_index.jsonl"
    )
    task_ids: dict[str, set[str]] = {}
    programs: dict[str, list[str]] = {}
    for row in index:
        benchmark = str(row["benchmark"])
        task_ids.setdefault(benchmark, set()).add(str(row["formal_task_id"]))
        programs.setdefault(benchmark, list(row["source_program_sha256"]))
    extractors = {
        "webshop": lambda: _webshop_receipts(
            task_ids["webshop"], programs["webshop"][0]
        ),
        "discoveryworld": lambda: _discoveryworld_receipts(
            task_ids["discoveryworld"], programs["discoveryworld"][0]
        ),
        "tirbench": lambda: _tir_receipts(
            task_ids["tirbench"], programs["tirbench"][0]
        ),
        "alfworld": lambda: _alfworld_receipts(
            task_ids["alfworld"], programs["alfworld"][0]
        ),
        "clevrer": lambda: _clevrer_receipts(
            task_ids["clevrer"], programs["clevrer"][0]
        ),
        "agqa2": lambda: _agqa_receipts(
            task_ids["agqa2"], programs["agqa2"]
        ),
    }
    for benchmark, extractor in extractors.items():
        receipts, integrity = extractor()
        assert set(receipts) == task_ids[benchmark]
        assert all(row["receipt_available"] for row in receipts.values())
        assert all(
            row["old_route_programs"] == programs[benchmark]
            for row in receipts.values()
        )
        assert all(
            value
            for key, value in integrity.items()
            if key.endswith("self_hash_valid")
        )

    alfworld, _ = extractors["alfworld"]()
    assert sum(
        row["receipt_kind"] == "ZERO_ADMISSION_ABSTENTION_EPISODE_RECEIPT"
        for row in alfworld.values()
    ) == 2


@pytest.mark.parametrize("reserve_dir", [
    "harness_9b_six_benchmark_substitution_v1",
    "harness_9b_six_benchmark_substitution_fresh_v2",
])
def test_exact_routes_bridge_to_all_six_formal_benchmarks(
    tmp_path: Path, monkeypatch, reserve_dir: str,
) -> None:
    prereg_path = (
        REPO
        / "runs" / reserve_dir / "preregistration.json"
    )
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    dataset_path = _artifact(prereg["route_selector_replay"]["path"])
    index_path = _artifact(prereg["native_replay_index"]["path"])
    activation_path = tmp_path / "activation.json"
    activation = {
        "status": "FROZEN_SIX_BENCHMARK_SUBSTITUTION_EVALUATION_READY",
        "gates": {"synthetic_test_activation": True},
        "target_preregistration": {
            "path": str(prereg_path),
            "sha256": action_audit._sha(prereg_path),
        },
        "evaluation_file": {
            "path": str(dataset_path),
            "sha256": action_audit._sha(dataset_path),
        },
        "native_replay_index": {
            "path": str(index_path),
            "sha256": action_audit._sha(index_path),
        },
    }
    activation_path.write_text(
        json.dumps(activation, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = tmp_path / "route_report.json"
    report_path.write_text(json.dumps({
        "status": "SIX_BENCHMARK_MODEL_SUBSTITUTION_ROUTE_GATE_PASSED",
        "gates": {"all_exact": True},
        "evaluation_manifest": {"sha256": action_audit._sha(activation_path)},
    }, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps({
            "example_id": row["example_id"],
            "regime": "CONTROLLER_LORA",
            "exact_json": True,
            "parsed": json.loads(row["completion"]),
        }, sort_keys=True) + "\n"
        for row in _rows(dataset_path)
    ), encoding="utf-8")
    output_path = tmp_path / "action_equivalence.json"
    monkeypatch.setattr(sys, "argv", [
        "audit_harness_9b_six_benchmark_action_equivalence_v1.py",
        "--activation", str(activation_path),
        "--route-report", str(report_path),
        "--route-predictions", str(predictions_path),
        "--output", str(output_path),
    ])
    assert action_audit.main() == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == (
        "SIX_BENCHMARK_9B_SUBSTITUTION_ACTION_EQUIVALENCE_VALIDATED"
    )
    assert result["summary"]["formal_tasks"] == 1346
    assert result["summary"]["route_decisions"] == 2246
    assert result["summary"]["action_equivalence"] == 1.0
    assert result["summary"]["divergence_episode_count"] == 0
    assert all(result["gates"].values())


def test_paper_table_reads_the_six_authoritative_formal_results() -> None:
    rows = formal_success_rows()
    assert {
        name: (
            row["tasks"], row["neural_correct"], row["source_correct"],
            row["delta_correct"],
        )
        for name, row in rows.items()
    } == {
        "webshop": (32, 7, 23, 16),
        "alfworld": (24, 13, 20, 7),
        "discoveryworld": (12, 3, 12, 9),
        "tirbench": (18, 6, 12, 6),
        "clevrer": (360, 236, 252, 16),
        "agqa2": (900, 249, 290, 41),
    }
    assert sum(row["tasks"] for row in rows.values()) == 1346
    assert sum(row["delta_correct"] for row in rows.values()) == 95


def test_paper_artifact_write_is_idempotent_but_rejects_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "paper.md"
    _write_if_absent_or_identical(output, "frozen result\n")
    _write_if_absent_or_identical(output, "frozen result\n")
    assert output.read_text(encoding="utf-8") == "frozen result\n"

    try:
        _write_if_absent_or_identical(output, "different result\n")
    except FileExistsError:
        pass
    else:
        raise AssertionError("divergent paper artifact was overwritten")
