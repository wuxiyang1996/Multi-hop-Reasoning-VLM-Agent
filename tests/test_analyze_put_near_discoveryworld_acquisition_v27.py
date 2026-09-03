from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/analyze_put_near_discoveryworld_acquisition_v27.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase15_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_second_family_audit_passes_all_gates() -> None:
    module = _load_module()
    report = module.run(
        REPO / "configs/put_near_discoveryworld_acquisition_v27.json"
    )

    assert report["status"] == (
        "SECOND_DISTINCT_PROGRAM_FAMILY_ACQUISITION_VALIDATED"
    )
    assert all(report["gates"].values())
    assert report["source"]["first_structurally_complete_budget"] == 2
    assert report["target"]["target_k0"]["status"].startswith("ABSTAIN")
    robustness = report["target"]["single_demo_robustness"]
    assert robustness["independent_single_demo_programs"] == 3
    assert robustness["all_contain_source_finite_subprogram"] is True
    assert robustness["all_precedence_permuted_support_zero"] is True


def test_finite_family_is_not_relabeled_recurrence() -> None:
    module = _load_module()
    report = module.run(
        REPO / "configs/put_near_discoveryworld_acquisition_v27.json"
    )
    distinction = report["family_distinction"]

    assert distinction["put_near_ir_kind"] == (
        "FINITE_STRUCTURAL_DELTA_SEQUENCE"
    )
    assert distinction["put_near_recurrent"] is False
    assert distinction["alfworld_recurrent_acquisition_control"] is True
    assert distinction["alfworld_recurrent_relation_update"] is True
    assert distinction["same_program_family"] is False
