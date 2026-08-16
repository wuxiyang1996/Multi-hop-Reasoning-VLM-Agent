import copy

import pytest

from motif_transfer.contracts import stable_hash
from scripts.run_phase3_discoveryworld_formal_acquisition_v2 import (
    validate_formal_manifest,
)


def _manifest():
    body = {
        "status": "FROZEN_BEFORE_ANY_PHASE3_V2_TARGET_RESET_OR_OUTCOME",
        "role": "formal_reserve",
        "tasks": [
            {"task_id": f"proteomics.easy.seed{seed}", "seed": seed}
            for seed in range(121, 145)
        ],
        "structured_acquisition_qualification": {
            "status": "DISCOVERYWORLD_STRUCTURED_ACQUISITION_QUALIFICATION_PASSED",
            "gates": {"ready": True, "outcome_blind": True},
        },
        "formal_target_outcome_read_for_freeze": False,
        "formal_reserve_task_opened": False,
    }
    return body | {"manifest_sha256": stable_hash(body)}


def test_formal_acquisition_v2_requires_self_hashed_new_reserve_and_qualification():
    validate_formal_manifest(_manifest())
    broken = copy.deepcopy(_manifest())
    broken["tasks"][0]["seed"] = 120
    broken_body = dict(broken); broken_body.pop("manifest_sha256")
    broken["manifest_sha256"] = stable_hash(broken_body)
    with pytest.raises(ValueError, match="seeds121-144"):
        validate_formal_manifest(broken)

    failed = copy.deepcopy(_manifest())
    failed["structured_acquisition_qualification"]["gates"]["ready"] = False
    failed_body = dict(failed); failed_body.pop("manifest_sha256")
    failed["manifest_sha256"] = stable_hash(failed_body)
    with pytest.raises(ValueError, match="failed gate"):
        validate_formal_manifest(failed)
