from pathlib import Path

from motif_transfer.contracts import stable_hash
from motif_transfer.direct_prospective_matrix_v1 import (
    SOURCE_GAMES,
    TARGET_DOMAINS,
    audit_cell_receipts,
    make_cell_execution_receipt,
)


def _decision(source_hash: str, index: int = 0) -> dict:
    body = {
        "source_artifact_sha256": source_hash,
        "source_action": "EXPLORE_UNTRIED",
        "admitted": True,
        "decision_index": index,
    }
    return body | {"receipt_sha256": stable_hash(body)}


def _manifest() -> dict:
    cells = []
    sources = {}
    for source_index, game in enumerate(SOURCE_GAMES):
        source_hash = f"{source_index + 1:064x}"
        sources[game] = {"artifact_sha256": source_hash}
        for domain in TARGET_DOMAINS:
            cells.append({
                "cell_id": f"{game}__to__{domain}",
                "source_game": game,
                "target_domain": domain,
                "target_task_id": f"{domain}.{game}",
                "source_artifact_sha256": source_hash,
            })
    body = {"sources": sources, "cells": cells}
    return body | {"manifest_sha256": stable_hash(body)}


def test_cell_receipt_requires_online_route_and_matched_state():
    manifest = _manifest()
    cell = manifest["cells"][0]
    conditions = ("raw", "authentic", "control")
    receipt = make_cell_execution_receipt(
        manifest_sha256=manifest["manifest_sha256"],
        cell=cell,
        source_artifact_sha256=cell["source_artifact_sha256"],
        conditions_executed=conditions,
        expected_conditions=conditions,
        target_initial_state_hashes=["state"] * 3,
        authentic_source_decisions=[_decision(cell["source_artifact_sha256"])],
        target_native_grounding_used=True,
        target_reset_or_sample_open_count=1,
    )
    assert receipt["status"] == "DIRECT_PROSPECTIVE_CELL_PASSED"
    assert all(receipt["gates"].values())

    failed = make_cell_execution_receipt(
        manifest_sha256=manifest["manifest_sha256"],
        cell=cell,
        source_artifact_sha256=cell["source_artifact_sha256"],
        conditions_executed=conditions,
        expected_conditions=conditions,
        target_initial_state_hashes=["state"] * 3,
        authentic_source_decisions=[],
        target_native_grounding_used=True,
        target_reset_or_sample_open_count=1,
    )
    assert failed["status"] == "DIRECT_PROSPECTIVE_CELL_FAILED"
    assert not failed["gates"]["source_routed_online"]


def test_audit_needs_24_unique_execution_fingerprints():
    manifest = _manifest()
    receipts = []
    for index, cell in enumerate(manifest["cells"]):
        receipts.append(make_cell_execution_receipt(
            manifest_sha256=manifest["manifest_sha256"],
            cell=cell,
            source_artifact_sha256=cell["source_artifact_sha256"],
            conditions_executed=("raw", "authentic"),
            expected_conditions=("raw", "authentic"),
            target_initial_state_hashes=["same", "same"],
            authentic_source_decisions=[
                _decision(cell["source_artifact_sha256"], index)
            ],
            target_native_grounding_used=True,
            target_reset_or_sample_open_count=1,
        ))
    report = audit_cell_receipts(manifest, receipts)
    assert report["status"] == "DIRECT_PROSPECTIVE_24_OF_24_VALIDATED"
    assert report["passed_cells"] == 24

    duplicate = list(receipts)
    copied = dict(duplicate[-1])
    copied["authentic_source_decision_receipt_sha256"] = duplicate[-2][
        "authentic_source_decision_receipt_sha256"
    ]
    body = dict(copied)
    body.pop("cell_receipt_sha256")
    copied["cell_receipt_sha256"] = stable_hash(body)
    duplicate[-1] = copied
    # The different source+target identity still makes this execution distinct;
    # receipt copying alone cannot erase the manifest-level identity binding.
    assert audit_cell_receipts(manifest, duplicate)["passed_cells"] == 24
