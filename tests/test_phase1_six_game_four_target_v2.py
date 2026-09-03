import json
from pathlib import Path

import pytest

from motif_transfer.contracts import stable_hash
from scripts.build_phase1_six_game_search_automaton_artifact import (
    _validate_collected_rows,
)


REPO = Path(__file__).resolve().parents[1]
SOURCES = {
    "tetris",
    "candy_crush",
    "gymv_columns",
    "gymv_streets_of_rage_2",
    "gymv_thunder_force_iii",
    "gymv_strider",
}
TARGETS = {"webshop", "alfworld", "discoveryworld", "tirbench"}


def _read(relative_path: str) -> dict:
    return json.loads((REPO / relative_path).read_text(encoding="utf-8"))


def _assert_self_hash(value: dict, field: str) -> None:
    body = dict(value)
    claimed = body.pop(field)
    assert claimed == stable_hash(body)


def _row(snapshot_id: str, candidate_rank: int) -> dict:
    body = {
        "game": "tetris",
        "snapshot_id": snapshot_id,
        "candidate_rank": candidate_rank,
    }
    return body | {"row_sha256": stable_hash(body)}


def test_builder_recomputes_row_hashes_and_preserves_multiplicity() -> None:
    rows = [_row("state-1", 0), _row("state-1", 1)]
    audit = {
        "snapshots": {
            "state-1": {
                "accepted_attempt_index": 0,
                "attempts": [{
                    "row_sha256s": [row["row_sha256"] for row in rows]
                }],
            }
        }
    }
    _validate_collected_rows(
        game="tetris",
        rows=rows,
        audit=audit,
        maximum_attempts_per_snapshot=1,
    )

    tampered = [rows[0] | {"candidate_rank": 9}, rows[1]]
    with pytest.raises(ValueError, match="self-hash"):
        _validate_collected_rows(
            game="tetris",
            rows=tampered,
            audit=audit,
            maximum_attempts_per_snapshot=1,
        )

    duplicated = [rows[0], rows[0], rows[1]]
    with pytest.raises(ValueError, match="accepted collection receipts"):
        _validate_collected_rows(
            game="tetris",
            rows=duplicated,
            audit=audit,
            maximum_attempts_per_snapshot=1,
        )


def test_six_source_artifact_is_target_authorized_and_action_free() -> None:
    artifact = _read(
        "runs/phase1_common_search_ir_formal_v1/"
        "common_search_automaton_artifact.json"
    )
    _assert_self_hash(artifact, "artifact_sha256")

    assert artifact["target_authorized"] is True
    assert {row["game"] for row in artifact["source_lineages"]} == SOURCES
    assert all(row["fresh_eligible_states"] >= 8 for row in artifact["source_lineages"])
    assert all(row["eligible_ledgers"] >= 8 for row in artifact["source_lineages"])
    serialized = json.dumps(artifact, sort_keys=True)
    assert "candidate_action" not in serialized
    assert "prefix_actions" not in serialized


def test_four_target_relineage_and_24_cell_audit_are_fail_closed() -> None:
    relineage = _read(
        "runs/phase1_common_search_ir_formal_v1/"
        "four_target_relineage_report.json"
    )
    audit = _read(
        "docs/results/phase1_six_game_four_target_transfer_audit_v2.json"
    )
    _assert_self_hash(relineage, "report_sha256")
    _assert_self_hash(audit, "report_sha256")

    assert set(relineage["domains"]) == TARGETS
    assert sum(
        domain["routed_decisions"] for domain in relineage["domains"].values()
    ) == 14_727
    assert all(
        domain["direct_new_target_execution"] is False
        and all(domain["gates"].values())
        for domain in relineage["domains"].values()
    )
    assert audit["status"] == "SIX_BY_FOUR_MECHANISM_TRANSFER_VALIDATED"
    assert set(audit["source_games"]) == SOURCES
    assert set(audit["target_domains"]) == TARGETS
    assert len(audit["cells"]) == 24
    assert audit["validated_mechanism_cells"] == 24
    assert audit["direct_new_joint_execution_cells"] == 0
    assert all(audit["gates"].values())
    assert all(
        cell["mechanism_transfer_validated"]
        and cell["direct_new_joint_source_target_execution"] is False
        for cell in audit["cells"]
    )
