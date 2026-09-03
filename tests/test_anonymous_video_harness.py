from __future__ import annotations

import copy
import json
from pathlib import Path

from motif_transfer.anonymous_video_harness import (
    compile_anonymous_source_controller,
    route_grounded_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
LINEAGES = Path("runs/phase3_source_induction_v1_development/lineages")


def test_compiler_uses_heldout_source_controls_and_no_target_data() -> None:
    artifact = compile_anonymous_source_controller(root=ROOT, lineage_directory=LINEAGES)
    assert artifact["status"] == "ANONYMOUS_SOURCE_VIDEO_HARNESS_QUALIFIED"
    assert artifact["target_data_read"] is False
    assert artifact["heldout_control"] == {
        "authentic_successes": 58,
        "effect_shuffled_successes": 0,
        "total_ledgers": 58,
    }
    assert len(artifact["lineage_receipts"]) == 6
    assert len(artifact["operators"]) == 3
    assert artifact["transitions"]


def test_alpha_renaming_supplied_operator_ids_cannot_change_artifact(tmp_path: Path) -> None:
    source = ROOT / LINEAGES
    renamed = tmp_path / "lineages"
    renamed.mkdir()
    for path in sorted(source.glob("*.json")):
        report = json.loads(path.read_text())
        program = report["authentic_program"]
        mapping = {
            row["operator_id"]: f"ARBITRARY_LABEL_{index}"
            for index, row in enumerate(reversed(program["operators"]))
        }
        for row in program["operators"]:
            row["operator_id"] = mapping[row["operator_id"]]
        for section in ("authentic", "authentic_closed_loop"):
            value = report["heldout"][section]
            if "operator_route_counts" in value:
                value["operator_route_counts"] = {
                    mapping[key]: count for key, count in value["operator_route_counts"].items()
                }
            for ledger in value.get("per_ledger", []):
                ledger["route_operator_ids"] = [mapping[key] for key in ledger["route_operator_ids"]]
        (renamed / path.name).write_text(json.dumps(report))

    authentic = compile_anonymous_source_controller(root=ROOT, lineage_directory=LINEAGES)
    alpha = compile_anonymous_source_controller(root=ROOT, lineage_directory=renamed)
    assert [row["operator_id"] for row in alpha["operators"]] == [
        row["operator_id"] for row in authentic["operators"]
    ]
    assert alpha["transitions"] == authentic["transitions"]
    assert alpha["heldout_control"] == authentic["heldout_control"]


def test_target_adapter_executes_anonymous_attempt_commit_or_fallback() -> None:
    artifact = compile_anonymous_source_controller(root=ROOT, lineage_directory=LINEAGES)
    committed = route_grounded_candidate(artifact, candidate_qualified=True)
    fallback = route_grounded_candidate(artifact, candidate_qualified=False)
    assert committed[-1] == "COMMIT"
    assert fallback[-1] == "FALLBACK"
    assert committed[0] == fallback[0]
    assert committed[1] != fallback[1]


def test_unqualified_controller_fails_closed() -> None:
    artifact = compile_anonymous_source_controller(root=ROOT, lineage_directory=LINEAGES)
    broken = copy.deepcopy(artifact)
    broken["status"] = "ANONYMOUS_SOURCE_VIDEO_HARNESS_ABSTAINED"
    assert route_grounded_candidate(broken, candidate_qualified=True) == ("ABSTAIN",)
