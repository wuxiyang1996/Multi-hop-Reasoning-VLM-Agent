from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_four_domain_release_v1 import audit_release
from scripts.materialize_four_domain_release_v1 import materialize


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs/four_domain_neurosymbolic_release_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_portable_release_audit_validates_all_routes_and_bundle() -> None:
    report = audit_release(MANIFEST)
    assert report["status"] == (
        "PORTABLE_FOUR_DOMAIN_AUDIT_AND_ALFWORLD_BUNDLE_VALIDATED"
    )
    assert report["domains"] == ["webshop", "alfworld", "discoveryworld", "tir"]
    assert report["positive_dispatches"] == 4
    assert report["negative_abstentions"] == 3
    assert report["target_native_action_authority"]
    assert report["alfworld_full_artifact_bundle_present"]


def test_materializer_uses_external_resource_paths_without_changing_artifacts(
    tmp_path: Path,
) -> None:
    alfworld_config = tmp_path / "alfworld_base_config.yaml"
    alfworld_config.write_text("logic_type: pddl\n", encoding="utf-8")
    alfworld_data = tmp_path / "alfworld_data"
    (alfworld_data / "json_2.1.1" / "valid_unseen").mkdir(parents=True)
    output = tmp_path / "release"
    result = materialize(
        manifest_path=MANIFEST,
        output_dir=output,
        alfworld_config=alfworld_config,
        alfworld_data=alfworld_data,
    )
    assert result["status"] == "PORTABLE_ALFWORLD_REPLAY_MATERIALIZED"
    config = json.loads(Path(result["replay_config"]).read_text(encoding="utf-8"))
    assert config["target"]["alfworld_config"] == str(alfworld_config.resolve())
    assert config["target"]["alfworld_data"] == str(alfworld_data.resolve())
    assert config["portable_reproduction"]["scientific_parameters_changed"] is False
    expected = {
        row["role"]: row["uncompressed_sha256"]
        for row in json.loads(MANIFEST.read_text())["bundled_artifacts"]
    }
    assert _sha256(output / "alfworld_candidate.json") == expected[
        "alfworld_frozen_candidate"
    ]
    assert _sha256(output / "alfworld_development_report.json") == expected[
        "alfworld_development_report"
    ]
    assert _sha256(output / "alfworld_reference_final_report.json") == expected[
        "alfworld_final_report"
    ]


def test_release_manifest_contains_no_machine_local_paths() -> None:
    payload = MANIFEST.read_text(encoding="utf-8")
    assert "/fs/" not in payload
    assert "/nfshomes/" not in payload
