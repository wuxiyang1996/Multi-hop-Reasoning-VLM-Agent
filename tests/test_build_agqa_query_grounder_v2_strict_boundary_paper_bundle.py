import json
from pathlib import Path

from scripts.build_agqa_query_grounder_v2_strict_boundary_paper_bundle import (
    _sha256,
    build_bundle,
)


def test_bundle_preserves_failed_status(tmp_path: Path):
    protocol = {
        "claim": "claim", "claim_boundary": {}, "shared_arm_contract": {},
        "arms": ["neural_only", "source_induced"],
    }
    values = {
        "protocol": protocol,
        "manifest": {"status": "AGQA_QUERY_GROUNDER_V2_STRICT_BOUNDARY_FRESH_FORMAL_FROZEN"},
        "qualification": {
            "status": "QUERY_GROUNDER_V2_STRICT_BOUNDARY_QUALIFIED",
            "qualification_rows": 2, "metrics": {}, "preoutcome_metrics": {},
            "gates": {"pass": True}, "report_sha256": "q",
        },
        "preoutcome": {"status": "ALL_FIVE_ARM_STRICT_BOUNDARY_DECISIONS_FROZEN_BEFORE_FORMAL_OUTCOMES"},
        "cost": {"provider_cost_usd": 0.0},
    }
    paths = {}
    for key, value in values.items():
        path = tmp_path / f"{key}.json"; path.write_text(json.dumps(value)); paths[key] = path
    formal = {
        "status": "AGQA_QUERY_GROUNDER_V2_STRICT_BOUNDARY_FRESH_FORMAL_GATES_FAILED",
        "claim_scope": "controlled", "protocol_file_sha256": _sha256(paths["protocol"]),
        "manifest_file_sha256": _sha256(paths["manifest"]),
        "preoutcome_file_sha256": _sha256(paths["preoutcome"]),
        "rows": [{}, {}],
        "summaries": {
            "neural_only": {"correct": 1, "total": 2, "accuracy": .5, "symbolic_commits": 0},
            "source_induced": {"correct": 1, "total": 2, "accuracy": .5, "symbolic_commits": 1},
        },
        "comparisons": {}, "gates": {"source_beats_neural": False},
        "secondary_target": {}, "failure_taxonomy": {}, "ablations": {},
        "report_sha256": "f",
    }
    path = tmp_path / "formal.json"; path.write_text(json.dumps(formal)); paths["formal"] = path
    bundle = build_bundle(paths)
    assert bundle["status"].endswith("GATES_FAILED")
    assert not bundle["formal"]["gates"]["source_beats_neural"]
