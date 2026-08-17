from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/audit_phase8_four_domain_unified_neurosymbolic_v2.py"


def test_phase8_four_domain_audit_passes() -> None:
    spec = importlib.util.spec_from_file_location("phase8_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report()
    assert report["selected_route_count"] == 4
    assert len(report["route_results"]) == 4
    assert all(report["gates"].values())
    assert report["alfworld_v13_integrity"][
        "authority_receipt_hashes_verified"
    ] == 291


def test_phase8_audit_reads_gzip_fallback(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("phase8_audit_gzip", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw_path = tmp_path / "report.json"
    payload = b'{"status":"ok"}\n'
    Path(str(raw_path) + ".gz").write_bytes(gzip.compress(payload, mtime=0))
    assert module._bytes(raw_path) == payload
