from pathlib import Path

from motif_transfer.vtb_capabilities import (
    OFFICIAL_REQUIRED_MODULES,
    OFFICIAL_TOOL_NAMES,
    audit_vtb_runtime,
)
from motif_transfer.vtb_evaluator import OFFICIAL_COMMIT


def _checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "official"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text(OFFICIAL_COMMIT + "\n")
    functions = "\n".join(
        f"def {'safe_calculator' if name == 'calculator' else name}(*args, **kwargs): pass"
        for name in OFFICIAL_TOOL_NAMES
    )
    (scripts / "tools.py").write_text(functions + "\n")
    (scripts / "model_inference.py").write_text("# pinned runner\n")
    return repo


def test_full_runtime_requires_all_tools_modules_and_external_keys(tmp_path: Path) -> None:
    repo = _checkout(tmp_path)
    modules = {name: True for name in OFFICIAL_REQUIRED_MODULES}
    audit = audit_vtb_runtime(
        repo,
        key_presence={"SERP_API_KEY": True, "OPENWEATHER_API_KEY": True},
        module_availability=modules,
    )
    assert audit.official_inference_ready
    assert audit.paper_faithful_full_tool_ready
    assert audit.observed_tools == OFFICIAL_TOOL_NAMES


def test_missing_external_keys_fails_full_tool_cell_but_not_local_imports(tmp_path: Path) -> None:
    repo = _checkout(tmp_path)
    modules = {name: True for name in OFFICIAL_REQUIRED_MODULES}
    audit = audit_vtb_runtime(repo, key_presence={}, module_availability=modules)
    assert audit.official_inference_ready
    assert not audit.paper_faithful_full_tool_ready
    assert set(audit.blockers) == {
        "missing external capability key: SERP_API_KEY",
        "missing external capability key: OPENWEATHER_API_KEY",
    }


def test_unknown_checkout_or_missing_tool_fails_closed(tmp_path: Path) -> None:
    repo = _checkout(tmp_path)
    (repo / ".git" / "HEAD").write_text("bad-commit\n")
    tools = repo / "scripts" / "tools.py"
    tools.write_text(tools.read_text().replace("def google_search", "def absent_google_search"))
    modules = {name: True for name in OFFICIAL_REQUIRED_MODULES}
    audit = audit_vtb_runtime(
        repo,
        key_presence={"SERP_API_KEY": True, "OPENWEATHER_API_KEY": True},
        module_availability=modules,
    )
    assert not audit.official_inference_ready
    assert "google_search" in audit.missing_tools
    assert any("commit mismatch" in item for item in audit.blockers)
