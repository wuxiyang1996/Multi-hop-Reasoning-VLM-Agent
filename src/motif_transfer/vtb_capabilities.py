from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping

from .vtb_evaluator import OFFICIAL_COMMIT, OFFICIAL_REPOSITORY


OFFICIAL_TOOL_NAMES = (
    "python_image_processing",
    "python_interpreter",
    "google_search",
    "browser_get_page_text",
    "historical_weather",
    "calculator",
)

# These are runtime capabilities of the pinned official implementation, not a
# hand-authored task ontology. The two key names are read directly from tools.py.
OFFICIAL_REQUIRED_KEYS = ("SERP_API_KEY", "OPENWEATHER_API_KEY")
OFFICIAL_REQUIRED_MODULES = (
    "PIL",
    "bs4",
    "cv2",
    "litellm",
    "matplotlib",
    "numpy",
    "pandas",
    "requests",
    "scipy",
    "simpy",
    "sklearn",
    "tabulate",
    "yfinance",
)


@dataclass(frozen=True)
class VTBRuntimeAudit:
    official_repository: str
    expected_commit: str
    observed_commit: str
    commit_matches: bool
    tools_py_sha256: str
    tool_contract_sha256: str
    observed_tools: tuple[str, ...]
    missing_tools: tuple[str, ...]
    module_available: Mapping[str, bool]
    key_present: Mapping[str, bool]
    official_inference_ready: bool
    paper_faithful_full_tool_ready: bool
    blockers: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head(repo: Path) -> str:
    head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = repo / ".git" / head.removeprefix("ref: ")
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
        packed = repo / ".git" / "packed-refs"
        if packed.exists():
            suffix = head.removeprefix("ref: ")
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("#") and line.endswith(f" {suffix}"):
                    return line.split(" ", 1)[0]
        raise ValueError(f"cannot resolve git ref {head}")
    return head


def _official_functions(tools_py: Path) -> set[str]:
    tree = ast.parse(tools_py.read_text(encoding="utf-8"), filename=str(tools_py))
    functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
    # The public descriptor is calculator, while the implementation method is
    # safe_calculator at the pinned commit.
    if "safe_calculator" in functions:
        functions.add("calculator")
    return functions


def audit_vtb_runtime(
    official_repo: Path,
    *,
    key_presence: Mapping[str, bool],
    module_availability: Mapping[str, bool] | None = None,
) -> VTBRuntimeAudit:
    tools_py = official_repo / "scripts" / "tools.py"
    inference_py = official_repo / "scripts" / "model_inference.py"
    if not tools_py.is_file() or not inference_py.is_file():
        raise ValueError("official checkout must contain scripts/tools.py and scripts/model_inference.py")
    observed_commit = _git_head(official_repo)
    source = tools_py.read_bytes()
    functions = _official_functions(tools_py)
    observed_tools = tuple(name for name in OFFICIAL_TOOL_NAMES if name in functions)
    missing_tools = tuple(name for name in OFFICIAL_TOOL_NAMES if name not in functions)
    modules = dict(module_availability) if module_availability is not None else {
        name: importlib.util.find_spec(name) is not None for name in OFFICIAL_REQUIRED_MODULES
    }
    keys = {name: bool(key_presence.get(name, False)) for name in OFFICIAL_REQUIRED_KEYS}
    commit_matches = observed_commit == OFFICIAL_COMMIT
    missing_modules = tuple(name for name in OFFICIAL_REQUIRED_MODULES if not modules.get(name, False))
    missing_keys = tuple(name for name in OFFICIAL_REQUIRED_KEYS if not keys.get(name, False))
    blockers = []
    if not commit_matches:
        blockers.append(f"official commit mismatch: {observed_commit}")
    blockers.extend(f"missing official tool: {name}" for name in missing_tools)
    blockers.extend(f"missing Python module: {name}" for name in missing_modules)
    inference_ready = not blockers
    blockers.extend(f"missing external capability key: {name}" for name in missing_keys)
    full_tool_ready = not blockers
    contract = "\n".join((OFFICIAL_COMMIT, *OFFICIAL_TOOL_NAMES)).encode("utf-8") + source
    return VTBRuntimeAudit(
        official_repository=OFFICIAL_REPOSITORY,
        expected_commit=OFFICIAL_COMMIT,
        observed_commit=observed_commit,
        commit_matches=commit_matches,
        tools_py_sha256=_sha256(source),
        tool_contract_sha256=_sha256(contract),
        observed_tools=observed_tools,
        missing_tools=missing_tools,
        module_available=modules,
        key_present=keys,
        official_inference_ready=inference_ready,
        paper_faithful_full_tool_ready=full_tool_ready,
        blockers=tuple(blockers),
    )
