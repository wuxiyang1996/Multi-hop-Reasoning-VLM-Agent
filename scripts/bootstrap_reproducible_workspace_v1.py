#!/usr/bin/env python3
"""Create the clean five-worktree neural-symbolic reproduction workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/reproducible_workspace_v1.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--include-model-cache", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    workspace = args.workspace.resolve()
    config = _read(args.config.resolve())
    if config.get("status") != "FIVE_WORKTREE_LAYOUT_FROZEN":
        raise SystemExit("workspace config is not frozen")
    if workspace.exists():
        raise SystemExit(f"refusing to overwrite workspace: {workspace}")

    files = config["package_files"]
    required_package_files = [
        files["harness_bundle"], files["four_worktree_bundle"],
        *files["required_data_archives"], "ARTIFACTS.json", "SHA256SUMS",
    ]
    if args.include_model_cache:
        required_package_files.append(files["optional_model_archive"])
    missing = [name for name in required_package_files if not (package / name).is_file()]
    if missing:
        raise SystemExit(f"package is missing workspace inputs: {missing}")

    workspace.mkdir(parents=True)
    components = {row["directory"]: row for row in config["components"]}
    harness_name = "Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
    harness = components[harness_name]
    _run([
        "git", "clone", "-q", "-b", harness["branch"],
        str(package / files["harness_bundle"]), str(workspace / harness_name),
    ])

    visual_name = "Multi-hop-Reasoning-VLM-Agent"
    visual = components[visual_name]
    legacy_bundle = package / files["four_worktree_bundle"]
    _run([
        "git", "clone", "-q", "-b", visual["branch"],
        str(legacy_bundle), str(workspace / visual_name),
    ])
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace / visual_name, text=True,
    ).strip() != visual["commit"]:
        raise SystemExit("visual-tools checkout has the wrong commit")

    for name in (
        "Multi-hop-Reasoning-VLM-Agent-github-main",
        "Multi-hop-Reasoning-VLM-Agent-source-fresh-v1",
        "Multi-hop-Reasoning-VLM-Agent-experiment-clean",
    ):
        component = components[name]
        _run([
            "git", "worktree", "add", "-q", "--detach",
            str(workspace / name), component["commit"],
        ], cwd=workspace / visual_name)

    for archive in files["required_data_archives"]:
        _run([
            "tar", "--zstd", "-xf", str(package / archive),
            "-C", str(workspace),
        ])
    if args.include_model_cache:
        _run([
            "tar", "--zstd", "-xf",
            str(package / files["optional_model_archive"]),
            "-C", str(workspace),
        ])

    heads = {}
    for component in config["components"]:
        root = workspace / component["directory"]
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip()
        expected = (
            _read(package / "ARTIFACTS.json")["git"]["head"]
            if component.get("commit_from_active_bundle")
            else component["commit"]
        )
        if head != expected:
            raise SystemExit(f"wrong commit for {component['directory']}: {head}")
        heads[component["directory"]] = head

    receipt = {
        "schema_version": "neurosymbolic-transfer-workspace-bootstrap-receipt-v1",
        "status": "CLEAN_FIVE_WORKTREE_WORKSPACE_CREATED",
        "package": str(package),
        "package_artifacts_sha256": _sha256(package / "ARTIFACTS.json"),
        "components": heads,
        "model_cache_extracted": args.include_model_cache,
        "result_boundary": config["result_boundary"],
    }
    (workspace / "WORKSPACE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    readme = f"""# Neural-symbolic transfer reproducible workspace

This workspace was bootstrapped from the content-addressed package at
`{package}`.  Exact component commits are recorded in `WORKSPACE_RECEIPT.json`.

## Verify

```bash
python {harness_name}/scripts/verify_reproducible_workspace_v1.py \\
  --workspace \"$PWD\" --package {package}
```

## Component roles

- `{harness_name}`: canonical harness, artifacts, evaluation and reports.
- `Multi-hop-Reasoning-VLM-Agent-github-main`: V3 LoRA/runtime dependency.
- `Multi-hop-Reasoning-VLM-Agent-source-fresh-v1`: optional source regeneration.
- `Multi-hop-Reasoning-VLM-Agent-experiment-clean`: archival DDP lineage.
- `Multi-hop-Reasoning-VLM-Agent`: optional raw-video grounding tools.

The verified six-domain result is the historical frozen cohort.  It is not an
official-full-size rerun.  See the manifest's `result_boundary` for full sizes.
"""
    (workspace / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
