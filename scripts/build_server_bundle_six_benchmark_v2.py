#!/usr/bin/env python3
"""Build a portable, dependency-complete six-benchmark server bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/server_bundle_six_benchmark_v2.json"


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


def _files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file())


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _write_git_archive(repo: Path, output: Path, prefix: str) -> None:
    git = subprocess.Popen(
        ["git", "archive", "--format=tar", f"--prefix={prefix}/", "HEAD"],
        cwd=repo,
        stdout=subprocess.PIPE,
    )
    assert git.stdout is not None
    zstd = subprocess.run(
        ["zstd", "-T0", "-19", "-o", str(output)],
        stdin=git.stdout,
        check=True,
    )
    git.stdout.close()
    git_status = git.wait()
    if git_status != 0 or zstd.returncode != 0:
        raise subprocess.CalledProcessError(git_status, "git archive")


def build(
    *, repo: Path, config_path: Path, base_bundle: Path, output_dir: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    base_bundle = base_bundle.resolve()
    config = _read(config_path)
    if config.get("status") != "PORTABLE_DEPENDENCY_CLOSURE_DECLARED":
        raise ValueError("bundle config is not frozen")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    roots = [repo / value for value in config["dependency_roots"]]
    missing = [str(path) for path in roots if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dependency roots: {missing}")
    reused = [base_bundle / value for value in config["archives_reused_from_v1"]]
    missing_reused = [str(path) for path in reused if not path.is_file()]
    if missing_reused:
        raise FileNotFoundError(f"missing reusable archives: {missing_reused}")

    output_dir.mkdir(parents=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True,
    ).strip()
    if not branch:
        raise ValueError("bundle must be built from a named branch")

    repository_bundle = output_dir / "00a-two-agent-clean-repository.bundle"
    _run(["git", "bundle", "create", str(repository_bundle), branch], cwd=repo)
    code_archive = output_dir / f"01-code-two-agent-clean-{head[:7]}.tar.zst"
    _write_git_archive(repo, code_archive, repo.name)
    for source in reused:
        _link_or_copy(source, output_dir / source.name)

    dependency_archive = output_dir / config["dependency_archive"]
    archive_members = [f"{repo.name}/{path.relative_to(repo)}" for path in roots]
    _run([
        "tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0",
        "--numeric-owner", "--zstd", "-cf", str(dependency_archive),
        "-C", str(repo.parent), *archive_members,
    ])

    dependency_files = []
    for root in roots:
        for path in _files(root):
            dependency_files.append({
                "path": str(path.relative_to(repo)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    artifact_manifest = {
        "schema_version": "server-bundle-artifact-manifest-v2",
        "status": "DEPENDENCY_CLOSURE_PACKED",
        "git": {"branch": branch, "head": head},
        "config": str(config_path.relative_to(repo)),
        "config_sha256": _sha256(config_path),
        "dependency_archive": dependency_archive.name,
        "dependency_files": dependency_files,
        "claim_boundary": config["claim_boundary"],
    }
    (output_dir / "ARTIFACTS.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Portable six-benchmark server bundle V2

Git commit: `{head}`

This package repairs the missing dependency closure in the 2026-09-03 V1
package.  It reproduces the historical frozen-cohort Qwen3.5-9B controller
substitution and native-action equivalence result.  It does **not** run the
full official benchmark sizes.

Verify and extract:

```bash
sha256sum -c SHA256SUMS
git clone -b {branch} 00a-two-agent-clean-repository.bundle workspace/{repo.name}
git clone -b agent/agent-native-skill-harness-v2 00b-source-runtime-repository.bundle workspace/Multi-hop-Reasoning-VLM-Agent-github-main
for f in 10-harness-source-only-core.tar.zst 11-target-adapted-baseline.tar.zst 13-six-benchmark-portable-dependencies.tar.zst; do
  tar --zstd -xf "$f" -C workspace
done
python workspace/{repo.name}/scripts/verify_server_bundle_six_benchmark_v2.py --workspace workspace --package .
```

Historical absolute paths retained inside signed receipts are provenance only.
Runtime readers remap repository-anchored paths into the active checkout.
See `ARTIFACTS.json` for the complete content-addressed dependency inventory.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    checksum_paths = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return {
        "status": "PORTABLE_SERVER_BUNDLE_V2_BUILT",
        "output": str(output_dir),
        "git_head": head,
        "dependency_files": len(dependency_files),
        "archives": len(checksum_paths),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        repo=args.repo, config_path=args.config.resolve(),
        base_bundle=args.base_bundle, output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
