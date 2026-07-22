#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash the exact source files used by Phase-1 collection")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--clean-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--submitted-batch-script")
    args = parser.parse_args()
    source = Path(args.source_repo).resolve()
    clean = Path(args.clean_repo).resolve()
    paths = [
        source / "trainer/coevolution/episode_runner.py",
        source / "trainer/coevolution/vllm_client.py",
        source / "harness/agent_reasoning_cycle.py",
        source / "scripts/run_instrumented_source_smoke.py",
        clean / "patches/source_policy_receipts.patch",
        clean / "cluster/collect_phase1_complete.sbatch",
    ]
    if args.submitted_batch_script:
        paths.append(Path(args.submitted_batch_script).resolve())
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing provenance inputs: {missing}")
    payload = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_limit": (
            "Hashes prove file identity at receipt time; for jobs started earlier, "
            "they do not prove that files were unchanged before this receipt."
        ),
        "files": {
            str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
