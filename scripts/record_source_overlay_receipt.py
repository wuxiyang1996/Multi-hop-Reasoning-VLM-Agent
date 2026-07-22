#!/usr/bin/env python3
"""Hash the exact patched source runner used by one collection job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-runner", type=Path, required=True)
    parser.add_argument("--overlay-vllm-client", type=Path, required=True)
    parser.add_argument("--overlay-smoke-runner", type=Path, required=True)
    parser.add_argument("--policy-receipts-patch", type=Path, required=True)
    parser.add_argument("--request-seed-control-patch", type=Path, required=True)
    parser.add_argument("--shadow-observer-patch", type=Path)
    parser.add_argument("--no-human-hints-patch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    overlay_files = [
        args.overlay_runner,
        args.overlay_vllm_client,
        args.overlay_smoke_runner,
    ]
    patch_files = [
        args.policy_receipts_patch,
        args.request_seed_control_patch,
    ]
    if args.shadow_observer_patch is not None:
        patch_files.append(args.shadow_observer_patch)
    if args.no_human_hints_patch is not None:
        patch_files.append(args.no_human_hints_patch)
    inputs = overlay_files + patch_files
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError([str(path) for path in inputs if not path.is_file()])
    payload = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reasoning_observer": args.shadow_observer_patch is not None,
        "human_policy_hints_excluded": args.no_human_hints_patch is not None,
        "request_seed_control": True,
        "overlay_runner": {
            "path": str(args.overlay_runner),
            "sha256": _sha256(args.overlay_runner),
        },
        "overlay_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in overlay_files
        },
        "applied_patch_sha256": [_sha256(path) for path in patch_files],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
