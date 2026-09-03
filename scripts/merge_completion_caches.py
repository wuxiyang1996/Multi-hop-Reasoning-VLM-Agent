#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_caches(paths: list[Path]) -> dict:
    payloads = [json.loads(path.read_text()) for path in paths]
    identities = {row["backend_identity_sha256"] for row in payloads}
    if len(identities) != 1:
        raise ValueError("completion caches have different backend identities")
    entries = {}
    for path, payload in zip(paths, payloads, strict=True):
        for key, value in payload.get("entries", {}).items():
            if key in entries and entries[key] != value:
                raise ValueError(f"conflicting completion for {key} in {path}")
            entries[key] = value
    return {
        "schema_version": 1,
        "backend_identity_sha256": next(iter(identities)),
        "entries": dict(sorted(entries.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_caches(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"entries": len(merged["entries"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
