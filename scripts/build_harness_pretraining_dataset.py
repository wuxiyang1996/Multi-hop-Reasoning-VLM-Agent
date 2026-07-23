#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

from motif_transfer.harness_training import (
    build_harness_training_examples,
    summarize_harness_training_examples,
)
from motif_transfer.instrumented_import import import_native_source_batch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build receipt-derived source-only Harness pretraining data"
    )
    parser.add_argument("evidence_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    episodes = []
    inputs = []
    for evidence_dir in args.evidence_dirs:
        imported = import_native_source_batch(evidence_dir)
        episodes.extend(imported)
        inputs.append({
            "evidence_dir": str(evidence_dir.resolve()),
            "episodes": len(imported),
            "manifest_sha256": _sha256(evidence_dir / "manifest.json"),
            "events_sha256": _sha256(evidence_dir / "events.jsonl"),
            "episodes_sha256": _sha256(evidence_dir / "episodes.jsonl"),
        })
    examples = build_harness_training_examples(episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(
                asdict(example), sort_keys=True, ensure_ascii=False,
            ) + "\n")
    manifest = {
        "schema_version": 1,
        "authority": "SOURCE_RECEIPT_DERIVED_SUPERVISION_ONLY",
        "inputs": inputs,
        "dataset_file": str(args.output.resolve()),
        "dataset_sha256": _sha256(args.output),
        "summary": summarize_harness_training_examples(examples),
        "claim_boundary": (
            "This dataset trains transition prediction, recorded-lineage "
            "recognition and abstention. It does not establish far-domain transfer."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
