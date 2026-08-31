#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SYSTEM = (
    "You are a selective adaptation Harness. Use only the supplied observable "
    "evidence. Do not invent predicates, hidden state, domain mappings, or actions. "
    "When evidence is missing, return the specified abstention target. Return JSON only."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format structured receipt supervision as prompt/completion SFT"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-game-id-hash",
        action="store_true",
        help=(
            "Expose a stable source-game hash. Disabled by default to prevent "
            "the Harness from learning a game-identity shortcut."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    streams = {
        split: (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "source_held_out")
    }
    counts = {split: 0 for split in streams}
    try:
        with args.dataset.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                split = str(row["split"])
                game_identity = (
                    "GAME_ID_HASH="
                    + hashlib.sha256(str(row["game"]).encode()).hexdigest()
                    + "\n"
                    if args.include_game_id_hash else ""
                )
                prompt = (
                    f"{SYSTEM}\n\nOBJECTIVE={row['objective']}\n"
                    + game_identity
                    + "EVIDENCE_INPUT=\n"
                    + json.dumps(
                        row["input_payload"], sort_keys=True, ensure_ascii=False,
                    )
                    + "\nOUTPUT_JSON="
                )
                completion = json.dumps(
                    row["target_payload"], sort_keys=True, ensure_ascii=False,
                )
                streams[split].write(json.dumps({
                    "example_id": row["example_id"],
                    "objective": row["objective"],
                    "prompt": prompt,
                    "completion": completion,
                    "evidence_receipt_ids": row["evidence_receipt_ids"],
                }, sort_keys=True, ensure_ascii=False) + "\n")
                counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    files = {
        split: {
            "path": str((args.output_dir / f"{split}.jsonl").resolve()),
            "sha256": _sha256(args.output_dir / f"{split}.jsonl"),
            "examples": counts[split],
        }
        for split in streams
    }
    manifest = {
        "schema_version": 1,
        "source_dataset": str(args.dataset.resolve()),
        "source_dataset_sha256": _sha256(args.dataset),
        "files": files,
        "prompt_policy": {
            "target_action_authority": False,
            "human_predicates": False,
            "source_target_mapping": False,
            "game_identity_exposed": args.include_game_id_hash,
            "json_only": True,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
