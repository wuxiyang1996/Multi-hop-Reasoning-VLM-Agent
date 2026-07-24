#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.control_priors import (  # noqa: E402
    compile_weak_prior_controls,
    knowledge_from_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile matched weak control-prior treatments from two qualified artifacts."
    )
    parser.add_argument("--authentic", type=Path, required=True)
    parser.add_argument("--other-game", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    authentic = knowledge_from_mapping(
        json.loads(args.authentic.read_text(encoding="utf-8"))
    )
    other = knowledge_from_mapping(
        json.loads(args.other_game.read_text(encoding="utf-8"))
    )
    treatments = compile_weak_prior_controls(authentic, other)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for condition, payload in treatments.items():
        path = args.output_dir / f"{condition}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "status": "COMPILED",
        "transfer_object": "WEAK_CONTROL_PRIOR",
        "conditions": sorted(treatments),
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
