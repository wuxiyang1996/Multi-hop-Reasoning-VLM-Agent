#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.vtb_treatments import compile_vtb_treatments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile five frozen VTB treatment artifacts.")
    parser.add_argument("--authentic-bundle", type=Path, required=True)
    parser.add_argument("--other-game-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    authentic = json.loads(args.authentic_bundle.read_text(encoding="utf-8"))
    other = json.loads(args.other_game_bundle.read_text(encoding="utf-8"))
    treatments = compile_vtb_treatments(authentic, other)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for condition, payload in treatments.items():
        destination = args.output_dir / f"{condition}.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "COMPILED",
        "conditions": sorted(treatments),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
