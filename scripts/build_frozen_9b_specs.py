#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from motif_transfer.model_specs import build_coevolved_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen co-evolved 9B adapter identities")
    parser.add_argument("checkpoint_root")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = build_coevolved_specs(args.checkpoint_root)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
