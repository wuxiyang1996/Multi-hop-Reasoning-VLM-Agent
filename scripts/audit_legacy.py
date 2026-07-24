#!/usr/bin/env python3
import argparse
import json

from motif_transfer.legacy_import import audit_legacy, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a legacy skill/mega-skill JSONL as lineage only")
    parser.add_argument("jsonl")
    args = parser.parse_args()
    print(json.dumps(audit_legacy(load_jsonl(args.jsonl)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
