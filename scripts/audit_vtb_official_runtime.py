#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.vtb_capabilities import OFFICIAL_REQUIRED_KEYS, audit_vtb_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed audit of the pinned official VTB runtime.")
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--keys", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    key_file = runpy.run_path(str(args.keys)) if args.keys else {}
    presence = {
        name: bool(os.environ.get(name) or key_file.get(name)) for name in OFFICIAL_REQUIRED_KEYS
    }
    audit = audit_vtb_runtime(args.official_repo.resolve(), key_presence=presence)
    payload = {"schema_version": 1, "audit": audit.to_json()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "official_inference_ready": audit.official_inference_ready,
        "paper_faithful_full_tool_ready": audit.paper_faithful_full_tool_ready,
        "tool_contract_sha256": audit.tool_contract_sha256,
        "blockers": audit.blockers,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
