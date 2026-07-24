#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.vtb_capabilities import OFFICIAL_REQUIRED_KEYS, audit_vtb_runtime


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_row(parquet: Path, sample_id: str) -> dict[str, Any]:
    import duckdb

    if not sample_id.startswith("row:"):
        raise ValueError("v2 uses frozen row:<zero-based index> IDs")
    index = int(sample_id.split(":", 1)[1])
    connection = duckdb.connect()
    columns = [row[0] for row in connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [str(parquet)]
    ).fetchall()]
    row = connection.execute(
        "SELECT * FROM read_parquet(?) LIMIT 1 OFFSET ?", [str(parquet), index]
    ).fetchone()
    if row is None:
        raise ValueError(f"sample does not exist: {sample_id}")
    return dict(zip(columns, row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned official VTB single-turn inference wrapper.")
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "configs/vtb_single_turn_manifest_v2.json")
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--model", default="openrouter/qwen/qwen3.5-35b-a3b")
    parser.add_argument("--max-tool-rounds", type=int, default=20)
    parser.add_argument("--allow-degraded-adaptation", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    allowed = {manifest["adaptation_id"], *manifest["test_ids"]}
    if args.sample_id not in allowed:
        raise SystemExit("sample is outside the frozen v2 manifest")
    keys = runpy.run_path(str(args.keys))
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", *OFFICIAL_REQUIRED_KEYS):
        if keys.get(name) and not os.environ.get(name):
            os.environ[name] = str(keys[name])
    audit = audit_vtb_runtime(
        args.official_repo.resolve(),
        key_presence={name: bool(os.environ.get(name)) for name in OFFICIAL_REQUIRED_KEYS},
    )
    degraded = not audit.paper_faithful_full_tool_ready
    if degraded and not (
        args.allow_degraded_adaptation and args.sample_id == manifest["adaptation_id"]
    ):
        raise SystemExit(
            "NOT_RUNNABLE: official full-tool preflight failed; degraded execution is restricted "
            "to the frozen adaptation item"
        )
    if not audit.official_inference_ready:
        raise SystemExit("NOT_RUNNABLE: official inference imports or checkout are invalid")
    if args.max_tool_rounds != int(manifest["official_tool_call_cap"]):
        raise SystemExit("matched v2 experiment must use the frozen official cap")

    record = _load_row(args.parquet, args.sample_id)
    prompts = list(record.get("turn_prompts") or [])
    if str(record.get("turncase")) != "single-turn" or len(prompts) != 1:
        raise SystemExit("NOT_RUNNABLE: v2 wrapper accepts exactly one official single-turn item")
    images = list(record.get("images") or [])
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    image_hashes = []
    for index, image in enumerate(images):
        raw = image.get("bytes") if isinstance(image, dict) else None
        if not raw:
            raise SystemExit(f"NOT_RUNNABLE: image {index} has no inline bytes")
        suffix = Path(str(image.get("path") or "image.png")).suffix or ".png"
        path = args.asset_dir / f"input_{index}{suffix}"
        path.write_bytes(raw)
        image_paths.append(str(path.resolve()))
        image_hashes.append(_sha(raw))

    official_scripts = args.official_repo.resolve() / "scripts"
    sys.path.insert(0, str(official_scripts))
    from model_inference import FunC_with_tools

    answer, official_round_counter, tool_trace, content_trace = FunC_with_tools(
        question_id=args.sample_id,
        prompt=str(prompts[0]),
        image_list=image_paths,
        max_tool_calls=args.max_tool_rounds,
        model_name=args.model,
        tool_observation_save_path=str(args.asset_dir.resolve()),
        system_prompt_level="high",
    )
    hit_cap = str(answer or "").strip() == "Model hit the maximum number of tool calls"
    # The pinned implementation increments its counter once per model round,
    # while one round may contain multiple function calls. Preserve both values.
    payload = {
        "schema_version": 2,
        "executor": "pinned_visualtoolbench_official_model_inference",
        "sample_id": args.sample_id,
        "split": "adaptation" if args.sample_id == manifest["adaptation_id"] else "test",
        "condition": "target_only",
        "claim_label": "CAPABILITY_DEGRADED_ADAPTATION_DIAGNOSTIC" if degraded else "TARGET_ONLY",
        "model": args.model,
        "official_commit": audit.expected_commit,
        "tool_contract_sha256": audit.tool_contract_sha256,
        "paper_faithful_full_tool_ready": audit.paper_faithful_full_tool_ready,
        "cap_semantics": "pinned implementation counts model tool rounds; a round may contain multiple function calls",
        "max_tool_rounds": args.max_tool_rounds,
        "official_round_counter": official_round_counter,
        "executed_function_calls": len(tool_trace),
        "termination_reason": "OFFICIAL_CAP_EXHAUSTED" if hit_cap else "MODEL_FINAL_ANSWER",
        "final_answer_present": bool(str(answer or "").strip()) and not hit_cap,
        "image_sha256": image_hashes,
        "answer": answer,
        "tool_trace": tool_trace,
        "content_trace": content_trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_id": args.sample_id,
        "claim_label": payload["claim_label"],
        "official_round_counter": official_round_counter,
        "executed_function_calls": len(tool_trace),
        "answer_present": payload["final_answer_present"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
