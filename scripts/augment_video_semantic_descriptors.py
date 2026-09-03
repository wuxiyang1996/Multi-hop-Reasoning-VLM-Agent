#!/usr/bin/env python3
"""Outcome-blind semantic descriptor backfill for adaptation receipts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import runpy
import sys

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from motif_transfer.active_video_transfer import stable_hash  # noqa: E402
from run_active_video_wrapper_transfer import (  # noqa: E402
    _questions,
    _semantic_candidate_embeddings,
    _semantic_candidate_text,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    receipts = json.loads(args.receipts.read_text(encoding="utf-8"))
    expected = list(map(str, config["splits"]["adaptation"]))
    observed = [str(receipt["sample_id"]) for receipt in receipts]
    if observed != expected:
        raise SystemExit("receipt IDs/order do not match frozen adaptation IDs")
    questions = _questions(args.dataset_root, "train")
    embedding_config = config["semantic_embedding"]
    keys = runpy.run_path(str(args.keys))
    key_name = str(embedding_config["api_key_name"])
    api_key = keys.get(key_name)
    if not api_key:
        raise SystemExit(f"{key_name} is missing")

    texts: list[str] = []
    references: list[tuple[int, int]] = []
    output = copy.deepcopy(receipts)
    for receipt_index, receipt in enumerate(output):
        row = questions[str(receipt["sample_id"])]
        for candidate_index, candidate in enumerate(receipt["candidates"]):
            proposal = {
                "arguments": candidate["arguments"],
                "hypothesis": candidate["hypothesis"],
            }
            texts.append(_semantic_candidate_text(
                row, proposal, receipt["audio_overview"],
            ))
            references.append((receipt_index, candidate_index))

    embeddings, embedding_receipt = _semantic_candidate_embeddings(
        OpenAI(
            api_key=str(api_key),
            base_url=str(embedding_config["base_url"]),
            timeout=float(embedding_config["timeout_seconds"]),
            max_retries=int(embedding_config["max_retries"]),
        ),
        model=str(embedding_config["model"]),
        dimensions=int(embedding_config["dimensions"]),
        texts=texts,
    )
    for text, embedding, (receipt_index, candidate_index) in zip(
        texts, embeddings, references,
    ):
        candidate = output[receipt_index]["candidates"][candidate_index]
        numeric = list(map(float, candidate["descriptor"][:8]))
        candidate["descriptor"] = numeric + embedding
        candidate["semantic_text_sha256"] = stable_hash(text)
    contract = {
        "kind": "OUTCOME_BLIND_TARGET_NATIVE_SEMANTIC_DESCRIPTOR_BACKFILL",
        "config_sha256": stable_hash(config),
        "input_receipts_sha256": stable_hash(receipts),
        "forbidden_embedding_inputs": [
            "gold_answer", "baseline.answer", "candidate.answer",
        ],
        "embedding_receipt": embedding_receipt,
        "numeric_descriptor_width": 8,
        "semantic_descriptor_width": int(embedding_config["dimensions"]),
    }
    contract["contract_sha256"] = stable_hash(contract)
    for receipt in output:
        receipt["semantic_descriptor_backfill"] = contract
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "samples": len(output),
        "descriptor_width": len(output[0]["candidates"][0]["descriptor"]),
        "contract_sha256": contract["contract_sha256"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
