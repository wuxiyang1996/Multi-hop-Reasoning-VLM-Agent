#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _select_balanced(rows: list[dict], per_objective: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["objective"])].append(row)
    selected = []
    for objective in sorted(grouped):
        ranked = sorted(
            grouped[objective],
            key=lambda row: hashlib.sha256(
                str(row["example_id"]).encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:per_objective])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched base-vs-LoRA held-out completion-NLL audit"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--per-objective", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = _select_balanced(_read_jsonl(args.dataset), args.per_objective)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    def row_nll(row: dict, *, adapter_enabled: bool) -> tuple[float, int]:
        completion_ids = tokenizer(
            row["completion"] + tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, args.max_length - 1),
        )["input_ids"]
        prompt_budget = max(1, args.max_length - len(completion_ids))
        prompt_ids = tokenizer(
            row["prompt"], add_special_tokens=True,
            truncation=True, max_length=prompt_budget,
        )["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:args.max_length]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        tensor_ids = torch.tensor([input_ids], device=model.device)
        tensor_labels = torch.tensor([labels], device=model.device)
        context = nullcontext() if adapter_enabled else model.disable_adapter()
        with context, torch.inference_mode():
            loss = model(input_ids=tensor_ids, labels=tensor_labels).loss
        tokens = sum(value != -100 for value in labels)
        return float(loss.item()), tokens

    aggregates = {
        regime: defaultdict(lambda: {"weighted_nll": 0.0, "tokens": 0, "rows": 0})
        for regime in ("BASE", "GAME_RECEIPT_LORA")
    }
    for row in rows:
        for regime, enabled in (("BASE", False), ("GAME_RECEIPT_LORA", True)):
            nll, tokens = row_nll(row, adapter_enabled=enabled)
            bucket = aggregates[regime][str(row["objective"])]
            bucket["weighted_nll"] += nll * tokens
            bucket["tokens"] += tokens
            bucket["rows"] += 1

    metrics = {}
    for regime, by_objective in aggregates.items():
        objective_metrics = {}
        total_nll = total_tokens = 0
        for objective, values in sorted(by_objective.items()):
            mean_nll = values["weighted_nll"] / values["tokens"]
            objective_metrics[objective] = {
                "rows": values["rows"],
                "completion_tokens": values["tokens"],
                "mean_token_nll": mean_nll,
                "perplexity": math.exp(min(mean_nll, 20.0)),
            }
            total_nll += values["weighted_nll"]
            total_tokens += values["tokens"]
        metrics[regime] = {
            "overall_mean_token_nll": total_nll / total_tokens,
            "overall_perplexity": math.exp(min(total_nll / total_tokens, 20.0)),
            "by_objective": objective_metrics,
        }
    base_nll = metrics["BASE"]["overall_mean_token_nll"]
    lora_nll = metrics["GAME_RECEIPT_LORA"]["overall_mean_token_nll"]
    payload = {
        "schema_version": 1,
        "authority": "FROZEN_SOURCE_HELD_OUT_TEACHER_FORCED_AUDIT",
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "dataset": str(args.dataset.resolve()),
        "selection": {
            "method": "sha256_example_id_balanced_by_objective",
            "per_objective": args.per_objective,
            "rows": len(rows),
            "example_ids": [row["example_id"] for row in rows],
        },
        "metrics": metrics,
        "lora_minus_base_mean_token_nll": lora_nll - base_nll,
        "sanity_gate_passed": lora_nll < base_nll,
        "claim_boundary": (
            "A source-held-out NLL gain is a training sanity check, not evidence "
            "of far-domain Harness transfer."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(rows),
        "base_nll": base_nll,
        "lora_nll": lora_nll,
        "sanity_gate_passed": payload["sanity_gate_passed"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
