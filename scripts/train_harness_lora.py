#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a source-receipt selective Harness LoRA"
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--source-repo", type=Path,
        default=Path(
            "/fs/gamma-projects/vlm-robot/"
            "Multi-hop-Reasoning-VLM-Agent-github-main"
        ),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if str(args.source_repo) not in sys.path:
        sys.path.insert(0, str(args.source_repo))
    from trainer.SFT.lora_targets import (  # type: ignore
        assert_lora_coverage,
        resolve_target_modules,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    target_modules = resolve_target_modules(model_name_or_arch=args.model)
    peft_model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    ))
    peft_model.enable_input_require_grads()
    architecture = str(
        getattr(getattr(model.config, "text_config", None), "model_type", "")
        or getattr(model.config, "model_type", "")
    )
    coverage = assert_lora_coverage(
        peft_model, model_arch=architecture, require_strict=True,
    )

    class PromptCompletionDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
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
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * prompt_length + input_ids[prompt_length:],
            }

    def collate(rows):
        maximum = max(len(row["input_ids"]) for row in rows)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            padding = maximum - len(row["input_ids"])
            result["input_ids"].append(
                row["input_ids"] + [tokenizer.pad_token_id] * padding
            )
            result["attention_mask"].append(
                row["attention_mask"] + [0] * padding
            )
            result["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value) for key, value in result.items()}

    train_path = args.dataset_dir / "train.jsonl"
    validation_path = args.dataset_dir / "validation.jsonl"
    train_rows = _read_jsonl(train_path)
    validation_rows = _read_jsonl(validation_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=PromptCompletionDataset(train_rows),
        eval_dataset=PromptCompletionDataset(validation_rows),
        data_collator=collate,
    )
    result = trainer.train()
    adapter_dir = args.output_dir / "adapter"
    peft_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    receipt = {
        "schema_version": 1,
        "model": args.model,
        "train_file_sha256": _sha256(train_path),
        "validation_file_sha256": _sha256(validation_path),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_coverage": coverage,
        "train_metrics": dict(result.metrics),
        "target_data_used": False,
    }
    (args.output_dir / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
