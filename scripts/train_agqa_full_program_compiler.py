#!/usr/bin/env python3
"""Fine-tune a small target-native seq2seq question-to-program compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("compiler checkpoint directory is immutable")
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["formal_programs_read"] or manifest["formal_answers_read"]:
        raise ValueError("compiler supervision crossed formal boundary")
    data = load_dataset("json", data_files={
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl"),
    })
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def encode(batch):
        inputs = tokenizer(
            ["compile AGQA: " + value for value in batch["question"]],
            max_length=192, truncation=True,
        )
        labels = tokenizer(text_target=batch["program"], max_length=512, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    encoded = data.map(
        encode, batched=True, num_proc=8,
        remove_columns=data["train"].column_names,
        desc="tokenizing compiler supervision",
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps, per_device_train_batch_size=16,
        gradient_accumulation_steps=2, learning_rate=3e-4,
        warmup_ratio=0.03, lr_scheduler_type="cosine", weight_decay=0.01,
        bf16=True, logging_steps=25, save_steps=500, save_total_limit=2,
        eval_strategy="steps", eval_steps=500,
        per_device_eval_batch_size=32, predict_with_generate=False,
        eval_accumulation_steps=8, dataloader_num_workers=8,
        report_to="none", seed=args.seed, data_seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"].select(range(min(10000, len(encoded["validation"])))),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))
    (args.output_dir / "training_receipt.json").write_text(json.dumps({
        "schema_version": "agqa-full-program-compiler-training-v1",
        "status": "TRAINING_COMPLETE", "model": args.model,
        "max_steps": args.max_steps, "seed": args.seed,
        "train_metrics": result.metrics,
        "formal_programs_read": False, "formal_answers_read": False,
        "supervision_manifest_sha256": manifest["manifest_sha256"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
