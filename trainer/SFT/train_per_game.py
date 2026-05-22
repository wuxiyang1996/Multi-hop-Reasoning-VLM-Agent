#!/usr/bin/env python3
"""Per-game SFT training: one LoRA adapter per (game, adapter_type) pair.

Distributes 24 per-game adapters across 8 GPUs using LPT bin-packing.
Each GPU loads the base model once, then trains its assigned adapters
sequentially (LoRA swap is ~10s vs ~150s for a full model reload).

Output layout::

    <output_dir>/
      twenty_forty_eight/
        skill_selection/   # adapter_config.json + adapter_model.safetensors
        action_taking/
      tetris/
        skill_selection/
        action_taking/
      ...

Usage::

    python -m trainer.SFT.train_per_game \
        --decision_data_dir ../SFT_Data/high_reward/decision_sft \
        --skillbank_data_dir ../SFT_Data/high_reward/skillbank_sft \
        --output_dir runs/sft_per_game \
        --gpus 0 1 2 3 4 5 6 7
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train_per_game")

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# ── Job definition ────────────────────────────────────────────────────

ADAPTER_TYPES = ("skill_selection", "action_taking")

DEFAULTS = {
    "model_name": "Qwen/Qwen3.5-9B",
    "epochs": 3,
    "batch_size": 16,
    "grad_accum": 1,
    "lr": 2e-4,
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "warmup_ratio": 0.05,
    "eval_fraction": 0.05,
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 2,
}


def _discover_games(
    decision_dir: Path, v2_dir: Path,
) -> List[Tuple[str, str, Path]]:
    """Return (game, adapter_type, jsonl_path) for every available job."""
    jobs = []

    # skill_selection from v2
    if v2_dir.is_dir():
        for game_dir in sorted(v2_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            p = game_dir / "skill_selection.jsonl"
            if p.exists() and p.stat().st_size > 0:
                jobs.append((game_dir.name, "skill_selection", p))

    # action_taking from decision_dir
    if decision_dir.is_dir():
        for game_dir in sorted(decision_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            p = game_dir / "action_taking.jsonl"
            if p.exists() and p.stat().st_size > 0:
                jobs.append((game_dir.name, "action_taking", p))

    return jobs


def _count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _lpt_schedule(
    jobs: List[Tuple[str, str, Path, int]], n_gpus: int,
) -> List[List[Tuple[str, str, Path, int]]]:
    """LPT (Longest Processing Time) bin-packing across GPUs."""
    jobs_sorted = sorted(jobs, key=lambda j: -j[3])
    gpu_load = [0] * n_gpus
    gpu_jobs: List[List] = [[] for _ in range(n_gpus)]

    for job in jobs_sorted:
        i = gpu_load.index(min(gpu_load))
        gpu_jobs[i].append(job)
        gpu_load[i] += job[3]

    return gpu_jobs


# ── Per-GPU worker ────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _worker(
    gpu_id: int,
    jobs: List[Tuple[str, str, Path, int]],
    output_dir: Path,
    params: Dict[str, Any],
):
    """Single-GPU worker: loads model once, trains all assigned adapters."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    from trainer.SFT.speed_utils import enable_tf32, apply_liger_kernel, pick_optim
    enable_tf32()

    model_name = params["model_name"]
    probe_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = getattr(probe_cfg, "text_config", probe_cfg)
    probe_arch = (getattr(text_cfg, "model_type", "") or "").lower()

    if params.get("use_liger_kernel", True):
        try:
            apply_liger_kernel(probe_arch)
        except Exception as exc:
            logger.warning("GPU %d: liger-kernel failed: %s", gpu_id, exc)

    dtype = torch.bfloat16 if params["bf16"] else torch.float32
    logger.info("GPU %d: loading %s (AutoModelForCausalLM)...", gpu_id, model_name)
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True,
    )
    base_model = base_model.to("cuda:0")
    base_model.config.use_cache = False
    logger.info(
        "GPU %d: model loaded in %.1fs (%.1f GB)",
        gpu_id, time.time() - t0, torch.cuda.memory_allocated() / 1e9,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from trainer.SFT.lora_targets import resolve_target_modules
    target_modules = resolve_target_modules(model_name_or_arch=model_name)

    from trainer.SFT.train import format_for_sft

    for game, adapter_type, data_path, n_rows in jobs:
        adapter_label = f"{game}/{adapter_type}"
        peft_adapter_name = f"{game}__{adapter_type}"
        logger.info("GPU %d: === Training %s (%d rows) ===", gpu_id, adapter_label, n_rows)
        t1 = time.time()

        examples = _read_jsonl(data_path)
        formatted = format_for_sft(examples, tokenizer)

        n_eval = max(1, int(len(formatted) * params["eval_fraction"]))
        eval_data = formatted[:n_eval]
        train_data = formatted[n_eval:]

        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        train_ds = Dataset.from_list(train_data)
        eval_ds = Dataset.from_list(eval_data)

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=params["max_seq_length"],
                padding=False,
            )

        train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            target_modules=target_modules,
            bias="none",
        )

        peft_model = get_peft_model(base_model, lora_config, adapter_name=peft_adapter_name)
        peft_model.enable_input_require_grads()

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        logger.info("GPU %d: %s — trainable %.2f%%", gpu_id, adapter_label, 100 * trainable / total)

        out_path = output_dir / game / adapter_type
        out_path.mkdir(parents=True, exist_ok=True)
        hf_out = str(out_path / "hf_trainer")

        resolved_optim = pick_optim(prefer_8bit=True)

        steps_per_epoch = math.ceil(len(train_data) / params["batch_size"])
        total_steps = steps_per_epoch * params["epochs"]

        # Save every epoch so that a mid-training crash leaves a usable
        # checkpoint behind (the previous policy was to save only at the
        # very last step, which lost ~3h of GPU work when GPU 2 was killed
        # at step 950/1137 during the sft_per_game_xml run).
        training_args = TrainingArguments(
            output_dir=hf_out,
            num_train_epochs=params["epochs"],
            per_device_train_batch_size=params["batch_size"],
            gradient_accumulation_steps=params["grad_accum"],
            learning_rate=params["lr"],
            warmup_ratio=params["warmup_ratio"],
            logging_steps=params["logging_steps"],
            save_strategy="epoch",
            save_total_limit=params["save_total_limit"],
            eval_strategy="epoch",
            bf16=params["bf16"],
            gradient_checkpointing=params["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=resolved_optim,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            report_to="none",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )

        # Auto-resume from the most recent epoch checkpoint if one exists
        # (paired with save_strategy="epoch" above). Crash → relaunch the
        # same command and training picks up from the last completed epoch
        # instead of restarting from scratch.
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = (
            get_last_checkpoint(hf_out) if os.path.isdir(hf_out) else None
        )
        if last_ckpt:
            logger.info(
                "GPU %d: %s — resuming from %s", gpu_id, adapter_label, last_ckpt,
            )
        trainer.train(resume_from_checkpoint=last_ckpt)

        peft_model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))

        meta = {
            "adapter_name": adapter_label,
            "game": game,
            "adapter_type": adapter_type,
            "base_model": model_name,
            "n_train": len(train_data),
            "n_eval": len(eval_data),
            "epochs": params["epochs"],
            "lr": params["lr"],
            "lora_r": params["lora_r"],
            "total_steps": total_steps,
        }
        with open(out_path / "adapter_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        elapsed = time.time() - t1
        logger.info(
            "GPU %d: %s done — %d steps in %.1fs (%.1f st/s)",
            gpu_id, adapter_label, total_steps, elapsed, total_steps / elapsed,
        )

        base_model_unwrapped = peft_model.unload()
        base_model_unwrapped.config.use_cache = False

        del trainer, peft_model, train_ds, eval_ds
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("GPU %d: all %d adapters complete", gpu_id, len(jobs))


# ── Parallel launcher ─────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Per-game SFT for decision adapters")
    p.add_argument("--decision_data_dir", type=str,
                    default="../SFT_Data/high_reward/decision_sft")
    p.add_argument("--v2_data_dir", type=str, default=None,
                    help="skill_selection v2 dir (default: <decision_data_dir>_v2)")
    p.add_argument("--skillbank_data_dir", type=str,
                    default="../SFT_Data/high_reward/skillbank_sft")
    p.add_argument("--output_dir", type=str, default="runs/sft_per_game")
    p.add_argument("--model_name", type=str, default=DEFAULTS["model_name"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--gpus", type=int, nargs="*", default=None)
    p.add_argument("--games", type=str, nargs="*", default=None,
                    help="Subset of games (default: all discovered)")
    p.add_argument("--adapters", type=str, nargs="*", default=None,
                    help="Subset: skill_selection, action_taking (default: both)")
    p.add_argument("--dry-run", action="store_true",
                    help="Print schedule only, don't train")
    args = p.parse_args()

    decision_dir = Path(args.decision_data_dir)
    v2_dir = Path(args.v2_data_dir) if args.v2_data_dir else Path(str(decision_dir) + "_v2")
    output_dir = Path(args.output_dir)

    # Discover jobs
    raw_jobs = _discover_games(decision_dir, v2_dir)
    if args.games:
        game_set = set(args.games)
        raw_jobs = [(g, a, p) for g, a, p in raw_jobs if g in game_set]
    if args.adapters:
        adapter_set = set(args.adapters)
        raw_jobs = [(g, a, p) for g, a, p in raw_jobs if a in adapter_set]

    jobs_with_size = []
    for game, adapter_type, path in raw_jobs:
        n = _count_lines(path)
        if n > 0:
            jobs_with_size.append((game, adapter_type, path, n))

    if not jobs_with_size:
        logger.error("No training data found!")
        return

    # GPU detection
    if args.gpus:
        gpu_ids = args.gpus
    else:
        try:
            import torch
            gpu_ids = list(range(torch.cuda.device_count()))
        except Exception:
            gpu_ids = [0]

    n_gpus = len(gpu_ids)
    schedule = _lpt_schedule(jobs_with_size, n_gpus)

    # Print schedule
    total_rows = sum(j[3] for j in jobs_with_size)
    print(f"\n{'='*70}")
    print(f"Per-Game SFT: {len(jobs_with_size)} adapters, {total_rows:,} rows, {n_gpus} GPUs")
    print(f"Config: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}")
    print(f"{'='*70}")

    for i, gpu_jobs in enumerate(schedule):
        if not gpu_jobs:
            continue
        gpu_steps = sum(
            math.ceil(int(j[3] * 0.95) / args.batch_size) * args.epochs
            for j in gpu_jobs
        )
        job_strs = [f"{j[0]}/{j[1]}({j[3]})" for j in gpu_jobs]
        print(f"  GPU {gpu_ids[i]}: {gpu_steps:>5} steps | {', '.join(job_strs)}")

    makespan = max(
        sum(math.ceil(int(j[3] * 0.95) / args.batch_size) * args.epochs for j in gpu_jobs)
        for gpu_jobs in schedule if gpu_jobs
    )
    print(f"\n  Makespan: {makespan} steps")
    print(f"  Est. time: ~{makespan / 1.3 / 60 + 3:.0f} min\n")

    if args.dry_run:
        print("(dry-run mode, not training)")
        return

    # Build params dict
    params = dict(DEFAULTS)
    params["model_name"] = args.model_name
    params["epochs"] = args.epochs
    params["batch_size"] = args.batch_size
    params["lr"] = args.lr

    # Launch one subprocess per GPU
    output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    t0 = time.time()

    for i, gpu_jobs in enumerate(schedule):
        if not gpu_jobs:
            continue
        gpu_id = gpu_ids[i]

        # Serialize job list to a temp file for the subprocess
        job_file = output_dir / f"_gpu{gpu_id}_jobs.json"
        with open(job_file, "w") as f:
            json.dump([
                {"game": g, "adapter_type": a, "data_path": str(p), "n_rows": n}
                for g, a, p, n in gpu_jobs
            ], f)

        log_file_path = output_dir / f"gpu{gpu_id}.log"
        log_fh = open(log_file_path, "w")

        cmd = [
            sys.executable, "-m", "trainer.SFT.train_per_game",
            "--_worker", str(gpu_id),
            "--_job_file", str(job_file),
            "--_output_dir", str(output_dir),
            "--_params_json", json.dumps(params),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        processes.append((gpu_id, gpu_jobs, proc, log_fh, log_file_path))
        logger.info("Launched GPU %d: %d adapters → %s", gpu_id, len(gpu_jobs), log_file_path)

    # Wait for all
    failed = []
    while processes:
        still_running = []
        for gpu_id, gpu_jobs, proc, log_fh, log_path in processes:
            ret = proc.poll()
            if ret is None:
                still_running.append((gpu_id, gpu_jobs, proc, log_fh, log_path))
            else:
                log_fh.close()
                if ret == 0:
                    logger.info("GPU %d finished successfully", gpu_id)
                else:
                    logger.error("GPU %d FAILED (exit %d) — see %s", gpu_id, ret, log_path)
                    failed.append(gpu_id)
        processes = still_running
        if processes:
            time.sleep(10)

    elapsed = time.time() - t0
    n_ok = len(gpu_ids) - len(failed)

    # Summary
    summary = {
        "adapters": len(jobs_with_size),
        "total_rows": total_rows,
        "gpus": gpu_ids,
        "epochs": args.epochs,
        "elapsed_min": round(elapsed / 60, 2),
        "failed_gpus": failed,
        "output_dir": str(output_dir),
        "model_name": args.model_name,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Per-game SFT complete: %d adapters on %d GPUs in %.1f min (%d failed)",
        len(jobs_with_size), len(gpu_ids), elapsed / 60, len(failed),
    )


def _worker_main():
    """Entry point for subprocess workers (called with --_worker)."""
    p = argparse.ArgumentParser()
    p.add_argument("--_worker", type=int, required=True)
    p.add_argument("--_job_file", type=str, required=True)
    p.add_argument("--_output_dir", type=str, required=True)
    p.add_argument("--_params_json", type=str, required=True)
    args = p.parse_args()

    with open(args._job_file) as f:
        raw_jobs = json.load(f)

    jobs = [
        (j["game"], j["adapter_type"], Path(j["data_path"]), j["n_rows"])
        for j in raw_jobs
    ]
    params = json.loads(args._params_json)
    output_dir = Path(args._output_dir)

    _worker(args._worker, jobs, output_dir, params)


if __name__ == "__main__":
    if "--_worker" in sys.argv:
        _worker_main()
    else:
        main()
