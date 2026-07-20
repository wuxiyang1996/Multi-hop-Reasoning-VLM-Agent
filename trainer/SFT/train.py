#!/usr/bin/env python3
"""SFT cold-start trainer for all 5 LoRA adapters.

Supports **parallel training** across multiple GPUs — each adapter gets
its own GPU and process, so all 5 train simultaneously.  With 5+ GPUs
the total wall-clock time equals the slowest single adapter rather than
the sum of all five.

Output checkpoints are written in the exact layout expected by the
co-evolution GRPO pipeline::

    <output_dir>/
    ├── decision/
    │   ├── skill_selection/   # adapter_config.json + adapter_model.safetensors
    │   └── action_taking/
    └── skillbank/
        ├── segment/
        ├── contract/
        └── curator/

Usage::

    # Sequential (1 GPU, adapters trained one after another)
    python -m trainer.SFT.train

    # Parallel (each adapter on a separate GPU, all at once)
    python -m trainer.SFT.train --parallel

    # Parallel on specific GPUs
    python -m trainer.SFT.train --parallel --gpus 0 1 2 3 4

    # Subset + parallel
    python -m trainer.SFT.train --parallel --adapters segment contract curator
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SFT cold-start training for decision + skill-bank LoRA adapters",
    )
    p.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3.5-9B",
        help="Base model (must match co-evolution config)",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Root output directory (default: runs/sft_coldstart)",
    )
    p.add_argument(
        "--decision_data_dir", type=str, default=None,
        help="Path to gpt54_skill_labeled/grpo_coldstart",
    )
    p.add_argument(
        "--skillbank_data_dir", type=str, default=None,
        help="Path to gpt54_skillbank_grpo",
    )
    p.add_argument(
        "--adapters", type=str, nargs="*", default=None,
        help="Subset of adapters to train (default: all 5)",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--eval_fraction", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="bf16")
    p.add_argument(
        "--games", type=str, nargs="*", default=None,
        help="Subset of games to include in training data",
    )
    p.add_argument(
        "--require_source_only_manifest", type=str, default=None,
        help=(
            "Fail unless this manifest proves source_only=true, zero target "
            "examples/updates, and hashes every decision JSONL input."
        ),
    )
    p.add_argument(
        "--parallel", action="store_true", default=False,
        help="Train adapters in parallel, one per GPU",
    )
    p.add_argument(
        "--gpus", type=int, nargs="*", default=None,
        help="GPU IDs for parallel training (default: 0..N-1 where N = #adapters)",
    )
    p.add_argument(
        "--gpu", type=int, default=None,
        help="(internal) Pin this process to a specific GPU (used by parallel launcher)",
    )
    p.add_argument(
        "--gpus_per_adapter", type=int, default=1,
        help=(
            "How many GPUs each adapter gets (default 1).  When >1 every "
            "adapter is launched under ``accelerate launch --num_processes N`` "
            "with DDP — HF Trainer auto-detects the multi-process env and "
            "data-parallels its way through.  Effective batch becomes "
            "``per_device_bs × N × grad_accum``; pair with ``--scale_lr N`` "
            "for the linear-scale rule.  With 8 H200s and 5 adapters: "
            "``--gpus_per_adapter 1`` uses 5 GPUs (3 idle), "
            "``--gpus_per_adapter 2`` runs 4 adapters at a time on 8 GPUs, "
            "``--gpus_per_adapter 4`` runs 2 adapters at a time."
        ),
    )
    # ── Speed / kernel knobs (T2.11 closure) ─────────────────────────
    p.add_argument(
        "--use_liger_kernel", dest="use_liger_kernel",
        action="store_true", default=True,
        help="Apply liger-kernel fused Qwen3.5 patches (default: True).",
    )
    p.add_argument(
        "--no_liger_kernel", dest="use_liger_kernel", action="store_false",
        help="Disable liger-kernel even if installed.",
    )
    p.add_argument(
        "--gradient_checkpointing", dest="gradient_checkpointing",
        action="store_true", default=True,
        help=(
            "Activation checkpointing (default: True).  Required for "
            "single-GPU bs=16 + 9B; disable only when memory permits."
        ),
    )
    p.add_argument(
        "--no_gradient_checkpointing", dest="gradient_checkpointing",
        action="store_false",
        help=(
            "Disable activation checkpointing — buys ~30-40 %% throughput "
            "but requires either smaller bs or multi-GPU per adapter "
            "(``--gpus_per_adapter 2+``)."
        ),
    )
    p.add_argument(
        "--optim", type=str, default=None,
        help=(
            "HF Trainer optim string.  Defaults to 'paged_adamw_8bit' when "
            "bitsandbytes is available, else 'adamw_torch_fused'."
        ),
    )
    p.add_argument(
        "--dataloader_workers", type=int, default=4,
        help="DataLoader worker count (default: 4).",
    )
    p.add_argument(
        "--strict_lora_coverage", action="store_true", default=False,
        help=(
            "Abort if any required projection has zero LoRA-wrapped layers "
            "(catches T2.11-style recipe drift)."
        ),
    )
    p.add_argument(
        "--scale_effective_batch", type=float, default=1.0,
        help=(
            "Multiply every per-adapter ``batch_size`` by this factor "
            "(default 1.0 keeps effective batch = 16).  Increasing past "
            "1.0 is **Lever B** in T2.12 — fewer optimizer steps, "
            "faster wall-clock, but you should also scale LR roughly "
            "linearly and watch the loss curve.  Use 2.0 for "
            "effective batch 32, 4.0 for 64."
        ),
    )
    p.add_argument(
        "--scale_lr", type=float, default=1.0,
        help=(
            "Multiply every per-adapter ``lr`` by this factor.  Pair "
            "with ``--scale_effective_batch`` for the linear-scale rule "
            "(matched factor = noise-equivalent training)."
        ),
    )
    return p.parse_args()


def _detect_text_arch(model) -> str:
    """Return ``model.config.text_config.model_type`` (or fallback)."""
    cfg = getattr(model, "config", None)
    if cfg is None and hasattr(model, "base_model"):
        cfg = getattr(model.base_model, "config", None)
    if cfg is None:
        return ""
    text_cfg = getattr(cfg, "text_config", cfg)
    return (getattr(text_cfg, "model_type", "") or "").lower()


def _build_config(args: argparse.Namespace):
    from trainer.SFT.config import SFTConfig
    kwargs = {
        "model_name": args.model_name,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "warmup_ratio": args.warmup_ratio,
        "eval_fraction": args.eval_fraction,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "bf16": args.bf16,
        "adapters": args.adapters,
        "use_liger_kernel": getattr(args, "use_liger_kernel", True),
        "gradient_checkpointing": getattr(args, "gradient_checkpointing", True),
        "optim": getattr(args, "optim", None),
        "dataloader_num_workers": getattr(args, "dataloader_workers", 4),
        "strict_lora_coverage": getattr(args, "strict_lora_coverage", False),
        "scale_effective_batch": getattr(args, "scale_effective_batch", 1.0),
        "scale_lr": getattr(args, "scale_lr", 1.0),
    }
    if args.output_dir:
        kwargs["output_dir"] = args.output_dir
    if args.decision_data_dir:
        kwargs["decision_data_dir"] = args.decision_data_dir
    if args.skillbank_data_dir:
        kwargs["skillbank_data_dir"] = args.skillbank_data_dir
    if args.games:
        kwargs["games"] = args.games
    return SFTConfig(**kwargs)


def format_for_sft(examples: list, tokenizer) -> list:
    """Convert prompt/completion dicts to chat-formatted text for SFT.

    Uses the model's chat template (``apply_chat_template``) when
    available so the SFT data matches the format the model sees during
    GRPO inference.
    """
    formatted = []
    for ex in examples:
        prompt = ex.get("prompt", "")
        completion = ex.get("completion", "")
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                text = f"{prompt}\n{completion}"
        else:
            text = f"{prompt}\n{completion}"
        formatted.append({"text": text})
    return formatted


def _validate_source_only_manifest(path: str, decision_data_dir: str) -> str:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "source_only": payload.get("source_only") is True,
        "target_examples": int(payload.get("target_examples", -1)) == 0,
        "target_gradient_updates": int(payload.get("target_gradient_updates", -1)) == 0,
        "semantic_clustering": payload.get("semantic_clustering") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"source-only manifest invariant failed: {checks}")
    root = Path(decision_data_dir).resolve()
    output_hashes = dict(payload.get("output_sha256") or {})
    if not output_hashes:
        raise ValueError("source-only manifest has no output hashes")
    for relative, expected in output_hashes.items():
        input_path = root / str(relative)
        if not input_path.is_file():
            raise FileNotFoundError(f"manifested SFT input missing: {input_path}")
        actual = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if actual != str(expected):
            raise ValueError(f"source-only SFT input hash mismatch: {input_path}")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    os.environ["SFT_SOURCE_ONLY_MANIFEST"] = str(manifest_path)
    os.environ["SFT_SOURCE_ONLY_MANIFEST_SHA256"] = digest
    logger.info("Verified source-only SFT manifest %s (%s)", manifest_path, digest)
    return digest


def train_single_adapter(
    adapter_name: str,
    examples: list,
    base_model,
    tokenizer,
    config,
) -> str:
    """Train one LoRA adapter via HuggingFace Trainer and return save path.

    The base model is wrapped with PEFT, trained, saved, then unwrapped
    so the same base model instance can be reused for the next adapter.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    params = config.effective_params(adapter_name)
    output_path = config.adapter_output_path(adapter_name)
    output_path.mkdir(parents=True, exist_ok=True)
    out_str = str(output_path)

    logger.info(
        "=== Training LoRA adapter '%s' === (%d examples, lr=%.2e, epochs=%d)",
        adapter_name, len(examples), params["lr"], params["epochs"],
    )

    formatted = format_for_sft(examples, tokenizer)

    n_eval = max(1, int(len(formatted) * config.eval_fraction))
    eval_data = formatted[:n_eval]
    train_data = formatted[n_eval:]
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

    target_modules = config.resolve_target_modules()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config, adapter_name=adapter_name)
    peft_model.enable_input_require_grads()

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    # ── T2.11 sanity check: did every architecturally-required leg
    # actually get LoRA-wrapped?  Fail-fast on recipe drift so no
    # future SFT silently ships a 23 % adapter again.
    from trainer.SFT.lora_targets import assert_lora_coverage
    text_arch = _detect_text_arch(peft_model)
    assert_lora_coverage(
        peft_model,
        model_arch=text_arch,
        require_strict=getattr(config, "strict_lora_coverage", False),
        logger_=logger,
    )

    from trainer.SFT.speed_utils import pick_optim
    resolved_optim = config.optim or pick_optim(prefer_8bit=True)
    logger.info(
        "Trainer optimizer = %s, gradient_checkpointing=%s, "
        "dataloader_workers=%d",
        resolved_optim, config.gradient_checkpointing, config.dataloader_num_workers,
    )

    hf_output = str(output_path / "hf_trainer")
    training_args = TrainingArguments(
        output_dir=hf_output,
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_accum"],
        learning_rate=params["lr"],
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=resolved_optim,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        # ``group_by_length`` was removed in transformers 5.x in favour
        # of explicit ``LengthGroupedSampler`` use.  Skipped here — the
        # pad-savings optimisation is small for our seq lengths and
        # liger-kernel + paged-AdamW are doing the heavy lifting.
        report_to="none",
        remove_unused_columns=False,
        # LoRA-only training never has true unused params (LoRA is
        # additive on every targeted projection that's exercised in the
        # forward pass).  Default ``find_unused_parameters=True`` adds an
        # extra autograd-graph traversal per step (HF warns explicitly:
        # "find_unused_parameters=True ... did not find any unused
        # parameters in the forward pass").  Disabling buys 5-10% per
        # step on 8x DDP.
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

    trainer.train()

    # PEFT >=0.19 writes a named adapter below ``save_directory/name``.
    # Saving to ``output_path`` would therefore create
    # ``decision/action_taking/action_taking`` while vLLM expects the exact
    # ``decision/action_taking`` layout. Save at the group parent and assert
    # the production-facing files exist.
    peft_model.save_pretrained(
        str(output_path.parent), selected_adapters=[adapter_name],
    )
    tokenizer.save_pretrained(out_str)
    if not (output_path / "adapter_config.json").is_file():
        raise RuntimeError(f"PEFT adapter layout mismatch: {output_path}")

    meta = {
        "adapter_name": adapter_name,
        "base_model": config.model_name,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": target_modules,
        "n_train": len(train_data),
        "n_eval": len(eval_data),
        "epochs": params["epochs"],
        "lr": params["lr"],
        "training_type": "sft_coldstart",
        "source_only_manifest": os.environ.get("SFT_SOURCE_ONLY_MANIFEST"),
        "source_only_manifest_sha256": os.environ.get("SFT_SOURCE_ONLY_MANIFEST_SHA256"),
        "target_examples": 0 if os.environ.get("SFT_SOURCE_ONLY_MANIFEST") else None,
        "target_gradient_updates": 0 if os.environ.get("SFT_SOURCE_ONLY_MANIFEST") else None,
    }
    with open(output_path / "adapter_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved adapter '%s' to %s", adapter_name, out_str)

    # Unwrap to reuse base model for next adapter
    base_model_unwrapped = peft_model.unload()
    base_model_unwrapped.config.use_cache = False

    del trainer, peft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_str


def train_all_adapters(config=None, gpu: Optional[int] = None, **kwargs) -> dict:
    """Train all requested LoRA adapters from cold-start data.

    Parameters
    ----------
    config : SFTConfig, optional
        If not given, a default config is created.
    gpu : int, optional
        Pin training to a specific GPU (sets ``CUDA_VISIBLE_DEVICES``
        before loading the model).
    **kwargs
        Override fields on ``SFTConfig``.

    Returns
    -------
    dict
        ``{adapter_name: output_path}`` for each trained adapter.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore
    except ImportError:  # pragma: no cover — older transformers <5
        AutoModelForImageTextToText = None  # type: ignore
    from trainer.SFT.config import SFTConfig
    from trainer.SFT.data_loader import load_all_adapter_datasets

    if config is None:
        config = SFTConfig(**kwargs)

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        logger.info("Pinned to GPU %d (CUDA_VISIBLE_DEVICES=%s)", gpu, gpu)

    logger.info("SFT cold-start config: model=%s, output=%s", config.model_name, config.output_dir)
    logger.info("Adapters to train: %s", config.adapters_to_train)

    # ── Speed-up scaffolding ─────────────────────────────────────────
    # Order matters: TF32 is a no-op if tensors aren't fp32 already, but
    # liger-kernel patches model classes — must run before
    # ``AutoModelForCausalLM.from_pretrained``.
    from trainer.SFT.speed_utils import enable_tf32, apply_liger_kernel
    enable_tf32()
    if config.use_liger_kernel:
        try:
            from transformers import AutoConfig
            _probe_cfg = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            _probe_arch = (
                getattr(getattr(_probe_cfg, "text_config", _probe_cfg), "model_type", "")
                or ""
            ).lower()
            apply_liger_kernel(_probe_arch)
        except Exception as exc:
            logger.warning("liger-kernel probe failed: %s — proceeding without it.", exc)

    t0 = time.time()

    datasets = load_all_adapter_datasets(config)

    empty = [name for name, data in datasets.items() if not data]
    if empty:
        logger.warning("No training data for adapters: %s — skipping", empty)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if config.bf16 else torch.float32
    # T2.13 fix (2026-05-03): on multimodal Qwen3.5 the umbrella config has
    # ``text_config`` + ``vision_config`` and the architecture is
    # ``Qwen3_5ForConditionalGeneration``.  ``AutoModelForCausalLM`` returns
    # the language-only sub-model (``Qwen3_5ForCausalLM``) and PEFT saves
    # LoRA keys without the ``language_model.`` prefix — but
    # ``trainer/coevolution/prepare_adapters`` and the vLLM serve path load
    # the full multimodal model where the LM lives under
    # ``model.language_model.layers.*``.  The structural prefix mismatch
    # silently zero-inits the LoRA at load time (lora_B = 0 → delta = 0),
    # dropping the cold-start signal at the SFT→GRPO/vLLM boundary.  Match
    # the production loader so saved keys carry the ``language_model.``
    # prefix natively.  See ``evaluation/fix_lora_keys_for_vlm_loader.py``
    # for a one-shot remap of legacy LM-only-keyed checkpoints.
    _probe_cfg2 = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
    is_multimodal = hasattr(_probe_cfg2, "text_config") or hasattr(_probe_cfg2, "vision_config")
    if is_multimodal and AutoModelForImageTextToText is not None:
        loader_cls = AutoModelForImageTextToText
        logger.info(
            "Multimodal Qwen3.5 detected (text_config+vision_config) — using "
            "AutoModelForImageTextToText to keep LoRA keys in lock-step with "
            "vLLM/GRPO production loaders."
        )
    else:
        loader_cls = AutoModelForCausalLM
    logger.info("Loading base model '%s' (dtype=%s, loader=%s) …",
                config.model_name, dtype, loader_cls.__name__)
    # When ``SFT_DEEPSPEED_CONFIG_FILE`` is set, instantiate
    # ``HfDeepSpeedConfig`` BEFORE ``from_pretrained`` so transformers
    # routes weight allocation through ``deepspeed.zero.Init()`` and
    # shards params at load time instead of materialising the full 9B+
    # base on a single GPU.  Without this, ZeRO-3 becomes an expensive
    # no-op (each rank holds the full model and we only save on optim
    # sharding — useless for LoRA where optim is tiny).
    _ds_cfg_path = os.environ.get("SFT_DEEPSPEED_CONFIG_FILE", "")
    _ds_cfg_keepalive = None
    if (
        _ds_cfg_path
        and os.environ.get("SFT_USE_DEEPSPEED", "0") == "1"
    ):
        # ``HfDeepSpeedConfig.__init__`` runs DeepSpeed's batch-related
        # assertion, which needs ``world_size`` to match the launcher's
        # actual world.  DeepSpeed has its own comm backend (``cdb``)
        # separate from ``torch.distributed`` — call
        # ``deepspeed.init_distributed`` so DS reads ``world_size = 4``
        # instead of falling back to 1.  Otherwise the
        # ``train_batch_size == micro * accum * world_size`` invariant
        # fails when HF's Trainer later cross-checks the config.
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            try:
                import deepspeed as _ds
                _ds.init_distributed()
                logger.info(
                    "deepspeed.init_distributed pre-run (rank=%s/%s) for ZeRO-3 init.",
                    os.environ.get("RANK", "?"), os.environ.get("WORLD_SIZE", "?"),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Pre-init of deepspeed.distributed failed: %s", exc)
        try:
            from transformers.integrations.deepspeed import HfDeepSpeedConfig  # type: ignore
        except ImportError:
            from transformers.integrations import HfDeepSpeedConfig  # type: ignore
        _ds_cfg_keepalive = HfDeepSpeedConfig(_ds_cfg_path)  # noqa: F841
        logger.info(
            "ZeRO-3 init context active via %s — model weights will be "
            "sharded at load time across DDP ranks.",
            _ds_cfg_path,
        )
    base_model = loader_cls.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if _ds_cfg_keepalive is None:
        # When accelerate launches N>1 ranks via torchrun, every rank's
        # ``cuda`` defaults to ``cuda:0`` until ``set_device`` is called.
        # That means all N ranks would copy the 9B model onto GPU 0
        # (28 GB × N) → instant OOM, even though each rank should own a
        # distinct GPU.  Pin to ``cuda:LOCAL_RANK`` BEFORE ``.to`` so
        # each replica lands on its own card.  Single-process / pinned
        # mode (``CUDA_VISIBLE_DEVICES`` filtered to one device) works
        # too — set_device(0) is a no-op when only one GPU is visible.
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(_local_rank)
        base_model = base_model.to(f"cuda:{_local_rank}")
    base_model.config.use_cache = False
    logger.info(
        "Model loaded on %s — %.1f GB GPU memory allocated",
        next(base_model.parameters()).device,
        torch.cuda.memory_allocated() / 1e9,
    )

    results: dict = {}
    for adapter_name in config.adapters_to_train:
        data = datasets.get(adapter_name, [])
        if not data:
            logger.warning("Skipping '%s' — no data", adapter_name)
            continue

        save_path = train_single_adapter(
            adapter_name=adapter_name,
            examples=data,
            base_model=base_model,
            tokenizer=tokenizer,
            config=config,
        )
        results[adapter_name] = save_path

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    logger.info(
        "SFT cold-start training complete: %d adapters in %.1f min",
        len(results), elapsed / 60,
    )
    logger.info("Adapter paths:")
    for name, path in results.items():
        logger.info("  %s → %s", name, path)

    summary_path = Path(config.output_dir) / "sft_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "adapters": results,
            "model_name": config.model_name,
            "elapsed_min": round(elapsed / 60, 2),
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.resolve_target_modules(),
            "source_only_manifest": os.environ.get("SFT_SOURCE_ONLY_MANIFEST"),
            "source_only_manifest_sha256": os.environ.get("SFT_SOURCE_ONLY_MANIFEST_SHA256"),
        }, f, indent=2)

    return results


def _print_progress(processes: list, t0: float):
    """Tail the last progress line from each chunk's log file."""
    import re
    elapsed = time.time() - t0
    parts = [f"[{elapsed/60:.0f}m]"]
    for chunk, adapter_list, proc, _lf, log_path in processes:
        status = "running" if proc.poll() is None else (
            "done" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        )
        progress = ""
        try:
            with open(log_path, "r") as f:
                content = f.read()
            last_incomplete = None
            last_any = None
            for m in re.finditer(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)", content):
                last_any = m
                if int(m.group(1)) < 100:
                    last_incomplete = m
            best = last_incomplete or last_any
            if best:
                progress = f" {best.group(1)}% ({best.group(2)}/{best.group(3)})"
        except Exception:
            pass
        names = "+".join(adapter_list)
        chunk_label = ",".join(map(str, chunk))
        parts.append(f"GPU[{chunk_label}][{names}]:{status}{progress}")
    logger.info("  ".join(parts))


def _train_parallel(config, gpu_ids: List[int], gpus_per_adapter: int = 1) -> dict:
    """Launch one subprocess per adapter, each pinned to a chunk of GPUs.

    With ``gpus_per_adapter == 1`` (default) each adapter gets one GPU
    and one ``python -m trainer.SFT.train --gpu <id>`` subprocess.

    With ``gpus_per_adapter > 1`` GPUs are grouped into contiguous
    chunks of that size and each adapter gets one chunk; the
    subprocess is launched under ``accelerate launch
    --num_processes <N>`` so HF Trainer data-parallels across the chunk.
    Effective batch becomes ``per_device_bs × N × grad_accum``; users
    should pair with ``--scale_lr <N>`` for the linear-scale rule.

    Wall-clock time = slowest adapter (across waves if there are more
    adapters than chunks).
    """
    from trainer.SFT.config import SFTConfig

    adapters = config.adapters_to_train
    n_adapters = len(adapters)
    n_gpus = len(gpu_ids)
    gpa = max(1, int(gpus_per_adapter))

    if n_gpus < gpa:
        raise ValueError(
            f"--gpus_per_adapter={gpa} but only {n_gpus} GPU IDs given. "
            f"Need at least {gpa} GPUs."
        )

    # Build GPU chunks: contiguous slices of size ``gpa``.
    chunks: List[List[int]] = []
    for i in range(0, n_gpus - n_gpus % gpa, gpa):
        chunks.append(gpu_ids[i : i + gpa])
    n_chunks = len(chunks)

    if n_chunks == 0:
        raise ValueError(
            f"No usable GPU chunks for --gpus_per_adapter={gpa} on "
            f"--gpus={gpu_ids}",
        )

    if n_chunks < n_adapters:
        logger.warning(
            "Only %d chunk(s) of %d GPU(s) for %d adapters — chunks will "
            "train multiple adapters sequentially in waves.",
            n_chunks, gpa, n_adapters,
        )

    # Round-robin adapters across chunks.
    chunk_assignment: List[List[str]] = [[] for _ in chunks]
    for i, adapter in enumerate(adapters):
        chunk_assignment[i % n_chunks].append(adapter)

    logger.info(
        "Parallel training plan: gpus_per_adapter=%d, %d chunks across %d GPUs",
        gpa, n_chunks, n_gpus,
    )
    for chunk, adapter_list in zip(chunks, chunk_assignment):
        if adapter_list:
            launcher = "accelerate(DDP)" if gpa > 1 else "python"
            logger.info(
                "  chunk GPU %s [%s]: %s",
                ",".join(map(str, chunk)), launcher, ", ".join(adapter_list),
            )

    # Split base launcher into "pre-script" args (consumed by accelerate
    # or the shell) and the "-m trainer.SFT.train" tail.  Per-chunk
    # ``--main_process_port`` is injected between them below.
    if gpa > 1:
        # Resolve ``accelerate`` from the same env as the parent
        # interpreter so we don't rely on PATH (the launcher may run
        # from a shell whose conda env differs from the trainer's).
        _accel_bin = Path(sys.executable).parent / "accelerate"
        accelerate_cmd = (
            str(_accel_bin) if _accel_bin.exists() else "accelerate"
        )
        launcher_prefix = [
            accelerate_cmd, "launch",
            "--num_processes", str(gpa),
            "--num_machines", "1",
            "--mixed_precision", "bf16" if config.bf16 else "no",
        ]
        # Opt-in DeepSpeed ZeRO-3 sharding via env var (no behavior change
        # without it).  ZeRO-3 shards params/grads/optim across the
        # ``gpus_per_adapter`` ranks so the 9B+ base model fits on
        # H100-class hardware with LoRA + grad-ckpt.  When
        # ``SFT_DEEPSPEED_CONFIG_FILE`` points at a JSON, we hand it to
        # accelerate verbatim (lets us flip CPU offload on for long
        # sequences without code changes).
        if os.environ.get("SFT_USE_DEEPSPEED", "0") == "1":
            ds_cfg = os.environ.get("SFT_DEEPSPEED_CONFIG_FILE", "")
            if ds_cfg:
                launcher_prefix += [
                    "--use_deepspeed",
                    "--deepspeed_config_file", ds_cfg,
                    "--zero3_init_flag", "true",
                ]
            else:
                launcher_prefix += [
                    "--use_deepspeed",
                    "--zero_stage", os.environ.get("SFT_ZERO_STAGE", "3"),
                    "--gradient_clipping", "1.0",
                    "--zero3_save_16bit_model", "true",
                    "--zero3_init_flag", "true",
                ]
        script_invocation = ["-m", "trainer.SFT.train"]
    else:
        launcher_prefix = [sys.executable]
        script_invocation = ["-m", "trainer.SFT.train"]

    shared_args = [
        "--model_name", config.model_name,
        "--output_dir", config.output_dir,
        "--decision_data_dir", config.decision_data_dir,
        "--skillbank_data_dir", config.skillbank_data_dir,
        "--lr", str(config.lr),
        "--epochs", str(config.epochs),
        "--batch_size", str(config.batch_size),
        "--grad_accum", str(config.grad_accum),
        "--max_seq_length", str(config.max_seq_length),
        "--lora_r", str(config.lora_r),
        "--lora_alpha", str(config.lora_alpha),
        "--lora_dropout", str(config.lora_dropout),
        "--warmup_ratio", str(config.warmup_ratio),
        "--eval_fraction", str(config.eval_fraction),
        "--logging_steps", str(config.logging_steps),
        "--save_steps", str(config.save_steps),
    ]
    if config.bf16:
        shared_args.append("--bf16")
    else:
        shared_args.append("--no_bf16")
    if config.games:
        shared_args.extend(["--games"] + config.games)
    if config.scale_effective_batch != 1.0:
        shared_args.extend(["--scale_effective_batch", str(config.scale_effective_batch)])
    if config.scale_lr != 1.0:
        shared_args.extend(["--scale_lr", str(config.scale_lr)])
    if config.gradient_checkpointing:
        shared_args.append("--gradient_checkpointing")
    else:
        shared_args.append("--no_gradient_checkpointing")
    if not config.use_liger_kernel:
        shared_args.append("--no_liger_kernel")
    if config.optim is not None:
        shared_args.extend(["--optim", config.optim])
    if config.dataloader_num_workers != 4:
        shared_args.extend(["--dataloader_workers", str(config.dataloader_num_workers)])
    if config.strict_lora_coverage:
        shared_args.append("--strict_lora_coverage")

    t0 = time.time()
    processes: List[tuple] = []

    for chunk, adapter_list in zip(chunks, chunk_assignment):
        if not adapter_list:
            continue
        # cmd shape:  [launcher_prefix...] [accelerate launch flags]
        #             [-m trainer.SFT.train] [--adapters ...] [shared args]
        cmd = list(launcher_prefix)
        if gpa > 1:
            # Inject a chunk-unique master port so concurrent accelerate
            # launches on the same node don't collide on rendezvous.
            cmd += ["--main_process_port", str(29500 + chunk[0])]
        cmd += script_invocation
        cmd += shared_args + ["--adapters"] + adapter_list
        # Single-GPU path keeps the legacy ``--gpu N`` flag for log
        # discoverability; multi-GPU path lets accelerate handle pinning
        # via CUDA_VISIBLE_DEVICES (the ``--gpu`` flag would conflict).
        if gpa == 1:
            cmd += ["--gpu", str(chunk[0])]

        chunk_label = "_".join(map(str, chunk))
        log_path = Path(config.output_dir) / f"sft_chunk_{chunk_label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, chunk))

        logger.info(
            "Launching GPUs %s (n=%d) for adapter(s) %s → %s",
            chunk, gpa, adapter_list, log_path,
        )
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        processes.append((chunk, adapter_list, proc, log_file, log_path))

    results: dict = {}
    failed: List[str] = []

    active = list(processes)
    while active:
        still_running = []
        for chunk, adapter_list, proc, log_file, log_path in active:
            ret = proc.poll()
            if ret is None:
                still_running.append((chunk, adapter_list, proc, log_file, log_path))
            else:
                log_file.close()
                if ret == 0:
                    logger.info("GPUs %s finished: %s", chunk, adapter_list)
                    for adapter in adapter_list:
                        results[adapter] = str(config.adapter_output_path(adapter))
                else:
                    logger.error(
                        "GPUs %s FAILED (exit %d): %s — see %s",
                        chunk, ret, adapter_list, log_path,
                    )
                    failed.extend(adapter_list)
        active = still_running
        if not active:
            break
        _print_progress(processes, t0)
        time.sleep(15)

    elapsed = time.time() - t0

    if failed:
        logger.error("Failed adapters: %s", failed)
    logger.info(
        "Parallel SFT complete: %d/%d adapters in %.1f min "
        "(gpus_per_adapter=%d, %d chunk(s) across %d GPUs)",
        len(results), n_adapters, elapsed / 60,
        gpa, n_chunks, n_gpus,
    )

    summary_path = Path(config.output_dir) / "sft_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "adapters": results,
            "failed": failed,
            "model_name": config.model_name,
            "elapsed_min": round(elapsed / 60, 2),
            "parallel": True,
            "gpus_per_adapter": gpa,
            "chunk_assignment": [
                {"gpus": chunk, "adapters": list(adapter_list)}
                for chunk, adapter_list in zip(chunks, chunk_assignment)
                if adapter_list
            ],
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "target_modules": config.resolve_target_modules(),
            "source_only_manifest": os.environ.get("SFT_SOURCE_ONLY_MANIFEST"),
            "source_only_manifest_sha256": os.environ.get("SFT_SOURCE_ONLY_MANIFEST_SHA256"),
        }, f, indent=2)

    return results


def _detect_gpu_count() -> int:
    """Return the number of available CUDA GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def main():
    args = parse_args()
    if args.require_source_only_manifest:
        _validate_source_only_manifest(
            args.require_source_only_manifest,
            args.decision_data_dir or str(
                __import__("trainer.SFT.config", fromlist=["DEFAULT_DECISION_DATA_DIR"])
                .DEFAULT_DECISION_DATA_DIR
            ),
        )
    config = _build_config(args)

    if args.parallel:
        adapters = config.adapters_to_train
        gpa = max(1, int(getattr(args, "gpus_per_adapter", 1)))
        if args.gpus:
            gpu_ids = args.gpus
        else:
            n_gpus = _detect_gpu_count()
            if n_gpus == 0:
                logger.error("--parallel requested but no GPUs detected; falling back to sequential")
                train_all_adapters(config)
                return
            # Default: enough GPUs to fit one chunk per adapter, capped by hardware.
            gpu_ids = list(range(min(n_gpus, len(adapters) * gpa)))
        logger.info(
            "Parallel mode: %d adapters across GPUs %s (gpus_per_adapter=%d)",
            len(adapters), gpu_ids, gpa,
        )
        results = _train_parallel(config, gpu_ids, gpus_per_adapter=gpa)
        missing = sorted(set(adapters) - set(results))
        if missing:
            raise SystemExit(f"parallel SFT failed or skipped adapters: {missing}")
    else:
        train_all_adapters(config, gpu=args.gpu)


if __name__ == "__main__":
    main()
