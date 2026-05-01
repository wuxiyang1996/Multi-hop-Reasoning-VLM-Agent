"""Configuration for the ``schema_gen`` LoRA SFT pipeline.

Mirrors ``trainer.SFT.config.SFTConfig`` but for the **vision** track:
multimodal LoRA over the LLM transformer (and optionally the vision
projector).  Hyper-parameters track PLAN-VISUAL-GROUNDING-MILESTONES
§5.1 and are deliberately conservative.

Default base model: ``Qwen/Qwen3.5-35B-A3B``
--------------------------------------------
Per the official model card, ``Qwen/Qwen3.5-35B-A3B`` is a unified
**vision-language foundation** (early-fusion training on multimodal
tokens, with image + video support), built on a sparse Mixture-of-
Experts spine (35B total params, ~3B active).  It exposes the
standard Qwen-VL chat-template tokens (``<|vision_start|>``,
``<|image_pad|>``, ``<|video_pad|>``, ``<|vision_end|>``) and is
loaded through ``transformers.AutoModelForVision2Seq`` /
``transformers.Qwen3_5ForConditionalGeneration``.

Smaller / cheaper alternatives are available via ``--model_name``:

* ``Qwen/Qwen3-VL-8B-Instruct`` — single-A100 smoke runs.
* ``Qwen/Qwen3-VL-32B``         — dense 32B vision sibling.
* ``Qwen/Qwen3-VL-235B-A22B``   — larger MoE teacher (Phase-F).

Memory / hardware notes
-----------------------
* The 35B-A3B base needs the full 35B params resident for forward
  passes (only ~3B are *active* per token, but the routing layer can
  hit any expert).  Plan for ZeRO-3 / FSDP across ≥2× H100-80GB or
  ≥4× A100-80GB.  ``run_schema_gen.sh`` exposes a deepspeed/FSDP flag
  that wires this in via ``accelerate launch`` if requested.
* The 8B variant fits on one A100-80GB with the conservative defaults
  below (batch_size=1, grad_accum=16, gradient_checkpointing=True).

LoRA targets for the MoE
------------------------
For the 35B-A3B MoE, PEFT's ``target_modules`` matching is by *suffix
name*, so listing ``q_proj`` / ``gate_proj`` etc. adapts the linear
inside *every* expert (``model.layers.<i>.mlp.experts.<j>.gate_proj``).
We adapt the LM-side linears by default.  The vision tower stays
frozen — visual features are already strong from pre-training, and
only the LM has to learn to *describe* them in our schema dialect.
The vision projector path differs across Qwen-VL minor versions (8B
uses ``visual.merger.mlp.0/2``; 35B-A3B's projector identifier may
differ); leave it unset by default and opt in via
``--lora_target_modules`` once the actual nn.Module names have been
inspected on the loaded model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---------------------------------------------------------------------------
# Default data sources
# ---------------------------------------------------------------------------
# These dirs are populated by ``labeling/build_schema_gen_triples.py``
# (gymv + env_wrappers from cold-start rollouts) and the legacy
# ``labeling/grounding/collect_browser.py`` / benchmark parsers
# (browser + image_qa + video_qa).
DEFAULT_GYMV_TRIPLE_ROOT = REPO_ROOT / "labeling" / "output" / "grounding" / "gymv"
DEFAULT_ENV_WRAPPERS_TRIPLE_ROOT = (
    REPO_ROOT / "labeling" / "output" / "grounding" / "env_wrappers"
)
DEFAULT_BROWSER_TRIPLE_ROOT = REPO_ROOT / "labeling" / "output" / "grounding" / "browser"
DEFAULT_IMAGE_QA_JSONL = REPO_ROOT / "labeling" / "output" / "grounding" / "image_qa" / "labels.jsonl"
DEFAULT_VIDEO_QA_JSONL = REPO_ROOT / "labeling" / "output" / "grounding" / "video_qa" / "labels.jsonl"


# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------
# Default LoRA targets: the seven LM-side linears found in every
# Qwen3-family transformer block (attention + MLP).  In a MoE such as
# Qwen3.5-35B-A3B PEFT matches by suffix, so each named projection is
# adapted across *all* experts of every layer.
#
# The vision tower stays frozen — pre-training gives strong visual
# features and we only need the LM to learn the schema dialect.  To
# adapt the vision projector additionally, override
# ``lora_target_modules`` after inspecting the loaded model's
# ``named_modules()`` (the projector path differs across Qwen-VL minor
# versions, e.g. ``visual.merger.mlp.{0,2}`` on Qwen3-VL-8B).
QWEN3_LM_LORA_TARGETS: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
# Backwards-compat alias — older configs imported this name.
QWEN3_VL_LORA_TARGETS: List[str] = list(QWEN3_LM_LORA_TARGETS)


@dataclass
class SchemaGenConfig:
    """Configuration for one ``schema_gen`` SFT run.

    A single config trains one adapter on the union of the requested
    domains.  The Phase-1 milestone calls for a single multi-domain
    adapter (``gymv`` + ``browser`` + ``image_qa`` + ``video_qa``)
    because cross-domain transfer is part of what we're measuring;
    ablation runs can override ``domains`` to train per-domain heads.
    """

    # ── Base model ────────────────────────────────────────────────────
    # Default = Qwen3.5-35B-A3B (vision-language MoE; see module
    # docstring).  Override with ``--model_name``:
    #   * Qwen/Qwen3-VL-8B-Instruct   — fast smoke run on one A100.
    #   * Qwen/Qwen3-VL-32B           — dense 32B vision sibling.
    #   * Qwen/Qwen3-VL-235B-A22B     — larger MoE teacher (Phase-F).
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    trust_remote_code: bool = True
    bf16: bool = True

    # ── Adapter identity ──────────────────────────────────────────────
    adapter_name: str = "schema_gen"
    run_id: str = field(
        default_factory=lambda: datetime.now().strftime("schema_gen_%Y%m%d_%H%M%S"),
    )

    # ── Data sources ──────────────────────────────────────────────────
    # Default to the two corpora we actually have on disk (gymv +
    # env_wrappers). browser / image_qa / video_qa are still supported
    # but opt-in via ``--domains``.
    domains: List[str] = field(
        default_factory=lambda: ["gymv", "env_wrappers"],
    )
    gymv_triple_root: str = str(DEFAULT_GYMV_TRIPLE_ROOT)
    env_wrappers_triple_root: str = str(DEFAULT_ENV_WRAPPERS_TRIPLE_ROOT)
    browser_triple_root: str = str(DEFAULT_BROWSER_TRIPLE_ROOT)
    image_qa_jsonl: str = str(DEFAULT_IMAGE_QA_JSONL)
    video_qa_jsonl: str = str(DEFAULT_VIDEO_QA_JSONL)

    # When both heuristic and vision schemas are present we default to
    # the gpt-5.5 vision schema as the SFT target — it's what Phase-1
    # is teaching the student to mimic.  Setting this to ``"heuristic"``
    # is useful for an ablation that measures how much extra signal the
    # SFT vision teacher contributes.
    target_source: str = "vision"  # "vision" | "heuristic" | "auto"

    # Filter out triples flagged as Path-B / Path-C hard cases by
    # ``cross_validate.py``.  Default True for Phase-1; flip to False
    # for the §9 ablation that includes the entire collected pool.
    drop_hard_cases: bool = True
    hard_cases_jsonl: Optional[str] = None  # auto-detected when None

    max_samples_per_domain: Optional[int] = None  # cap for smoke tests

    # ── Training hyper-parameters ─────────────────────────────────────
    output_dir: str = str(REPO_ROOT / "runs" / "sft_schema_gen")
    lr: float = 1e-4
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 16
    max_seq_length: int = 4096
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    eval_fraction: float = 0.05
    seed: int = 42

    # ── LoRA ──────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: list(QWEN3_LM_LORA_TARGETS),
    )

    # ── Vision pre-processing ─────────────────────────────────────────
    image_max_side: int = 1024
    video_num_frames: int = 8

    # ── Logging / checkpointing ───────────────────────────────────────
    logging_steps: int = 5
    save_steps: int = 200
    save_total_limit: int = 3
    eval_steps: int = 200
    report_to: List[str] = field(default_factory=lambda: ["none"])

    # ── Optional accelerate config ────────────────────────────────────
    gradient_checkpointing: bool = True
    # Flash-Attention 2 is preferred when the `flash-attn` wheel is available.
    # When it is not (e.g. torch 2.11+cu130 has no prebuilt wheel yet) the
    # trainer transparently falls back to PyTorch SDPA, which is fast on
    # H200/bf16 and needs no extra dependency.  Set to False to skip the
    # flash-attn import probe entirely.
    use_flash_attention: bool = True

    # ------------------------------------------------------------------
    def adapter_output_dir(self) -> Path:
        """Resolved output directory for the adapter checkpoint."""
        return Path(self.output_dir) / self.run_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "run_id": self.run_id,
            "domains": list(self.domains),
            "gymv_triple_root": self.gymv_triple_root,
            "env_wrappers_triple_root": self.env_wrappers_triple_root,
            "browser_triple_root": self.browser_triple_root,
            "image_qa_jsonl": self.image_qa_jsonl,
            "video_qa_jsonl": self.video_qa_jsonl,
            "target_source": self.target_source,
            "drop_hard_cases": self.drop_hard_cases,
            "hard_cases_jsonl": self.hard_cases_jsonl,
            "max_samples_per_domain": self.max_samples_per_domain,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_seq_length": self.max_seq_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
            "image_max_side": self.image_max_side,
            "video_num_frames": self.video_num_frames,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_flash_attention": self.use_flash_attention,
            "seed": self.seed,
        }
