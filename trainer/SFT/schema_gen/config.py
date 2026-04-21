"""Configuration for the ``schema_gen`` LoRA SFT pipeline.

Mirrors ``trainer.SFT.config.SFTConfig`` but for the **vision** track
(Qwen3-VL-8B + multimodal LoRA over the LLM transformer + the vision
projector).  Hyper-parameters track PLAN-VISUAL-GROUNDING-MILESTONES
§5.1 — they are deliberately conservative so a first run on a single
A100 80 GB completes in ≤24 h.
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
# Default data sources (PLAN-V-G-MILESTONES §5)
# ---------------------------------------------------------------------------
DEFAULT_GYMV_TRIPLE_ROOT = REPO_ROOT / "labeling" / "output" / "grounding" / "gymv"
DEFAULT_BROWSER_TRIPLE_ROOT = REPO_ROOT / "labeling" / "output" / "grounding" / "browser"
DEFAULT_IMAGE_QA_JSONL = REPO_ROOT / "labeling" / "output" / "grounding" / "image_qa" / "labels.jsonl"
DEFAULT_VIDEO_QA_JSONL = REPO_ROOT / "labeling" / "output" / "grounding" / "video_qa" / "labels.jsonl"


# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------
# Qwen3-VL exposes the same transformer linears as Qwen3 plus the
# vision projector (``visual.merger.mlp.*``).  We adapt the LM head
# linears (q/k/v/o + gate/up) and the projector — leaving the patch
# embedding / vision encoder frozen which keeps the parameter count
# small while still letting the model learn how to *describe* what the
# vision tower extracts.
QWEN3_VL_LORA_TARGETS: List[str] = [
    # LM transformer
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj",
    # Vision projector (Qwen3-VL merger MLP) — gives us a small
    # additional capacity to learn pos/bbox-aware visual features
    # without unfreezing the ViT.
    "visual.merger.mlp.0", "visual.merger.mlp.2",
]


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
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    trust_remote_code: bool = True
    bf16: bool = True

    # ── Adapter identity ──────────────────────────────────────────────
    adapter_name: str = "schema_gen"
    run_id: str = field(
        default_factory=lambda: datetime.now().strftime("schema_gen_%Y%m%d_%H%M%S"),
    )

    # ── Data sources ──────────────────────────────────────────────────
    domains: List[str] = field(
        default_factory=lambda: ["gymv", "browser", "image_qa", "video_qa"],
    )
    gymv_triple_root: str = str(DEFAULT_GYMV_TRIPLE_ROOT)
    browser_triple_root: str = str(DEFAULT_BROWSER_TRIPLE_ROOT)
    image_qa_jsonl: str = str(DEFAULT_IMAGE_QA_JSONL)
    video_qa_jsonl: str = str(DEFAULT_VIDEO_QA_JSONL)

    # When both heuristic and vision schemas are present we default to
    # the GPT-4o vision schema as the SFT target — it's what Phase-1
    # is teaching the student to mimic.  Setting this to ``"heuristic"``
    # is useful for an ablation that measures how much extra signal the
    # GPT-4o teacher contributes.
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
        default_factory=lambda: list(QWEN3_VL_LORA_TARGETS),
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
