"""Configuration for the co-evolution training loop."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))


SKILL_BANK_GAMES = [
    "twenty_forty_eight",
    "tetris",
    "candy_crush",
    "super_mario",
    # Phase-1 cross-game curriculum: 4 Gym-V Temporal/* games dispatched
    # via env_wrappers/gymv_temporal_nl_wrapper.py.  These games run at
    # frame_skip=8 (StochasticFrameSkip), so each agent step ≈ 8 emulator
    # frames and the per-episode cap below is in agent steps.
    "gymv_space_harrier_ii",
    "gymv_streets_of_rage_2",
    "gymv_columns",
    "gymv_strider",
]

# Evaluation-only: rollouts collected for metrics but NOT fed into GRPO training.
EVAL_ONLY_GAMES: List[str] = []

GAME_MAX_STEPS: Dict[str, int] = {
    "twenty_forty_eight": 200,
    "tetris": 200,
    "candy_crush": 50,
    "super_mario": 500,
    # Gym-V Temporal/* @ frame_skip=8.  200 agent steps ≈ 1600 emulator
    # frames ≈ 27 s of in-game time at 60 Hz, which comfortably covers
    # the ~100-step paper Table-3 anchor episodes for all 4 games while
    # still bounding the worst case (long Streets-of-Rage 2 stages).
    "gymv_space_harrier_ii": 200,
    "gymv_streets_of_rage_2": 200,
    "gymv_columns": 200,
    "gymv_strider": 200,
}

EMULATOR_GAMES: set = set()

# ── Multi-role rollout constants ─────────────────────────────────────
# When ``unified_role_rollouts`` is enabled, the same decision agent
# plays ALL roles and each rollout is tagged with role / side metadata.
# Per-game episode overrides ensure sufficient role coverage.

EPISODES_PER_GAME_MULTIROLE: Dict[str, int] = {}


def resolve_bank_key(game: str, role: str = "", side: str = "") -> str:
    """Return the skill-bank routing key for an episode.

    In unified-role mode the key may encode a role dimension. For standard
    games the key is just the game name.

    Examples::

        resolve_bank_key("tetris") -> "tetris"
    """
    return game


def bank_keys_for_game(game: str) -> List[str]:
    """Return all possible bank keys for a game.

    Used by ``PerGameSkillBankManager`` to pre-create sub-bank pipelines.
    """
    return [game]


GAME_DURATION_ORDER = [
    "twenty_forty_eight",
    "tetris",
    "candy_crush",
    "super_mario",
]

ADAPTER_NAMES = [
    "skill_selection",
    "action_taking",
    "segment",
    "contract",
    "curator",
]

# ── Curriculum presets ───────────────────────────────────────────────
# Each maps step thresholds → active game lists.

CURRICULUM_GRADUAL: Dict[int, List[str]] = {
    0: ["twenty_forty_eight", "tetris", "candy_crush"],
    10: ["twenty_forty_eight", "tetris", "candy_crush", "super_mario"],
}

CURRICULUM_FOCUSED: Dict[int, List[str]] = {
    0: ["twenty_forty_eight", "tetris", "candy_crush", "super_mario"],
}

CURRICULUM_PRESETS: Dict[str, Optional[Dict[int, List[str]]]] = {
    "gradual": CURRICULUM_GRADUAL,
    "focused": CURRICULUM_FOCUSED,
    "none": None,
}


def _model_short_name(model_name: str) -> str:
    """Extract a filesystem-safe short name from a model identifier.

    ``"Qwen/Qwen3.5-9B"`` → ``"Qwen3.5-9B"``
    ``"meta-llama/Llama-3-8B"`` → ``"Llama-3-8B"``
    """
    short = model_name.rsplit("/", 1)[-1]
    return re.sub(r"[^\w\-.]", "_", short)


def _generate_run_dir(model_name: str) -> str:
    """Generate a unique run directory name from model name + timestamp.

    Example: ``runs/Qwen3.5-9B_20260427_131800``
    """
    short = _model_short_name(model_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("runs") / f"{short}_{ts}")


@dataclass
class CoEvolutionConfig:
    """Top-level configuration for the co-evolution loop."""

    games: List[str] = field(default_factory=lambda: list(SKILL_BANK_GAMES))
    eval_games: List[str] = field(default_factory=lambda: list(EVAL_ONLY_GAMES))
    episodes_per_game: int = 4
    eval_episodes_per_game: int = 3

    # ── Unified multi-role rollout mode ──────────────────────────
    # When True, per-game episode counts follow ``episodes_per_game_overrides``.
    # When False (default), ``episodes_per_game`` applies uniformly to every game.
    unified_role_rollouts: bool = False
    episodes_per_game_overrides: Dict[str, int] = field(
        default_factory=lambda: dict(EPISODES_PER_GAME_MULTIROLE),
    )

    max_concurrent_episodes: int = 64
    total_steps: int = 60

    # GPU allocation — split between persistent vLLM and FSDP training.
    # vLLM instances (TP=1) run on vllm_gpu_ids and stay up for the
    # entire training run.  After each GRPO step, adapters are hot-reloaded
    # via the vLLM API (no restart required).
    vllm_gpu_ids: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3],
    )
    grpo_devices: List[int] = field(
        default_factory=lambda: [4, 5, 6, 7],
    )

    # vLLM inference — base model that GRPO trains and that vLLM serves.
    # ``Qwen/Qwen3.5-9B`` is the multimodal hybrid Gated-DeltaNet + Gated-Attention
    # text decoder we LoRA-tune on.  For text-only rollouts the vision tower
    # is skipped via ``--language-model-only`` (see vllm_server.py).
    model_name: str = "Qwen/Qwen3.5-9B"
    temperature: float = 0.5
    max_tokens: int = 512
    vllm_base_url: str = "http://localhost:8000/v1"  # used only when manage_vllm=False
    vllm_base_port: int = 8000
    vllm_gpu_util: float = 0.95

    # Speculative decoding.  Two methods are supported:
    #   "mtp"          — use the model's built-in Multi-Token-Prediction head
    #                    (Qwen3.5-9B / Qwen3.5-MoE / Qwen3-Next ship one).
    #                    No draft model needed.  Recommended for Qwen3.5.
    #   "draft_model"  — use a small external model (e.g. Qwen3-0.6B) as the
    #                    drafter.  Required for plain Qwen3 (no MTP head).
    speculative_method: str = "mtp"
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 1

    # Inference-only secondary model (e.g. ``Qwen/Qwen3.5-35B-A3B``) — NOT
    # trained, served separately for evaluation, baselines, or as a teacher.
    # See ``inference/serve_qwen35_35b_a3b.sh`` for the standalone launcher;
    # this field is purely informational so config snapshots record which
    # inference model the run was paired with.
    inference_only_model: Optional[str] = "Qwen/Qwen3.5-35B-A3B"

    # When True, the orchestrator manages vLLM server lifecycle
    # (persistent instances on vllm_gpu_ids, hot-reload after GRPO).
    manage_vllm: bool = True

    # External opponent API (reserved for future multi-agent games).
    opponent_model: Optional[str] = None
    opponent_api_base: Optional[str] = "https://openrouter.ai/api/v1"

    # Skill bank EM
    em_max_iterations: int = 3
    em_micro_batch_size: int = 8

    # GRPO
    grpo_enabled: bool = True
    # Deprecated: kept for backward compat only.
    grpo_decision_devices: List[int] = field(default_factory=list)
    grpo_skillbank_devices: List[int] = field(default_factory=list)

    # ── Crafter + Promotion online hooks (Phase 1, off by default) ─────
    # Spec: implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md §3
    # When enabled, after every co-evolution step's Phase B finalize() the
    # trainer additionally runs:
    #   1. ``trainer.coevolution._crafter_hook.run_crafter_step`` —
    #      synthesizes FailureTraces from the just-finished EpisodeResults,
    #      calls SkillCrafterService.reflect_on_episode, dumps proposals
    #      in the offline-mirror JSONL schema.
    #   2. ``trainer.coevolution._promotion_hook.run_promotion_step`` —
    #      subprocess-invokes ``decide_promotion_gpt54.py --gate-mode
    #      offline-synthetic`` over those proposals, then writes the
    #      promoted (PROVISIONAL) skills back into each per-game
    #      ``skill_bank.jsonl`` via ``skill_bank.legacy_writeback``.
    # The hook is one-way (D8 Option A): the legacy 4-stage skill_agents
    # pipeline keeps writing to per-game banks unchanged, the new path
    # only *appends* promoted skills.
    crafter_promotion_enabled: bool = False
    # Per-batch Crafter cycle cadence. ``0`` disables cycle() entirely
    # (per-episode reactive pass still runs).  Recommend 5-10 in steady
    # state — runs Composer / Generalizer over accumulated failures.
    crafter_cycle_every_k_steps: int = 0
    # Total reward at/below this threshold counts as OUTCOME_FAILURE.
    # ``0.0`` matches the offline mirror's default.
    crafter_outcome_failure_threshold: float = 0.0
    # Hard wall-clock cap on the decide_promotion subprocess invocation.
    # Phase-0 Airstriker baseline is ~0.3s; 300s leaves room for the
    # 13-game sweep + future Stage-1 replay overhead.
    crafter_promotion_timeout_s: float = 300.0
    # Lane-(a) feature flag. ``False`` (default) parks the Repairer /
    # PatchProposal mint path: under
    # ``implementation_notes/legacy/skill-lane-decision.md`` skills are
    # retrieval payloads, not runnable programs, so live protocol-edit
    # proposals would be edits to a contract no live runtime executes.
    # The Crafter dispatcher's existing ``_STATUS_NO_OP`` →
    # Hypothesizer fall-through carries the failure signal through.
    # Set ``True`` only when running explicit lane-(b) experiments.
    crafter_enable_protocol_patching: bool = False

    # ── Cross-domain transfer gate (Layer A) ────────────────────────
    # When ``True`` (and ``crafter_promotion_enabled=True``), the
    # trainer inserts ``trainer.coevolution._transfer_hook.run_transfer_gate_step``
    # after the promotion writeback. The gate re-evaluates each
    # just-promoted skill's cross-domain admit rate against the
    # configured ``crafter_transfer_targets`` and rolls back
    # promotions that fall below ``crafter_transfer_admit_band[0]``
    # on every target.
    #
    # The matrix subprocess is the same
    # ``labeling_supplement/_phase4_transfer_matrix.py`` driver the
    # Stage-6 measurement uses. Per-step gating only makes sense when
    # ``crafter_promotion_enabled`` already runs at low cadence (e.g.
    # every K steps); see
    # ``implementation_notes/coevolution-cross-domain-integration.md``
    # §4 for the full contract + acceptance criteria.
    crafter_transfer_gate_enabled: bool = False
    # Target corpora the matrix driver evaluates against. Subset of
    # the canonical Stage-6 cluster set (video / visual_reasoning /
    # osworld / browser). Empty list disables the gate at runtime
    # without flipping the master flag.
    crafter_transfer_targets: Tuple[str, ...] = ("video", "visual_reasoning")
    # ``(lower, upper)`` admit-rate band per §11.5.4 of the cross-
    # domain measurement plan. Skills failing ``admit_rate < lower``
    # on every target are DEMOTED. ``upper`` is informational (Layer
    # D dashboard surfaces it for the Generalizer's signal).
    crafter_transfer_admit_band: Tuple[float, float] = (0.15, 0.60)
    # Minimum number of targets a skill must clear ``band[0]`` on to
    # KEEP. ``1`` is the most permissive: a single transferable
    # target is enough to keep the promotion.
    crafter_transfer_min_targets_in_band: int = 1
    # Forwarded to ``_phase4_transfer_matrix.py --max-skills``. The
    # synthetic bank-run is already filtered to just-promoted skills
    # so this caps the per-cell sweep within that subset.
    crafter_transfer_max_skills_per_cell: int = 5
    # Hard wall-clock cap on the matrix subprocess. The driver loads
    # 1-2 demos per cell; conda-env helpers (browser) cold-boot
    # adds ~30s; 1800s is a generous ceiling for K=4 targets × N=5
    # skills.
    crafter_transfer_timeout_s: float = 1800.0

    # ── Cross-domain dashboard (Layer D) ────────────────────────────
    # Periodic offline pass that snapshots the trainer's banks and
    # runs the full N×N Stage-6 transfer matrix to surface the live
    # cross-domain transfer health (G1-G5 acceptance gates, per-
    # cluster admit rates) to wandb / TensorBoard. Decoupled from
    # the per-step transfer gate (Layer A) so dashboards can run at
    # low frequency (every 100 steps) while the gate runs at
    # higher frequency without doubling the subprocess load.
    #
    # Spec:  implementation_notes/coevolution-cross-domain-integration.md §5
    crafter_dashboard_enabled: bool = False
    # Cadence: dashboard fires when ``step % crafter_dashboard_every_k_steps == 0``.
    # ``0`` disables the dashboard via cadence (master flag stays
    # untouched). Recommend 100-500 in steady state — a full N×N
    # sweep takes ~1-30 minutes depending on bank size + N targets.
    crafter_dashboard_every_k_steps: int = 0
    # Target corpora the dashboard's matrix sweep evaluates against.
    # Typically the full canonical Stage-6 set so the wandb dashboard
    # shows complete G3-G5 verdicts.
    crafter_dashboard_targets: Tuple[str, ...] = (
        "video", "visual_reasoning", "osworld", "browser",
    )
    # Forwarded to ``_phase4_transfer_matrix.py --max-skills``.
    crafter_dashboard_max_skills_per_cell: int = 5
    # Hard wall-clock cap on the matrix subprocess. Default 1h leaves
    # room for the full N×N sweep with conda-env helpers (browser)
    # cold-booting.
    crafter_dashboard_timeout_s: float = 3600.0

    # ── Harness wire-up (Day-10) ────────────────────────────────────
    # When enabled, the trainer wires the harness's two LLM-free
    # surfaces into Phase A rollouts:
    #   1. ``SkillHarness.select_eligible_skills`` filters the cold-
    #      start RAG candidates *before* the skill_selection LLM picks.
    #   2. ``SkillHarness.validate_invocation`` second-pass-vetoes the
    #      LLM's chosen skill, falling back to the next eligible one
    #      when vetoed.
    # The aggregated rejection patterns ride into Phase B′ via
    # ``RejectedSkillSink → SkillLifecycleManager.record_false_binding_pattern``
    # so the Crafter's Repairer sees richer evidence on each
    # ``SkillRecord.false_binding_patterns`` (PLAN-SKILL-BANK §4.3b).
    # See ``trainer/coevolution/_harness_hook.py`` for the full
    # contract; ``harness/README.md`` §22 for the spec gap this closes.
    harness_enabled: bool = False
    # Whether SHADOW skills are admitted by the eligibility filter.
    # Trainer default ``True`` mirrors ``HarnessConfig.allow_shadow``.
    # Disable with ``--no-harness-allow-shadow`` for runs that must
    # bind only fully-validated (ACTIVE / PROVISIONAL) skills.
    harness_allow_shadow: bool = True

    # Run directory — all other dirs are relative to this.
    # Auto-generated from model_name + timestamp if None.
    run_dir: Optional[str] = None

    # Directories (rebased under run_dir by resolve_paths())
    bank_dir: str = "skillbank"
    adapter_dir: str = "lora_adapters"  # parent; decision/ and skillbank/ live under this
    checkpoint_dir: str = "checkpoints"
    log_dir: str = ""  # root of run_dir
    grpo_data_dir: str = "grpo_data"
    rewards_dir: str = "rewards"
    tensorboard_dir: str = "tensorboard"
    debug_io_dir: str = "debug_io"

    # T2.4 — single reward sink. When set (auto-resolved by
    # ``resolve_paths`` to ``{rewards_dir}/reward_log.jsonl`` if left
    # empty), the orchestrator constructs a ``harness.RewardLogger``
    # against this path and threads it through every rollout. Each
    # ``GRPORecord`` written by the trainer is mirrored into the same
    # JSONL (``kind="grpo_step"``) so eval and training read from one
    # source. Set to ``""`` to disable.
    reward_log_path: str = ""

    # T2.7 — curator overfit mitigation. The CURATOR LoRA receives a
    # scaled GRPO reward equal to ``base * min(1, step / warmup) *
    # curator_weight``. With ``curator_warmup_steps=0`` (default) the
    # ramp is disabled and the scalar reward passes through unchanged
    # (``curator_weight`` still applies as a constant). Set
    # ``curator_warmup_steps`` to a small positive integer (e.g. 50)
    # to dampen early-training noise; set ``curator_weight`` < 1.0
    # for a permanent down-weight (the curator becomes a tie-breaker
    # rather than a hard gate). Wired via
    # ``skill_agents.bank_maintenance.llm_curator.set_curator_warmup``
    # at the start of every outer-loop step.
    curator_weight: float = 1.0
    curator_warmup_steps: int = 0

    # Debug: log every LLM I/O and GRPO sample to disk for inspection
    debug_io: bool = False

    # Checkpointing
    checkpoint_interval: int = 5
    # How many per-step checkpoints to keep on disk.  Set to 0 to keep ALL.
    # The "best" checkpoint (step_99999) is never deleted regardless.
    checkpoint_keep_last: int = 0

    # Number of consecutive reward declines before rolling back to the
    # best checkpoint.  Higher values give the optimizer more room to
    # recover from temporary dips; lower values are more conservative.
    rollback_patience: int = 4

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "game-ai-coevolution"
    wandb_run_name: Optional[str] = None

    # Start mode:
    #   "from_scratch" — random-init all LoRA adapters, ignore any checkpoint
    #   "resume"       — resume from latest (or specific) checkpoint
    #   "auto"         — resume if checkpoint exists, else from scratch
    start_mode: str = "auto"
    resume_from_step: Optional[int] = None

    # Load pre-trained adapters instead of random init.
    # Maps adapter name → path to an existing adapter directory.
    # Only used when start_mode != "resume" (resume loads from checkpoint).
    # Example: {"skill_selection": "prev_run/lora/skill_selection", ...}
    pretrained_adapter_paths: Dict[str, str] = field(default_factory=dict)

    # Seed each per-game skill bank from a cold-start directory on first
    # launch.  Expected layout: ``<seed_bank_dir>/<game>/skill_bank.jsonl``.
    # Skills are copied only when the game's bank is empty; once the
    # co-evolution loop adds its own skills, the seed is never re-applied.
    seed_bank_dir: Optional[str] = None

    # Thread/process executors
    thread_workers: int = 64
    process_workers: int = 8

    # Early episode termination
    stuck_window: int = 15
    min_steps_before_stuck_check: int = 20

    # Rollout batching synchronizer — prevents episodes from
    # desynchronizing and losing vLLM request batching (which causes
    # 10-20x throughput loss due to the GPU batch-size cliff).
    rollout_sync_timeout_s: float = 0.10

    # LoRA adapter defaults (matches skill_agents.lora.config)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    # "gaussian" → both A and B get small random init (better for GRPO)
    # True       → Kaiming A + zero B (standard LoRA, B gets no grad initially)
    lora_init_weights: Any = "gaussian"

    # Curriculum learning — two-phase focused training.
    # Phase 1 (steps 0-39): 4 simpler single-player games to build core skills.
    # Phase 2 (steps 40-59): focus on harder social/strategic games (Avalon, Diplomacy).
    # Set to None for no curriculum (all games from step 0).
    # Use CURRICULUM_PRESETS["gradual"] for the old incremental schedule.
    curriculum_schedule: Optional[Dict[int, List[str]]] = field(
        default_factory=lambda: dict(CURRICULUM_FOCUSED)
    )

    # GRPO from-scratch schedule — higher exploration early, then anneal.
    # Only applied when start_mode == "from_scratch" (otherwise GRPO
    # configs use the default values from StageGRPOConfig).
    scratch_warmup_steps: int = 20
    scratch_initial_lr: float = 1e-4
    scratch_steady_lr: float = 5e-5
    scratch_initial_temperature: float = 1.0
    scratch_steady_temperature: float = 0.7
    scratch_initial_kl_coeff: float = 0.01
    scratch_steady_kl_coeff: float = 0.05

    # Per-run GRPO overrides (set via CLI, leave None to use defaults)
    grpo_clip_ratio: float = 0.2
    grpo_max_epochs: int = 4
    grpo_adv_clip: Optional[float] = None

    _resolved: bool = field(default=False, repr=False)

    def resolve_paths(self) -> "CoEvolutionConfig":
        """Rebase all directory paths under ``run_dir``.

        If ``run_dir`` is ``None``, generates one from the model name
        and current timestamp (e.g. ``runs/Qwen3.5-9B_20260427_131800``).

        Idempotent — calling twice is safe.
        """
        if self._resolved:
            return self

        if self.run_dir is None:
            self.run_dir = _generate_run_dir(self.model_name)

        root = Path(self.run_dir).resolve()
        self.run_dir = str(root)

        def _rebase(rel: str) -> str:
            p = Path(rel)
            if p.is_absolute():
                return rel
            return str(root / rel) if rel else str(root)

        self.bank_dir = _rebase(self.bank_dir)
        self.adapter_dir = _rebase(self.adapter_dir)
        self.checkpoint_dir = _rebase(self.checkpoint_dir)
        self.log_dir = _rebase(self.log_dir) if self.log_dir else str(root)
        self.grpo_data_dir = _rebase(self.grpo_data_dir)
        self.rewards_dir = _rebase(self.rewards_dir)
        self.tensorboard_dir = _rebase(self.tensorboard_dir)
        self.debug_io_dir = _rebase(self.debug_io_dir)
        # T2.4 — auto-place the unified reward log under ``rewards_dir``
        # when the field was left at its default empty string. Explicit
        # ``""`` after ``resolve_paths()`` means the user disabled the
        # sink (already-resolved paths are not re-rebased).
        if self.reward_log_path == "":
            self.reward_log_path = str(Path(self.rewards_dir) / "reward_log.jsonl")
        elif not Path(self.reward_log_path).is_absolute():
            self.reward_log_path = str(root / self.reward_log_path)

        self._resolved = True
        return self

    @property
    def effective_grpo_devices(self) -> List[int]:
        """GPU IDs used for GRPO training (FSDP data-parallel)."""
        return self.grpo_devices

    @property
    def vllm_base_urls(self) -> List[str]:
        """vLLM base URLs for the inference client.

        Returns one URL per vLLM GPU when managed, or a single URL
        when the user runs vLLM externally.
        """
        if self.manage_vllm:
            return [
                f"http://localhost:{self.vllm_base_port + i}/v1"
                for i in range(len(self.vllm_gpu_ids))
            ]
        return [self.vllm_base_url]

    @property
    def decision_adapter_dir(self) -> str:
        return str(Path(self.adapter_dir) / "decision")

    @property
    def skillbank_adapter_dir(self) -> str:
        return str(Path(self.adapter_dir) / "skillbank")

    def adapter_path(self, name: str) -> str:
        if name in ("skill_selection", "action_taking"):
            return str(Path(self.decision_adapter_dir) / name)
        return str(Path(self.skillbank_adapter_dir) / name)

    def get_episodes_for_game(self, game: str) -> int:
        """Return the episode count for *game*.

        In unified-role mode, per-game overrides are applied
        (5 for Avalon, 7 for Diplomacy by default).  Otherwise
        the global ``episodes_per_game`` is returned for all games.
        """
        if self.unified_role_rollouts:
            return self.episodes_per_game_overrides.get(
                game, self.episodes_per_game,
            )
        return self.episodes_per_game

    def active_games(self, step: int) -> List[str]:
        """Return the list of active games for the given training step.

        Uses ``curriculum_schedule`` when set, otherwise returns all games.
        """
        if not self.curriculum_schedule:
            return list(self.games)
        thresholds = sorted(k for k in self.curriculum_schedule if k <= step)
        if not thresholds:
            return list(self.games)
        return list(self.curriculum_schedule[thresholds[-1]])

    def curriculum_description(self) -> str:
        """Human-readable summary of the curriculum schedule."""
        if not self.curriculum_schedule:
            return "none (all games every step)"
        phases = sorted(self.curriculum_schedule.items())
        parts = []
        for i, (start, games) in enumerate(phases):
            end = phases[i + 1][0] - 1 if i + 1 < len(phases) else self.total_steps - 1
            parts.append(f"  steps {start}–{end}: {', '.join(games)}")
        return "focused curriculum\n" + "\n".join(parts)

    def grpo_schedule(self, step: int) -> Dict[str, float]:
        """Return GRPO hyperparameters for the current step.

        During from-scratch training, the first ``scratch_warmup_steps``
        use higher learning rate, higher sampling temperature (more
        exploration), and lower KL penalty (allow larger policy shifts).
        After warmup, LR follows cosine decay to a minimum of 10% of
        steady-state.  Temperature and KL hold at steady values.
        """
        import math as _math

        if self.start_mode != "from_scratch":
            total = max(1, self.total_steps)
            progress = min(1.0, step / total)
            lr_min = self.scratch_steady_lr * 0.3
            lr = lr_min + 0.5 * (self.scratch_steady_lr - lr_min) * (
                1.0 + _math.cos(_math.pi * progress)
            )
            kl = self.scratch_steady_kl_coeff
            return {
                "lr": lr,
                "temperature": self.scratch_steady_temperature,
                "kl_coeff": kl,
            }

        w = self.scratch_warmup_steps
        total = self.total_steps

        if w <= 0 or step >= w:
            warmup_alpha = 1.0
        else:
            warmup_alpha = step / w

        def _lerp(init: float, steady: float) -> float:
            return init + warmup_alpha * (steady - init)

        if step < w:
            lr = _lerp(self.scratch_initial_lr, self.scratch_steady_lr)
        else:
            decay_steps = max(1, total - w)
            progress = min(1.0, (step - w) / decay_steps)
            lr_min = self.scratch_steady_lr * 0.1
            lr = lr_min + 0.5 * (self.scratch_steady_lr - lr_min) * (
                1.0 + _math.cos(_math.pi * progress)
            )

        return {
            "lr": lr,
            "temperature": _lerp(
                self.scratch_initial_temperature,
                self.scratch_steady_temperature,
            ),
            "kl_coeff": _lerp(
                self.scratch_initial_kl_coeff,
                self.scratch_steady_kl_coeff,
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict for config.json persistence."""
        d = asdict(self)
        d.pop("_resolved", None)
        return d


DECISION_ADAPTERS = ["skill_selection", "action_taking"]
SKILLBANK_ADAPTERS = ["segment", "contract", "curator"]


def prepare_adapters(config: CoEvolutionConfig) -> Dict[str, str]:
    """Ensure every adapter directory is populated and ready for vLLM/GRPO.

    Two paths:

    **Load pre-trained** (``config.pretrained_adapter_paths`` is non-empty):
        Copy the 2 decision + 3 skill-bank adapters from the given paths
        into ``config.adapter_dir``.  Any adapter not listed in the dict
        will be random-initialised as a fallback.

    **Train from scratch** (``config.start_mode == "from_scratch"`` or
    no pre-trained paths and no existing adapters):
        Create random-initialised adapters (``init_lora_weights="gaussian"``
        by default).  Both the A and B LoRA matrices receive small random
        values so that gradients flow to all parameters from step 1.

    Returns a dict mapping adapter name → resolved directory path.
    """
    import gc
    import logging
    import shutil

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        # transformers 5.x exposes the multimodal "image-text-to-text" auto class
        # used by Qwen3.5 / Qwen3-VL.  Fall back gracefully for text-only setups.
        from transformers import AutoModelForImageTextToText  # type: ignore
    except ImportError:  # pragma: no cover — older transformers
        AutoModelForImageTextToText = None  # type: ignore

    logger = logging.getLogger(__name__)
    force = config.start_mode == "from_scratch"
    pretrained = config.pretrained_adapter_paths or {}
    result: Dict[str, str] = {}

    # ── Phase 1: copy pre-trained adapters ────────────────────────
    copied: List[str] = []
    for name in ADAPTER_NAMES:
        dst = Path(config.adapter_path(name))
        src = pretrained.get(name)
        if src is not None:
            src_path = Path(src)
            if not (src_path / "adapter_config.json").exists():
                # PEFT save_pretrained nests files under <adapter_name>/;
                # check one level deeper before giving up.
                nested = src_path / name
                if (nested / "adapter_config.json").exists():
                    logger.info(
                        "Pre-trained adapter '%s': found nested layout at %s",
                        name, nested,
                    )
                    src_path = nested
                else:
                    # Also try any single subdirectory that has the file
                    found = False
                    if src_path.is_dir():
                        for child in src_path.iterdir():
                            if child.is_dir() and (child / "adapter_config.json").exists():
                                logger.info(
                                    "Pre-trained adapter '%s': found nested layout at %s",
                                    name, child,
                                )
                                src_path = child
                                found = True
                                break
                    if not found:
                        logger.warning(
                            "Pre-trained adapter '%s' not found at %s "
                            "(checked top-level and subdirectories) — will random-init",
                            name, src,
                        )
                        continue
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src_path), str(dst))
            logger.info("Loaded pre-trained adapter '%s': %s → %s", name, src, dst)
            copied.append(name)
            result[name] = str(dst)

    # ── Phase 2: random-init any remaining missing adapters ───────
    need_init: List[str] = []
    for name in ADAPTER_NAMES:
        if name in copied:
            continue
        dst = Path(config.adapter_path(name))
        marker = dst / "adapter_config.json"
        if marker.exists() and not force:
            logger.info("LoRA adapter '%s' already exists: %s", name, dst)
            result[name] = str(dst)
        else:
            if force and dst.exists():
                logger.info("Force re-init: removing existing adapter '%s'", name)
                shutil.rmtree(dst)
            need_init.append(name)

    if not need_init:
        if copied:
            logger.info(
                "Loaded %d pre-trained adapter(s), all adapters ready", len(copied),
            )
        return result

    # ── Resolve target_modules from model architecture ────────────
    model_cfg = AutoConfig.from_pretrained(
        config.model_name, trust_remote_code=True,
    )
    # Multimodal Qwen3.5 / Qwen3-VL configs nest the language-model spec under
    # ``text_config``.  Use it for arch-detection so we identify the LM type
    # (e.g. ``qwen3_5_text``) rather than the umbrella multimodal type.
    text_cfg = getattr(model_cfg, "text_config", model_cfg)
    text_arch = (getattr(text_cfg, "model_type", "") or "").lower()
    is_multimodal = hasattr(model_cfg, "text_config") or hasattr(model_cfg, "vision_config")

    # ── T2.11 closure: single-source-of-truth target_modules resolver ──
    # SFT and GRPO must write/read the same LoRA shape, otherwise legs
    # missing in one recipe silently drop deltas at the boundary.  We
    # therefore delegate to ``trainer.SFT.lora_targets`` for both
    # pipelines — see ``implementation_notes/pre-training-readiness-audit.md``
    # §0.3.  Qwen3.5 hybrid stack reaches ALL GatedDeltaNet legs incl.
    # ``in_proj_z/b/a`` (``in_proj_z`` is hidden×value_dim, NOT tiny — the
    # earlier "skip the gating legs" rationale undercounted it).
    from trainer.SFT.lora_targets import resolve_target_modules as _resolve_targets

    target_modules = _resolve_targets(
        text_arch=text_arch,
        explicit=config.lora_target_modules,
    )

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=config.lora_init_weights,
    )

    init_desc = (
        "gaussian (A & B both random — GRPO-ready)"
        if config.lora_init_weights == "gaussian"
        else f"standard ({config.lora_init_weights})"
    )
    logger.info(
        "Loading base model '%s' on CPU to initialise %d adapter(s) "
        "[init=%s, r=%d, alpha=%d, arch=%s, multimodal=%s]: %s",
        config.model_name, len(need_init), init_desc,
        config.lora_r, config.lora_alpha,
        text_arch or "unknown", is_multimodal, need_init,
    )
    # For multimodal Qwen3.5 / Qwen3-VL the umbrella ``Qwen3_5Config`` lacks
    # ``vocab_size`` (lives under ``text_config``), so ``AutoModelForCausalLM``
    # crashes when fed the outer config.  Use ``AutoModelForImageTextToText``
    # in that case — the LoRA target_modules above still only match text
    # decoder linears, leaving the (frozen) vision tower untouched.
    if is_multimodal and AutoModelForImageTextToText is not None:
        loader_cls = AutoModelForImageTextToText
    else:
        loader_cls = AutoModelForCausalLM
    base_model = loader_cls.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    for name in need_init:
        out = Path(config.adapter_path(name))
        out.mkdir(parents=True, exist_ok=True)
        logger.info("Random-init LoRA adapter '%s' → %s", name, out)
        try:
            peft_model = get_peft_model(base_model, lora_cfg)
            peft_model.save_pretrained(str(out))
            result[name] = str(out)
            base_model = peft_model.unload()
        except Exception as exc:
            logger.error("Failed to create adapter '%s': %s", name, exc)

    del base_model
    gc.collect()
    # Do NOT call torch.cuda.empty_cache() here — it initializes a CUDA
    # context on GPU 0, wasting ~6 GB that the vLLM instance needs.

    logger.info(
        "Adapter summary: %d pre-trained, %d random-init, %d total ready",
        len(copied), len(need_init), len(result),
    )
    return result


# Keep the old name as an alias for backward compatibility
def init_lora_adapters(
    config: CoEvolutionConfig,
    force: bool = False,
) -> List[str]:
    """Backward-compatible wrapper around :func:`prepare_adapters`."""
    if force:
        config.start_mode = "from_scratch"
    result = prepare_adapters(config)
    return list(result.keys())
