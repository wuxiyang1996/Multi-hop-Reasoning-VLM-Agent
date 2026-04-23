# Decision agents: dummy language agent and LLM decision-making agent.

from .dummy_agent import (
    language_agent_action,
    detect_game,
    run_episode_with_experience_collection,
    AgentBufferManager,
    GAME_AVALON,
    GAME_DIPLOMACY,
    GAME_GAMINGAGENT,
)

from .agent_helper import (
    get_state_summary,
    compact_structured_state,
    compact_text_observation,
    DEFAULT_SUMMARY_CHAR_BUDGET,
    HARD_SUMMARY_CHAR_LIMIT,
    infer_intention,
    EpisodicMemoryStore,
    skill_bank_to_text,
    select_skill_from_bank,
    query_skill_bank,
)

from .agent import (
    VLMDecisionAgent,
    LLMDecisionAgent,
    AgentState,
    run_tool,
    run_episode_vlm_agent,
    run_episode_llm_agent,
    TOOL_TAKE_ACTION,
    TOOL_GET_STATE_SUMMARY,
    TOOL_GET_INTENTION,
    TOOL_SELECT_SKILL,
    TOOL_REWARD,
)

from .reward_func import (
    RewardConfig,
    RewardResult,
    RewardComputer,
    compute_reward,
)

# Schema-native Actor Agent stack (PLAN-ACTION-AGENT.md).  These are the
# classes the future GRPO training pipeline will target.  The older
# ``VLMDecisionAgent`` path above is kept for backward compatibility with
# scripts/qwen3_decision_agent.py; new code should prefer ``ActorAgent``.
from .schema_parser import (
    Answer,
    Entity,
    Hop,
    Relation,
    ResolvedAction,
    StateFlags,
    StateSchema,
    Targets,
    parse_state_schema,
    resolve_entity_action,
)
from .skill_interface import (
    NullSkillProvider,
    SkillBankProvider,
    SkillGuidance,
    SkillProvider,
)
from .skill_tracker import (
    ActivationCheck,
    SkillTracker,
    TrackerState,
)
from .inner_mdp import (
    HeuristicHopPolicy,
    HopAction,
    HopPolicy,
    HopStep,
    HopTrace,
    parse_hop_action,
)
from .actor_agent import (
    ActorAgent,
    ActorDecision,
    ActorState,
    run_actor_episode,
)

# ── New flavours of the Actor Agent (PLAN-ACTION-AGENT.md §2.3) ──
#
# Two specialised subclasses live in their own sub-packages so the
# GPT-4o data-collection path stays cleanly separated from the
# Qwen3-VL inference + GRPO path.  They are imported lazily through
# attribute access below so that ``import decision_agents`` does not
# pull in ``openai`` / ``trainer.coevolution.vllm_client`` /
# ``trainer.common.metrics`` unless the caller actually asks for them.
from .core import (
    VisualInput,
    build_openai_vision_messages,
    build_qwen_vl_messages,
    load_image_as_data_url,
)

# PEP 562 lazy attribute access so importing ``decision_agents`` stays
# light: ``openai`` (SFT actor) and ``trainer.coevolution.vllm_client``
# (GRPO actor) are pulled in only when the matching symbol is touched.
_LAZY_ATTRS = {
    "GPT4oCollectorActor": ("decision_agents.SFT", "GPT4oCollectorActor"),
    "SFTRecorder":         ("decision_agents.SFT", "SFTRecorder"),
    "SFTRecord":           ("decision_agents.SFT", "SFTRecord"),
    "QwenVLActor":         ("decision_agents.grpo", "QwenVLActor"),
    "GRPORolloutLogger":   ("decision_agents.grpo", "GRPORolloutLogger"),
    "DEFAULT_QWEN_VL_MODEL": ("decision_agents.grpo", "DEFAULT_QWEN_VL_MODEL"),
}


def __getattr__(name: str):  # pragma: no cover — thin shim
    """Lazy import for the SFT / GRPO actor flavours."""
    if name in _LAZY_ATTRS:
        import importlib
        mod_name, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(mod_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'decision_agents' has no attribute {name!r}")


__all__ = [
    "language_agent_action",
    "detect_game",
    "run_episode_with_experience_collection",
    "AgentBufferManager",
    "GAME_AVALON",
    "GAME_DIPLOMACY",
    "GAME_GAMINGAGENT",
    "get_state_summary",
    "compact_structured_state",
    "compact_text_observation",
    "DEFAULT_SUMMARY_CHAR_BUDGET",
    "HARD_SUMMARY_CHAR_LIMIT",
    "infer_intention",
    "EpisodicMemoryStore",
    "skill_bank_to_text",
    "select_skill_from_bank",
    "query_skill_bank",
    "VLMDecisionAgent",
    "LLMDecisionAgent",
    "AgentState",
    "run_tool",
    "run_episode_vlm_agent",
    "run_episode_llm_agent",
    "TOOL_TAKE_ACTION",
    "TOOL_GET_STATE_SUMMARY",
    "TOOL_GET_INTENTION",
    "TOOL_SELECT_SKILL",
    "TOOL_REWARD",
    "RewardConfig",
    "RewardResult",
    "RewardComputer",
    "compute_reward",
    # Schema parser
    "Answer",
    "Entity",
    "Hop",
    "Relation",
    "ResolvedAction",
    "StateFlags",
    "StateSchema",
    "Targets",
    "parse_state_schema",
    "resolve_entity_action",
    # Skill interface
    "NullSkillProvider",
    "SkillBankProvider",
    "SkillGuidance",
    "SkillProvider",
    # Skill tracker
    "ActivationCheck",
    "SkillTracker",
    "TrackerState",
    # Inner MDP
    "HeuristicHopPolicy",
    "HopAction",
    "HopPolicy",
    "HopStep",
    "HopTrace",
    "parse_hop_action",
    # Actor agent
    "ActorAgent",
    "ActorDecision",
    "ActorState",
    "run_actor_episode",
    # Multimodal scaffolding (decision_agents/core/)
    "VisualInput",
    "build_openai_vision_messages",
    "build_qwen_vl_messages",
    "load_image_as_data_url",
    # SFT collection flavour (decision_agents/SFT/) — lazy-loaded
    "GPT4oCollectorActor",
    "SFTRecorder",
    "SFTRecord",
    # GRPO + LoRA flavour (decision_agents/grpo/) — lazy-loaded
    "QwenVLActor",
    "GRPORolloutLogger",
    "DEFAULT_QWEN_VL_MODEL",
]
