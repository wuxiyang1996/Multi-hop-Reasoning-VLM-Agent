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
]
