"""
Env wrappers: convert environment state to/from natural language for language-model agents.

Available wrappers:
  - OvercookedNLWrapper: Overcooked cooperative cooking (2 agents)
  - AvalonNLWrapper:     Avalon hidden-role deduction (5-10 agents)
  - DiplomacyNLWrapper:  Diplomacy strategic negotiation (7 agents / powers)
"""

from env_wrappers.overcooked_nl_wrapper import (
    OvercookedNLWrapper,
    joint_action_to_indices,
    natural_language_to_action_index,
    state_to_natural_language,
    state_to_natural_language_for_all_agents,
)

from env_wrappers.avalon_nl_wrapper import (
    AvalonNLWrapper,
    state_to_natural_language as avalon_state_to_nl,
    state_to_natural_language_for_all as avalon_state_to_nl_all,
    parse_vote as avalon_parse_vote,
    parse_team as avalon_parse_team,
    parse_target as avalon_parse_target,
)

from env_wrappers.diplomacy_nl_wrapper import (
    DiplomacyNLWrapper,
    state_to_natural_language as diplomacy_state_to_nl,
    state_to_natural_language_for_all as diplomacy_state_to_nl_all,
    parse_orders as diplomacy_parse_orders,
)

__all__ = [
    # Overcooked
    "OvercookedNLWrapper",
    "joint_action_to_indices",
    "natural_language_to_action_index",
    "state_to_natural_language",
    "state_to_natural_language_for_all_agents",
    # Avalon
    "AvalonNLWrapper",
    "avalon_state_to_nl",
    "avalon_state_to_nl_all",
    "avalon_parse_vote",
    "avalon_parse_team",
    "avalon_parse_target",
    # Diplomacy
    "DiplomacyNLWrapper",
    "diplomacy_state_to_nl",
    "diplomacy_state_to_nl_all",
    "diplomacy_parse_orders",
]
