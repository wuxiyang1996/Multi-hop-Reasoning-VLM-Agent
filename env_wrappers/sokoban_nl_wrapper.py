"""
Sokoban-specialized NL wrapper with spatial grid observations, rolling
memory, periodic reflection, and domain-specific prompts.

Ported from ``cold_start/generate_cold_start_sokoban.py`` so that the
same rich prompt architecture is available during Qwen3 evaluation.

Usage::

    from env_wrappers.sokoban_nl_wrapper import SokobanNLWrapper
    from evaluate_gamingagent.gym_like import make_gaming_env

    base_env = make_gaming_env("sokoban", max_steps=200)
    env = SokobanNLWrapper(base_env)
    obs, info = env.reset()

    # In the agent loop:
    system  = env.system_prompt
    user    = env.build_user_prompt()
    # ... send to LLM ...
    action  = env.parse_action(reply)
    obs, reward, term, trunc, info = env.step(action)
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MEMORY_STEPS = 8

VALID_ACTIONS = [
    "up", "down", "left", "right",
    "push up", "push down", "push left", "push right",
]

CHAR_LEGEND = {
    "#": "Wall",
    " ": "Floor",
    "@": "Player",
    "$": "Box",
    "?": "Target (empty)",
    "*": "Box on Target (solved!)",
    "+": "Player on Target",
}


# ---------------------------------------------------------------------------
# Sokoban-specific prompts
# ---------------------------------------------------------------------------

SOKOBAN_SYSTEM_PROMPT = """\
You are an expert AI player solving Sokoban puzzles. Your goal is to push ALL \
boxes ($) onto ALL target locations (?). A box on a target becomes *.

GRID CHARACTERS:
  #  Wall (impassable)
  .  Floor (empty space, shown as space in the grid)
  @  Player (you)
  $  Box on floor
  ?  Target location (empty — needs a box)
  *  Box on a target (SOLVED — do not move this off the target!)
  +  Player standing on a target

ACTIONS — choose exactly one:
  Movement only:   up, down, left, right
    → Moves the player one step if the destination is floor or empty target.
  Push:            push up, push down, push left, push right
    → Player must be adjacent to a box in that direction AND the cell beyond \
the box must be floor or empty target. Player and box both move one step.

CRITICAL STRATEGY:
1. PLAN AHEAD: Before moving, trace the path from each box to the nearest \
unoccupied target. Decide which box to push first.
2. AVOID DEADLOCKS:
   - Never push a box into a corner (two perpendicular walls) unless it is a target.
   - Never push a box against a wall if there is no target along that wall.
   - Once a box is in a deadlock position the puzzle is UNSOLVABLE.
3. POSITIONING: Move the player around boxes to approach from the correct side \
before pushing. Use "move" actions (up/down/left/right) to reposition.
4. DO NOT oscillate: If you find yourself repeating the same 2-3 moves, STOP \
and rethink your plan completely.
5. PUSH vs MOVE: Use "push X" ONLY when you are adjacent to a box and want to \
push it. Use plain movement (up/down/left/right) to navigate without pushing.
6. Once a box is on a target (*), try not to move it off.

Respond with EXACTLY this format:
REASONING: <your step-by-step reasoning about the current board, which box to \
target, where to position, and what to push>
ACTION: <action>"""

SOKOBAN_USER_TEMPLATE = """\
## Current Sokoban Board (row, col from top-left = 0,0)
```
{grid}
```

## Element Summary
{element_summary}

## Recent History ({n_history} steps)
{history}

## Reflection
{reflection}

## Task
Push all boxes ($) onto targets (?). Boxes on targets show as *.
Choose ONE action from: {actions}

REASONING: <reasoning>
ACTION: <action>"""

REFLECTION_PROMPT = """\
Analyze the last {n} steps of this Sokoban game:

{trajectory}

Current board:
{board}

Briefly reflect (under 60 words):
1. Did recent actions make progress (box moved closer to target)?
2. Any oscillation or repeated moves?
3. Any deadlock risk (box pushed to wall/corner)?
4. What should the next priority be?"""


# ---------------------------------------------------------------------------
# Board parsing — convert the flat table back to a spatial grid
# ---------------------------------------------------------------------------

def table_obs_to_grid(obs_text: str) -> Optional[List[List[str]]]:
    """Parse the text-table observation back into a 2D character grid.

    The env produces a table like::

        ID  | Item Type    | Position
        --------------------------------
        1   | Wall         | (0, 0)
        2   | Wall         | (1, 0)
        ...

    We reconstruct the grid from these rows.
    """
    rows: Dict[Tuple[int, int], str] = {}
    type_to_char = {
        "wall": "#",
        "worker": "@",
        "box": "$",
        "dock": "?",
        "box on dock": "*",
        "empty": " ",
    }

    for line in obs_text.splitlines():
        line = line.strip()
        if not line or line.startswith("ID") or line.startswith("-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        item_type = parts[1].strip().lower()
        pos_match = re.search(r"\((\d+)\s*,\s*(\d+)\)", parts[2])
        if not pos_match:
            continue
        col, row = int(pos_match.group(1)), int(pos_match.group(2))
        char = type_to_char.get(item_type, "?")
        rows[(row, col)] = char

    if not rows:
        return None

    max_r = max(r for r, _ in rows) + 1
    max_c = max(c for _, c in rows) + 1
    grid = [[" " for _ in range(max_c)] for _ in range(max_r)]
    for (r, c), ch in rows.items():
        grid[r][c] = ch
    return grid


def grid_to_string(grid: List[List[str]]) -> str:
    """Render grid with row/col indices for spatial reasoning."""
    if not grid:
        return "(empty board)"
    max_c = max(len(row) for row in grid)
    col_header = "    " + "".join(f"{c}" for c in range(max_c))
    lines = [col_header]
    for r, row in enumerate(grid):
        lines.append(f"{r:>3} " + "".join(row))
    return "\n".join(lines)


def summarize_elements(grid: List[List[str]]) -> str:
    """List positions of key elements for quick reference."""
    player_pos = []
    boxes = []
    targets = []
    boxes_on_target = []

    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "@":
                player_pos.append((r, c))
            elif ch == "+":
                player_pos.append((r, c))
                targets.append((r, c))
            elif ch == "$":
                boxes.append((r, c))
            elif ch == "?":
                targets.append((r, c))
            elif ch == "*":
                boxes_on_target.append((r, c))

    parts = []
    if player_pos:
        parts.append(f"Player: {player_pos[0]}")
    if boxes:
        parts.append(f"Boxes on floor: {boxes}")
    if targets:
        parts.append(f"Empty targets: {targets}")
    if boxes_on_target:
        parts.append(f"Boxes on target (solved): {boxes_on_target}")
    total_boxes = len(boxes) + len(boxes_on_target)
    total_targets = len(targets) + len(boxes_on_target)
    parts.append(f"Progress: {len(boxes_on_target)}/{total_targets} targets filled")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Rolling memory
# ---------------------------------------------------------------------------

class SokobanMemory:
    """Rolling window of recent (action, board_summary, reward) tuples."""

    def __init__(self, max_steps: int = MAX_MEMORY_STEPS):
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        self.last_reflection: str = "Game just started. Survey the board and plan."

    def add(self, step: int, action: str, reward: float, board_summary: str):
        self.history.append({
            "step": step,
            "action": action,
            "reward": reward,
            "summary": board_summary,
        })
        if len(self.history) > self.max_steps:
            self.history = self.history[-self.max_steps:]

    def format_history(self) -> str:
        if not self.history:
            return "(no previous actions)"
        lines = []
        for h in self.history:
            r_str = f"{h['reward']:+.2f}"
            lines.append(f"  Step {h['step']}: {h['action']} → reward {r_str}")
        return "\n".join(lines)

    def format_trajectory_for_reflection(self) -> str:
        if not self.history:
            return "(no actions yet)"
        lines = []
        for h in self.history:
            lines.append(f"Step {h['step']}: action={h['action']}, reward={h['reward']:+.2f}")
            lines.append(f"  Board: {h['summary']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_sokoban_action(reply: str) -> Optional[str]:
    """Extract a valid Sokoban action from free-form LLM reply."""
    if not reply:
        return None

    lower = reply.lower()

    action_m = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", reply, re.IGNORECASE)
    if action_m:
        candidate = action_m.group(1).strip().lower()
        for v in VALID_ACTIONS:
            if v == candidate:
                return v

    move_m = re.search(r"move:\s*(.+?)(?:\n|$)", lower)
    if move_m:
        candidate = move_m.group(1).strip()
        for v in VALID_ACTIONS:
            if v == candidate:
                return v

    for v in sorted(VALID_ACTIONS, key=len, reverse=True):
        if v in lower:
            return v

    return None


# ---------------------------------------------------------------------------
# SokobanNLWrapper
# ---------------------------------------------------------------------------

class SokobanNLWrapper:
    """
    Wraps GamingAgentEnv (Sokoban) so observations are spatial-grid NL
    strings with rolling memory and domain-specific prompt support.

    Provides:
    - ``system_prompt``: Domain-specific system prompt.
    - ``build_user_prompt()``: Rich user prompt with grid, elements, history, reflection.
    - ``parse_action(reply)``: Extract valid Sokoban action from LLM reply.
    - ``generate_reflection(ask_fn, model)``: Ask the agent model to reflect.
    """

    def __init__(
        self,
        env: Any,
        max_memory: int = MAX_MEMORY_STEPS,
        reflect_every: int = 3,
    ):
        self._env = env
        self._memory = SokobanMemory(max_steps=max_memory)
        self._reflect_every = reflect_every
        self._step_count = 0
        self._last_grid: Optional[List[List[str]]] = None
        self._last_obs_nl: str = ""
        self._action_names = VALID_ACTIONS

    @property
    def env(self):
        return self._env

    @property
    def system_prompt(self) -> str:
        return SOKOBAN_SYSTEM_PROMPT

    @property
    def action_names(self) -> List[str]:
        return list(self._action_names)

    @property
    def memory(self) -> SokobanMemory:
        return self._memory

    @property
    def last_grid(self) -> Optional[List[List[str]]]:
        return self._last_grid

    def _process_obs(self, obs_nl: str) -> Tuple[str, Optional[List[List[str]]]]:
        """Parse table obs into grid; return (obs_text, grid_or_none)."""
        grid = table_obs_to_grid(obs_nl)
        if grid is not None:
            self._last_grid = grid
        return obs_nl, grid

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        self._step_count = 0
        self._memory = SokobanMemory(max_steps=self._memory.max_steps)

        obs_nl = obs if isinstance(obs, str) else str(obs.get("text", obs))
        self._last_obs_nl, self._last_grid = self._process_obs(obs_nl)

        info["action_names"] = self._action_names
        info["env_name"] = "gamingagent"
        info["game_name"] = "sokoban"
        if self._last_grid is not None:
            info["grid_string"] = grid_to_string(self._last_grid)
            info["element_summary"] = summarize_elements(self._last_grid)
        return self._last_obs_nl, info

    def step(
        self,
        action: Union[str, int, np.integer],
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        obs_nl = obs if isinstance(obs, str) else str(obs.get("text", obs))
        self._last_obs_nl, self._last_grid = self._process_obs(obs_nl)

        board_summary = ""
        if self._last_grid is not None:
            board_summary = summarize_elements(self._last_grid)

        self._memory.add(self._step_count, str(action), float(reward), board_summary)

        info["action_names"] = self._action_names
        info["env_name"] = "gamingagent"
        info["game_name"] = "sokoban"
        info["step"] = self._step_count
        if self._last_grid is not None:
            info["grid_string"] = grid_to_string(self._last_grid)
            info["element_summary"] = board_summary
        return obs_nl, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_user_prompt(self, reflection_override: Optional[str] = None) -> str:
        """Build the rich user prompt with grid, elements, history, and reflection."""
        if self._last_grid is not None:
            grid_str = grid_to_string(self._last_grid)
            element_summary = summarize_elements(self._last_grid)
        else:
            grid_str = self._last_obs_nl
            element_summary = "(could not parse grid)"

        reflection = reflection_override or self._memory.last_reflection

        return SOKOBAN_USER_TEMPLATE.format(
            grid=grid_str,
            element_summary=element_summary,
            n_history=len(self._memory.history),
            history=self._memory.format_history(),
            reflection=reflection,
            actions=", ".join(VALID_ACTIONS),
        )

    def should_reflect(self) -> bool:
        """Whether it's time for a reflection call (every N steps, after warmup)."""
        return (
            self._step_count > 0
            and len(self._memory.history) >= 3
            and self._step_count % self._reflect_every == 0
        )

    def generate_reflection(
        self,
        ask_fn: Callable[..., str],
        model: str,
    ) -> str:
        """Ask the LLM to reflect on recent history. Updates internal state.

        ``ask_fn`` should have signature ``ask_fn(prompt, model=..., temperature=..., max_tokens=...) -> str``.
        """
        if len(self._memory.history) < 3:
            return self._memory.last_reflection

        board_str = grid_to_string(self._last_grid) if self._last_grid else "(no grid)"

        prompt = (
            "You are a Sokoban game analyst. Give brief, actionable reflections.\n\n"
            + REFLECTION_PROMPT.format(
                n=len(self._memory.history),
                trajectory=self._memory.format_trajectory_for_reflection(),
                board=board_str,
            )
        )

        try:
            reflection = ask_fn(prompt, model=model, temperature=0.2, max_tokens=120)
            if reflection and not reflection.startswith("Error"):
                self._memory.last_reflection = reflection
                return reflection
        except Exception:
            pass
        return self._memory.last_reflection

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_action(reply: str) -> str:
        """Extract a valid Sokoban action from LLM reply; falls back to 'up'."""
        action = parse_sokoban_action(reply)
        return action if action else "up"

    @staticmethod
    def parse_reasoning(reply: str) -> Optional[str]:
        """Extract REASONING from LLM reply."""
        m = re.search(
            r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)", reply, re.DOTALL | re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
        m = re.search(
            r"thought\s*:\s*(.+?)(?=\nmove|\Z)", reply, re.DOTALL | re.IGNORECASE
        )
        if m:
            return m.group(1).strip()
        return None

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)
