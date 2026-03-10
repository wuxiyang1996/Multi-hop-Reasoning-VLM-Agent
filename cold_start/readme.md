# Cold-Start Data Generation

Generate initial trajectory data and skill seeds for the Game-AI-Agent system.

## Goal

1. **Prompt decision agents** (VLMDecisionAgent or dummy language agent) powered by GPT-5-mini to generate unlabeled trajectories from game environments.
2. **Label trajectories** with GPT-5-mini to produce initial seeds for the skill database (summaries, intentions, sub-task labels).

## Setup

```bash
# 1. Activate the cold-start conda environment
conda activate cold-start-agent

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 3. Set PYTHONPATH (from Game-AI-Agent root)
cd /path/to/Game-AI-Agent
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
```

## Batch Rollouts (100 per game)

The primary workflow generates 100 rollout episodes per game, with output formatted
for direct ingestion by the co-evolution framework (skill pipeline + trainer).

```bash
# All available games, 100 episodes each (default)
python cold_start/run_100_rollouts.py

# Specific games only
python cold_start/run_100_rollouts.py --games twenty_forty_eight sokoban tetris candy_crush

# Fewer episodes for testing
python cold_start/run_100_rollouts.py --episodes 5 --max_steps 30

# Use VLM decision agent (richer Experience fields)
python cold_start/run_100_rollouts.py --agent_type vlm

# Resume interrupted run (skips completed episodes)
python cold_start/run_100_rollouts.py --resume

# Skip labeling for faster generation
python cold_start/run_100_rollouts.py --no_label
```

### Output (cold_start/output/)

```
cold_start/output/
├── batch_rollout_summary.json          # Master run summary
├── twenty_forty_eight/
│   ├── episode_000.json ... episode_099.json   # Individual episodes
│   ├── episode_buffer.json                      # Episode_Buffer (loadable)
│   ├── rollouts.jsonl                           # JSONL: one Episode per line
│   └── rollout_summary.json                     # Per-game stats
├── sokoban/
│   └── ...
├── candy_crush/
│   └── ...
├── tetris/
│   └── ...
└── ... (all available games)
```

### Loading Rollouts into the Co-Evolution Framework

```python
from cold_start.load_rollouts import (
    load_episodes_from_jsonl,
    load_episode_buffer,
    episodes_to_rollout_records,
    load_all_game_rollouts,
)

# --- Skill pipeline ingestion ---
from skill_agents.pipeline import SkillBankAgent

episodes = load_episodes_from_jsonl("cold_start/output/tetris/rollouts.jsonl")
agent = SkillBankAgent(bank_path="skills/bank.jsonl")
agent.ingest_episodes(episodes)
agent.run_until_stable(max_iterations=3)

# --- Trainer ingestion (Episode → RolloutRecord) ---
from trainer.skillbank.ingest_rollouts import ingest_rollouts

records = episodes_to_rollout_records(episodes)
trajectories = ingest_rollouts(records)

# --- Load all games at once ---
all_rollouts = load_all_game_rollouts("cold_start/output")
for game_name, episodes in all_rollouts.items():
    print(f"{game_name}: {len(episodes)} episodes")
```

## Single-Game Generation (generate_cold_start.py)

For smaller-scale generation or single-game runs:

```bash
# Generate cold-start data for 2048 (3 episodes, 50 steps, GPT-5-mini)
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight \
    --episodes 3 --max_steps 50 --model gpt-5-mini

# Use VLM decision agent
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight \
    --agent_type vlm --episodes 3 --max_steps 50

# Generate for all available games
python cold_start/generate_cold_start.py --all_games --episodes 3 --max_steps 50

# Skip trajectory labeling
python cold_start/generate_cold_start.py \
    --game twenty_forty_eight --episodes 5 --max_steps 50 --no_label
```

Output goes to `cold_start/data/<game_name>/` by default.

## Available Games

| Game | Registry Key | Description |
|------|-------------|-------------|
| 2048 | `twenty_forty_eight` | Tile merging puzzle |
| Sokoban | `sokoban` | Box-pushing puzzle |
| Candy Crush | `candy_crush` | Match-3 tile puzzle |
| Tetris | `tetris` | Falling block puzzle |
| Doom | `doom` | FPS shooting |
| Pokemon Red | `pokemon_red` | RPG exploration (needs ROM) |
| Super Mario Bros | `super_mario_bros` | Platformer (needs retro ROM) |
| Ace Attorney | `ace_attorney` | Investigation/dialogue (needs retro ROM) |
| 1942 | `nineteen_forty_two` | Vertical shooter (needs retro ROM) |
| Tic-Tac-Toe | `tic_tac_toe` | Classic board game |
| Texas Hold'em | `texas_holdem` | Poker card game |

Games requiring ROMs or special setup will be automatically skipped if their
environment classes are not importable.

## Agent Types

- **`dummy`** (default): Uses `language_agent_action` with GPT function calling. Simpler, single-turn action selection per step.
- **`vlm`**: Uses `run_episode_vlm_agent()` which returns `Episode` objects with fully-populated Experience fields (`summary_state`, `intentions`, `sub_tasks`, `reward_details`, `action_type`). These can be fed directly into the skill pipeline.

## Output Format

Each episode JSON contains:
- `episode_id` — Unique UUID
- `env_name` — Platform name (`"gamingagent"`)
- `game_name` — Specific game (e.g. `"tetris"`, `"sokoban"`)
- `experiences` — List of Experience objects:
  - `state`, `action`, `reward`, `next_state`, `done`
  - `intentions`, `tasks`, `sub_tasks`
  - `summary`, `summary_state`
  - `reward_details` (r_env, r_follow, r_cost, r_total)
  - `action_type` (primitive, QUERY_MEM, QUERY_SKILL, CALL_SKILL)
  - `idx` (step index)
- `task` — Task description
- `outcome` — Episode outcome
- `summary` — Episode summary
- `metadata` — Rollout stats (steps, total_reward, model, agent_type, etc.)

This format is directly compatible with:
- `Episode.from_dict()` / `Episode_Buffer.load_from_json()` for data loading
- `SkillBankAgent.ingest_episodes()` for skill pipeline ingestion
- `episodes_to_rollout_records()` → `ingest_rollouts()` for trainer ingestion
