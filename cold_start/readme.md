# Cold-Start Data Generation

Cold-start data generation for the **COS-PLAY** co-evolution framework
(COLM 2026, Section 5).  The SFT teacher (``gpt-5.5`` by default — pulled
from ``common.models.BACKBONE_SFT_TEACHER_MODEL``) generates seed
trajectories per game, bootstrapping the co-evolution training loop
between the Decision Agent and the Skill Bank Agent.

> **Model mapping (current phase):** SFT teacher = ``gpt-5.5``; trained
> actor + skill-bank backbone = ``Qwen/Qwen3.5-9B``; frozen control plane
> = ``Qwen/Qwen3.5-35B-A3B``.  See [`../common/models.py`](../common/models.py).

## Directory Contents

| File | Purpose |
|------|---------|
| `generate_cold_start.py` | Core module: game registry, env wrapper, episode runners, labeling |
| `generate_cold_start_gpt54.py` | SFT-teacher (``gpt-5.5``) agent for LMGame-Bench (2048, Candy Crush, Tetris).  Filename retained for path compatibility. |
| `generate_cold_start_evolver.py` | SFT-teacher (``gpt-5.5``) agent for Avalon & Diplomacy |
| `generate_cold_start_orak.py` | SFT-teacher (``gpt-5.5``) agent for Super Mario (Orak env) |
| `generate_cold_start_actor.py` | **Actor-agent** rollouts via ``env_wrappers``: gpt-5.5 vision converts each rendered frame → ``<state>`` schema (per `vlm_wrapper.schema`), then gpt-5.5 picks one action from the schema with function calling.  Covers 2048 / Candy Crush / Tetris (macro) / Super Mario. |
| `load_rollouts.py` | Utility: load rollout outputs into Episode / RolloutRecord |
| `run_coldstart_gpt54.sh` | Shell launcher for LMGame-Bench rollouts |
| `run_coldstart_evolver.sh` | Shell launcher for Avalon & Diplomacy rollouts |
| `run_coldstart_orak_mario.sh` | Shell launcher for Super Mario (conda + Xvfb) |
| `run_coldstart_actor.sh` | Shell launcher for the actor-agent rollouts (env_wrappers + gpt-5.5 visual schema → action). |
| `run_coldstart_actor_all_games.sh` | **Multi-env wrapper** — runs `run_coldstart_actor.sh` twice in one command: 2048 / Candy Crush / Tetris under `game-ai-agent`, then Super Mario under `orak-mario` (with Xvfb).  Use this when you want all 4 games in one shot. |

## Games Covered (6 total)

### LMGame-Bench (`generate_cold_start_gpt54.py`)

| Game | Registry Key | Actions |
|------|--------------|---------|
| **2048** | `twenty_forty_eight` | `up`, `down`, `left`, `right` |
| **Candy Crush** | `candy_crush` | coordinate swaps, e.g. `((0,5),(1,5))` |
| **Tetris** | `tetris` | `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `hard_drop`, `soft_drop` |

### AgentEvolver (`generate_cold_start_evolver.py`)

| Game | Registry Key | Actions |
|------|--------------|---------|
| **Avalon** | `avalon` | team proposals, votes, pass/fail, assassination |
| **Diplomacy** | `diplomacy` | unit orders (move, hold, support, convoy, retreat, build, disband) |

### Orak (`generate_cold_start_orak.py`)

| Game | Registry Key | Actions |
|------|--------------|---------|
| **Super Mario** | `super_mario` | `Jump Level : 0` ... `6` |

## Setup

```bash
conda activate game-ai-agent
export OPENROUTER_API_KEY="sk-or-..."   # or OPENAI_API_KEY
cd /path/to/Game-AI-Agent
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
```

## Quick Start

### LMGame-Bench (2048, Candy Crush, Tetris)

```bash
# All 3 games, 60 episodes each
bash cold_start/run_coldstart_gpt54.sh --episodes 60

# Specific games
bash cold_start/run_coldstart_gpt54.sh --games tetris candy_crush --episodes 60

# Resume interrupted run
bash cold_start/run_coldstart_gpt54.sh --episodes 60 --resume

# Python directly
python cold_start/generate_cold_start_gpt54.py --games tetris --episodes 5 --resume
```

Output: `cold_start/output/gpt54/<game_name>/`

### Avalon & Diplomacy

```bash
# Both games, 20 episodes each (default)
bash cold_start/run_coldstart_evolver.sh

# Avalon only, 60 episodes
bash cold_start/run_coldstart_evolver.sh --games avalon --episodes 60

# Resume + verbose
bash cold_start/run_coldstart_evolver.sh --resume -v

# Python directly
python cold_start/generate_cold_start_evolver.py --games diplomacy --episodes 10 -v
```

Output: `cold_start/output/gpt54_evolver/<game_name>/`

### Super Mario

Requires the `orak-mario` conda environment and Xvfb for headless rendering.

```bash
bash cold_start/run_coldstart_orak_mario.sh --episodes 10

# Or manually
source env_wrappers/setup_orak_mario.sh
python cold_start/generate_cold_start_orak.py --games super_mario --episodes 10
```

Output: `cold_start/output/gpt54_orak/<game_name>/`

### Actor agent (env_wrappers, gpt-5.5 visual schema → action)

The actor pipeline can run any subset of the four supported games
(2048 / Candy Crush / Tetris / Super Mario).  Choose **one** of the two
launchers depending on which games you need.

#### Single-env launcher — `run_coldstart_actor.sh`

Use this when you want to drive the cold-start from **one already-active
conda env**.  Best for 2048 + Candy Crush + Tetris (run from
`game-ai-agent`), or Super Mario alone (run from `orak-mario`).

```bash
conda activate game-ai-agent  # or orak-mario for super_mario only

# All games supported by the active env (defaults: --resume --save_frames -v)
bash cold_start/run_coldstart_actor.sh

# Just 2048 + tetris, 10 episodes each, max 50 steps, save VLM frames
bash cold_start/run_coldstart_actor.sh \
    --games twenty_forty_eight tetris --episodes 10 --max_steps 50 --save_frames

# Skip the vision call (cheaper canonical-schema baseline; deterministic)
bash cold_start/run_coldstart_actor.sh --no_vision --episodes 3

# Python directly (no Xvfb / conda env activation)
python cold_start/generate_cold_start_actor.py \
    --games twenty_forty_eight candy_crush --episodes 5 -v
```

When `super_mario` is in the requested games AND the `orak-mario` conda
env exists, the launcher auto-activates it and starts Xvfb on `:99` so
nes-py / pyglet can render headlessly.

#### Multi-env wrapper — `run_coldstart_actor_all_games.sh`

Use this when you want **all 4 games in a single command**.  The
wrapper dispatches each game via `conda run -n <env>` so the right
conda env is used per game (no manual `conda activate` swaps):

| Game | Env |
|---|---|
| `twenty_forty_eight`, `candy_crush`, `tetris` | `game-ai-agent` |
| `super_mario` | `orak-mario` |

All games share the same `Cold-start-out/` directory at the repo root
(`<codebase_root>/Cold-start-out/`), all extra CLI args are forwarded
verbatim, and vision is on by default.

```bash
# Sequential (default): one game at a time, ~20-30 min for --episodes 1
bash cold_start/run_coldstart_actor_all_games.sh \
    --episodes 1 --model gpt-5.5 --save_frames -v

# PARALLEL: all 4 games dispatched concurrently as separate processes.
# Bottleneck is the longest game; ~3-4x speedup for --episodes 1.
# Per-game stdout streams to Cold-start-out/_logs/<game>.log so output
# isn't interleaved.  Tail any game live with:
#   tail -f Cold-start-out/_logs/tetris.log
bash cold_start/run_coldstart_actor_all_games.sh --parallel \
    --episodes 1 --model gpt-5.5 --save_frames -v

# Skip super_mario (e.g. orak-mario env not installed yet)
bash cold_start/run_coldstart_actor_all_games.sh --no_mario \
    --episodes 2 --save_frames -v

# Cheap baseline (skip vision call; canonical schema only)
bash cold_start/run_coldstart_actor_all_games.sh --parallel \
    --episodes 3 --no_vision
```

Wrapper-only flags (consumed by the wrapper, NOT forwarded):

| Flag | Effect |
|---|---|
| `--parallel`, `-P` | Dispatch all games concurrently as background processes |
| `--no_mario` | Skip `super_mario` (and orak-mario env activation) |
| `--games <g>...` | Restrict to a subset of games (default: all 4) |
| `--run_id <id>` | Override auto-generated `YYYY-MM-DD_HH-MM-SS` run id (useful for `--resume` into an existing run) |
| `--output_dir <path>` | Override base dir (default: `<repo>/Cold-start-out`) |

Pre-reqs the wrapper expects:

```bash
# game-ai-agent — already required for everything else
bash install/install_main_env.sh
conda activate game-ai-agent
pip install tile_match_gym                # adds Candy Crush backend
                                          # (downgrades numpy to 1.26.x;
                                          #  fine for actor cold-start,
                                          #  conflicts with vLLM stack)

# orak-mario — required for Super Mario
git clone https://github.com/krafton-ai/Orak.git /path/to/Orak
bash install/install_orak_mario.sh
```

#### Outputs

The `run_coldstart_actor_all_games.sh` wrapper writes timestamped runs
grouped by conda env:

```
<codebase_root>/Cold-start-out/
├── <run_id>/                              # YYYY-MM-DD_HH-MM-SS
│   ├── game-ai-agent/                     # env subdir
│   │   ├── twenty_forty_eight/
│   │   │   ├── episode_NNN.json
│   │   │   ├── episode_buffer.json
│   │   │   ├── rollouts.jsonl
│   │   │   ├── rollout_summary.json
│   │   │   └── frames/ep_NNN/step_NNN.png   # only with --save_frames
│   │   ├── candy_crush/...
│   │   └── tetris/...
│   ├── orak-mario/
│   │   └── super_mario/...
│   ├── _logs/<game>.log                  # per-game stdout/stderr
│   └── _run_meta.json                    # run config + per-game rc
└── latest -> <run_id>                     # symlink to most recent run
```

The single-env launcher `run_coldstart_actor.sh` writes a flat layout
when invoked directly (no `<run_id>/<env>` wrapping):

```
<codebase_root>/Cold-start-out/
└── <game>/
    ├── episode_NNN.json
    ├── episode_buffer.json
    ├── rollouts.jsonl
    ├── rollout_summary.json
    └── frames/ep_NNN/step_NNN.png
```

`Cold-start-out/` lives at the top of the `Multi-hop-Reasoning-VLM-Agent`
repo and is git-ignored.  Override with `--output_dir <path>` if you
want to write somewhere else.

Each step persists the visual schema, the raw VLM output, the action +
reasoning, the canonical (deterministic) schema fallback, and (with
`--save_frames`) the actual PNG frame sent to the VLM.

Tetris is **always** wrapped with `TetrisMacroActionWrapper` so each
LLM call commits one full piece placement (rotation + column +
landing) instead of a primitive key press; this keeps every step
informative for skill mining / GRPO.

API keys are auto-loaded from `<workspace>/api_keys.py` (or
`$COSPLAY_API_KEYS_FILE`) at import time — no `export` needed. See
`generate_cold_start_actor.py::_bootstrap_api_keys_from_file` for the
lookup order.

## Output Structure

All generators produce the same per-game layout:

```
<output_root>/<game_name>/
├── episode_000.json ... episode_NNN.json   # Individual episodes
├── episode_buffer.json                      # Episode_Buffer (loadable)
├── rollouts.jsonl                           # JSONL: one Episode per line
└── rollout_summary.json                     # Per-game stats
```

Default `<output_root>` per launcher:

| Launcher | `<output_root>` | Notes |
|---|---|---|
| `run_coldstart_gpt54.sh` | `cold_start/output/gpt54/` | LMGame-Bench (2048 / Candy Crush / Tetris) |
| `run_coldstart_evolver.sh` | `cold_start/output/gpt54_evolver/` | Avalon / Diplomacy |
| `run_coldstart_orak_mario.sh` | `cold_start/output/gpt54_orak/super_mario/` | Super Mario |
| `run_coldstart_actor.sh` & `run_coldstart_actor_all_games.sh` | **`Cold-start-out/`** at the repo root (git-ignored) | env_wrappers actor agent — visual-schema-driven action selection across all four games |

Override the actor pipeline's output dir with `--output_dir <path>`.

### Per-step extras (actor pipeline)

Each `Experience.extras` dict captured by `generate_cold_start_actor.py`
carries the schema metadata so trajectories can be replayed and parse
failures can be debugged offline:

| key | meaning |
|---|---|
| `schema` | The `<state>...</state>` block fed to the action LLM (parsed VLM output, or canonical fallback). |
| `schema_source` | `vlm` &nbsp;— parsed from the visual call.<br>`fallback_canonical` &nbsp;— parser failed, used the deterministic canonical schema.<br>`canonical` &nbsp;— vision was disabled (`--no_vision`).<br>`text_only` &nbsp;— no canonical helper available either. |
| `schema_recovery` | When the strict parser fails, how the schema was salvaged: `strict` / `fenced` / `truncated` / `untagged`.  Absent on `fallback_canonical`. |
| `schema_finish_reason` | OpenAI/OpenRouter `finish_reason` for the visual call (`stop`, `length`, …).  `length` ⇒ the response was truncated by the token cap; raise `_SCHEMA_MAX_TOKENS_REASONING` if you see this often. |
| `schema_raw_excerpt` | First 4 KB of the raw VLM response (only present when the call succeeded).  Use this to diagnose `fallback_canonical`. |
| `schema_raw_full_len` | Full length of the raw response (so you can detect truncation even after the excerpt is capped). |
| `schema_canonical` | The deterministic canonical schema for this step (the fallback that *would* be used). |
| `schema_error` | Exception repr if the visual API call itself raised. |
| `action_raw`, `action_error` | Raw output / exception from the action call. |
| `frame_path` | Path to the saved PNG frame fed to the VLM (when `--save_frames`). |
| `valid_actions`, `is_noop` | Action list at this step + whether the action was a no-op. |

### Token budgets and reasoning models

`generate_cold_start_actor.py` recognises reasoning models (`gpt-5.x`,
`o1`, `o3`, `o4`) via `_is_reasoning_model` and routes them straight to
`max_completion_tokens` with a generous cap (`_SCHEMA_MAX_TOKENS_REASONING
= 12000`).  This is necessary because:

- Reasoning models charge **internal thinking tokens** against the same
  budget as the visible response.
- Dense games (candy_crush 8×8, tetris 20×10) emit long schemas: 30+
  entities, attributes, affordances, relations, targets, actions.
- With the previous 1500-token cap the response was truncated **before**
  the closing `</state>` tag, so the strict regex parser always failed
  and every step fell back to canonical.  After the bump, parse rates
  for these games match 2048 (≥80 % `schema=vlm`).

The lenient parser (`_lenient_parse_schema`) is a defence-in-depth
backstop: it strips markdown code fences, closes a missing `</state>`
tag (`recovery=truncated`), and wraps loose `<entities>` blocks
(`recovery=untagged`).

## Loading Rollouts into the Training Pipeline

```python
from cold_start.load_rollouts import (
    load_episodes_from_jsonl,
    load_episode_buffer,
    episodes_to_rollout_records,
    load_all_game_rollouts,
)

# Load episodes from a single game
episodes = load_episodes_from_jsonl("cold_start/output/gpt54/tetris/rollouts.jsonl")

# Convert to RolloutRecord for the trainer
records = episodes_to_rollout_records(episodes)

# Load all games at once
all_rollouts = load_all_game_rollouts("cold_start/output/gpt54")
for game_name, eps in all_rollouts.items():
    print(f"{game_name}: {len(eps)} episodes")
```

## End Conditions

Episodes terminate at each game's **natural end condition**.  Step caps
mirror upstream COS-PLAY (`generate_cold_start.py::COLD_START_MAX_STEPS_NATURAL_END`
plus the `--max_steps 100` in `run_coldstart_orak_mario.sh`) and are
also used as the per-game defaults in `generate_cold_start_actor.py::DEFAULT_MAX_STEPS`:

| Game | Natural end condition | Default step cap |
|------|---------------------|------------------|
| **2048** | No valid moves, or reach 2048 tile; also 10 steps with no board change | 200 |
| **Candy Crush** | Run out of moves (50 in env config) | 50 |
| **Tetris** | Stack reaches top; or 30 steps with no change | 200 (primitive) / ≈60 macro placements is equivalent in the actor pipeline |
| **Avalon** | 3 quest failures or assassination resolves after 3 quest successes | (per-env) |
| **Diplomacy** | Solo victory or 20 phases elapsed | (per-env) |
| **Super Mario** | Level complete or game over | 100 |

## Design Notes

- **Diplomacy 20-phase limit**: Matches AgentEvolver's `DiplomacyConfig.max_phases`.
  Phases 1-20 contain the richest strategic diversity; more episodes at 20
  phases produces better seed skills than fewer at 50 phases.
- **Parallelized API calls**: Per-power (Diplomacy, 7 concurrent) and
  per-player (Avalon, up to 5 concurrent) calls are parallelized via
  `ThreadPoolExecutor` for ~7x speedup.
- **Labeling**: Off by default. Use the separate `labeling/` folder, or
  opt in with `--label` on the shell scripts.
