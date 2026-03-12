# Labeling — Episode Annotation with GPT-5.4

This folder contains code and scripts for annotating cold-start episode
trajectories with concise labels suitable for RAG retrieval, the manager
agent, and downstream skill extraction.

## What Gets Labeled

For **each experience step** in an episode:

| Field           | Description |
|-----------------|-------------|
| `summary_state` | Compact state summary (1-2 sentences, ≤60 words). Factual board/player/score description for retrieval. |
| `summary`       | Identical to `summary_state`. |
| `intentions`    | Concise reasoning (1-2 sentences, ≤40 words) for **why** the action was chosen. |
| `skills`        | Renamed from `sub_tasks`. Left `null` — populated by the skill pipeline downstream. |

### Design Principles

- **Concise for RAG**: Summaries and intentions are short so embedding-based
  retrieval stays accurate and context windows stay lean.
- **Deterministic pre-compression**: Uses `compact_text_observation` and
  `get_state_summary` from `decision_agents.agent_helper` before calling
  GPT-5.4, reducing token cost and improving consistency.
- **GPT-5.4 refinement**: The LLM produces natural-language annotations that
  go beyond what deterministic heuristics can achieve.

## Files

| File                      | Purpose |
|---------------------------|---------|
| `label_episodes_gpt54.py` | Main labeling script. Reads episode JSONs, calls GPT-5.4, writes labeled output. |
| `run_labeling.sh`         | Convenience shell wrapper (sets PYTHONPATH, runs the script). |
| `readme.md`               | This file. |

## Usage

```bash
# From Game-AI-Agent root
export OPENROUTER_API_KEY="sk-or-..."
export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

# Label all games (reads from cold_start/output/gpt54/, writes to labeling/output/gpt54/)
python labeling/label_episodes_gpt54.py

# Label specific game(s)
python labeling/label_episodes_gpt54.py --games tetris candy_crush

# Label a single episode file
python labeling/label_episodes_gpt54.py --input_file cold_start/output/gpt54/tetris/episode_000.json

# Dry run — preview labeling on one episode without saving
python labeling/label_episodes_gpt54.py --dry_run --games tetris --max_episodes 1

# Process exactly one rollout per game (quick test across all games)
python labeling/label_episodes_gpt54.py --one_per_game -v

# Write labels in-place (overwrite originals)
python labeling/label_episodes_gpt54.py --in_place

# Or use the shell wrapper:
bash labeling/run_labeling.sh --games tetris -v
```

## CLI Options

| Flag              | Default                           | Description |
|-------------------|-----------------------------------|-------------|
| `--input_dir`     | `cold_start/output/gpt54`        | Input directory with `<game>/episode_*.json` |
| `--input_file`    | —                                 | Label a single file instead of scanning a directory |
| `--output_dir`    | `labeling/output/gpt54`          | Output directory for labeled episodes |
| `--games`         | all found                         | Filter to specific game(s) |
| `--model`         | `gpt-5.4`                        | LLM model for labeling |
| `--max_episodes`  | all                               | Cap episodes per game |
| `--one_per_game`  | off                               | Process only the first episode for each game |
| `--delay`         | `0.1`                            | Seconds between API calls (rate limiting) |
| `--overwrite`     | off                               | Re-label already-labeled episodes |
| `--in_place`      | off                               | Write back to original input files |
| `--dry_run`       | off                               | Preview without saving |
| `--verbose / -v`  | off                               | Print per-step details |

## Output Structure

```
labeling/output/gpt54/
├── tetris/
│   ├── episode_000.json          # labeled episode
│   ├── episode_001.json
│   └── labeling_summary.json     # per-game stats
├── candy_crush/
│   └── ...
└── labeling_batch_summary.json   # overall run stats
```

## Pipeline Integration

The labeled episodes follow the same `Episode` / `Experience` schema from
`data_structure/experience.py` and can be loaded directly:

```python
from data_structure.experience import Episode
import json

with open("labeling/output/gpt54/tetris/episode_000.json") as f:
    ep = Episode.from_dict(json.load(f))

for exp in ep.experiences:
    print(exp.summary_state)  # concise state summary
    print(exp.intentions)     # concise action reasoning
```

## Functions Used from `decision_agents`

| Function | Source | Role in Labeling |
|----------|--------|-----------------|
| `compact_text_observation()` | `agent_helper.py` | Deterministic state pre-compression before GPT-5.4 |
| `get_state_summary()` | `agent_helper.py` | Structured/text state summarisation backbone |
| `infer_intention()` | `agent_helper.py` | Fallback intention inference when GPT-5.4 call fails |
| `strip_think_tags()` | `agent_helper.py` | Strip `<think>` blocks from reasoning model output |
