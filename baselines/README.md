# Baselines

LLM API baselines for evaluating frontier models on the 6 game environments (Table 1 in the paper). Each game has a single script that accepts a `--model` flag.

There is also a self-hosted **Qwen via vLLM** baseline that runs Qwen3.5-9B and Qwen3.5-35B-A3B over both the env_wrappers games and the gym-v Temporal envs (see `run_qwen_vllm_baselines.sh` below).

## Supported Models

| Model | Flag |
|-------|------|
| GPT-5.4 | `--model gpt-5.4` |
| GPT-OSS 120B | `--model openai/gpt-oss-120b` |
| Gemini 3.1 Pro | `--model google/gemini-3.1-pro-preview` |
| Claude 4.6 Sonnet | `--model anthropic/claude-4.6-sonnet-20260217` |
| Qwen3.5-9B (vLLM) | `--models 9B`  (see Qwen vLLM section) |
| Qwen3.5-35B-A3B (vLLM) | `--models 35B` (see Qwen vLLM section) |

## Usage

```bash
# Single-player games (no vLLM server needed, uses OpenRouter API)
bash baselines/run_tetris_baseline.sh                                      # GPT-5.4 default
bash baselines/run_tetris_baseline.sh --model openai/gpt-oss-120b
bash baselines/run_2048_baseline.sh --model google/gemini-3.1-pro-preview
bash baselines/run_candy_crush_baseline.sh --model anthropic/claude-4.6-sonnet-20260217

# Multi-agent games (controlled model vs GPT-5.4 opponents)
bash baselines/run_avalon_baseline.sh --model openai/gpt-oss-120b          # 40 eps (8/player × 5)
bash baselines/run_diplomacy_baseline.sh --model google/gemini-3.1-pro-preview  # 56 eps (8/power × 7)

# Super Mario (requires orak-mario conda env for NES emulator)
bash baselines/run_super_mario_baseline.sh --model openai/gpt-oss-120b

# Override episodes / temperature
EPISODES=16 bash baselines/run_tetris_baseline.sh --model gpt-5.4
EPISODES_PER_POWER=4 bash baselines/run_diplomacy_baseline.sh --model gpt-5.4
```

## Frontier sweep via OpenRouter (Claude 4.6 Sonnet + Gemini 3.1 Pro + Qwen3.5 Plus)

`run_openrouter_baselines.sh` runs the same COS-PLAY actor cold-start pipeline used by `run_qwen_vllm_baselines.sh`, but routes every call through **OpenRouter** so no GPUs are required. It dispatches all three frontier multimodal backbones across **all 17 games** (3 env_wrappers + super_mario + 13 gym-v Temporal envs) at **16 episodes per (model × env)** with **16 jobs in flight**.

Total work: 3 backbones × 17 games × 16 episodes = **816 episodes**. Estimated total spend: **≈ $2.1k** (≈ $1094 Claude + $817 Gemini + $164 Qwen3.5 Plus, vision ON).

| Backbone | Slug | Input ($/M) | Output ($/M) |
|----------|------|-------------|--------------|
| Claude Sonnet 4.6 | `anthropic/claude-4.6-sonnet-20260217` | 3.00 | 15.00 |
| Gemini 3.1 Pro    | `google/gemini-3.1-pro-preview`        | 2.00 | 12.00 |
| Qwen3.5 Plus      | `qwen/qwen3.5-plus-20260420`           | 0.40 |  2.40 |

```bash
# All 3 models × 17 games × 16 episodes, vision ON
bash baselines/run_openrouter_baselines.sh

# Drop a model from the sweep
bash baselines/run_openrouter_baselines.sh --models claude gemini

# Smoke run: 4 episodes, 2 gym-v envs only
bash baselines/run_openrouter_baselines.sh --episodes 4 \
    --gymv Temporal/Airstriker-v0 Temporal/Columns-v0 --skip_envwrappers

# Skip super_mario (no orak-mario conda env)
bash baselines/run_openrouter_baselines.sh --skip_mario

# Resume an interrupted run
bash baselines/run_openrouter_baselines.sh --run_id myrun --resume
```

Outputs land under `<codebase_root>/openrouter-baselines-out/<run_id>/<model_tag>/{env_wrappers,gymv}/<env>/` (mirrors the `qwen-baselines-out` layout). The `OPENROUTER_API_KEY` is auto-loaded from `<workspace>/api_keys.py` if not already in the environment.

## Qwen via vLLM (Qwen3.5-9B + Qwen3.5-35B-A3B)

`run_qwen_vllm_baselines.sh` reuses the COS-PLAY actor cold-start pipeline (visual grounding + schema-driven action selection) from `cold_start/generate_cold_start_actor.py` and `cold_start/generate_cold_start_actor_gymv.py`, but routes every call through a **vLLM OpenAI-compatible endpoint** instead of OpenAI / OpenRouter.

By default it runs **both backbones across both env families in parallel** with **16 episodes per (model × env) combo** and **16 in-flight jobs** — sized to keep an 8× H200 box (4 GPUs per backbone, TP=4) saturated:

| Backbone              | GPUs (default) | TP | Port | URL                          |
|-----------------------|----------------|----|------|------------------------------|
| `Qwen/Qwen3.5-9B`     | `0,1,2,3`      | 4  | 8000 | `http://localhost:8000/v1`   |
| `Qwen/Qwen3.5-35B-A3B`| `4,5,6,7`      | 4  | 8001 | `http://localhost:8001/v1`   |

Total work: 2 backbones × (3 env_wrappers + 13 gym-v Temporal envs) × 16 episodes = **512 episodes**.

### One-shot on an 8× H200 box: launch servers + run baselines

The orchestrator can spin both vLLM servers up in-script (and tears them down on exit) when given `--launch_servers`:

```bash
bash baselines/run_qwen_vllm_baselines.sh --launch_servers
```

Both servers warm up under `qwen-baselines-out/<run_id>/_logs/_servers/{9B,35B-A3B}.log`; the script blocks on each `/health` endpoint before dispatching baseline jobs. Override allocations with `--gpus_9b 0,1,2,3 --tp_9b 4 --gpus_35b 4,5,6,7 --tp_35b 4` (or any custom split). If a server is already running at the expected URL the launcher detects it and reuses it instead of double-launching.

### Bring up the vLLM servers manually (alternative)

If you'd rather manage the servers separately:

```bash
# Qwen3.5-9B  -> 4x H200 (TP=4)  :8000
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 4 --max-model-len 4096 \
    --enable-prefix-caching --enable-chunked-prefill \
    --gpu-memory-utilization 0.85 --dtype auto --trust-remote-code &

# Qwen3.5-35B-A3B  -> 4x H200 (TP=4 + expert-parallel)  :8001
GPUS=4,5,6,7 TENSOR_PARALLEL=4 PORT=8001 \
    bash inference/serve_qwen35_35b_a3b.sh &

# Once both health endpoints are green:
bash baselines/run_qwen_vllm_baselines.sh
```

Override the URLs with `VLLM_QWEN_9B_URL` / `VLLM_QWEN_35B_URL` env vars. Endpoints that are unreachable when the script starts are auto-skipped.

### Useful subsets

```bash
# Only Qwen3.5-9B, only env_wrappers games:
bash baselines/run_qwen_vllm_baselines.sh --models 9B --skip_gymv

# Only gym-v Temporal envs (both backbones), 8 episodes for a smoke run:
bash baselines/run_qwen_vllm_baselines.sh --skip_envwrappers --episodes 8

# Restrict gym-v to two envs:
bash baselines/run_qwen_vllm_baselines.sh \
    --skip_envwrappers --gymv Temporal/Airstriker-v0 Temporal/Columns-v0

# Include super_mario (requires the `orak-mario` conda env):
bash baselines/run_qwen_vllm_baselines.sh --include_mario

# Crank concurrency further on an idle box (32 in-flight, ~16 per server):
bash baselines/run_qwen_vllm_baselines.sh --max_parallel 32

# Resume an interrupted run (output dir is reused, finished episodes skipped):
bash baselines/run_qwen_vllm_baselines.sh --run_id qwen_vllm_smoke --resume
```

### Output layout

Both backbones share **one** run directory under `<codebase_root>/qwen-baselines-out/` (gitignored, mirrors the `Cold-start-out/` convention):

```
qwen-baselines-out/
└── <run_id>/
    ├── 9B/
    │   ├── env_wrappers/
    │   │   ├── twenty_forty_eight/{episode_000.json … episode_015.json, rollouts.jsonl, rollout_summary.json}
    │   │   ├── candy_crush/...
    │   │   └── tetris/...
    │   └── gymv/
    │       ├── Temporal_Airstriker-v0/...
    │       └── ...  (one dir per env)
    ├── 35B-A3B/
    │   └── ...   (same structure as 9B/)
    ├── _logs/<model_tag>__<envw|gymv>__<env>.log
    └── _run_meta.json
qwen-baselines-out/latest -> <run_id>
```

### Aggregate the results

```bash
python baselines/summarize_qwen_vllm_baselines.py            # latest run
python baselines/summarize_qwen_vllm_baselines.py --run_dir qwen-baselines-out/<run_id>
```

This prints a per-(model × env) table (mean reward, ±95% CI, mean steps, etc.) and writes `qwen_vllm_summary.json` next to the per-run output.

## Analysis

After collecting results, compute win rates and confidence intervals for the multi-agent baselines:

```bash
python baselines/analyze_baselines.py
```

## Files

| File | Purpose |
|------|---------|
| `run_tetris_baseline.sh` | Tetris baseline via macro-action wrapper |
| `run_2048_baseline.sh` | 2048 baseline |
| `run_candy_crush_baseline.sh` | Candy Crush baseline |
| `run_avalon_baseline.sh` | Avalon baseline (per-role cycling vs GPT-5.4) |
| `run_diplomacy_baseline.sh` | Diplomacy baseline (per-power cycling vs GPT-5.4) |
| `run_super_mario_baseline.sh` | Super Mario baseline (Xvfb + orak-mario env) |
| `run_gpt54_tetris_macro.py` | Python backend for Tetris (shared by all models) |
| `run_qwen_vllm_baselines.sh` | Qwen3.5-{9B, 35B-A3B} baselines via vLLM (env_wrappers + gym-v, parallel, 16 eps) |
| `run_openrouter_baselines.sh` | Claude-4.6-Sonnet + Gemini-3.1-Pro + Qwen3.5-Plus baselines via OpenRouter (all 17 games, parallel, 16 eps) |
| `summarize_qwen_vllm_baselines.py` | Aggregate per-(model × env) Qwen vLLM stats |
| `analyze_baselines.py` | Post-hoc analysis: win rates, CIs |
