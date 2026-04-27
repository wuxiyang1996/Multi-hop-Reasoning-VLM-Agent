# Installation Guide

> **Visual-grounding benchmarks (BrowserGym, OSWorld, VisualToolBench, TIR-Bench, Video-Holmes):**
> see [`INSTALL_BENCHMARKS.md`](INSTALL_BENCHMARKS.md). This file covers the
> training / actor / skill-bank stack and the bundled game environments.

COS-PLAY uses **three core conda environments** (plus one optional add-on for Super Mario):

| Environment | Purpose | Python | Torch / CUDA | Install Time |
|---|---|---|---|---|
| **`game-ai-agent`** | GRPO training, vLLM inference, baselines, plus 2048 / Tetris (GamingAgent), Avalon, Diplomacy, **gym-v** (179 visual envs incl. 13 stable-retro Sega Genesis games), and the four **visual-reasoning benchmark loaders** (TIR-Bench, VisualToolBench, Video-Holmes, SIV-Bench) | 3.11 | torch 2.11+cu130 | ~15 min |
| **`browsergym`** | MiniWoB++ / WebArena / VisualWebArena / AssistantBench (2,063 tasks total) — see [`INSTALL_BENCHMARKS.md`](INSTALL_BENCHMARKS.md) §2 | 3.11 | torch 2.4.1+cu121 | ~10 min |
| **`osworld`** | OSWorld desktop tasks (369 Ubuntu + 49 Windows examples) — see [`INSTALL_BENCHMARKS.md`](INSTALL_BENCHMARKS.md) §3 | 3.11 | torch 2.5.1+cu124 | ~10 min |
| **`orak-mario`** *(optional)* | Super Mario Bros — `nes-py` requires `numpy<2` and `gym==0.26.2` so it cannot share `game-ai-agent` | 3.11 | torch latest+cu124 | ~5 min |

> **Why three core envs?** `game-ai-agent` runs `gymnasium 1.3` + `numpy 2.x` (latest vLLM / Qwen3.5 stack); BrowserGym hard-pins `playwright==1.44`; OSWorld hard-pins `gymnasium~=0.28.1`, `transformers~=4.35`, `torch~=2.5`. They cannot co-resolve. See the incompatibility matrix in [`INSTALL_BENCHMARKS.md`](INSTALL_BENCHMARKS.md) §"TL;DR".

## Prerequisites

- **OS:** Linux (Ubuntu 20.04+ recommended). macOS for development only (no vLLM / FSDP).
- **GPU:** 8× A100 / H100 / B200 recommended for full GRPO co-evolution training. 1× GPU is sufficient for inference / baselines.
- **CUDA:** 12.8+ or 13.x driver on the host. The `game-ai-agent` env pulls `torch 2.11.0+cu130` transitively from vLLM, which is forward-compatible with any CUDA 12.8+ driver.
- **Conda:** Miniconda3 or Anaconda.
- **System libs (BrowserGym only):** `python -m playwright install-deps chromium` requires sudo on first install (handled by `install_browsergym.sh`).

```bash
# Install Miniconda if not present
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init
```

## Quick Start

```bash
# 1. Clone this repo + sibling repos under one parent
mkdir -p ~/cos-play && cd ~/cos-play

git clone https://github.com/<your-org>/Multi-hop-Reasoning-VLM-Agent.git
git clone https://github.com/lmgame-org/GamingAgent.git
git clone https://github.com/modelscope/AgentEvolver.git
git clone https://github.com/ModalMinds/gym-v.git              # for gym-v + Temporal/* envs
# Optional / per-domain:
git clone https://github.com/ServiceNow/BrowserGym.git         # for browsergym env
git clone https://github.com/xlang-ai/OSWorld.git              # for osworld env
git clone https://github.com/nicholascpark/orak.git Orak       # for Super Mario

# 2. Install the main environment (training + gym-v + GamingAgent)
#    Optional 2nd arg = path to a Sega Genesis ROM zip (enables Temporal/* envs).
bash Multi-hop-Reasoning-VLM-Agent/install/install_main_env.sh \
     "" \
     /path/to/Mega_Drive_Mini_Full_Set.zip

# 3. (Optional) Install the visual-grounding benchmark envs
bash Multi-hop-Reasoning-VLM-Agent/install/install_browsergym.sh
bash Multi-hop-Reasoning-VLM-Agent/install/install_osworld.sh

# 4. (Optional) Install the orak-mario env for Super Mario Bros
bash Multi-hop-Reasoning-VLM-Agent/install/install_orak_mario.sh

# 5. Configure API keys
cp Multi-hop-Reasoning-VLM-Agent/.env.example Multi-hop-Reasoning-VLM-Agent/.env
# Edit .env with your keys (OpenRouter, OpenAI, Anthropic, Google, Together, xAI, Z.AI)
```

---

## 1. Main Environment (`game-ai-agent`)

### What it covers

- **Training:** GRPO + FSDP + LoRA on Qwen3.5-9B (co-evolution + cold-start)
- **Inference server:** vLLM 0.20.0 serving Qwen3.5-9B (training-time) and Qwen3.5-35B-A3B (long-horizon eval)
- **Skill bank pipeline:** boundary proposal, segmentation, contracts, curation, RAG retrieval (Qwen3-Embedding-0.6B)
- **Baselines:** GPT-class, Claude-class, Gemini-class, GPT-OSS-class via OpenRouter / native SDKs
- **Game environments (in-process, no extra env switch):**
  - **GamingAgent / LMGame-Bench:** 2048, Tetris (plain). *Candy Crush is excluded — `tile_match_gym` pulls in `numba` which forces `numpy<2` and breaks the Qwen3.5/vLLM stack. Run it via `SubprocessEnv` if needed.*
  - **gym-v** (ModalMinds): 179 visual envs (Games, Spatial, Temporal, Reasoning) including the 13 `Temporal/*` Sega Genesis games when ROMs are imported via the bundled `gymv_temporal_patch/`.
  - **Avalon, Diplomacy** (via `PYTHONPATH`).
- **Visual-reasoning benchmark loaders** (`visual_reasoning_wrapper.benchmarks`):
  - **TIR-Bench** (`Agents-X/TIR-Bench`, ~1.2 GB) — image, 1 215 questions across 13 task families.
  - **VisualToolBench** (`ScaleAI/VisualToolBench`, HF stream) — image, 1 204 single-turn rows.
  - **Video-Holmes** (`TencentARC/Video-Holmes`, ~4 GB, gated) — video, 1 837 MCQs over 503 cropped clips.
  - **SIV-Bench** (`Fancylalala/SIV-Bench`, ~42 GB) — video, 8 728 social-interaction MCQs over 2 792 clips.

  All four iterators / parsers, plus `decord`-based frame sampling and `vlm_wrapper.ground.cascaded_ground` (API-driven), import cleanly inside `game-ai-agent`. Heavy in-process grounding heads (Florence-2 / GroundingDINO / OmniParser-v2 / EasyOCR / YOLO) are kept in the optional `vlm_benchmarks` env. See [`INSTALL_BENCHMARKS.md`](INSTALL_BENCHMARKS.md) §4 for the on-disk data layout and the `Benchmark/` wrapper that Video-Holmes expects.

### Install

```bash
cd ~/cos-play

# Minimal install (no Sega Genesis ROMs)
bash Multi-hop-Reasoning-VLM-Agent/install/install_main_env.sh

# With Sega Genesis ROMs → enables the 13 Temporal/* envs (StreetsOfRage2, GoldenAxe, …)
bash Multi-hop-Reasoning-VLM-Agent/install/install_main_env.sh \
     "" \
     /path/to/Mega_Drive_Mini_Full_Set.zip
```

Positional arguments:
1. `CONDA_PATH` — explicit `conda` binary (auto-detected if blank)
2. `ROM_ZIP` — path to a Sega Genesis ROM zip; passed to `gymv_temporal_patch/apply_patch.sh`. Skip to install gym-v without retro envs.

The script:
1. Creates `game-ai-agent` (Python 3.11) if missing.
2. Installs all pip dependencies from `install/requirements.txt` (vLLM transitively pulls torch 2.11+cu130).
3. Installs `GamingAgent` editable + the LLM-provider SDKs it imports (`google-generativeai`, `together`, `zai-sdk`, `xai-sdk`, `pygame`).
4. Clones `gym-v` (if absent) and installs `pip install -e gym-v[games,spatial]`.
5. If `ROM_ZIP` is provided, runs `gymv_temporal_patch/apply_patch.sh` to enable the 13 `Temporal/*` retro envs.
6. Verifies AgentEvolver is reachable via `PYTHONPATH`.
7. Runs ~35 import / version checks (`gym_v Temporal/* envs`, `vllm`, `transformers`, etc.).

### Activate

```bash
conda activate game-ai-agent
export PYTHONPATH=$(pwd)/Multi-hop-Reasoning-VLM-Agent:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
```

### Set API keys

```bash
cp Multi-hop-Reasoning-VLM-Agent/.env.example Multi-hop-Reasoning-VLM-Agent/.env
# Edit .env — at minimum set OPENROUTER_API_KEY for baselines
set -a && source Multi-hop-Reasoning-VLM-Agent/.env && set +a
```

### Verify

```bash
python -c "from API_func import api_call; print('API_func OK')"
python -c "import gym_v, gym_v.envs; print('Temporal/*:', len([e for e in gym_v.registry if e.startswith('Temporal/')]))"
python -c "from env_wrappers.gym_like import make_gaming_env; e=make_gaming_env('twenty_forty_eight'); o=e.reset(); print('2048 OK', list(o[0].keys()) if isinstance(o, tuple) else type(o).__name__)"
python -c "
from visual_reasoning_wrapper.benchmarks import (
    iter_tir_bench_samples, iter_visual_toolbench_samples,
    iter_video_holmes_samples, iter_siv_bench_samples,
)
print('TIR     ', next(iter_tir_bench_samples(limit=1)).task)
print('VHolmes ', next(iter_video_holmes_samples(limit=1)).question_type)
print('SIV     ', next(iter_siv_bench_samples(limit=1)).dimension)
"
pytest Multi-hop-Reasoning-VLM-Agent/tests/ -q
```

### Key dependencies (game-ai-agent)

| Package | Version | Purpose |
|---|---|---|
| torch | 2.11.0+cu130 | GRPO, FSDP, LoRA training (CUDA 12.8+ / 13.x) |
| transformers | 5.6.2 | Qwen3.5-9B / Qwen3.5-35B-A3B archs (`qwen3_5`, `qwen3_5_moe`) |
| vllm | 0.20.0 | Fast inference server with Qwen3.5 support |
| peft | 0.19.1 | LoRA adapters on transformers 5.x |
| accelerate | 1.13.0 | FSDP / multi-GPU helpers compatible with torch 2.11 |
| sentence-transformers | ≥5.0 | Qwen3-Embedding-0.6B retriever |
| numpy | 2.x | Latest stack (no longer pinned to 1.26) |
| gymnasium | 1.3.0 | Modern API used by gym-v + GamingAgent envs |
| gym-v | 0.1.0 | 179 visual envs (Games / Spatial / Temporal) |
| stable-retro | 1.0.0 | Sega Genesis backend for `Temporal/*` |
| google-generativeai / google-genai | ≥0.8 / ≥1.0 | Gemini API + GamingAgent's runtime |
| openai / anthropic | ≥2.0 / ≥0.97 | OpenRouter, OpenAI, Claude baselines |
| together / zai-sdk / xai-sdk | latest | Required by GamingAgent's `api_providers.py` |
| pygame | ≥2.6 | Tetris rendering inside GamingAgent |
| diplomacy | ≥1.1.2 | Diplomacy game engine |
| scikit-learn | ≥1.0 | Skill bank clustering |
| decord | 0.6.0 | Fast random-access video reader (Video-Holmes / SIV-Bench) |
| opencv-python-headless | ≥4.8 | `cv2` fallback for video frame decode (no X11 deps) |

Full list: [`install/requirements.txt`](requirements.txt)

---

## 2. Visual-grounding benchmark environments (`browsergym`, `osworld`)

These are interactive runtimes used by the `vlm_wrapper` grounding pipeline and have hard-pinned dependency sets that cannot co-resolve with `game-ai-agent`. Each gets its own conda env.

```bash
# BrowserGym  — 2,063 tasks (910 vwa + 812 wa + 215 ab + 125 miniwob + 1 openended)
bash install/install_browsergym.sh
#   (clones BrowserGym, creates `browsergym` env, pip install editable,
#    runs `playwright install-deps chromium && playwright install chromium`)

# OSWorld  — 369 Ubuntu + 49 Windows tasks (requires Docker / VMware / AWS to actually run)
bash install/install_osworld.sh
#   (clones OSWorld, creates `osworld` env, pip install -r requirements.txt + editable)
```

Full setup, VM backend selection, and a per-runtime task table: [`INSTALL_BENCHMARKS.md`](INSTALL_BENCHMARKS.md).

---

## 3. Super Mario Environment (`orak-mario`) *— optional*

### Why a separate environment?

`nes-py` (the NES emulator wrapper) requires **numpy<2** and **gym==0.26.2**, which conflict with the main env's stack (vLLM, transformers 5.x, gymnasium 1.3, numpy 2.x).

### Install

```bash
cd ~/cos-play
bash Multi-hop-Reasoning-VLM-Agent/install/install_orak_mario.sh
```

### Activate

```bash
source Multi-hop-Reasoning-VLM-Agent/env_wrappers/setup_orak_mario.sh   # also sets PYTHONPATH + DISPLAY (Xvfb)
```

Or manually:

```bash
conda activate orak-mario
export PYTHONPATH=$(pwd)/Multi-hop-Reasoning-VLM-Agent:$(pwd)/Orak/src:$PYTHONPATH
```

### Headless servers (no display)

`nes-py` / `pyglet` require an X display. The setup script starts Xvfb on `:99` automatically. If you see display errors:

```bash
sudo apt install -y xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

### Verify

```bash
python -c "import gym_super_mario_bros; print('Mario env OK')"
python -c "import nes_py, numpy; print(f'numpy {numpy.__version__}')"
```

### Key dependencies (orak-mario)

| Package | Version | Purpose |
|---|---|---|
| nes-py | ==8.2.1 | NES emulator |
| gym-super-mario-bros | ==7.4.0 | Mario environment |
| gym | ==0.26.2 | OpenAI Gym (nes-py compat) |
| numpy | <2 | nes-py incompatible with NumPy 2.x |
| torch + torchvision | latest | Orak object detection |
| opencv-python-headless | latest | Vision processing |
| openai | ≥1.0 | LLM baselines via OpenRouter |

Full list: [`install/requirements-orak-mario.txt`](requirements-orak-mario.txt)

---

## Troubleshooting

### `conda: command not found`

```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
# Or pass the path directly:
bash Multi-hop-Reasoning-VLM-Agent/install/install_main_env.sh $HOME/miniconda3/bin/conda
```

### `gamingagent 0.1.0 requires numpy==1.24.4`

Nominal warning only. We run `numpy 2.x` in `game-ai-agent` for the latest vLLM / torch 2.11 stack; GamingAgent's 2048 + plain Tetris work fine. Candy Crush is intentionally excluded (its `tile_match_gym` dep transitively forces `numpy<2` via `numba`).

### `ModuleNotFoundError: No module named 'google.generativeai' / 'together' / 'zai' / 'xai_sdk'`

These LLM SDKs are imported by `GamingAgent/tools/serving/api_providers.py` even when you don't use those providers. They are now in `requirements.txt`; re-run:

```bash
conda run -n game-ai-agent pip install -r install/requirements.txt
```

### `ModuleNotFoundError: No module named 'games'`

AgentEvolver is loaded via `PYTHONPATH`, not pip. Set it:

```bash
export PYTHONPATH=$(pwd)/Multi-hop-Reasoning-VLM-Agent:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
```

### `gym_v Temporal/* envs` count is 0

You skipped the ROM-import step. Re-run:

```bash
bash install/gymv_temporal_patch/apply_patch.sh \
     /path/to/gym-v \
     /path/to/Mega_Drive_Mini_Full_Set.zip
```

The script is idempotent. See [`gymv_temporal_patch/README.md`](gymv_temporal_patch/README.md).

### Visual-reasoning loaders raise `FileNotFoundError`

Each loader resolves data via `default_*_root()` which returns
`<repo>/data/<bench>`. Either symlink the data into the repo (cheap if
your benchmarks live elsewhere) or pass an explicit path:

```bash
ln -s /scratch/visual-reasoning-data /path/to/Multi-hop-Reasoning-VLM-Agent/data
# or
python -c "from visual_reasoning_wrapper.benchmarks import iter_video_holmes_samples
print(next(iter_video_holmes_samples(video_holmes_root='/scratch/Video-Holmes', limit=1)))"
```

Video-Holmes specifically needs a `Benchmark/` wrapper subdir
(`<root>/Benchmark/{annotations,annotation_training,videos/videos_cropped,*.json}`)
— see [`INSTALL_BENCHMARKS.md`](INSTALL_BENCHMARKS.md) §4.

### `vllm` installation fails

vLLM 0.20.0 requires CUDA 12.8+. Verify with `nvidia-smi`. CPU-only fallback (no local vLLM, API baselines only):

```bash
# Comment out vllm in requirements.txt, then:
pip install -r install/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### BrowserGym `playwright` system deps missing

The installer runs `python -m playwright install-deps chromium` which needs sudo on first install. If you don't have sudo:

```bash
sudo apt-get install -y libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2t64
```

### `nes_py` or `pyglet` display errors (Super Mario)

```bash
sudo apt install -y xvfb
# The setup script (env_wrappers/setup_orak_mario.sh) starts Xvfb automatically.
```

### CUDA version mismatch in orak-mario

The `orak-mario` install defaults to CUDA 12.4 wheels (matches the legacy `nes-py` stack). For a different CUDA version, edit `install_orak_mario.sh` step `[2/4]`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## File Structure

```
install/
├── README.md                       # This file (training stack + game-ai-agent + orak-mario)
├── INSTALL_BENCHMARKS.md           # Visual-grounding benchmarks (BrowserGym, OSWorld, vlm_benchmarks, gymv standalone)
├── install_main_env.sh             # game-ai-agent installer (training + gym-v + GamingAgent)
├── install_browsergym.sh           # browsergym installer (BrowserGym + Playwright)
├── install_osworld.sh              # osworld installer (OSWorld + desktop-env)
├── install_gymv.sh                 # standalone `gymv` env (rare — for VLM-eval-only setups)
├── install_orak_mario.sh           # orak-mario installer (Super Mario only)
├── requirements.txt                # game-ai-agent pip deps
├── requirements-orak-mario.txt     # orak-mario pip deps
├── browsergym.environment.yml      # browsergym conda env spec
├── osworld.environment.yml         # osworld conda env spec
├── gymv.environment.yml            # standalone `gymv` env spec (optional)
├── vlm_benchmarks.environment.yml  # vlm_benchmarks env spec (image/video QA + grounding)
├── *_smoke.py                      # one-line OK/FAIL/WARN smoke tests per env
└── gymv_temporal_patch/            # Sega Genesis ROM import + multimodal upgrades for Temporal/* envs
```

## Environment Summary

```
┌──────────────────────────────────────────────────────────────────────────┐
│  game-ai-agent  (main, ~9 GB on disk)                                    │
│  ├── Training:   GRPO, FSDP, LoRA on Qwen3.5-9B                          │
│  ├── Inference:  vLLM 0.20 serving Qwen3.5-9B / Qwen3.5-35B-A3B          │
│  ├── Skill bank: boundary, segmentation, contract, curation, RAG         │
│  ├── Games:      2048, Tetris, Avalon, Diplomacy, gym-v (179 envs)       │
│  ├── VR-bench:   TIR-Bench + VisualToolBench (image),                    │
│  │               Video-Holmes + SIV-Bench (video, decord)                │
│  └── Stack:      torch 2.11+cu130, transformers 5.6.2, numpy 2.x         │
├──────────────────────────────────────────────────────────────────────────┤
│  browsergym  (separate — playwright==1.44, transformers 4.57, ~6 GB)     │
│  └── Tasks:      MiniWoB / WebArena / VisualWebArena / AssistantBench    │
├──────────────────────────────────────────────────────────────────────────┤
│  osworld  (separate — gymnasium 0.28.1, transformers 4.35, ~9 GB)        │
│  └── Tasks:      OSWorld desktop (Ubuntu 369 + Windows 49)               │
├──────────────────────────────────────────────────────────────────────────┤
│  orak-mario  (separate — numpy<2, gym==0.26.2)                           │
│  └── Game:       Super Mario Bros (nes-py)                               │
└──────────────────────────────────────────────────────────────────────────┘
```
