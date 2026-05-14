# Installation

For the full install guide (troubleshooting, orak-mario, environment diagrams),
see [install/README.md](install/README.md).

## Requirements

- **Python**: 3.9–3.11 (3.11 recommended)
- **OS**: Linux (Ubuntu 20.04+ recommended). macOS for development only (no vLLM / FSDP).
- **GPU**: 8× A100-80GB (or H100 / H200) for full co-evolution training.
  1× GPU (24+ GB) sufficient for inference and baselines.
- **CUDA driver**: ≥ 570.x (CUDA 12.8). See [CUDA driver compatibility](#cuda-driver-compatibility) for version-specific guidance.

### Key software versions

| Package | Version | Notes |
|---|---|---|
| PyTorch | 2.11.0+cu128 / cu130 | Pulled transitively by vLLM |
| vLLM | ≥ 0.20.0 | Required for `qwen3_5` / `qwen3_5_moe` architecture support |
| Transformers | ≥ 5.6.0 | Required for Qwen3.5-9B and Qwen3.5-35B-A3B |
| PEFT | ≥ 0.17 | LoRA adapter loading / merging on Transformers 5.x |
| Accelerate | ≥ 1.0 | FSDP / multi-GPU helpers |
| SentenceTransformers | ≥ 5.0 | RAG embeddings (Qwen3-Embedding-0.6B) |

### Model stack

| Tier | Model | Role | Served on |
|---|---|---|---|
| Actor + Skill-Bank | **Qwen/Qwen3.5-9B** | LoRA-trained policy (5 adapters) | vLLM `:8000` |
| Teacher + Judge | **Qwen/Qwen3.5-35B-A3B** | Frozen MoE control-plane + eval judge | vLLM `:8001` |
| SFT Teacher | **gpt-5.5** | Cold-start data generation (one-time) | OpenAI API |

### GPU allocation (8× A100 co-evolution)

| GPUs | Role |
|---|---|
| 0–3 | vLLM inference servers (Qwen3.5-9B, TP=1 per GPU) |
| 4–7 | FSDP GRPO training **or** 35B-A3B judge/teacher (TP=4) |

## Quick Start

```bash
# 1. Clone all repos into the same parent directory
mkdir -p ~/cos-play && cd ~/cos-play

git clone <this-repo> Multi-hop-Reasoning-VLM-Agent
git clone https://github.com/lmgame-org/GamingAgent.git
git clone https://github.com/modelscope/AgentEvolver.git
git clone https://github.com/nicholascpark/orak.git Orak    # optional, for Super Mario

# 2. Install the main environment (creates conda env "game-ai-agent" + all deps)
bash Multi-hop-Reasoning-VLM-Agent/install/install_main_env.sh

# 3. (Optional) Install the orak-mario environment for Super Mario
bash Multi-hop-Reasoning-VLM-Agent/install/install_orak_mario.sh

# 4. Configure API keys
cp Multi-hop-Reasoning-VLM-Agent/.env.example Multi-hop-Reasoning-VLM-Agent/.env
# Edit .env — at minimum set OPENAI_API_KEY for gpt-5.5 SFT teacher

# 5. Activate
conda activate game-ai-agent
export PYTHONPATH=$(pwd)/Multi-hop-Reasoning-VLM-Agent:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
set -a && source Multi-hop-Reasoning-VLM-Agent/.env && set +a
```

## CUDA driver compatibility

The default `pip install vllm>=0.20` resolves to **`torch 2.11.0+cu130`**,
which requires a CUDA 13.0-capable driver (≥ 575.x). Many A100 clusters
ship with the **570.x driver** (CUDA 12.8), which causes PyTorch to report
`CUDA available: False` even though `nvidia-smi` shows all GPUs.

**Fix** — after running `install_main_env.sh`, replace the cu130 stack
with the cu128 variant:

```bash
conda activate game-ai-agent

# 1. Remove the cu130 packages
pip uninstall -y vllm torch torchvision torchaudio

# 2. Reinstall with cu128 (requires uv for --torch-backend selection)
pip install uv
uv pip install vllm --torch-backend=cu128
```

This installs `torch==2.11.0+cu128` + `vllm==0.20.2` compiled against
CUDA 12.8, fully compatible with the 570.x driver family.

**How to check your driver:**

```bash
nvidia-smi | head -3
# Look for "CUDA Version: 12.8" → use cu128
# Look for "CUDA Version: 13.0" → default cu130 is fine
```

| Driver version | Max CUDA | PyTorch variant | Install method |
|---|---|---|---|
| ≥ 575.x | 13.0 | `torch+cu130` (default) | `pip install vllm` |
| 570.x | 12.8 | `torch+cu128` | `uv pip install vllm --torch-backend=cu128` |
| 555.x–565.x | 12.4–12.6 | `torch+cu124` | `uv pip install vllm --torch-backend=cu124` |

## Alternative Install Methods

If you prefer not to use the automated install script:

```bash
cd Multi-hop-Reasoning-VLM-Agent

# Option A: pip install (editable mode — registers all packages)
pip install -e .

# Option B: pip install from requirements file
pip install -r install/requirements.txt
```

Both options require you to verify the PyTorch CUDA variant matches your
driver (see [CUDA driver compatibility](#cuda-driver-compatibility) above).

## Optional Dependencies

- **RAG (embeddings)**: `pip install -r rag/requirements.txt`
  See [rag/README.md](rag/README.md) for text and multimodal embedding setup.

- **Skill agents / boundary proposal**: `pip install -r skill_agents/boundary_proposal/requirements.txt`
  **Infer segmentation**: `pip install -r skill_agents/infer_segmentation/requirements.txt`

- **Game environments**: See [install/README.md](install/README.md),
  [env_wrappers/setup_gamingagent_eval_env.md](env_wrappers/setup_gamingagent_eval_env.md),
  [env_wrappers/README.md](env_wrappers/README.md).

## Verification

After installation, verify the environment:

```bash
conda activate game-ai-agent
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
import vllm; print(f'vLLM: {vllm.__version__}')
import transformers; print(f'Transformers: {transformers.__version__}')
"
```

Expected output on 8× A100 with CUDA 12.8:

```
PyTorch: 2.11.0+cu128
CUDA available: True
GPU count: 8
vLLM: 0.20.2
Transformers: 5.8.1
```
