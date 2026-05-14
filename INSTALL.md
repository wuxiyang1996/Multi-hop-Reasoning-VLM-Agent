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
which requires a CUDA 13.0-capable driver (≥ 580.x). Many A100 clusters
ship with the **570.x driver** (CUDA 12.8), which causes PyTorch to report
`CUDA available: False` even though `nvidia-smi` shows all GPUs.

### The cu128 wheel reality (important)

**vLLM 0.20.x does NOT publish `+cu128` wheels.** Checking the GitHub release
assets for v0.20.0 / v0.20.1 / v0.20.2 shows only two GPU variants:

| vLLM 0.20.x asset | Compiled against |
|---|---|
| `vllm-0.20.x+cu129-...whl` (PyPI default for GPU) | CUDA 12.9 |
| `vllm-0.20.x-...whl` (no suffix, default index on Docker) | CUDA 13.0 |

The vLLM install docs do mention "we also provide vLLM binaries compiled
with CUDA 12.8" — that statement is aspirational and only applied to
earlier minor releases. For the 0.20 line, **the lowest CUDA wheel
available is `+cu129`.**

The good news: a **570.x driver runs `+cu129` binaries fine** via NVIDIA's
[CUDA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility).
The vLLM wheel bundles its own CUDA 12.9 user-space runtime (`libcudart.so.12`,
cuBLAS, cuDNN), which forward-compats onto any driver ≥ 570.86.10 within
CUDA 12.x. PyTorch is independent and is pinned to `+cu128` to stay
bit-identical with the driver's `libcuda.so`.

### Recommended cu128 install (570.x driver, A100/H100/H200)

After running `install_main_env.sh`:

```bash
conda activate game-ai-agent

# 1. Remove anything the auto-install pulled (cu130 torch, etc.)
pip uninstall -y vllm torch torchvision torchaudio

# 2. Install matching PyTorch + matching vLLM
pip install uv
uv pip install \
  torch==2.11.0 torchvision torchaudio \
  --torch-backend=cu128
uv pip install vllm==0.20.2 --torch-backend=cu128
```

`--torch-backend=cu128` swaps the PyTorch index to
`download.pytorch.org/whl/cu128` (true CUDA 12.8 torch). For vLLM, uv
still resolves to the only published GPU wheel (`+cu129`); this is
expected — its bundled 12.9 user-space libs run on the 570.x driver via
forward-compat.

After install you should see:

```
torch:        2.11.0+cu128
torch.version.cuda: 12.8
vllm:         0.20.2  (vllm-0.20.2+cu129-...whl on disk — expected)
```

**How to check your driver:**

```bash
nvidia-smi | head -3
# "CUDA Version: 12.8" (driver 570.x)  → use cu128 torch + cu129 vllm (this guide)
# "CUDA Version: 12.9" (driver 575.x)  → use cu129 torch + cu129 vllm
# "CUDA Version: 13.0" (driver 580.x+) → default `pip install vllm` is fine
```

| Driver | Max CUDA | Torch wheel | vLLM wheel | Install method |
|---|---|---|---|---|
| ≥ 580.x | 13.0 | `torch+cu130` (default) | `+cu130` (no suffix) | `pip install vllm` |
| 575.x | 12.9 | `torch+cu129` | `+cu129` | `uv pip install vllm --torch-backend=cu129` |
| 570.x | 12.8 | `torch+cu128` | `+cu129` (forward-compat) | `uv pip install vllm --torch-backend=cu128` |
| 555.x–565.x | 12.4–12.6 | `torch+cu124` | — | vLLM 0.20 not supported; downgrade to vLLM 0.19 |

> **V100 / SM 7.0 users:** PyTorch 2.11 cu128 wheels dropped Volta support.
> Stay on `torch+cu126` and build vLLM from source. This repo does not
> support V100 by default.

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

Expected output on 8× A100 with 570.x driver / CUDA 12.8:

```
PyTorch: 2.11.0+cu128
CUDA available: True
GPU count: 8
vLLM: 0.20.2            # wheel on disk is vllm-0.20.2+cu129-...whl (expected)
Transformers: 5.8.1
```

On a 580.x driver / CUDA 13.0 host the torch line will instead read
`PyTorch: 2.11.0+cu130` and the vLLM wheel will have no `+cuXXX` suffix.
