# Visual-Grounding Benchmarks — Installation Guide

This document covers the **five task domains** targeted by the
`vlm_wrapper` grounding pipeline (see
[`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md`](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING.md) §2 and
[`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md`](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) §3).

For training / inference of the actor + skill stack, see the original
[`install/README.md`](README.md). This file is scoped to what's needed
to **drive the benchmark runtimes** and **run the grounding pipeline**
on each domain.

---

## TL;DR — four conda envs, one per runtime family

The three interactive runtimes have hard-pinned dependency sets that
cannot co-resolve (gymnasium 0.28 vs 1.2+, transformers 4.35 vs 4.51+,
playwright pinned to 1.44 for BrowserGym-core). They each get their own
env. Offline **video** benchmarks (Video-Holmes / SIV-Bench on disk) plus
**image** benchmarks pulled from HuggingFace (VisualToolBench, TIR-Bench)
share the same `vlm_benchmarks` env as the `vlm_wrapper` grounding pipeline.

| Domain     | Conda env        | Upstream source                                                     | Purpose                                                             |
|------------|------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|
| `gymv`     | `gymv`           | [ModalMinds/gym-v](https://github.com/ModalMinds/gym-v)             | 179 procedurally-generated visual envs                              |
| `browser`  | `browsergym`     | [ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym)   | MiniWoB++ / WebArena / VisualWebArena / AssistantBench              |
| `desktop`  | `osworld`        | [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)             | Desktop tasks + `desktop-env` runtime                               |
| `image_qa` | `vlm_benchmarks` | HF `datasets` + cache                                                 | VisualToolBench + TIR-Bench loaders + grounding pipeline              |
| `video_qa` | `vlm_benchmarks` | —                                                                   | Video-Holmes loader + grounding pipeline                            |

### Incompatibility matrix (why four envs, not one)

| Package        | gym-v        | BrowserGym    | OSWorld         | vlm_benchmarks |
|----------------|--------------|---------------|-----------------|----------------|
| `gymnasium`    | `>=1.2.2`    | `>=0.27`      | `~=0.28.1`      | any            |
| `playwright`   | —            | `==1.44`      | unpinned        | latest         |
| `transformers` | —            | —             | `~=4.35.2`      | `>=4.51,<4.56` |
| `torch`        | —            | (VWA scorer)  | `~=2.5.0`       | `2.4.1+cu121`  |
| `tqdm`         | —            | `>=4.66.2` *  | `~=4.65.0`      | latest         |

\* `browsergym-workarena` only — install it on top of the `browsergym`
env with `pip install --no-deps browsergym-workarena`.

---

## 0. Prerequisites

```bash
# Miniconda3 / Anaconda with conda on PATH
conda --version

# Workspace root
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent
```

For HuggingFace-gated datasets/models (Video-Holmes, OmniParser-v2):

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

---

## 1. `gymv` — ModalMinds/gym-v

### Install

```bash
bash install/install_gymv.sh            # creates env, clones gym-v, pip install -e .[games,spatial], smoke-tests
```

Or manually:

```bash
git clone https://github.com/ModalMinds/gym-v.git /fs/gamma-projects/vlm-robot/gym-v
conda env create -f install/gymv.environment.yml
conda activate gymv
pip install -e "/fs/gamma-projects/vlm-robot/gym-v[games,spatial]"
# optional: .[temporal] (stable-retro), .[vlmeval], .[reasoning-gym]
python install/gymv_smoke.py
```

### What's bundled

All 179 envs ship with the source install. Default enabled groups:

| Group      | Packages                              | Envs                                  |
|------------|---------------------------------------|---------------------------------------|
| `games`    | textarena, pettingzoo[classic]        | Chess, TicTacToe, TextArena games     |
| `spatial`  | minigrid, miniworld                   | 2D + 3D navigation                    |
| `temporal` | stable-retro                          | Retro arcade (not installed by default — needs ROMs) |
| `vlmeval`  | VLMEvalKit (git)                      | VLM evaluation benchmarks             |

### Quick-start

```python
import gym_v
env = gym_v.make("Games/TicTacToe-v0")
obs, info = env.reset(seed=0)
# obs = {"agent_0": Observation(image=PIL.Image, text="...", metadata={})}
env.close()
```

---

## 2. `browser` — ServiceNow/BrowserGym

### Install

```bash
bash install/install_browsergym.sh      # creates env, clones repo, pip installs each sub-package editable, playwright install
```

Or manually:

```bash
git clone https://github.com/ServiceNow/BrowserGym.git /fs/gamma-projects/vlm-robot/BrowserGym
conda env create -f install/browsergym.environment.yml
conda activate browsergym
cd /fs/gamma-projects/vlm-robot/BrowserGym
pip install -e ./browsergym/core \
            -e ./browsergym/miniwob \
            -e ./browsergym/webarena \
            -e ./browsergym/visualwebarena \
            -e ./browsergym/assistantbench \
            -e ./browsergym/experiments
python -m playwright install chromium
```

### Optional WorkArena

```bash
conda activate browsergym
pip install --no-deps browsergym-workarena   # needs playwright==1.44.0 too; compatible
```

### Self-hosted sites

MiniWoB++ HTML pages ship with the pip package. WebArena/VisualWebArena
ship only task definitions — the actual Shopping / Reddit / GitLab /
Classifieds Docker images must be hosted separately
(see <https://github.com/web-arena-x/webarena#setup-instructions>).
Visual grounding label collection can run against any Playwright page
without the hosted sites.

---

## 3. `desktop` — xlang-ai/OSWorld

### Install

```bash
bash install/install_osworld.sh         # creates env, clones OSWorld, pip install -r requirements.txt, pip install -e .
```

Or manually:

```bash
git clone https://github.com/xlang-ai/OSWorld.git /fs/gamma-projects/vlm-robot/OSWorld
conda env create -f install/osworld.environment.yml
conda activate osworld
cd /fs/gamma-projects/vlm-robot/OSWorld
pip install -r requirements.txt
pip install -e .
python install/osworld_smoke.py
```

### VM backend

To actually run desktop tasks you need one of:

| Backend  | Setup                                                                           |
|----------|---------------------------------------------------------------------------------|
| Docker   | `docker pull happysixd/osworld-docker` (requires KVM on host for performance)   |
| VMware   | Install VMware Workstation Pro, configure `vmrun`; see OSWorld's `INSTALL_VMWARE.md` |
| AWS      | See OSWorld's `SETUP_GUIDELINE.md` — Host-Client architecture, parallel eval    |

The `grounding_osworld_obs_to_schema` converter in `vlm_wrapper` works
against any captured screenshot+a11y dict, so offline schema labeling
does not require a VM.

---

## 4. `image_qa` + `video_qa` + grounding pipeline — `vlm_benchmarks`

Single env that covers:

* VisualToolBench / TIR-Bench loaders (`vlm_wrapper.visual_reasoning_wrapper.benchmarks.{visual_toolbench,tir_bench}`)
* Video-Holmes / SIV-Bench loaders (`vlm_wrapper.visual_reasoning_wrapper.benchmarks.{video_holmes,siv_bench}`)
* The full `vlm_wrapper` grounding pipeline (GroundingDINO, OmniParser-v2, Florence-2, EasyOCR, YOLO, decord)
* OpenAI / Anthropic / Google-Genai API clients for Head 2 labeling

### Install

```bash
conda env create -f install/vlm_benchmarks.environment.yml
conda activate vlm_benchmarks
python -m playwright install chromium     # optional; used by demo/labeling scripts
pip install -e .                           # makes vlm_wrapper importable
python install/vlm_benchmarks_smoke.py
```

### Image QA — HuggingFace (VisualToolBench + TIR-Bench)

CLEVR / GQA are **not** used anymore; image QA is **VisualToolBench** + **TIR-Bench** only.

**A. Cache-only (smallest disk footprint)** — warms the HuggingFace datasets cache on first import:

```bash
conda activate vlm_benchmarks
python - <<'PY'
from datasets import load_dataset
load_dataset("Agents-X/TIR-Bench", split="test", trust_remote_code=True)
load_dataset("ScaleAI/VisualToolBench", split="test", streaming=True, trust_remote_code=True)
print("HF caches warmed — image benchmarks ready.")
PY
```

**B. Full copy under `data/datasets/`** (repo `data/` is gitignored — safe for large files):

```bash
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent
mkdir -p data/datasets
pip install -U "huggingface_hub[cli]"
hf download Agents-X/TIR-Bench       --repo-type dataset --local-dir data/datasets/TIR-Bench
hf download ScaleAI/VisualToolBench  --repo-type dataset --local-dir data/datasets/VisualToolBench
```

The `vlm_wrapper` loaders still resolve rows through `datasets.load_dataset(...)` (Hub + cache). The on-disk copy is for backups, air-gapped transfer, or tooling that expects a folder tree.

Set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` if either dataset is gated for your org.

If `hf download` fails with **permission denied** under a shared default cache, point Hugging Face at the repo-local cache (gitignored):

```bash
cd /path/to/Multi-hop-Reasoning-VLM-Agent
export HF_HOME="$(pwd)/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1
```

Then re-run the `hf download` commands above.

### Video-Holmes data (~8 GB)

```bash
dst=/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/data/Video-Holmes
git clone https://github.com/TencentARC/Video-Holmes.git "$dst"
cd "$dst"
python download.py --hf_token "$HF_TOKEN"
cd Benchmark
for z in videos.zip annotations.zip annotation_training.zip; do
    unzip -q "$z" -d "${z%.zip}"
done
```

---

## 5. OmniParser-v2 weights (optional, used by Head 3)

OmniParser-v2 is lazy-loaded by `vlm_wrapper.grounding`. First call
downloads ~1.5 GB of weights to `~/.cache/omniparser-v2/` (override with
`OMNIPARSER_CACHE_DIR`). Already covered by the `ultralytics`,
`easyocr`, and `supervision` pins in the `vlm_benchmarks` env.

---

## 6. End-to-end verification

```bash
conda run -n gymv            python install/gymv_smoke.py
conda run -n browsergym      python install/browsergym_smoke.py
conda run -n osworld         python install/osworld_smoke.py
conda run -n vlm_benchmarks  python install/vlm_benchmarks_smoke.py
```

Each script prints one `[OK] / [FAIL] / [WARN]` line per check.

---

## 7. Mapping the install to the plan

| Plan item                                                                       | Status after this doc       |
|---------------------------------------------------------------------------------|-----------------------------|
| §3 escalation chain — `gymv`                                                    | Runtime ready (`gymv` env)  |
| §3 escalation chain — `browser`                                                 | Runtime ready (`browsergym` env) |
| §3 escalation chain — `desktop`                                                 | Runtime ready (`osworld` env); VM backend pulled separately |
| §3 escalation chain — `image_qa`                                                | Runtime ready (`vlm_benchmarks` env); HF image benchmarks via `datasets` |
| §3 escalation chain — `video_qa`                                                | Runtime ready (`vlm_benchmarks` env); Video-Holmes data on disk |
| Phase 0 — Gym-V data collection                                                 | Unblocked                   |
| Phase 0 — BrowserGym data collection                                            | Unblocked                   |
| Phase 1 — image-QA SFT                                                          | Unblocked after HF cache / `datasets` smoke |
| Phase 1 — video-QA SFT                                                          | Unblocked                   |

---

## 8. Troubleshooting

### `gym-v` install fails with `gymnasium` not found

`gymnasium>=1.2.2` only exists on PyPI as of 2025-Q4. Upgrade pip
(`pip install -U pip`) and re-run inside the `gymv` env.

### BrowserGym pages render blank after install

Re-run `python -m playwright install chromium` inside the env — only
the Chromium binary is needed; the Python package is already in
`browsergym-core`'s deps.

### OSWorld's `pip install -r requirements.txt` drags in ~1.5 GB of wheels

That's expected — OSWorld bundles Office-document processors (pymupdf,
python-docx, python-pptx, openpyxl, formulas, odfpy) plus audio libs
(librosa, mutagen, pyacoustid) for the eval checkers. If you only need
the `DesktopEnv` runtime and don't plan to run the evaluator locally,
`pip install desktop-env` is enough (done automatically by
`install/osworld.environment.yml`).

### `huggingface_hub` 401 on Video-Holmes

The repo is gated. Set `HF_TOKEN` in the **same shell** that runs
`download.py`, and click "Agree and access repository" on
<https://huggingface.co/datasets/TencentARC/Video-Holmes>.

### `decord` import fails on Linux

Use the prebuilt wheel: `pip install decord==0.6.0`. If that still
fails, substitute with `pip install av` and switch readers to `pyav`
(the `vlm_benchmarks` env already installs both).
