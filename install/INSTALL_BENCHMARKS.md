# Visual-Grounding Benchmarks — Installation Guide

This doc covers the **five task domains** that the `vlm_wrapper` grounding pipeline targets
(see [`plans/PLAN-VISUAL-GROUNDING.md`](../plans/PLAN-VISUAL-GROUNDING.md) §3 and
[`plans/PLAN-VISUAL-GROUNDING-MILESTONES.md`](../plans/PLAN-VISUAL-GROUNDING-MILESTONES.md) §3).

For training / inference of the actor + skill stack, see the original
[`install/README.md`](README.md). This file only covers what's needed to **run grounding
and generate labels** on each domain.

---

## TL;DR — current status on this machine

| Domain     | Conda env       | Status                                                                            | Benchmark      | Data status |
|------------|-----------------|-----------------------------------------------------------------------------------|----------------|-------------|
| `gymv`     | `gaming_eval`   | Ready — `GamingAgent` editable at `D:\ICML2026\GamingAgent`, `gymnasium`+`gym`    | 2048 / Sokoban / Minesweeper (in-process) | bundled |
| `browser`  | `browsergym`    | Ready — `browsergym-{core,miniwob,webarena,visualwebarena,workarena}` + Playwright/Chromium | MiniWoB++ / WebArena / VisualWebArena | tasks bundled, sites need self-host |
| `desktop`  | `osworld`       | Ready — `desktop_env==1.0.2`                                                      | OSWorld        | Docker images need to be pulled separately |
| `image_qa` | *missing*       | **TODO** — create env (`bench_vqa` recommended below)                             | **CLEVR**      | not downloaded |
| `video_qa` | `videochat-r1`  | Empty shell — needs `torch`, `transformers`, `decord`, etc. installed             | **Video-Holmes** | not downloaded |

The grounding pipeline itself (Heads 1/2 + cascaded escalation + re-observation +
tool loop) lives in `vlm_wrapper/` and only requires **one** of `openai`,
`anthropic`, or `google-genai` plus `Pillow` — every other dependency is opt-in
per domain. Head 3 (OmniParser-v2) needs additional model weights (see §6).

---

## 0. Prerequisites

```powershell
# Anaconda 3 with conda on PATH
conda --version

# Workspace root
cd D:\multi_hop_visual_reasoning
```

For HuggingFace-gated datasets / models (Video-Holmes, OmniParser-v2 weights):

```powershell
# Get a token at https://huggingface.co/settings/tokens (read scope is enough)
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# also export under the legacy var that some loaders still read
$env:HUGGING_FACE_HUB_TOKEN = $env:HF_TOKEN
```

---

## 1. `gymv` — already installed in `gaming_eval`

`vlm_wrapper.tools_gymv` and `vlm_wrapper.gymv_heuristic` rely on `gymnasium` plus
the in-tree GamingAgent wrappers. Both are present:

```powershell
conda run -n gaming_eval python -c "import gymnasium, gym, gamingagent; from importlib.metadata import version; print('gymnasium', version('gymnasium')); print('gym', version('gym')); print('GamingAgent OK')"
```

If `GamingAgent` import fails, reinstall it editable:

```powershell
conda run -n gaming_eval pip install -e D:\ICML2026\GamingAgent
```

No external data download is needed — the puzzle environments (2048, Sokoban,
Minesweeper) are generated in-process.

---

## 2. `browser` — BrowserGym (just repaired)

We reinstalled the full stack from PyPI on 2026-04-18 because the previous editable
installs were pointing at a deleted source tree. The env is now self-contained.

### Verify

```powershell
conda run -n browsergym python -c "import browsergym, browsergym.core, browsergym.miniwob, browsergym.webarena, browsergym.visualwebarena, browsergym.workarena; print('imports OK'); from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(headless=True); print('chromium OK'); b.close(); p.stop()"
```

Expected output:

```
imports OK
chromium OK
```

### Re-run the install (if the env is ever lost)

```powershell
conda create -n browsergym python=3.11 -y
conda run -n browsergym pip install browsergym browsergym-core browsergym-miniwob browsergym-webarena browsergym-visualwebarena browsergym-workarena
conda run -n browsergym python -m playwright install chromium
```

### Task-suite data

| Suite              | What's bundled in the pip package | What you must host yourself                   |
|--------------------|-----------------------------------|-----------------------------------------------|
| MiniWoB++          | All HTML pages                    | Static-file server (`python -m http.server` works) |
| WebArena           | Task definitions only             | Shopping/Reddit/GitLab Docker images          |
| VisualWebArena     | Task definitions only             | Same Docker images as WebArena + classifieds  |

Self-hosting WebArena sites is **not** required for visual grounding label
collection — you can drive any web page through a Playwright session and call
`browser_obs_to_schema(obs, ...)`. Only end-to-end task evaluation needs the
hosted sites.

---

## 3. `desktop` — OSWorld

`desktop_env` is already installed:

```powershell
conda run -n osworld python -c "import desktop_env; from importlib.metadata import version; print('desktop_env', version('desktop_env'))"
# desktop_env 1.0.2
```

To actually run desktop tasks you need the Docker image:

```powershell
docker pull happysixd/osworld-docker
```

(Skip if you only need the schema converter `grounding_osworld_obs_to_schema`,
which works against any captured screenshot dict.)

---

## 4. `image_qa` — CLEVR (recommended)

We picked **CLEVR** over GQA because:

* deterministic ground-truth scene graphs ⇒ ideal for the Head 1 heuristic +
  Head 2 vision agreement check used in Phase 0 labeling;
* synthetic images make object-level evaluation robust;
* 18 GB total, single archive, no scene-graph reconstruction needed.

### 4.1 Create a fresh env

There is no `image_qa` conda env yet. The recommended setup is one shared
`bench_vqa` env that also covers Video-Holmes (§5).

```powershell
conda create -n bench_vqa python=3.11 -y
conda run -n bench_vqa pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
conda run -n bench_vqa pip install transformers accelerate safetensors pillow numpy
conda run -n bench_vqa pip install openai anthropic google-genai          # at least one needed
conda run -n bench_vqa pip install huggingface_hub datasets requests tqdm
# video extras (used for §5; harmless to install here too)
conda run -n bench_vqa pip install decord av opencv-python
```

If you only want CPU inference replace the torch line with:

```powershell
conda run -n bench_vqa pip install torch torchvision
```

### 4.2 Download the dataset

```powershell
$dst = "D:\multi_hop_visual_reasoning\data\CLEVR"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$zip = Join-Path $dst "CLEVR_v1.0.zip"
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip" -OutFile $zip
Expand-Archive -Path $zip -DestinationPath $dst
Remove-Item $zip
```

After extraction the layout is:

```
data/CLEVR/CLEVR_v1.0/
├── images/{train,val,test}/CLEVR_*_NNNNNN.png
├── questions/CLEVR_{train,val,test}_questions.json
└── scenes/CLEVR_{train,val}_scenes.json   # ground-truth scene graphs
```

**Disk:** ~18 GB. **Download time:** 20–60 min depending on bandwidth.

### 4.3 Smoke-test against the grounding pipeline

```powershell
conda run -n bench_vqa python -c @"
from PIL import Image
from vlm_wrapper import GroundingRequest, cascaded_ground
img = Image.open(r'D:\multi_hop_visual_reasoning\data\CLEVR\CLEVR_v1.0\images\val\CLEVR_val_000000.png')
req = GroundingRequest(domain='image_qa', image=img, goal='Count the number of red cubes.')
res = cascaded_ground(req)
print('head:', res.head_used, 'valid:', res.validation.valid)
print(res.schema)
"@
```

(Optional alternative — **GQA**: ~75 GB total, real images, scene-graph plus
question files. Use this if you specifically need natural-image evaluation;
download from <https://cs.stanford.edu/people/dorarad/gqa/download.html>.)

---

## 5. `video_qa` — Video-Holmes

Video-Holmes is the chosen video benchmark:

* multi-hop reasoning questions (matches the project's `multi_hop_visual_reasoning` framing);
* 1 836 questions over 270 videos — small enough to host locally;
* well-typed schema (`question`, `options`, `answer`, `video_id`, `start_time`, `end_time`).

### 5.1 Use the `bench_vqa` env from §4

If you skipped §4, run only these (no torch needed for raw download):

```powershell
conda create -n bench_vqa python=3.11 -y
conda run -n bench_vqa pip install huggingface_hub decord av opencv-python pillow
```

### 5.2 Clone the repo (questions, eval scripts, prompts)

```powershell
$root = "D:\multi_hop_visual_reasoning\data\Video-Holmes"
git clone https://github.com/TencentARC/Video-Holmes.git $root
```

### 5.3 Download the dataset from HuggingFace

The repo ships its own downloader that uses `huggingface_hub.snapshot_download`
under the hood. Just run it from inside the cloned tree:

```powershell
$env:HF_TOKEN = "<your token>"
Set-Location "D:\multi_hop_visual_reasoning\data\Video-Holmes"
python download.py --hf_token $env:HF_TOKEN
```

This downloads 8 files (~4 GB) into `Benchmark/`:

| File | Purpose |
|---|---|
| `videos.zip` (4.0 GB) | 503 cropped clips, named by YouTube id |
| `annotations.zip` (0.3 MB) | per-clip eval annotations |
| `annotation_training.zip` (0.3 MB) | per-clip training annotations |
| `train_Video-Holmes.json` (1.2 MB) | training questions (~5 K items) |
| `test_Video-Holmes.json` (1.4 MB) | 1 837 multi-hop test questions |
| `config.json`, `README.md`, `.gitattributes` | metadata |

### 5.4 Extract the archives

```powershell
$bench = "D:\multi_hop_visual_reasoning\data\Video-Holmes\Benchmark"
foreach ($z in @('videos.zip','annotations.zip','annotation_training.zip')) {
    $name = $z -replace '\.zip$',''
    Expand-Archive -Path (Join-Path $bench $z) -DestinationPath (Join-Path $bench $name) -Force
}
```

After extraction, the relevant tree is:

```
data/Video-Holmes/Benchmark/
├── videos/videos_cropped/<youtube_id>.mp4    # 503 mp4s, ~4 GB
├── annotations/                               # 270 per-clip eval annotations
├── annotation_training/                       # 246 per-clip training annotations
├── train_Video-Holmes.json                    # training Qs
├── test_Video-Holmes.json                     # 1 837 eval Qs (keys: video ID, Question, Options, Answer, Explanation)
└── config.json
```

**Disk:** ~8 GB after extraction. **Download time:** 5–15 min.

### 5.5 Smoke-test

```powershell
conda run -n bench_vqa python -c @"
import decord, json, os
bench = r'D:\multi_hop_visual_reasoning\data\Video-Holmes\Benchmark'
with open(os.path.join(bench, 'test_Video-Holmes.json'), encoding='utf-8') as f:
    qs = json.load(f)
print('test questions:', len(qs))
v = decord.VideoReader(os.path.join(bench, 'videos', 'videos_cropped', qs[0]['video ID'] + '.mp4'))
print('first video frames:', len(v), 'shape:', v[0].shape)
"@
```

---

## 6. Optional — OmniParser-v2 (Head 3, used for `desktop` and as `browser` fallback)

`vlm_wrapper.grounding` lazy-loads OmniParser-v2 for Head 3 grounding. It is
**only** required if you want Head 3 in the cascade for browser/desktop schemas
(the validator + cascaded_ground will simply skip it if unavailable).

```powershell
# Pick the env that should host Head 3 (browsergym is a natural fit)
conda run -n browsergym pip install ultralytics easyocr supervision
# Florence-2 weights ship via transformers; weights auto-download on first call
```

Cache location: `%USERPROFILE%\.cache\omniparser-v2\` (override with
`OMNIPARSER_CACHE_DIR`). First call downloads ~1.5 GB.

---

## 7. End-to-end verification

After §1–§5, this single block should print one line per domain:

```powershell
conda run -n gaming_eval  python -c "import gymnasium, gamingagent; print('gymv: OK')"
conda run -n browsergym   python -c "import browsergym.core; from playwright.sync_api import sync_playwright; print('browser: OK')"
conda run -n osworld      python -c "import desktop_env; print('desktop: OK')"
conda run -n bench_vqa    python -c "import os; assert os.path.isdir(r'D:\multi_hop_visual_reasoning\data\CLEVR\CLEVR_v1.0\images\val'); print('image_qa: OK')"
conda run -n bench_vqa    python -c "import os, glob; assert glob.glob(r'D:\multi_hop_visual_reasoning\data\Video-Holmes\dataset\*.mp4'); print('video_qa: OK')"
```

---

## 8. Mapping the install to the plan

| Plan item                                                                       | Status after this doc |
|---------------------------------------------------------------------------------|-----------------------|
| §3 escalation chain — `gymv`                                                    | Runtime ready         |
| §3 escalation chain — `browser`                                                 | Runtime ready         |
| §3 escalation chain — `desktop`                                                 | Runtime ready (Docker pull pending if running real OSWorld tasks) |
| §3 escalation chain — `image_qa`                                                | Runtime ready once §4 download finishes |
| §3 escalation chain — `video_qa`                                                | Runtime ready once §5 download finishes |
| Phase 0 — Gym-V data collection                                                 | Unblocked             |
| Phase 0 — BrowserGym data collection                                            | Unblocked             |
| Phase 1 — image-QA SFT                                                          | Unblocked after §4    |
| Phase 1 — video-QA SFT                                                          | Unblocked after §5    |

The remaining open items in the plan
([`PLAN-VISUAL-GROUNDING-MILESTONES.md`](../plans/PLAN-VISUAL-GROUNDING-MILESTONES.md)
§6–§8) — benchmark loaders, evaluation harness, Qwen3-VL-8B training pipeline,
data-collection scripts, actor schema integration — are software TODOs, not
install TODOs.

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'browsergym.core'` after pip says it's installed

The `browsergym-*` packages were installed `-e` from a deleted source tree. Fix:

```powershell
conda run -n browsergym pip install --force-reinstall --no-deps `
    browsergym browsergym-core browsergym-miniwob browsergym-webarena `
    browsergym-visualwebarena browsergym-experiments browsergym-assistantbench `
    browsergym-webarena-verified browsergym-webarenalite
```

### Playwright launches but pages render blank

Re-run `python -m playwright install chromium` inside the env — only the
*chromium binary* is needed; the python package is already in the
`browsergym-core` deps tree.

### `huggingface_hub` `401 Unauthorized` on Video-Holmes

The repo is gated. Make sure `HF_TOKEN` is set in the **same shell** that runs
`snapshot_download`, and that you've clicked "Agree and access repository" on
<https://huggingface.co/datasets/TencentARC/Video-Holmes>.

### `decord` import fails on Windows

Use the prebuilt CPU wheel: `pip install decord==0.6.0`. If that still fails,
substitute `pip install av` and switch readers to `pyav`.

### `nvidia-smi` shows GPU but `torch.cuda.is_available()` is False in `bench_vqa`

You installed the CPU wheel. Reinstall with the CUDA index URL from §4.1.
