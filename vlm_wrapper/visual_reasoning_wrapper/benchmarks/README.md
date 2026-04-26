# `vlm_wrapper.visual_reasoning_wrapper.benchmarks` — image & video QA loaders

Streaming loaders and `parse_*_sample` helpers for four public benchmarks used with the `image_qa` / `video_qa` tool loops (`vlm_wrapper.tool_loop`). Each module returns a uniform result dict (schema, answer, tool trace, metadata) after running the configured vision model through `cascaded_ground` / parsers.

**Legacy note:** CLEVR and GQA loaders were removed in favour of **VisualToolBench** and **TIR-Bench** (tool-centric and thinking-with-images image benchmarks on HuggingFace).

**Design matrix** — see [`../__init__.py`](../__init__.py) (`PRIMARY_VISUAL_REASONING_BENCHMARKS`).

## Benchmarks

| Key | Modality | Module | Data source |
|-----|----------|--------|-------------|
| `visual_toolbench` | Image | `visual_toolbench.py` | HuggingFace `ScaleAI/VisualToolBench` |
| `tir_bench` | Image | `tir_bench.py` | HuggingFace `Agents-X/TIR-Bench` (`test` split) |
| `video_holmes` | Video | `video_holmes.py` | `data/Video-Holmes/Benchmark/` |
| `siv_bench` | Video | `siv_bench.py` | `data/SIV-Bench/` (see `default_siv_bench_root()` in code) |

## Download image benchmarks (HF)

Loaders call `datasets.load_dataset(...)` and use the HuggingFace **cache** by default (`~/.cache/huggingface/`). To also keep a **full on-disk copy** under the repo (ignored by git — see root `.gitignore` `data/`):

```bash
cd /fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent
mkdir -p data/datasets

# Option A — huggingface-cli (recommended; resumable)
pip install -U "huggingface_hub[cli]"
hf download Agents-X/TIR-Bench         --repo-type dataset --local-dir data/datasets/TIR-Bench
hf download ScaleAI/VisualToolBench    --repo-type dataset --local-dir data/datasets/VisualToolBench

# Option B — Python (same result)
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
root = Path("data/datasets")
root.mkdir(parents=True, exist_ok=True)
snapshot_download("Agents-X/TIR-Bench", repo_type="dataset", local_dir=str(root / "TIR-Bench"))
snapshot_download("ScaleAI/VisualToolBench", repo_type="dataset", local_dir=str(root / "VisualToolBench"))
print("Saved to", root.resolve())
PY
```

Set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` if your account must authenticate for either dataset.

**Cluster tip:** if downloads fail with permission errors under a global HF path, force a writable cache in the repo (also gitignored):

```bash
cd /path/to/Multi-hop-Reasoning-VLM-Agent
export HF_HOME="$(pwd)/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1   # optional: avoids xet log/tmp issues on some NFS mounts
```

Rough sizes: TIR-Bench ~1.2 GB; VisualToolBench ~7.5 GB — ensure disk space before downloading.

## Python API

```python
from vlm_wrapper.visual_reasoning_wrapper.benchmarks import (
    iter_visual_toolbench_samples,
    parse_visual_toolbench_sample,
    iter_tir_bench_samples,
    parse_tir_bench_sample,
    iter_video_holmes_samples,
    parse_video_holmes_sample,
    iter_siv_bench_samples,
    parse_siv_bench_sample,
)
```

Docstrings in `__init__.py` list exports and dependencies (`Pillow`, `datasets` for image HF rows; `decord` or `opencv-python` for video).

## CLI evaluation

Batch scoring and JSONL logging: [`../../eval/run_eval.py`](../../eval/run_eval.py) (see [`../../eval/README.md`](../../eval/README.md)).

## Environment

Install and paths: parent [`../README.md`](../README.md) and [`../../../install/INSTALL_BENCHMARKS.md`](../../../install/INSTALL_BENCHMARKS.md).
