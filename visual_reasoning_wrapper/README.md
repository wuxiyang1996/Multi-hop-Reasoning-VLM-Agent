# `visual_reasoning_wrapper`

Visual-reasoning slice of the wrapper: image / video tool registries **and** the
public benchmarks the agent is evaluated on. Every other adapter (Gym-V,
BrowserGym, OSWorld) reuses these tools — they are not benchmark-private.

## Layout

```
visual_reasoning_wrapper/
├── __init__.py          # XSkill-aligned design notes + PRIMARY_VISUAL_REASONING_BENCHMARKS
├── tools_visual.py      # Single-frame tools (detect_objects, describe_region, …)
├── tools_video.py       # Temporal navigation tools (get_frame, scene_changes, …)
├── tools_video_visual.py# Cross-frame tools (track_object, find_moment, summarize_clip)
└── benchmarks/          # Loaders + parse_*_sample for the standard 2×2 set
    ├── README.md
    ├── _hf_images.py
    ├── visual_toolbench.py   # HF ScaleAI/VisualToolBench
    ├── tir_bench.py          # HF Agents-X/TIR-Bench
    ├── video_holmes.py       # local data/Video-Holmes
    └── siv_bench.py          # local data/SIV-Bench
```

The package's `__init__.py` only carries documentation and the
`VisualReasoningBenchmark` dataclass / `PRIMARY_VISUAL_REASONING_BENCHMARKS`
tuple. Tool registries and benchmark loaders are imported directly:

```python
from visual_reasoning_wrapper.tools_visual import build_visual_registry
from visual_reasoning_wrapper.tools_video_visual import (
    build_video_visual_registry,
)
from visual_reasoning_wrapper.benchmarks.tir_bench import (
    iter_tir_bench_samples, parse_tir_bench_sample,
)
```

The convenience symbols (`build_visual_registry`, `build_video_registry`,
`build_video_visual_registry`, all `iter_*` / `parse_*` helpers) are also
re-exported at the top level via `vlm_wrapper/__init__.py`.

## Tools

* **`tools_visual.build_visual_registry(image, *, prefer_gdino=False)`** — single
  frame: `detect_objects`, `grounded_detect`, `describe_region`, `read_text`,
  `spatial_query`, `crop_and_zoom`, etc.
* **`tools_video.build_video_registry(frames=…, fps=…, current_index=…)`** —
  temporal navigation: `get_frame`, `compare_frames`, `detect_scene_changes`,
  `sample_frames`, `read_text_in_frame`, …
* **`tools_video_visual.build_video_visual_registry(frames=…, …)`** — superset
  that adds cross-frame tools (`track_object`, `find_moment`,
  `summarize_clip`, …) on top of the merged video + visual registries.

## Benchmarks

The standard coverage is a 2 image + 2 video set chosen to exercise the
agent's actual action space (`inspect`, `crop`, `zoom`, `compare`, `track`,
`OCR`, `verify`, `answer`) instead of only static visual-question answering.

| Key | Modality | Loader | Source links | Why it is covered |
|-----|----------|--------|--------------|-------------------|
| `visual_toolbench` | image | `benchmarks/visual_toolbench.py` | [HF dataset](https://huggingface.co/datasets/ScaleAI/VisualToolBench), [paper](https://arxiv.org/abs/2510.12712) | Tool-enabled image perception, transformation, and reasoning; closer to "think with images" than CLEVR/GQA-style QA. |
| `tir_bench` | image | `benchmarks/tir_bench.py` | [repo](https://github.com/agents-x-project/TIR-Bench), [HF dataset](https://huggingface.co/datasets/Agents-X/TIR-Bench), [paper](https://arxiv.org/abs/2511.01833) | Agentic thinking-with-images across 13 task families; stresses image operations inside a reasoning chain. |
| `video_holmes` | video | `benchmarks/video_holmes.py` | [repo](https://github.com/TencentARC/Video-Holmes), [project page](https://video-holmes.github.io/Page.github.io/), [HF dataset](https://huggingface.co/datasets/TencentARC/Video-Holmes), [paper](https://arxiv.org/abs/2505.21374) | Short-video multi-hop evidence-chain reasoning over clues scattered across clips. |
| `siv_bench` | video | `benchmarks/siv_bench.py` | [project page](https://kfq20.github.io/sivbench/), [HF dataset](https://huggingface.co/datasets/Fancylalala/SIV-Bench), [paper](https://arxiv.org/abs/2506.05425) | Social scene, state, intent, and interaction reasoning; complements Video-Holmes with social/latent-state tasks. |

The image pair intentionally replaces CLEVR + GQA for this wrapper: CLEVR/GQA are
useful static VQA checks, but they do not directly evaluate whether the agent
selects and uses visual tools during reasoning. For video, SIV-Bench is preferred
over a generic Video-MME short subset because it tests interaction, intent, and
social-state inference rather than broad video QA alone.

Download instructions and HF cache tips: [`benchmarks/README.md`](benchmarks/README.md).

## Why one folder

XSkill (arXiv:2603.12056) frames a multimodal agent as **observation → reasoning
+ tool calls → structured state**.  The tool registries and the benchmarks that
exercise them are two halves of the same loop, so they live together; pulling
either side into the agent is one import (`visual_reasoning_wrapper`).

The rest of `vlm_wrapper/` (`tool_loop.py`, `ground.py`, `schema.py`, the env
adapters and the heuristic / OmniParser heads) plugs into these registries —
e.g. `tool_loop.visual_generate_label_with_tools` calls
`visual_reasoning_wrapper.tools_visual.build_visual_registry` and
`tool_loop.video_visual_generate_label_with_tools` calls
`visual_reasoning_wrapper.tools_video_visual.build_video_visual_registry`.

## Evaluation

Wired up in [`../vlm_wrapper/eval/run_eval.py`](../vlm_wrapper/eval/run_eval.py) — pass
`--benchmark {visual_toolbench,tir_bench,video_holmes,siv_bench}`. See
[`../vlm_wrapper/eval/README.md`](../vlm_wrapper/eval/README.md) for the harness API and metrics.
