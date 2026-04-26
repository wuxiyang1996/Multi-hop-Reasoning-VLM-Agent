# `vlm_wrapper.visual_reasoning_wrapper`

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
from vlm_wrapper.visual_reasoning_wrapper.tools_visual import build_visual_registry
from vlm_wrapper.visual_reasoning_wrapper.tools_video_visual import (
    build_video_visual_registry,
)
from vlm_wrapper.visual_reasoning_wrapper.benchmarks.tir_bench import (
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

| Key | Modality | Loader | Source |
|-----|----------|--------|--------|
| `visual_toolbench` | image | `benchmarks/visual_toolbench.py` | HF `ScaleAI/VisualToolBench` |
| `tir_bench`        | image | `benchmarks/tir_bench.py`        | HF `Agents-X/TIR-Bench` (test split) |
| `video_holmes`     | video | `benchmarks/video_holmes.py`     | `data/Video-Holmes/Benchmark/` |
| `siv_bench`        | video | `benchmarks/siv_bench.py`        | `data/SIV-Bench/` |

Download instructions and HF cache tips: [`benchmarks/README.md`](benchmarks/README.md).

## Why one folder

XSkill (arXiv:2603.12056) frames a multimodal agent as **observation → reasoning
+ tool calls → structured state**.  The tool registries and the benchmarks that
exercise them are two halves of the same loop, so they live together; pulling
either side into the agent is one import (`vlm_wrapper.visual_reasoning_wrapper`).

The rest of `vlm_wrapper/` (`tool_loop.py`, `ground.py`, `schema.py`, the env
adapters and the heuristic / OmniParser heads) plugs into these registries —
e.g. `tool_loop.visual_generate_label_with_tools` calls
`visual_reasoning_wrapper.tools_visual.build_visual_registry` and
`tool_loop.video_visual_generate_label_with_tools` calls
`visual_reasoning_wrapper.tools_video_visual.build_video_visual_registry`.

## Evaluation

Wired up in [`../eval/run_eval.py`](../eval/run_eval.py) — pass
`--benchmark {visual_toolbench,tir_bench,video_holmes,siv_bench}`. See
[`../eval/README.md`](../eval/README.md) for the harness API and metrics.
