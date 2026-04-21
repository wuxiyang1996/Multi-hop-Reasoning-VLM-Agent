"""Benchmark loaders + GPT-4o parsers for the image/video QA domains.

Each module exposes two things:

  1. A ``load_*`` / ``iter_samples`` helper that streams questions (and
     their images / videos) off disk without consuming all of RAM.
  2. A ``parse_sample`` entry point that runs the GPT-4o vision parser
     through ``vlm_wrapper.ground.cascaded_ground`` and returns a
     uniform result dict with the schema, answer, tool trace, and
     metadata.

Currently supported:

* ``clevr``       — CLEVR v1.0 (image-QA). 1 M questions / 100 k images.
* ``gqa``         — GQA balanced (image-QA). 1.7 M questions / 113 k
                    real-world images, with scene-graph ground truth.
* ``video_holmes`` — Video-Holmes (video-QA). 1 837 multi-hop MCQs over
                     503 cropped clips.
* ``siv_bench``   — SIV-Bench (video-QA, social interactions).  8 728
                    MCQs over 2 792 clips.

The loaders deliberately have zero hard dependencies on the rest of the
project: they only need ``Pillow`` for the image benchmarks and
(optionally) one of ``decord`` / ``opencv-python`` for the video
benchmarks (SIV-Bench reuses Video-Holmes' frame-sampler).
"""

from .clevr import (
    CLEVRSample,
    default_clevr_root,
    iter_clevr_samples,
    load_clevr_image,
    load_clevr_questions,
    parse_clevr_sample,
)
from .gqa import (
    GQASample,
    default_gqa_root,
    iter_gqa_samples,
    load_gqa_image,
    load_gqa_questions,
    parse_gqa_sample,
)
from .siv_bench import (
    SIVBenchSample,
    default_siv_bench_root,
    iter_siv_bench_samples,
    load_siv_bench_questions,
    parse_siv_bench_sample,
)
from .video_holmes import (
    VideoHolmesSample,
    default_video_holmes_root,
    iter_video_holmes_samples,
    load_video_holmes_questions,
    parse_video_holmes_sample,
    sample_video_frames,
)

__all__ = [
    # CLEVR
    "CLEVRSample",
    "default_clevr_root",
    "iter_clevr_samples",
    "load_clevr_image",
    "load_clevr_questions",
    "parse_clevr_sample",
    # GQA
    "GQASample",
    "default_gqa_root",
    "iter_gqa_samples",
    "load_gqa_image",
    "load_gqa_questions",
    "parse_gqa_sample",
    # Video-Holmes
    "VideoHolmesSample",
    "default_video_holmes_root",
    "iter_video_holmes_samples",
    "load_video_holmes_questions",
    "parse_video_holmes_sample",
    "sample_video_frames",
    # SIV-Bench
    "SIVBenchSample",
    "default_siv_bench_root",
    "iter_siv_bench_samples",
    "load_siv_bench_questions",
    "parse_siv_bench_sample",
]
