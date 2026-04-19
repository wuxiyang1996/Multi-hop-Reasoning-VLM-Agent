"""Benchmark loaders + GPT-4o parsers for the image/video QA domains.

Each module exposes two things:

  1. A ``load_*`` / ``iter_samples`` helper that streams questions (and
     their images / videos) off disk without consuming all of RAM.
  2. A ``parse_sample`` entry point that runs the GPT-4o vision parser
     through ``vlm_wrapper.ground.ground`` and returns a uniform result
     dict with the schema, answer, tool trace, and metadata.

Currently supported:

* ``clevr``       — CLEVR v1.0 (image-QA). 1M questions over 100k images.
* ``video_holmes`` — Video-Holmes (video-QA). 1 837 multi-hop MCQs over
  503 cropped clips.

The loaders deliberately have zero hard dependencies on the rest of the
project: they only need ``Pillow`` for CLEVR and (optionally) one of
``decord`` / ``opencv-python`` for Video-Holmes frame sampling.
"""

from .clevr import (
    CLEVRSample,
    default_clevr_root,
    iter_clevr_samples,
    load_clevr_image,
    load_clevr_questions,
    parse_clevr_sample,
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
    # Video-Holmes
    "VideoHolmesSample",
    "default_video_holmes_root",
    "iter_video_holmes_samples",
    "load_video_holmes_questions",
    "parse_video_holmes_sample",
    "sample_video_frames",
]
