"""Test cases for the five GPT-4o VLM parsers shipped in ``vlm_wrapper``.

Two tiers of tests:

* **offline** — always run; import every adapter / loader, instantiate
  a synthetic sample, and verify the public contracts (dataclass
  fields, iterator pagination, required kwargs).  These protect
  against dependency or refactor regressions without burning API
  credits.

* **live**    — marked ``@pytest.mark.live``; skipped unless
  ``OPENAI_API_KEY`` (or ``VLM_TEST_API_KEY``) is set.  Each live test
  actually calls GPT-4o once and asserts that a parseable
  ``<state>…</state>`` schema comes back.  Schemas are saved under
  ``out/schemas/<case>.schema.txt`` so you can eyeball the output.

Run offline only (default)::

    pytest vlm_wrapper/tests/test_gpt4o_parsers.py

Run everything including live GPT-4o calls::

    pytest -m "live or not live" vlm_wrapper/tests/test_gpt4o_parsers.py

Run only live tests::

    pytest -m live vlm_wrapper/tests/test_gpt4o_parsers.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# We reuse the runner's synthesizers so the live cases exercise the
# *same* inputs a user would get from ``scripts/test_vlm_parsers.py``.
from scripts.test_vlm_parsers import (  # noqa: E402
    _synthesize_2048_board,
    _synthesize_desktop,
    _synthesize_wiki_page,
)

SCHEMAS_DIR = REPO_ROOT / "out" / "schemas"
API_KEY = os.environ.get("VLM_TEST_API_KEY") or os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("VLM_LABEL_MODEL", "gpt-4o")

live = pytest.mark.live
needs_api = pytest.mark.skipif(
    not API_KEY, reason="OPENAI_API_KEY / VLM_TEST_API_KEY not set"
)


def _save_schema(case: str, schema: str) -> None:
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    (SCHEMAS_DIR / f"{case}.schema.txt").write_text(schema, encoding="utf-8")


def _assert_valid_schema(schema: str | None, case: str) -> None:
    """Structural sanity: presence of ``<state>`` block and either
    ``<entities>`` (interactive/image-QA domains) or
    ``<events>``/``<answer>`` (video domain)."""
    assert schema, f"{case}: parser returned no schema"
    assert "<state>" in schema and "</state>" in schema, (
        f"{case}: missing <state>…</state>:\n{schema[:400]}"
    )
    has_entities = "<entities>" in schema
    has_events = "<events>" in schema
    has_answer = "<answer>" in schema
    assert has_entities or has_events or has_answer, (
        f"{case}: schema lacks any of <entities>/<events>/<answer>:\n"
        f"{schema[:400]}"
    )


# ======================================================================
# Offline contract tests
# ======================================================================

def test_offline_gymv_adapter_import() -> None:
    from vlm_wrapper.gymv_adapter import generate_label
    assert callable(generate_label)


def test_offline_browser_adapter_import() -> None:
    from vlm_wrapper.browser_adapter import generate_label
    assert callable(generate_label)


def test_offline_osworld_adapter_import() -> None:
    from vlm_wrapper.osworld_adapter import generate_label
    assert callable(generate_label)


def test_offline_clevr_loader_yields_samples() -> None:
    from vlm_wrapper.benchmarks.clevr import iter_clevr_samples, CLEVRSample

    samples = list(iter_clevr_samples(split="val", limit=2))
    assert len(samples) == 2
    for s in samples:
        assert isinstance(s, CLEVRSample)
        assert s.question and s.image_filename
        assert s.image_path and s.image_path.exists(), (
            f"CLEVR image missing on disk: {s.image_path}")


def test_offline_video_holmes_loader_yields_samples() -> None:
    from vlm_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples, VideoHolmesSample,
    )

    samples = list(iter_video_holmes_samples(split="test", limit=3))
    assert samples, "Video-Holmes loader returned no samples"
    for s in samples:
        assert isinstance(s, VideoHolmesSample)
        assert s.question and s.question_id
        assert s.options, "each Video-Holmes question should carry options"


def test_offline_synthesizers_produce_rgb_images() -> None:
    for fn in (_synthesize_2048_board, _synthesize_wiki_page,
               _synthesize_desktop):
        img = fn()
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert min(img.size) > 64


# ======================================================================
# Live GPT-4o schema tests (one per domain)
# ======================================================================

@live
@needs_api
def test_live_gymv_schema() -> None:
    from vlm_wrapper.gymv_adapter import generate_label

    image = _synthesize_2048_board()
    result = generate_label(
        image,
        goal="Reach 2048",
        task_id="Game2048-v0",
        step=0,
        game_rules="2048 — slide tiles [Up]/[Down]/[Left]/[Right]; equal tiles merge.",
        obs_text="| 2 | 4 | 0 | 0 |\n| 0 | 0 | 0 | 0 |\n"
                 "| 0 | 0 | 0 | 0 |\n| 0 | 0 | 0 | 0 |",
        max_entities=12,
        model=MODEL,
        api_key=API_KEY,
    )
    schema = result.get("schema")
    _assert_valid_schema(schema, "gymv")
    assert "<actions>" in schema, (
        "gymv schema should include an <actions> block for move proposals")
    _save_schema("gymv", schema)


@live
@needs_api
def test_live_browser_schema() -> None:
    from vlm_wrapper.browser_adapter import generate_label

    image = _synthesize_wiki_page()
    result = generate_label(
        image,
        goal="Identify the first section heading on the Wikipedia main page.",
        task_id="wiki.main_page.demo",
        step=0,
        url="https://en.wikipedia.org",
        max_entities=15,
        model=MODEL,
        api_key=API_KEY,
    )
    schema = result.get("schema")
    _assert_valid_schema(schema, "browser")
    assert "<entities>" in schema
    _save_schema("browser", schema)


@live
@needs_api
def test_live_desktop_schema() -> None:
    from vlm_wrapper.osworld_adapter import generate_label

    image = _synthesize_desktop()
    result = generate_label(
        image,
        instruction="Open the Files application.  What pyautogui action would you take first?",
        task_id="osworld.demo",
        step=0,
        max_entities=15,
        model=MODEL,
        api_key=API_KEY,
    )
    schema = result.get("schema")
    _assert_valid_schema(schema, "desktop")
    _save_schema("desktop", schema)


@live
@needs_api
def test_live_clevr_schema() -> None:
    from vlm_wrapper.benchmarks.clevr import (
        iter_clevr_samples, parse_clevr_sample,
    )

    sample = next(iter_clevr_samples(split="val", limit=1))
    out = parse_clevr_sample(
        sample,
        model=MODEL,
        api_key=API_KEY,
        max_entities=10,
        max_rounds=2,
    )
    schema = out.get("schema")
    _assert_valid_schema(schema, "clevr")
    assert "<answer>" in schema, "CLEVR schema must contain <answer>"
    assert out.get("answer"), "CLEVR parser returned empty answer"
    _save_schema("clevr", schema)


@live
@needs_api
def test_live_video_holmes_schema() -> None:
    from vlm_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples, parse_video_holmes_sample,
    )

    chosen = None
    for s in iter_video_holmes_samples(split="test", limit=50):
        if s.video_path and s.video_path.exists():
            chosen = s
            break
    if chosen is None:
        pytest.skip(
            "No Video-Holmes video clips on disk — see "
            "install/INSTALL_BENCHMARKS.md §5 to download them.")

    out = parse_video_holmes_sample(
        chosen,
        num_frames=4,
        max_rounds=2,
        max_entities=12,
        model=MODEL,
        api_key=API_KEY,
    )
    schema = out.get("schema")
    _assert_valid_schema(schema, "video_holmes")
    assert "<answer>" in schema, "Video-Holmes schema must contain <answer>"
    _save_schema("video_holmes", schema)
