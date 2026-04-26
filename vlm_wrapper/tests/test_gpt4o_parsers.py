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
    from gymv_wrapper.adapter import generate_label
    assert callable(generate_label)


def test_offline_browser_adapter_import() -> None:
    from vlm_wrapper.browser_adapter import generate_label
    assert callable(generate_label)


def test_offline_osworld_adapter_import() -> None:
    from vlm_wrapper.osworld_adapter import generate_label
    assert callable(generate_label)


def test_offline_tir_bench_loader_yields_samples() -> None:
    pytest.importorskip("datasets")
    from vlm_wrapper.visual_reasoning_wrapper.benchmarks.tir_bench import (
        TIRBenchSample,
        iter_tir_bench_samples,
    )

    try:
        samples = list(iter_tir_bench_samples(split="test", limit=1))
    except Exception as exc:
        pytest.skip(f"TIR-Bench HF load skipped: {exc}")
    assert samples
    s = samples[0]
    assert isinstance(s, TIRBenchSample)
    assert s.prompt and s.sample_id


def test_offline_video_holmes_loader_yields_samples() -> None:
    from vlm_wrapper.visual_reasoning_wrapper.benchmarks.video_holmes import (
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
    from gymv_wrapper.adapter import generate_label

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
def test_live_gymv_tool_loop_schema() -> None:
    """Gym-V via VLM + tool-calling (cascaded_ground, chain=['tool_loop']).

    Exercises the full multi-hop pipeline for an interactive env: the
    VLM sees the rendered 2048 board AND has access to the gymv tool
    registry (list_entities, query_entity_pos, check_relation,
    count_merge_candidates, get_state_flags, list_valid_actions, …).
    We provide obs_text alongside the image so the tool handlers can
    return ground-truth data — the point of this test is to prove the
    VLM actually uses tools to ground the schema, not to hide them.

    Asserts (in order):
      1. a parseable <state> schema comes back;
      2. cascaded_ground picks the tool_loop head;
      3. the semantic validator marks the schema valid;
      4. GPT-4o issued at least one tool call (tool_trace non-empty);
      5. the schema's <actions> block copies the env's valid actions
         verbatim (no "slide_left"-style hallucinations);
      6. an <evidence> block records the reasoning hops so downstream
         skill learning can cite them.
    """
    from PIL import ImageDraw, ImageFont

    from vlm_wrapper.ground import GroundingRequest, cascaded_ground

    # Hand-drawn mid-game 2048 board (richer than the early-game
    # synthesizer) so GPT-4o has several non-empty tiles to reason
    # about.  Keeps the test above the gymv entity-count floor (>=3)
    # without leaning on the `_synthesize_2048_board` default.
    board = [
        [2, 4, 0, 0],
        [0, 2, 4, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]
    tile_colors = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
    }
    image = Image.new("RGB", (480, 480), (187, 173, 160))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    cell, pad = 110, 10
    for r in range(4):
        for c in range(4):
            x0 = pad + c * (cell + pad)
            y0 = pad + r * (cell + pad)
            val = board[r][c]
            draw.rounded_rectangle(
                [x0, y0, x0 + cell, y0 + cell], radius=6,
                fill=tile_colors.get(val, (237, 224, 200)),
            )
            if val:
                draw.text(
                    (x0 + cell / 2, y0 + cell / 2), str(val),
                    fill=(119, 110, 101), anchor="mm", font=font,
                )
    obs_text = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in board
    )
    game_rules = (
        "You are playing 2048.  Slide tiles in one of the four "
        "directions; tiles with the same value merge.  Valid moves: "
        "[Up], [Down], [Left], [Right]."
    )
    valid_actions = ["[Up]", "[Down]", "[Left]", "[Right]"]

    req = GroundingRequest(
        images=image,
        goal="Reach 2048",
        domain="gymv",
        output_mode="actions",
        task_id="Game2048-v0",
        step=0,
        context={
            "description": game_rules,
            "obs_text": obs_text,             # wired into tool handlers
            "valid_actions": valid_actions,
            # Hide obs_text from the VLM prompt so GPT-4o has to call
            # tools (list_entities / get_grid_state / query_entity_pos
            # / check_relation / count_merge_candidates) to ground
            # its schema — otherwise it would just paraphrase the grid.
            "show_obs_text": False,
        },
        max_entities=12,
        max_rounds=6,
        model=MODEL,
        api_key=API_KEY,
    )
    result = cascaded_ground(
        req,
        image_size=image.size,
        chain=["tool_loop"],
    )

    schema = result.schema
    _assert_valid_schema(schema, "gymv_tool_loop")
    _save_schema("gymv_tool_loop", schema)

    assert result.head_used == "tool_loop", (
        f"expected tool_loop head, got {result.head_used!r}"
    )

    val = result.validation
    assert val is not None, "cascaded_ground should attach a ValidationResult"
    assert val.valid, (
        f"tool_loop schema failed semantic validation: "
        f"errors={val.errors} missing_slots={val.missing_slots}"
    )

    assert result.tool_trace, (
        "GPT-4o did not invoke any tools — tool_loop should produce "
        "at least one call in the trace"
    )
    # Every recorded call must dispatch to a real registered tool
    # (not GPT-4o's internal meta-names like `multi_tool_use.parallel`).
    gymv_tool_names = {
        "query_entity_pos", "list_entities", "check_relation",
        "get_state_flags", "list_valid_actions", "get_grid_state",
        "check_deadlock", "spatial_analysis", "count_merge_candidates",
    }
    visual_tool_names = {
        "detect_objects", "grounded_detect", "describe_region",
        "zoom_region", "visual_search", "count_objects",
        "classify_scene", "spatial_query", "measure_distance",
        "extract_colors", "read_text_region",
    }
    known_tools = gymv_tool_names | visual_tool_names
    called_tools = {
        (tc.get("call", {}) or {}).get("name", "") for tc in result.tool_trace
    }
    assert called_tools & known_tools, (
        f"tool_trace only contains unknown tool names {called_tools} — "
        f"expected at least one call to a gymv or visual tool"
    )

    # Prompt guarantees: actions must come from the env vocabulary.
    for action in valid_actions:
        if action in schema:
            break
    else:  # pragma: no cover — defensive
        raise AssertionError(
            f"none of the valid actions {valid_actions} appear in the "
            f"<actions> block of the schema:\n{schema[-400:]}"
        )
    assert "slide_" not in schema.lower(), (
        "schema contains invented 'slide_*' actions — the VLM ignored "
        "the valid-actions instruction"
    )

    # Multi-hop reasoning trace should be preserved in the schema too.
    assert "<evidence>" in schema, (
        "tool_loop schema should record reasoning hops in <evidence>"
    )


@live
@needs_api
def test_live_tir_bench_schema() -> None:
    pytest.importorskip("datasets")
    from vlm_wrapper.visual_reasoning_wrapper.benchmarks.tir_bench import (
        iter_tir_bench_samples,
        parse_tir_bench_sample,
    )

    try:
        sample = next(iter_tir_bench_samples(split="test", limit=1))
    except Exception as exc:
        pytest.skip(f"TIR-Bench HF load skipped: {exc}")
    out = parse_tir_bench_sample(
        sample,
        model=MODEL,
        api_key=API_KEY,
        max_entities=10,
        max_rounds=4,
    )
    schema = out.get("schema")
    _assert_valid_schema(schema, "tir_bench")
    assert "<answer>" in schema, "TIR-Bench schema must contain <answer>"
    assert out.get("answer"), "TIR-Bench parser returned empty answer"
    _save_schema("tir_bench", schema)


@live
@needs_api
def test_live_video_holmes_schema() -> None:
    from vlm_wrapper.visual_reasoning_wrapper.benchmarks.video_holmes import (
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
