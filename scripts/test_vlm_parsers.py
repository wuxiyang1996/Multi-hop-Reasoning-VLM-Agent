#!/usr/bin/env python3
"""End-to-end smoke-test of every GPT-4o parser in ``vlm_wrapper``.

Runs the Head-2 / tool-loop pipeline across the five supported domains
and writes each resulting ``<state>`` schema to ``out/schemas/`` so the
shape can be inspected by eye:

  1. ``gymv``         — vlm_wrapper.gymv_adapter.generate_label
  2. ``browser``      — vlm_wrapper.browser_adapter.generate_label
  3. ``desktop``      — vlm_wrapper.osworld_adapter.generate_label
  4. ``clevr``        — vlm_wrapper.benchmarks.clevr.parse_clevr_sample
  5. ``video_holmes`` — vlm_wrapper.benchmarks.video_holmes.parse_video_holmes_sample

Each case uses a real input that already ships with the repo (bundled
PNGs for the three interactive domains, ``data/CLEVR`` and
``data/Video-Holmes`` for the two benchmarks), so the script can be run
without any extra setup besides ``OPENAI_API_KEY`` in ``.env``.

Outputs per case::

    out/schemas/
    ├── <case>.schema.txt   # the <state>…</state> block
    ├── <case>.raw.json     # full adapter result dict
    └── summary.json        # one-line-per-case run report

Usage::

    python scripts/test_vlm_parsers.py                 # run all five
    python scripts/test_vlm_parsers.py --cases gymv browser
    python scripts/test_vlm_parsers.py --max-rounds 2  # cheaper CLEVR/VH
    python scripts/test_vlm_parsers.py --dry-run       # no API calls
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vlm_parser import _load_dotenv  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "out" / "schemas"
DEFAULT_CAPTURE_DIR = REPO_ROOT / "out" / "captures"

CASES = ("gymv", "browser", "desktop", "clevr", "video_holmes")

logger = logging.getLogger("test_vlm_parsers")


# ----------------------------------------------------------------------
# Input fixtures — try the real env first, fall back to a PIL-rendered
# synthetic image so the script always has something to send to GPT-4o.
# ----------------------------------------------------------------------

def _capture_or_synthesize_gymv(
    capture_dir: Path,
) -> tuple["Image.Image", dict[str, Any], str]:  # type: ignore[name-defined]
    """Return (image, extra_adapter_kwargs, source_tag)."""
    from PIL import Image

    try:
        import gym_v  # type: ignore
        env = gym_v.make("Games/Game2048-v0")
        obs_dict, _ = env.reset(seed=42)
        obs = obs_dict["agent_0"]
        image = obs.image if isinstance(obs.image, Image.Image) else Image.fromarray(obs.image)
        # Pull the env's actual action vocabulary so we can constrain
        # the VLM's <actions> output (avoids the "slide_left" hallucination).
        try:
            valid_actions = list(env.action_space[obs_dict.get("agent_0", "agent_0")])  # type: ignore[index]
        except Exception:
            valid_actions = ["[Up]", "[Down]", "[Left]", "[Right]"]
        env.close()
        extras = {
            "game_rules": getattr(env, "description", "") or (
                "You are playing 2048.  Slide tiles Up/Down/Left/Right; "
                "equal tiles merge."),
            "obs_text": obs.text or "",
            "valid_actions": valid_actions,
        }
        source = "captured_from_gymv"
    except Exception as exc:
        logger.warning("gym_v capture failed (%s); using synthetic 2048 board", exc)
        image = _synthesize_2048_board()
        extras = {
            "game_rules": (
                "You are playing 2048.  Slide tiles in one of the four "
                "directions; tiles with the same value merge.  "
                "Valid moves: [Up], [Down], [Left], [Right]."),
            "obs_text": (
                "| 2 | 4 | 0 | 0 |\n| 0 | 0 | 0 | 0 |\n"
                "| 0 | 0 | 0 | 0 |\n| 0 | 0 | 0 | 0 |"),
            "valid_actions": ["[Up]", "[Down]", "[Left]", "[Right]"],
        }
        source = "synthetic_2048_board"

    capture_dir.mkdir(parents=True, exist_ok=True)
    image.save(capture_dir / "gymv_2048.png")
    return image, extras, source


def _capture_or_synthesize_browser(
    capture_dir: Path,
) -> tuple["Image.Image", dict[str, Any], str]:  # type: ignore[name-defined]
    from PIL import Image
    import numpy as np

    try:
        import gymnasium as gym  # type: ignore
        import browsergym.core  # type: ignore  # noqa: F401
        env = gym.make(
            "browsergym/openended",
            task_kwargs={"start_url": "https://en.wikipedia.org"},
            headless=True,
        )
        obs, _ = env.reset()
        shot = obs["screenshot"]
        image = Image.fromarray(shot) if isinstance(shot, np.ndarray) else shot
        env.close()
        extras = {
            "url": obs.get("url", "https://en.wikipedia.org"),
        }
        source = "captured_from_browsergym"
    except Exception as exc:
        logger.warning("browsergym capture failed (%s); using synthetic page", exc)
        image = _synthesize_wiki_page()
        extras = {"url": "https://en.wikipedia.org"}
        source = "synthetic_wiki_page"

    capture_dir.mkdir(parents=True, exist_ok=True)
    image.save(capture_dir / "browser_wiki.png")
    return image, extras, source


def _capture_or_synthesize_desktop(
    capture_dir: Path,
) -> tuple["Image.Image", dict[str, Any], str]:  # type: ignore[name-defined]
    from PIL import Image

    image = _synthesize_desktop()
    extras: dict[str, Any] = {
        "a11y_tree_xml": "",
    }
    source = "synthetic_desktop"
    capture_dir.mkdir(parents=True, exist_ok=True)
    image.save(capture_dir / "desktop.png")
    return image, extras, source


def _synthesize_2048_board() -> "Image.Image":  # type: ignore[name-defined]
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 480), (187, 173, 160))
    draw = ImageDraw.Draw(img)
    board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    colors = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
    }
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    cell = 110
    pad = 10
    for r in range(4):
        for c in range(4):
            x0 = pad + c * (cell + pad)
            y0 = pad + r * (cell + pad)
            val = board[r][c]
            draw.rounded_rectangle(
                [x0, y0, x0 + cell, y0 + cell], radius=6,
                fill=colors.get(val, (237, 224, 200)),
            )
            if val:
                draw.text(
                    (x0 + cell / 2, y0 + cell / 2), str(val),
                    fill=(119, 110, 101), anchor="mm", font=font,
                )
    draw.text((10, 448), "2048  (score: 0)", fill=(250, 248, 239))
    return img


def _synthesize_wiki_page() -> "Image.Image":  # type: ignore[name-defined]
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1024, 640), "white")
    draw = ImageDraw.Draw(img)
    try:
        f_head = ImageFont.truetype("arialbd.ttf", 28)
        f_body = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        f_head = f_body = ImageFont.load_default()

    # chrome
    draw.rectangle([0, 0, 1024, 60], fill=(240, 240, 240))
    draw.rectangle([80, 16, 960, 44], fill="white", outline=(180, 180, 180))
    draw.text((92, 22), "https://en.wikipedia.org/wiki/Main_Page", fill=(40, 40, 40), font=f_body)

    # page content
    draw.text((40, 90), "Welcome to Wikipedia,", fill="black", font=f_head)
    draw.text((40, 130),
              "the free encyclopedia that anyone can edit.",
              fill=(30, 30, 30), font=f_body)
    draw.rectangle([40, 170, 990, 172], fill=(200, 200, 200))
    draw.text((40, 190), "From today's featured article",
              fill=(30, 30, 120), font=f_head)
    draw.text((40, 236),
              "The Great Barrier Reef is the world's largest coral reef system...",
              fill="black", font=f_body)
    draw.text((40, 280), "In the news", fill=(30, 30, 120), font=f_head)
    draw.text((40, 320),
              "- Scientists report a new exoplanet discovery.",
              fill="black", font=f_body)
    draw.text((40, 346),
              "- International Space Station crew completes spacewalk.",
              fill="black", font=f_body)
    return img


def _synthesize_desktop() -> "Image.Image":  # type: ignore[name-defined]
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1280, 800), (38, 47, 68))
    draw = ImageDraw.Draw(img)
    try:
        f_label = ImageFont.truetype("arial.ttf", 14)
        f_title = ImageFont.truetype("arialbd.ttf", 18)
    except Exception:
        f_label = f_title = ImageFont.load_default()

    # top menu bar
    draw.rectangle([0, 0, 1280, 28], fill=(25, 25, 25))
    draw.text((16, 6), "Activities    Applications    Places", fill="white", font=f_label)
    draw.text((1130, 6), "16:04   EN   100%", fill="white", font=f_label)

    # a terminal window
    draw.rounded_rectangle([120, 120, 720, 460], radius=6, fill=(30, 30, 30))
    draw.rectangle([120, 120, 720, 148], fill=(60, 60, 60))
    draw.text((130, 126), "Terminal - bash", fill="white", font=f_label)
    draw.text((135, 160), "user@linux:~$ ls", fill=(180, 255, 180), font=f_label)
    draw.text((135, 184), "Desktop  Documents  Downloads  Pictures",
              fill="white", font=f_label)
    draw.text((135, 208), "user@linux:~$", fill=(180, 255, 180), font=f_label)

    # desktop icons (column)
    for i, name in enumerate(["Home", "Trash", "Files", "Firefox", "Terminal"]):
        y = 80 + i * 110
        draw.rounded_rectangle([32, y, 96, y + 64], radius=6,
                               fill=(80, 110, 160))
        draw.text((64, y + 80), name, fill="white", font=f_label, anchor="mm")

    # taskbar
    draw.rectangle([0, 770, 1280, 800], fill=(20, 20, 20))
    draw.text((16, 776), "[Files]  [Firefox]  [Terminal]  [Settings]",
              fill="white", font=f_label)
    return img


# ======================================================================
# Per-case runners — each returns (case_result_dict, schema_string)
# ======================================================================

def _run_gymv(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    from vlm_wrapper.gymv_adapter import generate_label

    image, extras, source = _capture_or_synthesize_gymv(
        Path(args.capture_dir))
    result = generate_label(
        image,
        goal="Reach 2048",
        task_id="Game2048-v0",
        step=0,
        game_rules=extras["game_rules"],
        obs_text=extras["obs_text"],
        valid_actions=extras.get("valid_actions"),
        max_entities=16,
        model=args.model,
        api_key=args.api_key,
    )
    return (
        {
            "case": "gymv",
            "input_source": source,
            "image_size": list(image.size),
            "goal": "Reach 2048",
            "model": result.get("model"),
            "warnings": result.get("warnings"),
            "validation": result.get("validation"),
            "schema": result.get("schema"),
        },
        result.get("schema"),
    )


def _run_browser(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    from vlm_wrapper.browser_adapter import generate_label

    image, extras, source = _capture_or_synthesize_browser(
        Path(args.capture_dir))
    result = generate_label(
        image,
        goal="Identify the first section heading on the Wikipedia main page.",
        task_id="wiki.main_page.demo",
        step=0,
        url=extras["url"],
        max_entities=20,
        model=args.model,
        api_key=args.api_key,
    )
    return (
        {
            "case": "browser",
            "input_source": source,
            "image_size": list(image.size),
            "goal": "Identify the first section heading on the Wikipedia main page.",
            "url": extras["url"],
            "model": result.get("model"),
            "warnings": result.get("warnings"),
            "validation": result.get("validation"),
            "schema": result.get("schema"),
        },
        result.get("schema"),
    )


def _run_desktop(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    from vlm_wrapper.osworld_adapter import generate_label

    image, extras, source = _capture_or_synthesize_desktop(
        Path(args.capture_dir))
    result = generate_label(
        image,
        instruction=(
            "Open the Files application.  Plan the first pyautogui "
            "action you would take from this desktop state."
        ),
        task_id="osworld.demo.open-files",
        step=0,
        a11y_tree_xml=extras["a11y_tree_xml"],
        max_entities=20,
        model=args.model,
        api_key=args.api_key,
    )
    return (
        {
            "case": "desktop",
            "input_source": source,
            "image_size": list(image.size),
            "instruction": "Open the Files application (first pyautogui step).",
            "model": result.get("model"),
            "warnings": result.get("warnings"),
            "validation": result.get("validation"),
            "schema": result.get("schema"),
        },
        result.get("schema"),
    )


def _run_clevr(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    from vlm_wrapper.benchmarks.clevr import (
        iter_clevr_samples, parse_clevr_sample,
    )

    samples = list(iter_clevr_samples(split="val", limit=1))
    if not samples:
        return (
            {"case": "clevr", "error": "no CLEVR samples found"},
            None,
        )
    sample = samples[0]
    out = parse_clevr_sample(
        sample,
        model=args.model,
        api_key=args.api_key,
        max_entities=10,
        max_rounds=args.max_rounds,
    )
    return (
        {
            "case": "clevr",
            "question": sample.question,
            "image_filename": sample.image_filename,
            "ground_truth": out.get("ground_truth"),
            "answer": out.get("answer"),
            "correct": out.get("correct"),
            "model": out.get("model"),
            "rounds": out.get("rounds"),
            "schema": out.get("schema"),
            "tool_trace": out.get("tool_trace"),
            "warnings": out.get("warnings"),
            "validation": out.get("validation"),
        },
        out.get("schema"),
    )


def _run_video_holmes(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str | None]:
    from vlm_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples, parse_video_holmes_sample,
    )

    # Pick the first test question whose video is actually on disk.
    selected = None
    for sample in iter_video_holmes_samples(split="test", limit=50):
        if sample.video_path and sample.video_path.exists():
            selected = sample
            break
    if selected is None:
        return (
            {
                "case": "video_holmes",
                "error": "no Video-Holmes test sample had a matching "
                         "video clip on disk — run "
                         "install/INSTALL_BENCHMARKS.md §5 first.",
            },
            None,
        )

    # Video-Holmes needs a few extra rounds (sample_frames + a couple
    # detect_objects_at_frame calls) before the VLM is ready to emit
    # the schema, so clamp max_rounds to >= 6 regardless of --max-rounds.
    vh_max_rounds = max(args.max_rounds, 6)
    out = parse_video_holmes_sample(
        selected,
        num_frames=args.num_frames,
        model=args.model,
        api_key=args.api_key,
        max_entities=15,
        max_rounds=vh_max_rounds,
    )
    return (
        {
            "case": "video_holmes",
            "video_id": selected.video_id,
            "question_id": selected.question_id,
            "question_type": selected.question_type,
            "question": selected.question,
            "options": selected.options,
            "ground_truth": out.get("ground_truth"),
            "answer": out.get("answer"),
            "answer_raw": out.get("answer_raw"),
            "correct": out.get("correct"),
            "num_frames": out.get("num_frames"),
            "video_meta": out.get("video_meta"),
            "model": out.get("model"),
            "rounds": out.get("rounds"),
            "schema": out.get("schema"),
            "raw": out.get("raw"),
            "tool_trace": out.get("tool_trace"),
            "warnings": out.get("warnings"),
            "validation": out.get("validation"),
        },
        out.get("schema"),
    )


_CASE_RUNNERS: dict[str, Callable[[argparse.Namespace], tuple[dict[str, Any], str | None]]] = {
    "gymv": _run_gymv,
    "browser": _run_browser,
    "desktop": _run_desktop,
    "clevr": _run_clevr,
    "video_holmes": _run_video_holmes,
}


# ======================================================================
# Orchestration
# ======================================================================

def _ascii_safe(text: str) -> str:
    """Windows cp1252 can't print many unicode chars that GPT-4o emits."""
    try:
        text.encode("cp1252")
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode("ascii", "replace").decode("ascii")


def _print_schema_block(label: str, schema: str | None) -> None:
    line = "-" * max(0, 70 - len(label))
    print()
    print(f"-- {label} {line}")
    if not schema:
        print("<no schema produced>")
        return
    try:
        print(schema.rstrip())
    except UnicodeEncodeError:
        print(_ascii_safe(schema.rstrip()))


def _run_case(name: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    runner = _CASE_RUNNERS[name]
    print(f"\n==== Running case: {name} ====")
    t0 = time.perf_counter()
    try:
        case_result, schema = runner(args)
        error = None
    except Exception as exc:  # keep the batch going
        logger.exception("case %s failed", name)
        case_result, schema, error = {"case": name}, None, str(exc)

    elapsed = time.perf_counter() - t0
    case_result["elapsed_s"] = round(elapsed, 2)
    if error:
        case_result["error"] = error

    _print_schema_block(f"{name} schema", schema)

    # Validation scorecard — what the plan's §6/§12 checks actually say
    # about this schema (entity count, missing skill-context fields,
    # fabricated evidence, …).  Surfaces the issues the previous run
    # silently buried in `warnings`.
    val = case_result.get("validation")
    if val:
        print(
            f"validation: valid={val.get('valid')}  "
            f"entities={val.get('entity_count')}  "
            f"high_uncert={val.get('high_uncertainty_frac')}  "
            f"missing_slots={val.get('missing_slots') or []}"
        )
        for err in val.get("errors", []) or []:
            print(f"  ERROR : {err}")
        for w in val.get("warnings", []) or []:
            print(f"  warn  : {w}")

    extra_warnings = [
        w for w in (case_result.get("warnings") or [])
        if not val or w not in (val.get("warnings", []) + val.get("errors", []))
    ]
    if extra_warnings:
        print("other warnings:")
        for w in extra_warnings:
            print(f"  - {w}")

    if "correct" in case_result:
        print(
            f"prediction={case_result.get('answer')!r}  "
            f"ground_truth={case_result.get('ground_truth')!r}  "
            f"correct={case_result.get('correct')}"
        )
    print(f"elapsed: {elapsed:.1f}s")

    # Persist for later inspection.
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_path = out_dir / f"{name}.schema.txt"
    raw_path = out_dir / f"{name}.raw.json"

    if schema:
        schema_path.write_text(schema, encoding="utf-8")
    elif schema_path.exists():
        schema_path.unlink()

    raw_path.write_text(
        json.dumps(case_result, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "case": name,
        "ok": bool(schema),
        "elapsed_s": round(elapsed, 2),
        "schema_path": str(schema_path.relative_to(REPO_ROOT)) if schema else None,
        "raw_path": str(raw_path.relative_to(REPO_ROOT)),
        "warnings": case_result.get("warnings"),
        "validation": case_result.get("validation"),
        "error": case_result.get("error"),
        "extra": {
            k: case_result.get(k)
            for k in ("answer", "ground_truth", "correct",
                      "question_type", "rounds")
            if k in case_result
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run every GPT-4o VLM parser on a real sample and "
                    "dump the resulting schema to out/schemas/.",
    )
    p.add_argument(
        "--cases", nargs="+", default=list(CASES),
        choices=CASES, help="Which cases to run (default: all five).",
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="Directory to write schemas and raw JSON to.")
    p.add_argument("--capture-dir", default=str(DEFAULT_CAPTURE_DIR),
                   help="Directory to save synthesized or captured input "
                        "images for gymv/browser/desktop.")
    p.add_argument("--model", default=None,
                   help="Vision model (default: $VLM_LABEL_MODEL or gpt-4o).")
    p.add_argument("--api-key", default=None,
                   help="OpenAI API key (default: $OPENAI_API_KEY).")
    p.add_argument("--max-rounds", type=int, default=3,
                   help="VLM tool-call rounds for CLEVR / Video-Holmes.")
    p.add_argument("--num-frames", type=int, default=6,
                   help="Frame count for Video-Holmes.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip the actual runners; just list the plan.")
    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    logging.basicConfig(
        level=os.environ.get("VLM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    args = build_parser().parse_args(argv)
    args.api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    args.model = args.model or os.environ.get("VLM_LABEL_MODEL", "gpt-4o")

    if args.dry_run:
        print("Dry run. Cases that would run:")
        for c in args.cases:
            print(f"  - {c}")
        return 0

    if not args.api_key:
        print(
            "No OPENAI_API_KEY found. Add it to .env (see .env.example) "
            "or set the environment variable before running.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir)
    summaries: list[dict[str, Any]] = []
    for case in args.cases:
        summaries.append(_run_case(case, args, out_dir))

    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    ok = sum(1 for s in summaries if s["ok"])
    print("\n==== Summary ====")
    for s in summaries:
        tag = "OK" if s["ok"] else "FAIL"
        extra = ""
        if s["extra"]:
            extra = " | " + " ".join(
                f"{k}={v}" for k, v in s["extra"].items() if v is not None
            )
        print(f"  [{tag}] {s['case']:<13} {s['elapsed_s']:>5.1f}s{extra}")
    print(
        f"{ok}/{len(summaries)} cases produced a <state> schema. "
        f"Outputs written under {out_dir}/"
    )

    return 0 if ok == len(summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
