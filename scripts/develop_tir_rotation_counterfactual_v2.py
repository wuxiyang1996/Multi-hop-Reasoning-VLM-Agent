#!/usr/bin/env python3
"""Consumed-only probe for target-native counterfactual rotation grounding.

The neural grounder sees a deterministically shuffled panel of candidate image
interventions but no numeric angles, answer slots, or gold label.  It identifies
the intervention that restores physical uprightness.  The symbolic group
executor then binds that anonymous intervention to a target-native rotation.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402
from motif_transfer.tetris_rotation_transfer import (  # noqa: E402
    exact_sign_p,
    parse_rotation_options,
)


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _panel_order(sample_id: str, slots: list[str]) -> list[str]:
    return sorted(
        slots,
        key=lambda slot: hashlib.sha256(
            f"tir-rotation-counterfactual-v2\0{sample_id}\0{slot}".encode()
        ).hexdigest(),
    )


def _contact_sheet(
    image: Image.Image, *, sample_id: str, options: dict[str, float],
) -> tuple[Image.Image, dict[str, str]]:
    slots = _panel_order(sample_id, list(options))
    tokens = [f"P{index}" for index in range(len(slots))]
    mapping = dict(zip(tokens, slots, strict=True))
    tile_w, tile_h, label_h = 420, 420, 54
    sheet = Image.new("RGB", (tile_w * 3, (tile_h + label_h) * 2), "#e9ecef")
    draw = ImageDraw.Draw(sheet)
    font = _font(30)
    for index, token in enumerate(tokens):
        slot = mapping[token]
        # PIL uses positive counterclockwise angles.  Each panel applies one
        # target-native CLOCKWISE candidate intervention to the current image.
        corrected = image.convert("RGB").rotate(
            -float(options[slot]), expand=True, fillcolor="white",
            resample=Image.Resampling.BICUBIC,
        )
        rendered = ImageOps.contain(corrected, (tile_w - 16, tile_h - 16))
        canvas = Image.new("RGB", (tile_w, tile_h), "white")
        canvas.paste(rendered, ((tile_w - rendered.width) // 2, (tile_h - rendered.height) // 2))
        col, row = index % 3, index // 3
        x, y = col * tile_w, row * (tile_h + label_h)
        sheet.paste(canvas, (x, y + label_h))
        draw.rectangle((x, y, x + tile_w, y + label_h), fill="#263238")
        draw.text((x + 16, y + 7), token, fill="white", font=font)
    return sheet, mapping


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _ground(
    *, api_key: str, base_url: str, model: str, sheet: Image.Image,
) -> tuple[str, float, dict[str, Any]]:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=2)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a visual orientation verifier. You cannot see numeric "
                    "rotation angles, answer choices, or a gold label. Return strict JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Each panel is the same scene after a different anonymous "
                            "rotation intervention. Which ONE panel restores the scene to "
                            "its physically original upright orientation? Use gravity, "
                            "walls, floors, text, people, furniture, and architecture. "
                            "Return {\"panel_id\":\"P0-P5\",\"confidence\":0-1," 
                            "\"visual_cues\":[string]}."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": _data_url(sheet)}},
                ],
            },
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    token = str(payload.get("panel_id") or "").strip().upper()
    if token not in {f"P{i}" for i in range(6)}:
        raise ValueError(f"invalid panel token: {token!r}")
    confidence = float(payload.get("confidence", 0.0))
    usage = response.usage
    return token, confidence, {
        "model": str(response.model),
        "prompt_tokens": int(usage.prompt_tokens if usage else 0),
        "completion_tokens": int(usage.completion_tokens if usage else 0),
        "response_sha256": stable_hash(payload),
        "visual_cues": [str(x) for x in payload.get("visual_cues", [])][:8],
    }


def _one(
    row: dict[str, Any], *, dataset_root: Path, api_key: str, base_url: str,
    model: str,
) -> dict[str, Any]:
    sample_id = str(row["id"])
    options = parse_rotation_options(str(row["prompt"]))
    with Image.open(dataset_root / str(row["image_1"])) as image:
        sheet, mapping = _contact_sheet(image, sample_id=sample_id, options=options)
    token, confidence, usage = _ground(
        api_key=api_key, base_url=base_url, model=model, sheet=sheet,
    )
    slot = mapping[token]
    body = {
        "sample_id": sample_id,
        "panel_order": mapping,
        "selected_panel": token,
        "authentic_answer": slot,
        "gold_answer_evaluator_only": str(row["answer"]),
        "correct": slot == str(row["answer"]),
        "confidence": confidence,
        "usage": usage,
        "numeric_angles_or_slots_seen_by_neural_grounder": False,
    }
    return body | {"receipt_sha256": stable_hash(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads((REPO / "configs/tir_tetris_rotation_v1_frozen.json").read_text())
    consumed_ids = config["splits"]["consumed_development"] + config["splits"]["qualification"]
    if args.limit is not None:
        consumed_ids = consumed_ids[: args.limit]
    index = {
        str(row["id"]): row
        for row in json.loads((args.dataset_root / "TIR-Bench.json").read_text())
    }
    keys = runpy.run_path(str(args.keys))
    api_key = str(keys.get("OPENROUTER_API_KEY") or "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir = args.output_dir / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    pending = []
    for sample_id in consumed_ids:
        path = receipt_dir / f"{sample_id}.json"
        if path.is_file():
            rows[sample_id] = json.loads(path.read_text())
        else:
            pending.append(sample_id)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _one, index[sample_id], dataset_root=args.dataset_root,
                api_key=api_key, base_url="https://openrouter.ai/api/v1",
                model="openai/gpt-4.1-mini",
            ): sample_id
            for sample_id in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            value = future.result()
            (receipt_dir / f"{sample_id}.json").write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )
            rows[sample_id] = value
            print(json.dumps({"sample_id": sample_id, "correct": value["correct"]}), flush=True)
    ordered = [rows[sample_id] for sample_id in consumed_ids]
    correct = sum(bool(row["correct"]) for row in ordered)
    # Existing raw answers are frozen receipts from V1; no new baseline call or
    # target label is used to construct the V2 counterfactual grounder.
    raw_by_id: dict[str, dict[str, Any]] = {}
    for split in ("consumed_development", "qualification"):
        old = json.loads((REPO / f"runs/tir_tetris_rotation_v1/{split}_report.json").read_text())
        raw_by_id.update({str(row["sample_id"]): row for row in old["rows"]})
    wins = sum(row["correct"] and not raw_by_id[row["sample_id"]]["correct"]["raw_target_only"] for row in ordered)
    losses = sum((not row["correct"]) and raw_by_id[row["sample_id"]]["correct"]["raw_target_only"] for row in ordered)
    raw_correct = sum(raw_by_id[row["sample_id"]]["correct"]["raw_target_only"] for row in ordered)
    report = {
        "schema_version": "tir-rotation-counterfactual-development-v2",
        "status": "CONSUMED_DEVELOPMENT_COUNTERFACTUAL_GATE_PASSED" if correct > raw_correct and wins > losses else "CONSUMED_DEVELOPMENT_COUNTERFACTUAL_GATE_FAILED",
        "tasks": len(ordered),
        "authentic_correct": correct,
        "raw_correct": raw_correct,
        "paired": {"wins": wins, "losses": losses, "ties": len(ordered)-wins-losses, "exact_two_sided_p": exact_sign_p(wins, losses)},
        "rows": ordered,
        "claim_boundary": "Consumed-only redesign after V1 qualification failure. The target neural model chooses an anonymous counterfactual image intervention; numeric rotations and answer slots remain hidden from it. No V1 held-out image, prompt, or outcome is opened.",
    }
    report["report_sha256"] = stable_hash(report)
    (args.output_dir / "development_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "tasks", "authentic_correct", "raw_correct", "paired")}, indent=2))


if __name__ == "__main__":
    main()
