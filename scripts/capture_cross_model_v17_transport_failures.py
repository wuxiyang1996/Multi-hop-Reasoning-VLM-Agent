#!/usr/bin/env python3
"""Capture raw OpenRouter text for V17 rows rejected only by JSON transport parsing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any

from openai import OpenAI


REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(path))

import collect_natural_video_cross_model_proof_v17 as v17  # noqa: E402
import collect_natural_video_focused_verify_v16 as focused  # noqa: E402
import collect_natural_video_recovery_v15 as paired  # noqa: E402
import run_active_video_wrapper_transfer as media_helpers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--existing", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = json.loads(Path(config["input_receipts"]).read_text(encoding="utf-8"))
    complete = {
        (row["benchmark"], row["sample_id"])
        for row in json.loads(args.existing.read_text(encoding="utf-8"))
    }
    pending = [
        row for row in inputs
        if (row["benchmark"], row["sample_id"]) not in complete
    ]
    keys = runpy.run_path(str(args.keys))
    client = OpenAI(
        api_key=keys[config["model"]["api_key_name"]],
        base_url=config["model"]["base_url"],
        timeout=config["model"]["timeout_seconds"],
        max_retries=config["model"]["max_retries"],
    )
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for row in pending:
        sample_id = str(row["sample_id"])
        safe = sample_id.replace("/", "_")
        raw_path = args.raw_dir / f"{row['benchmark']}__{safe}.txt"

        def capture(
            client: OpenAI, *, model: str, system: str,
            content: list[dict[str, Any]], max_tokens: int,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            response = client.chat.completions.create(
                model=model, temperature=0, max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            )
            raw = response.choices[0].message.content or ""
            raw_path.write_text(raw, encoding="utf-8")
            return json.loads(raw), {
                "model": str(response.model),
                "finish_reason": str(response.choices[0].finish_reason),
                "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            }

        original = media_helpers._json_call
        media_helpers._json_call = capture
        try:
            panels = focused._proof_panels(row, config)
            paired._proof_call(
                client, sample=v17._RuntimeSample(row), panels=panels, config=config,
            )
            print(json.dumps({"unexpectedly_valid": sample_id, "raw": str(raw_path)}))
        except Exception as exc:
            print(json.dumps({
                "captured": sample_id, "error": f"{type(exc).__name__}: {exc}",
                "raw": str(raw_path),
            }))
        finally:
            media_helpers._json_call = original


if __name__ == "__main__":
    main()
