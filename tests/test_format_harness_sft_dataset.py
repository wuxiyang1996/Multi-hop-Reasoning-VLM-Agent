from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_formatter_hides_game_identity_by_default(tmp_path):
    dataset = tmp_path / "source.jsonl"
    dataset.write_text(json.dumps({
        "example_id": "example",
        "game": "shortcut-game-name",
        "episode_id": "episode",
        "split": "train",
        "objective": "OPERATIONAL_EFFECT_PROBE",
        "input_payload": {"before": {}, "after": {}},
        "target_payload": {"verdict": "OBSERVED_FROM_RECEIPT"},
        "evidence_receipt_ids": ["receipt"],
    }) + "\n", encoding="utf-8")
    output = tmp_path / "formatted"
    script = Path(__file__).parents[1] / "scripts" / "format_harness_sft_dataset.py"
    subprocess.run(
        [sys.executable, str(script), str(dataset), "--output-dir", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    row = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "GAME_ID_HASH" not in row["prompt"]
    assert "shortcut-game-name" not in row["prompt"]
    assert manifest["prompt_policy"]["game_identity_exposed"] is False
