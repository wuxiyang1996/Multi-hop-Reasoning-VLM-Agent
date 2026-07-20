"""Import legacy shared-bank records without trusting legacy verdicts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator


TARGET_TASK_TOKENS = {
    "alfworld",
    "browser",
    "miniwob",
    "webshop",
    "osworld",
    "tir_bench",
    "visual_toolbench",
    "video_holmes",
    "siv_bench",
}


def is_source_game_task(task: str) -> bool:
    normalized = task.strip().lower()
    return bool(normalized) and not any(token in normalized for token in TARGET_TASK_TOKENS)


def iter_legacy_bindings(root: str | Path) -> Iterator[Dict[str, Any]]:
    root = Path(root)
    for path in sorted(root.glob("by_task/*/bindings.jsonl")):
        task_from_path = path.parent.name
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                task = str(raw.get("task") or task_from_path)
                canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
                yield {
                    "legacy_id": str(raw.get("concrete_skill_id") or raw.get("skill_id") or ""),
                    "abstract_skill_id": str(raw.get("abstract_skill_id") or ""),
                    "task": task,
                    "source_only": is_source_game_task(task),
                    "status": "LEGACY_PROPOSAL",
                    "legacy_claimed_status": raw.get("binding_status"),
                    "legacy_n_episodes_verified": int(raw.get("n_episodes_verified") or 0),
                    "legacy_pass_rate": float(raw.get("pass_rate") or 0.0),
                    "binding_source": raw.get("binding_source"),
                    "source_path": str(path),
                    "source_line": line_no,
                    "record_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    "quarantine_reason": "legacy verdict lacks replay-qualified source receipts",
                }


__all__ = ["is_source_game_task", "iter_legacy_bindings"]
