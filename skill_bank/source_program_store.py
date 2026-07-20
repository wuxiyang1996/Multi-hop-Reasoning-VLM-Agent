"""Append-safe JSONL store for canonical source programs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from skill_bank.program_ir import CanonicalSkillProgram


class SourceProgramStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def replace(self, programs: Iterable[CanonicalSkillProgram]) -> int:
        """Atomically replace the store after validating every program."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        seen: set[str] = set()
        for program in programs:
            program.validate()
            if program.program_id in seen:
                raise ValueError(f"duplicate program_id: {program.program_id}")
            seen.add(program.program_id)
            rows.append(program.to_dict())
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
        return len(rows)


__all__ = ["SourceProgramStore"]
