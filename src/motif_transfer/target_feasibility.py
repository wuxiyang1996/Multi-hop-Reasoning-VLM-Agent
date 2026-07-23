from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CellAudit:
    domain: str
    cell: str
    data_ready: bool
    official_evaluator_ready: bool
    real_executor_ready: bool
    adaptation_split_ready: bool
    test_split_ready: bool
    stub_fallback_possible: bool
    target_content_in_source_treatment: bool
    evidence: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        return all(
            (
                self.data_ready,
                self.official_evaluator_ready,
                self.real_executor_ready,
                self.adaptation_split_ready,
                self.test_split_ready,
                not self.stub_fallback_possible,
                not self.target_content_in_source_treatment,
            )
        )

    @property
    def status(self) -> str:
        if self.target_content_in_source_treatment:
            return "CONTAMINATED"
        if self.stub_fallback_possible:
            return "STUB_FALLBACK_BLOCKED"
        if not self.data_ready:
            return "DATA_MISSING"
        if not self.official_evaluator_ready:
            return "OFFICIAL_EVALUATOR_NOT_INTEGRATED"
        if not self.real_executor_ready:
            return "REAL_EXECUTOR_MISSING"
        if not (self.adaptation_split_ready and self.test_split_ready):
            return "SPLIT_NOT_FROZEN"
        return "RUNNABLE"

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runnable"] = self.runnable
        payload["status"] = self.status
        return payload


def validate_matrix(rows: Iterable[CellAudit]) -> tuple[CellAudit, ...]:
    values = tuple(rows)
    expected = {
        "visual_toolbench",
        "tir_bench",
        "video_holmes",
        "miniwob",
        "webshop",
        "alfworld_valid_seen",
        "alfworld_valid_unseen",
    }
    names = {row.cell for row in values}
    if len(values) != 7 or names != expected:
        raise ValueError(f"target diagnosis must contain exactly the 7 frozen cells; got {sorted(names)}")
    if any(row.cell == "siv_bench" for row in values):
        raise ValueError("SIV-Bench is excluded from the frozen diagnosis matrix")
    return values


def summarize_matrix(rows: Iterable[CellAudit]) -> dict[str, Any]:
    values = validate_matrix(rows)
    domains = {row.domain for row in values}
    by_status: dict[str, int] = {}
    for row in values:
        by_status[row.status] = by_status.get(row.status, 0) + 1
    return {
        "schema_version": 1,
        "matrix": "4-domain/7-cell",
        "domains": sorted(domains),
        "cells": [row.to_json() for row in values],
        "runnable_cells": sum(row.runnable for row in values),
        "status_counts": by_status,
        "claim_limits": [
            "Video evidence is one benchmark only.",
            "A deterministic or synthetic executor is never a target result.",
            "Target-derived skills or mega-skill members are never source treatment.",
            "No source treatment is called transferable before matched controls show incremental value.",
        ],
    }


def write_summary(path: str | Path, rows: Iterable[CellAudit]) -> dict[str, Any]:
    payload = summarize_matrix(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
