#!/usr/bin/env python3
"""Collect immutable Slurm resource receipts without inventing USD prices."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

from motif_transfer.contracts import stable_hash


def _gpu_count(alloc_tres: str) -> int:
    match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", alloc_tres)
    if match:
        return int(match.group(1))
    typed = re.findall(r"(?:^|,)gres/gpu:[^=,]+=(\d+)(?:,|$)", alloc_tres)
    return max((int(value) for value in typed), default=0)


def _gpu_types(alloc_tres: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"gres/gpu:([^=,]+)=\d+", alloc_tres))))


def parse_sacct(text: str, phase_by_job: dict[str, str]) -> tuple[list[dict], dict]:
    rows = []
    totals = defaultdict(int)
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("|").split("|", 4)
        if len(fields) != 5:
            raise ValueError("unexpected sacct row")
        job_id, name, state, elapsed_raw, alloc_tres = fields
        parent = job_id.split("_", 1)[0].split(".", 1)[0]
        if parent not in phase_by_job or "." in job_id:
            continue
        seconds = int(elapsed_raw or 0)
        gpu_count = _gpu_count(alloc_tres)
        gpu_seconds = seconds * gpu_count
        phase = phase_by_job[parent]
        for gpu_type in _gpu_types(alloc_tres):
            totals[gpu_type] += gpu_seconds
        rows.append({
            "job_id": job_id,
            "parent_job_id": parent,
            "phase": phase,
            "job_name": name,
            "state": state,
            "elapsed_seconds": seconds,
            "allocated_gpu_count": gpu_count,
            "gpu_types": list(_gpu_types(alloc_tres)),
            "gpu_seconds": gpu_seconds,
            "alloc_tres": alloc_tres,
        })
    if not rows:
        raise ValueError("sacct returned no experiment jobs")
    return rows, {
        "total_gpu_seconds": sum(row["gpu_seconds"] for row in rows),
        "gpu_seconds_by_type": dict(sorted(totals.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True, help="phase=job_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("formal cost receipt is immutable")
    mapping = {}
    for value in args.job:
        phase, job_id = value.split("=", 1)
        if not phase or not job_id.isdigit() or job_id in mapping:
            raise ValueError("invalid or duplicate phase=job_id")
        mapping[job_id] = phase
    command = [
        "sacct", "-X", "-j", ",".join(mapping),
        # JobIDRaw gives each array element its internal Slurm allocation ID
        # (for example 7438470), which cannot be related back to the submitted
        # parent ID (for example 7438425).  JobID preserves the parent/task
        # spelling (7438425_0), which parse_sacct intentionally understands.
        "--format=JobID,JobName,State,ElapsedRaw,AllocTRES", "-n", "-P",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    rows, totals = parse_sacct(completed.stdout, mapping)
    seen = {row["parent_job_id"] for row in rows}
    if seen != set(mapping):
        raise ValueError(f"cost receipt missing jobs: {sorted(set(mapping)-seen)}")
    body = {
        "schema_version": "agqa-query-grounder-v2-formal-slurm-cost-v1",
        "status": "RESOURCE_RECEIPTS_COLLECTED",
        "jobs": rows,
        **totals,
        "provider_calls": 0,
        "provider_cost_usd": 0.0,
        "local_gpu_usd_cost_claimed": False,
        "target_outcomes_read": False,
    }
    body["report_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": body["status"], "jobs": len(rows),
        "total_gpu_seconds": body["total_gpu_seconds"],
        "gpu_seconds_by_type": body["gpu_seconds_by_type"],
        "report_sha256": body["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
