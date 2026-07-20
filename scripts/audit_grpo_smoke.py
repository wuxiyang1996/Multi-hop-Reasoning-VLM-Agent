#!/usr/bin/env python3
"""Fail unless a short co-evolution run performed real GRPO updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from skill_agents.grpo.advantage_utils import compute_grpo_group_advantages


DECISION_ADAPTERS = ("skill_selection", "action_taking")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warm-adapter-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    trainer_log = (run_dir / "trainer.log").read_text(errors="replace")
    fatal_markers = (
        "GRPO training failed",
        "Final GRPO training failed",
        "CUDA out of memory",
        "Traceback (most recent call last)",
    )
    found_fatal = [marker for marker in fatal_markers if marker in trainer_log]
    if found_fatal:
        raise SystemExit(f"fatal trainer markers: {found_fatal}")

    step_rows = _read_jsonl(run_dir / "step_log.jsonl")
    expected = list(range(args.expected_steps))
    actual = [int(row["step"]) for row in step_rows]
    if actual != expected:
        raise SystemExit(f"expected finalized GRPO steps {expected}, got {actual}")
    if any(float(row.get("phase_c_grpo_time_s", 0)) <= 0 for row in step_rows):
        raise SystemExit("one or more steps did not record positive GRPO wall time")

    evidence: dict[str, dict] = {}
    for step in expected:
        step_dir = run_dir / "grpo_data" / f"step_{step:04d}"
        step_evidence: dict[str, dict] = {}
        for adapter in DECISION_ADAPTERS:
            path = step_dir / f"{adapter}.jsonl"
            rows = _read_jsonl(path)
            rewards = [float(row["reward"]) for row in rows]
            completions = [str(row.get("completion", "")) for row in rows]
            advantages = compute_grpo_group_advantages(
                rewards, completions=completions
            )
            n_nonzero = sum(abs(value) > 1e-8 for value in advantages)
            if len(rows) < 16 or n_nonzero == 0:
                raise SystemExit(
                    f"step {step} {adapter}: rows={len(rows)}, "
                    f"nonzero_advantages={n_nonzero}"
                )
            step_evidence[adapter] = {
                "n_records": len(rows),
                "n_nonzero_advantages": n_nonzero,
                "reward_min": min(rewards),
                "reward_max": max(rewards),
            }
        checkpoint = run_dir / "checkpoints" / f"step_{step:04d}" / "metadata.json"
        if not checkpoint.is_file():
            raise SystemExit(f"missing checkpoint metadata: {checkpoint}")
        evidence[str(step)] = step_evidence

    completed_updates: dict[str, int] = {}
    changed_adapters: dict[str, dict[str, str | bool]] = {}
    for adapter in DECISION_ADAPTERS:
        pattern = re.compile(rf"FSDP GRPO \[{re.escape(adapter)}\] done:")
        completed_updates[adapter] = len(pattern.findall(trainer_log))
        if completed_updates[adapter] < args.expected_steps:
            raise SystemExit(
                f"{adapter}: only {completed_updates[adapter]} completed FSDP updates"
            )
        source = args.warm_adapter_dir / "decision" / adapter / "adapter_model.safetensors"
        output = run_dir / "lora_adapters" / "decision" / adapter / "adapter_model.safetensors"
        source_hash = _sha256(source)
        output_hash = _sha256(output)
        changed = source_hash != output_hash
        if not changed:
            raise SystemExit(f"adapter weights did not change: {adapter}")
        changed_adapters[adapter] = {
            "source_sha256": source_hash,
            "output_sha256": output_hash,
            "changed": changed,
        }

    result = {
        "status": "ok",
        "run_dir": str(run_dir),
        "expected_steps": args.expected_steps,
        "finalized_steps": actual,
        "completed_fsdp_updates": completed_updates,
        "step_evidence": evidence,
        "adapter_hashes": changed_adapters,
        "phase_c_wall_time_s": [
            float(row["phase_c_grpo_time_s"]) for row in step_rows
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
