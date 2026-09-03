#!/usr/bin/env python3
"""Merge scalar-executor and multi-IR selector supervision without target data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    config = _read(args.config)
    components = config["components"]
    if set(components) != {"SCALAR_EXECUTOR", "MULTI_IR_SELECTOR"}:
        raise ValueError("mixed SFT requires the two declared objectives")

    component_audit = {}
    component_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, spec in components.items():
        root = _resolve(spec["dataset_dir"])
        manifest_path = root / "manifest.json"
        manifest = _read(manifest_path)
        if manifest.get("status") not in {
            "FROZEN_SOURCE_ONLY_CONTROLLER_SUPERVISION",
            "FROZEN_SOURCE_ONLY_MULTI_IR_SELECTOR_SUPERVISION",
        } or not all((manifest.get("gates") or {}).values()):
            raise ValueError(f"component is not source-only and gate-clean: {name}")
        if manifest.get("target_data_used", False) is not False:
            raise ValueError(f"target data entered component: {name}")
        by_split = {}
        files = {}
        for split in ("train", "validation", "source_held_out"):
            path = root / f"{split}.jsonl"
            expected = (
                manifest["sft_files"][split]["sha256"]
                if "sft_files" in manifest else
                manifest["files"][split]["sha256"]
            )
            if _sha(path) != expected:
                raise ValueError(f"component split hash mismatch: {name}:{split}")
            by_split[split] = _rows(path)
            files[split] = {
                "path": str(path.resolve()), "sha256": _sha(path),
                "rows": len(by_split[split]),
            }
        component_rows[name] = by_split
        component_audit[name] = {
            "dataset_dir": str(root.resolve()),
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _sha(manifest_path),
                "status": manifest["status"],
            },
            "files": files,
        }

    args.output_dir.mkdir(parents=True)
    output_files = {}
    all_prompts: dict[str, set[str]] = {}
    for split in ("train", "validation", "source_held_out"):
        merged = []
        for objective, by_split in component_rows.items():
            for row in by_split[split]:
                merged.append(dict(row) | {
                    "training_objective_audit_only": objective,
                })
        ids = [str(row["example_id"]) for row in merged]
        prompts = [str(row["prompt"]) for row in merged]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate example IDs in mixed {split}")
        if len(prompts) != len(set(prompts)):
            raise ValueError(f"duplicate prompts in mixed {split}")
        all_prompts[split] = set(prompts)
        merged.sort(key=lambda row: hashlib.sha256(
            str(row["example_id"]).encode("utf-8")
        ).hexdigest())
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in merged:
                stream.write(json.dumps(
                    row, sort_keys=True, ensure_ascii=False,
                ) + "\n")
        output_files[split] = {
            "path": str(path.resolve()), "sha256": _sha(path),
            "rows": len(merged),
            "by_objective": dict(sorted(Counter(
                row["training_objective_audit_only"] for row in merged
            ).items())),
            "by_decision": dict(sorted(Counter(
                json.loads(row["completion"])["decision"] for row in merged
            ).items())),
        }

    splits = list(all_prompts.values())
    gates = {
        "exact_two_source_only_objectives": len(component_rows) == 2,
        "all_splits_nonempty": all(
            output_files[split]["rows"] > 0 for split in output_files
        ),
        "no_prompt_overlap_across_splits": all(
            left.isdisjoint(right)
            for index, left in enumerate(splits)
            for right in splits[index + 1:]
        ),
        "both_objectives_in_every_split": all(
            set(output_files[split]["by_objective"])
            == {"SCALAR_EXECUTOR", "MULTI_IR_SELECTOR"}
            for split in output_files
        ),
        "target_data_absent": True,
        "target_outcomes_absent": True,
        "native_actions_absent": all(
            "native_action" not in prompt
            for prompts in all_prompts.values() for prompt in prompts
        ),
        "named_policy_templates_absent": all(
            all(token not in prompt for token in (
                "EXPLORE_UNTRIED", "BACKTRACK_REPLAN", "COMMIT_VERIFY",
            ))
            for prompts in all_prompts.values() for prompt in prompts
        ),
    }
    if not all(gates.values()):
        raise SystemExit(f"mixed source SFT gates failed: {gates}")
    body = {
        "schema_version": "harness-mixed-source-sft-v1",
        "status": "FROZEN_SOURCE_ONLY_MIXED_HARNESS_SUPERVISION",
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha(args.config),
        "authority": (
            "UNION_OF_FROZEN_SOURCE_ONLY_SCALAR_EXECUTOR_AND_MULTI_IR_SELECTOR;"
            "NO_TARGET_EXAMPLE;NO_TARGET_OUTCOME;NO_NATIVE_ACTION"
        ),
        "components": component_audit,
        "files": output_files,
        "target_data_used": False,
        "target_outcome_used_for_controller_labels": False,
        "formal_or_qualification_targets_used": False,
        "video_target_data_used": False,
        "target_grounder_training_used_target_outcomes": False,
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
    }
    manifest = body | {"manifest_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()}
    _write(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"], "files": output_files,
        "gates": gates, "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
