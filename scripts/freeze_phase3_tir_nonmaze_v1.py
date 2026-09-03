#!/usr/bin/env python3
"""Freeze disjoint color-family TIR Phase-3 splits before target calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from motif_transfer.contracts import stable_hash  # noqa: E402


TIR_FAMILIES = frozenset({
    "color", "contrast", "instrument", "jigsaw", "math", "maze", "ocr",
    "refcoco", "rotation_game", "spot_difference", "symbolic",
    "visual_search", "word_search",
})
AUDIT_ONLY_DEVELOPMENT_IDS = ("23", "33", "58")
SOURCE_PROGRAM_DIR = "configs/phase3_source_induction_v3/frozen_reserve/programs"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_contextual_ids(
    value: Any, *, known_ids: set[str], path: Path,
    ancestor_is_tir: bool = False,
) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        family = str(value.get("family") or value.get("task") or "")
        local_tir = ancestor_is_tir or family in TIR_FAMILIES
        for key, nested in value.items():
            key_text = str(key).lower()
            child_tir = local_tir or "tir" in key_text
            id_key = "id" in key_text or key_text in {
                "adaptation", "development", "qualification", "heldout",
                "held_out", "formal", "selection",
            }
            if child_tir and id_key:
                candidates = nested if isinstance(nested, list) else [nested]
                for candidate in candidates:
                    if not isinstance(candidate, (dict, list)):
                        text = str(candidate).removeprefix("row:")
                        if text in known_ids:
                            output.add(text)
            output |= _collect_contextual_ids(
                nested, known_ids=known_ids, path=path,
                ancestor_is_tir=child_tir,
            )
    elif isinstance(value, list):
        for nested in value:
            output |= _collect_contextual_ids(
                nested, known_ids=known_ids, path=path,
                ancestor_is_tir=ancestor_is_tir,
            )
    return output


def _historical_tir_reservations(
    roots: list[Path], known_ids: set[str], *,
    ignored_path_substrings: tuple[str, ...] = (),
) -> dict[str, list[str]]:
    """Conservatively find TIR-specific reservations/receipts across checkouts.

    Broad benchmark index manifests are ignored: listing the public population
    without opening content or outcomes is not exposure.  TIR-named configs,
    receipts, and reports are included even if an old run later aborted.
    """

    found: dict[str, set[str]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        candidates = []
        for subdir in ("configs", "runs", "docs/results"):
            directory = root / subdir
            if not directory.is_dir():
                continue
            candidates.extend(directory.rglob("*.json"))
        for path in candidates:
            relative = str(path.relative_to(root)).lower()
            if "tir" not in relative:
                continue
            if any(value.lower() in relative for value in ignored_path_substrings):
                # Idempotent regeneration must not treat this script's own
                # prospective reservation or partial development receipts as
                # pre-existing target exposure.
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            ids = _collect_contextual_ids(
                payload, known_ids=known_ids, path=path,
                ancestor_is_tir=True,
            )
            # Per-sample receipt paths sometimes carry the only explicit ID.
            if path.stem in known_ids and "receipt" in relative:
                ids.add(path.stem)
            for sample_id in ids:
                found.setdefault(sample_id, set()).add(str(path))
    return {key: sorted(value) for key, value in sorted(found.items(), key=lambda x: int(x[0]))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/fs/gamma-projects/vlm-robot/datasets/TIR-Bench"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "configs/phase3_tir_nonmaze_v1_splits.json",
    )
    args = parser.parse_args()
    dataset_file = args.dataset_root / "TIR-Bench.json"
    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    known_ids = {str(row["id"]) for row in rows}
    roots = [
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent"),
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-experiment-clean"),
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-github-main"),
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-source-fresh-v1"),
        Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent-two-agent-clean"),
    ]
    historical = _historical_tir_reservations(
        roots, known_ids,
        ignored_path_substrings=("phase3_tir_nonmaze_v1",),
    )
    eligible = [
        str(row["id"]) for row in rows
        if str(row.get("task")) == "color" and not row.get("image_2")
    ]
    if len(eligible) != 100:
        raise SystemExit(f"unexpected TIR color population: {len(eligible)}")
    excluded = set(historical) | set(AUDIT_ONLY_DEVELOPMENT_IDS)
    available = [sample_id for sample_id in eligible if sample_id not in excluded]
    ordered = sorted(available, key=lambda sample_id: hashlib.sha256(
        f"phase3-tir-nonmaze-v1\0{sample_id}".encode()
    ).hexdigest())
    if len(ordered) < 57:
        raise SystemExit("insufficient unreserved TIR color IDs")
    development_train = [*AUDIT_ONLY_DEVELOPMENT_IDS, *ordered[:13]]
    development_validation = ordered[13:21]
    qualification = ordered[21:33]
    formal = ordered[33:57]
    reserve = ordered[57:]
    splits = {
        "development_train": development_train,
        "development_validation": development_validation,
        "qualification": qualification,
        "formal": formal,
        "unopened_reserve": reserve,
    }
    flattened = [sample_id for values in splits.values() for sample_id in values]
    if len(flattened) != len(set(flattened)):
        raise SystemExit("TIR Phase-3 splits overlap")

    source_programs = []
    for path in sorted((REPO / SOURCE_PROGRAM_DIR).glob("*.json")):
        payload = json.loads(path.read_text())
        source_programs.append({
            "path": str(path.relative_to(REPO)),
            "file_sha256": file_sha256(path),
            "artifact_sha256": payload["artifact_sha256"],
        })
    body = {
        "schema_version": "phase3-tir-nonmaze-split-manifest-v1",
        "status": "FROZEN_BEFORE_ANY_PHASE3_TIR_TARGET_CALL",
        "claim_boundary": (
            "PROSPECTIVE_SAME_SOURCE_IR_REPLICATION_ON_SINGLE_IMAGE_NON_MAZE_"
            "TIR_COLOR_TASKS;DEVELOPMENT_ONLY_GROUNDER_ACQUISITION;FORMAL_LOCKED"
        ),
        "dataset": {
            "path": str(dataset_file.resolve()),
            "sha256": file_sha256(dataset_file),
            "benchmark": "TIR-Bench",
            "family": "color",
            "population": len(eligible),
            "formal_prompt_image_answer_read_before_freeze": False,
        },
        "selection": {
            "rule": (
                "Exclude TIR-specific historical reservations/receipts across five "
                "checkouts and three explicitly audited examples; sort remaining "
                "color IDs by sha256('phase3-tir-nonmaze-v1\\0'+id), then allocate "
                "16 train, 8 validation, 12 qualification, 24 formal. Selection "
                "reads only id, task family, and image-count metadata."
            ),
            "target_outcome_used": False,
            "prompt_or_image_used": False,
            "audit_only_development_ids": list(AUDIT_ONLY_DEVELOPMENT_IDS),
            "historically_reserved_ids": sorted(historical, key=int),
            "historical_reservation_receipt_sha256": stable_hash(historical),
            "checkout_roots": list(map(str, roots)),
        },
        "splits": splits,
        "conditions": [
            "neural_only", "source_induced", "source_permuted",
            "generic_scaffold", "target_native_ceiling",
        ],
        "source_programs": source_programs,
        "source_ir": {
            "runtime": "src/motif_transfer/phase3_attempt_runtime.py",
            "portfolio": "src/motif_transfer/phase3_source_portfolio.py",
            "typed_effect_induction": "src/motif_transfer/phase3_typed_effect_induction.py",
            "program_updated_for_tir": False,
            "source_identity_used_as_runtime_feature": False,
        },
        "target_mdp": {
            "state": "ACCUMULATED_TARGET_NATIVE_WRAPPER_EVIDENCE",
            "actions": ["zoom_region", "extract_colors"],
            "transition_horizons": [1, 4, 8],
            "budget": 8,
            "unavailable_wrapper_tools_fail_closed": [
                "read_text_region", "describe_region",
            ],
        },
        "wrapper": {
            "root": "/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent",
            "execution_authority": "visual_reasoning_wrapper.tools_visual registry",
        },
        "model": {
            "provider": "openrouter",
            "id": "qwen/qwen3-vl-32b-instruct",
            "base_url": "https://openrouter.ai/api/v1",
            "timeout_seconds": 240,
            "max_retries": 3,
            "temperature": 0,
        },
        "media": {
            "maximum_source_pixels": 300000000,
            "native_working_max_side": 2048,
            "overview_max_side": 768,
            "evidence_max_side": 1280,
            "jpeg_quality": 88,
        },
        "qualification_gates": {
            "expected_tasks": 12,
            "minimum_ceiling_successes": 9,
            "minimum_source_action_contrasts": 3,
            "minimum_permuted_action_contrasts": 3,
            "minimum_selected_effect_types": 2,
            "maximum_negative_transfer_rate": 0.0,
            "required_gate_names": [
                "expected_task_count", "target_native_ceiling_capable",
                "source_changes_target_policy", "authentic_differs_from_permuted",
                "multiple_source_effect_types_selected", "maximum_negative_transfer",
                "source_not_below_neural", "source_strictly_beats_neural",
                "source_strictly_beats_permuted", "source_strictly_beats_generic",
            ],
        },
        "formal_gates": {
            "expected_tasks": 24,
            "minimum_ceiling_successes": 18,
            "minimum_source_action_contrasts": 6,
            "minimum_permuted_action_contrasts": 6,
            "minimum_selected_effect_types": 2,
            "maximum_negative_transfer_rate": 0.0,
            "required_gate_names": [
                "expected_task_count", "target_native_ceiling_capable",
                "source_changes_target_policy", "authentic_differs_from_permuted",
                "multiple_source_effect_types_selected", "maximum_negative_transfer",
                "source_not_below_neural", "source_strictly_beats_neural",
                "source_strictly_beats_permuted", "source_strictly_beats_generic",
            ],
        },
        "integrity": {
            "code_sha256": {
                path: file_sha256(REPO / path) for path in (
                    "scripts/collect_phase3_tir_nonmaze.py",
                    "src/motif_transfer/phase3_tir_nonmaze.py",
                    "src/motif_transfer/phase3_attempt_runtime.py",
                    "src/motif_transfer/phase3_source_portfolio.py",
                    "src/motif_transfer/phase3_typed_effect_induction.py",
                )
            },
        },
    }
    output = body | {"config_sha256": stable_hash(body)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": output["status"],
        "splits": {key: len(value) for key, value in splits.items()},
        "historically_reserved": len(historical),
        "config_sha256": output["config_sha256"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
