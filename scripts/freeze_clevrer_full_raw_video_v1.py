#!/usr/bin/env python3
"""Freeze outcome-blind, history-disjoint CLEVRER development/reserve cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from motif_transfer.contracts import stable_hash


FAMILIES = ("descriptive", "explanatory", "predictive", "counterfactual")
FORBIDDEN_KEYS = {"answer", "program", "correct", "ground_truth", "motion_trajectory", "collision"}


def _prior_video_ids(root: Path) -> tuple[set[int], list[dict[str, Any]]]:
    paths = (
        list((root / "configs").glob("*clevrer*.json"))
        + list((root / "configs").glob("sokoban_clevrer*.json"))
        + list((root / "runs").glob("*clevrer*/*.json"))
        + list((root / "runs/typed_clevrer_v4b_gpt54").glob("*.json"))
        + list((root / "runs/structured_clevrer_v4").glob("*.json"))
    )
    ids: set[int] = set(); receipts = []
    for path in sorted(set(paths)):
        # New V1 protocol/config files contain paths and hashes but no runtime
        # video IDs; including them is harmless and makes reruns deterministic.
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = {int(x) for x in re.findall(r"(?:video_|sim_)(1[0-4]\d{3})", text)}
        if found:
            ids.update(found)
            receipts.append({"path": str(path.relative_to(root)), "video_id_count": len(found),
                             "video_ids_sha256": stable_hash(sorted(found))})
    return ids, receipts


def _public_projection(question: dict[str, Any]) -> dict[str, Any]:
    family = str(question.get("question_type") or "").casefold()
    if family not in FAMILIES:
        raise ValueError(f"unknown public CLEVRER question family: {family}")
    choices = []
    for choice in question.get("choices") or ():
        choices.append({
            "choice_id": int(choice["choice_id"]),
            "choice": str(choice["choice"]),
        })
    return {
        "question_id": int(question["question_id"]),
        "question": str(question["question"]),
        "question_type": family,
        "question_subtype": str(question.get("question_subtype") or ""),
        "choices": choices,
    }


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_KEYS & {str(key).casefold() for key in value}) or any(
            _contains_forbidden(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden(child) for child in value)
    return False


def _rank(nonce: str, video_id: int) -> str:
    return stable_hash({"nonce": nonce, "video_id": video_id})


def _select_questions(row: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
    grouped = {family: [] for family in FAMILIES}
    for question in row["questions"]:
        public = _public_projection(question)
        grouped[public["question_type"]].append(public)
    output = []
    for family in FAMILIES:
        selected = sorted(grouped[family], key=lambda q: stable_hash(q))[:maximum]
        output.extend(selected)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("CLEVRER V1 cohort freeze is immutable")
    root = args.root.resolve()
    protocol = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if protocol.get("status") not in {
        "FROZEN_BEFORE_PUBLIC_COHORT_PROJECTION_OR_ANY_NEW_OUTCOME_READ",
        "FROZEN_BEFORE_V1B_COHORT_SELECTION_OR_ANY_NEW_OUTCOME_READ",
        "FROZEN_BEFORE_V2_COHORT_SELECTION_OR_ANY_NEW_OUTCOME_READ",
    }:
        raise ValueError("CLEVRER V1 preregistration is not frozen")
    raw = json.loads(args.validation.read_text(encoding="utf-8"))
    prior_ids, prior_receipts = _prior_video_ids(root)
    candidates = [row for row in raw if int(row["scene_index"]) not in prior_ids]
    if protocol.get("require_all_families_per_video") is True:
        candidates = [
            row for row in candidates
            if {str(q.get("question_type") or "").casefold() for q in row["questions"]}
            >= set(FAMILIES)
        ]
    candidates.sort(key=lambda row: _rank(protocol["selection_nonce"], int(row["scene_index"])))
    dev_n = int(protocol["development_video_count"])
    reserve_n = int(protocol["reserve_video_count"])
    if len(candidates) < dev_n + reserve_n:
        raise ValueError("insufficient history-disjoint CLEVRER videos")
    partitions = {"development": candidates[:dev_n], "reserve": candidates[dev_n:dev_n + reserve_n]}
    projected = {}
    for split, rows in partitions.items():
        projected[split] = [{
            "video_id": int(row["scene_index"]),
            "video_filename": str(row["video_filename"]),
            "selection_rank_sha256": _rank(protocol["selection_nonce"], int(row["scene_index"])),
            "questions": _select_questions(row, int(protocol["maximum_questions_per_family_per_video"])),
        } for row in rows]
    if _contains_forbidden(projected):
        raise RuntimeError("public CLEVRER cohort projection leaked outcomes/programs")
    dev_ids = {row["video_id"] for row in projected["development"]}
    reserve_ids = {row["video_id"] for row in projected["reserve"]}
    gates = {
        "development_count_exact": len(dev_ids) == dev_n,
        "reserve_count_exact": len(reserve_ids) == reserve_n,
        "history_video_disjoint": not bool((dev_ids | reserve_ids) & prior_ids),
        "development_reserve_video_disjoint": not bool(dev_ids & reserve_ids),
        "all_four_public_families_present_per_video": all(
            {q["question_type"] for q in row["questions"]} == set(FAMILIES)
            for split in projected.values() for row in split
        ),
        "no_outcome_or_program_in_projection": not _contains_forbidden(projected),
    }
    body = {
        "schema_version": "clevrer-full-raw-video-public-cohort-v1",
        "status": "FROZEN_PUBLIC_COHORTS" if all(gates.values()) else "COHORT_FREEZE_FAILED",
        "preregistration_sha256": stable_hash(protocol),
        "selection_authority": "PUBLIC_TEXT_AND_HISTORY_IDS_ONLY;NO_OUTCOME_OR_PROGRAM_FEATURE",
        "source_validation_file_sha256": stable_hash(args.validation.read_text(encoding="utf-8")),
        "prior_video_count": len(prior_ids),
        "prior_video_ids_sha256": stable_hash(sorted(prior_ids)),
        "prior_receipts": prior_receipts,
        "development": projected["development"],
        "reserve": projected["reserve"],
        "gates": gates,
        "answers_projected": False,
        "functional_programs_projected": False,
    }
    body["cohort_sha256"] = stable_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": body["status"], "prior_video_count": len(prior_ids),
        "development_videos": len(dev_ids), "reserve_videos": len(reserve_ids),
        "development_tasks": sum(len(row["questions"]) for row in projected["development"]),
        "reserve_tasks": sum(len(row["questions"]) for row in projected["reserve"]),
        "gates": gates, "cohort_sha256": body["cohort_sha256"],
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
