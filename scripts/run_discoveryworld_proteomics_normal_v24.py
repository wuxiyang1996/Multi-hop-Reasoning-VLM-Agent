#!/usr/bin/env python3
"""Target-native acquisition controller for Proteomics Normal.

This is deliberately source-blind.  It strengthens only the target policy that
must reach a native DROP decision before the separately qualified Sokoban
effect guard can be evaluated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from motif_transfer.discoveryworld_env import DETERMINISM_PROTOCOL, stable_hash  # noqa: E402
from motif_transfer.frozen_motif_agent import (  # noqa: E402
    MemoizedCompletionBackend,
    OpenAICompatibleBackend,
)
from scripts.run_discoveryworld_target_only_v1 import (  # noqa: E402
    file_sha256,
    run_episode,
)


PROTEOMICS_NORMAL_SYSTEM_PROMPT = """You are a target-native DiscoveryWorld
scientific agent solving Proteomics Normal. You receive no source-game advice
and never see the hidden scorecard.

Execute the task as a disciplined scientific survey:
1. Get the proteomics meter from Instrument Table.
   At Instrument Table, do not PICKUP it while it is only salient/nearby: first
   TELEPORT_TO_OBJECT using the meter UUID, then PICKUP after it appears in
   accessibleEnvironmentObjects.
2. Visit the listed Possible Animal Area locations systematically. Never wander
   manually between named areas. If an unmeasured animal is in the supplied
   salient_relative_objects, TELEPORT_TO_OBJECT using its exact visible UUID;
   once accessible, USE the inventory meter on it.
3. Persist an exact table of measured animal name, UUID, Protein A, and Protein B
   parsed from last_action_message. Never remeasure a completed UUID. Continue
   until all five distinct species have measurements. Use every numbered animal
   area as needed; do not cycle only through Areas 1 and 2.
4. Identify the single protein-vector outlier from the five measurements. Then
   obtain the red flag, TELEPORT_TO_LOCATION for that species' statue, inspect
   exact spatial relations, and place the flag directly beside the statue.

TELEPORT_TO_OBJECT is valid for any UUID explicitly listed in
salient_relative_objects, even when distance is greater than one. Prefer it over
manual MOVE_DIRECTION when pursuing a visible scientific object. Manipulate an
object only when it is in inventory or accessibleEnvironmentObjects. If an
action fails, repair its precondition instead of repeating it. Choose exactly
one native action per turn and preserve all measurements in compact memory.

Return one JSON object. Outside dialog use:
{\"action\":\"...\", \"arg1\":..., \"arg2\":..., \"memory\":\"...\",
 \"running_hypotheses\":[\"...\"], \"expected_effect\":\"...\", \"reason\":\"...\"}
Include only required action arguments. During dialog return
chosen_dialog_option_int plus the same memory fields. No prose or Markdown."""


class PromptOverrideBackend:
    def __init__(self, backend: MemoizedCompletionBackend) -> None:
        self.backend = backend

    @property
    def last_usage(self):
        return self.backend.last_usage

    def complete(self, model: str, system_prompt: str, user_prompt: Any) -> str:
        if model != "decision":
            raise ValueError(f"unexpected model role: {model}")
        return self.backend.complete(model, PROTEOMICS_NORMAL_SYSTEM_PROMPT, user_prompt)


ANIMAL_NAMES = (
    "animaplant", "echojelly", "prismatic beast", "spheroid", "vortisquid",
)
PROTEIN_PATTERN = re.compile(r"Protein ([AB]):\s*([0-9.]+)")


class ProteomicsSurveyBackend:
    """Deterministic target-native survey executor over public UI facts."""

    def __init__(self) -> None:
        self.last_usage: dict[str, Any] = {
            "provider": "TARGET_NATIVE_SYMBOLIC_SURVEY", "cost": 0.0,
        }
        self.measured: dict[str, tuple[float, float]] = {}
        self.pending_measure: str | None = None
        self.area_index = 0
        self.at_target_statue = False
        self.commit_attempted = False

    @staticmethod
    def _named(rows: list[dict[str, Any]], needle: str) -> dict[str, Any] | None:
        return next(
            (row for row in rows if needle in str(row.get("name") or "").lower()),
            None,
        )

    def _update_measurement(self, facts: dict[str, Any]) -> None:
        if self.pending_measure is None:
            return
        values = {name: float(value) for name, value in PROTEIN_PATTERN.findall(
            str(facts.get("last_action_message") or "")
        )}
        if set(values) == {"A", "B"}:
            self.measured[self.pending_measure] = (values["A"], values["B"])
            self.pending_measure = None

    def _anomaly(self) -> str:
        if len(self.measured) != 5:
            raise ValueError("anomaly requested before five species were measured")
        ordered = sorted(self.measured)
        first = sorted(value[0] for value in self.measured.values())[2]
        second = sorted(value[1] for value in self.measured.values())[2]
        return max(
            ordered,
            key=lambda name: (
                math.dist(self.measured[name], (first, second)), name,
            ),
        )

    def complete(self, model: str, system_prompt: str, user_prompt: Any) -> str:
        del system_prompt
        if model != "decision" or not isinstance(user_prompt, dict):
            raise ValueError("survey backend accepts only decision payloads")
        facts = dict(user_prompt["target_native_facts"])
        self._update_measurement(facts)
        inventory = [dict(row) for row in facts.get("inventory") or ()]
        accessible = [dict(row) for row in facts.get("accessible_objects") or ()]
        salient = [dict(row) for row in facts.get("salient_relative_objects") or ()]
        meter_inventory = self._named(inventory, "proteomics meter")
        flag_inventory = self._named(inventory, "flag")
        action: dict[str, Any]
        reason: str
        if meter_inventory is None:
            accessible_meter = self._named(accessible, "proteomics meter")
            visible_meter = self._named(salient, "proteomics meter")
            if accessible_meter is not None:
                action = {"action": "PICKUP", "arg1": int(accessible_meter["uuid"])}
                reason = "Acquire the accessible target-native measurement tool."
            elif visible_meter is not None:
                action = {"action": "TELEPORT_TO_OBJECT", "arg1": int(visible_meter["uuid"])}
                reason = "Localize the visible target-native measurement tool."
            else:
                action = {"action": "TELEPORT_TO_LOCATION", "arg1": "Instrument Table"}
                reason = "Move to the official target-native instrument location."
        elif len(self.measured) < 5:
            animals = [
                row for row in accessible
                if str(row.get("name") or "").lower() in ANIMAL_NAMES
                and str(row.get("name") or "").lower() not in self.measured
            ]
            if animals:
                target = sorted(animals, key=lambda row: int(row["uuid"]))[0]
                species = str(target["name"]).lower()
                self.pending_measure = species
                action = {
                    "action": "USE", "arg1": int(meter_inventory["uuid"]),
                    "arg2": int(target["uuid"]),
                }
                reason = f"Measure the unobserved species {species}."
            else:
                visible = [
                    row for row in salient
                    if str(row.get("name") or "").lower() in ANIMAL_NAMES
                    and str(row.get("name") or "").lower() not in self.measured
                ]
                if visible:
                    target = sorted(
                        visible,
                        key=lambda row: (
                            int(row.get("distance") or 10**9), int(row["uuid"]),
                        ),
                    )[0]
                    action = {"action": "TELEPORT_TO_OBJECT", "arg1": int(target["uuid"])}
                    reason = f"Localize unmeasured visible species {target['name']}."
                else:
                    self.area_index = (self.area_index % 12) + 1
                    action = {
                        "action": "TELEPORT_TO_LOCATION",
                        "arg1": f"Possible Animal Area {self.area_index}",
                    }
                    reason = "Advance the exhaustive twelve-location survey."
        else:
            anomaly = self._anomaly()
            if flag_inventory is None:
                accessible_flag = self._named(accessible, "flag")
                visible_flag = self._named(salient, "flag")
                if accessible_flag is not None:
                    action = {"action": "PICKUP", "arg1": int(accessible_flag["uuid"])}
                    reason = "Acquire the target-native result marker."
                elif visible_flag is not None:
                    action = {"action": "TELEPORT_TO_OBJECT", "arg1": int(visible_flag["uuid"])}
                    reason = "Localize the visible target-native result marker."
                else:
                    action = {"action": "TELEPORT_TO_LOCATION", "arg1": "Instrument Table"}
                    reason = "Return to the official result-marker location."
            elif not self.at_target_statue:
                statue = f"Statue of a {anomaly}"
                action = {"action": "TELEPORT_TO_LOCATION", "arg1": statue}
                self.at_target_statue = True
                reason = f"Navigate to the statue for robust protein outlier {anomaly}."
            elif not self.commit_attempted:
                action = {"action": "DROP", "arg1": int(flag_inventory["uuid"])}
                self.commit_attempted = True
                reason = "Issue the first native commit; transfer remains source-blind here."
            else:
                action = {"action": "DISCOVERY_FEED_GET_UPDATES"}
                reason = "Commit already attempted; preserve the observable terminal state."
        memory = json.dumps({
            "measured": self.measured,
            "survey_area_index": self.area_index,
            "anomaly": self._anomaly() if len(self.measured) == 5 else None,
        }, sort_keys=True)
        return json.dumps({
            **action, "memory": memory,
            "running_hypotheses": [
                "the robust protein-vector outlier is the non-native species"
            ],
            "expected_effect": reason, "reason": reason,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] not in {
        "CONSUMED_DEVELOPMENT_TARGET_ACQUISITION",
        "FROZEN_BEFORE_NORMAL_QUALIFICATION",
        "FROZEN_BEFORE_NORMAL_FORMAL",
    }:
        raise SystemExit("invalid Proteomics Normal protocol status")
    runpy.run_path(str(args.keys))  # retain the common CLI contract; no key is used.
    runtime_hashes = {
        "config": file_sha256(args.config),
        "runner": file_sha256(Path(__file__)),
        "base_runner": file_sha256(REPO / "scripts/run_discoveryworld_target_only_v1.py"),
        "environment_wrapper": file_sha256(REPO / "src/motif_transfer/discoveryworld_env.py"),
        "policy_payload": file_sha256(REPO / "src/motif_transfer/discoveryworld_policy.py"),
        "target_controller": "DETERMINISTIC_PROTEOMICS_SURVEY_V1",
        "official_environment_commit": str(config["official_environment_commit"]),
    }
    receipts = []
    for index, task in enumerate(config["tasks"]):
        if task != {"scenario": "Proteomics", "difficulty": "Normal", "seed": int(task["seed"])}:
            raise SystemExit(f"invalid task for V24 controller: {task}")
        receipts.append(run_episode(
            task=task, config=config, backend=ProteomicsSurveyBackend(),
            output_dir=args.output_dir, runtime_hashes=runtime_hashes,
            thread_id=97000 + index,
        ))
    first_commit = []
    for receipt in receipts:
        commits = [
            row for row in receipt["steps"]
            if str(row["action"].get("action")) in {"DROP", "PUT"}
        ]
        first_commit.append({
            "task_id": receipt["task_id"],
            "commit_reached": bool(commits),
            "first_commit_step": commits[0]["episode_step"] if commits else None,
            "first_commit_action": commits[0]["action"] if commits else None,
            "official_success": bool(receipt["evaluation"]["official_success"]),
            "steps": len(receipt["steps"]),
            "episode_sha256": receipt["episode_sha256"],
        })
    required = int(config["acquisition_gate"]["minimum_commit_coverage"])
    coverage = sum(row["commit_reached"] for row in first_commit)
    gate = coverage >= required
    body = {
        "schema_version": "discoveryworld-proteomics-normal-acquisition-v24",
        "status": "TARGET_ACQUISITION_GATE_PASSED" if gate else "TARGET_ACQUISITION_GATE_FAILED",
        "claim_boundary": config["claim_boundary"],
        "tasks": len(receipts),
        "commit_coverage": coverage,
        "minimum_commit_coverage": required,
        "episodes": first_commit,
        "zero_source_input": True,
        "zero_policy_oracle_scorecard_use": all(not row["policy_runtime_saw_oracle_scorecard"] for row in receipts),
        "runtime_hashes": runtime_hashes,
        "determinism_protocol": DETERMINISM_PROTOCOL,
    }
    body["summary_sha256"] = stable_hash(body)
    path = args.output_dir / "acquisition_summary.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body, indent=2, sort_keys=True))
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
