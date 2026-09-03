from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from motif_transfer.visual_intervention_receipts import (
    ContentAddressedFrameStore,
    build_visual_intervention_plan,
    collect_plan_split,
    observable_sha256,
    validate_plan,
)


PNG = b"\x89PNG\r\n\x1a\nsmall-test-frame"


def _event(episode_id, sequence, kind, payload):
    return {
        "episode_id": episode_id,
        "sequence": sequence,
        "kind": kind,
        "payload": payload,
    }


def _evidence(tmp_path: Path) -> Path:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    events = []
    episodes = []
    for episode_index in range(6):
        episode_id = f"episode-{episode_index}"
        episodes.append({"episode_id": episode_id})
        sequence = 0
        events.append(_event(
            episode_id, sequence, "RESET", {"requested_seed": 100 + episode_index},
        ))
        sequence += 1
        actions = ("LEFT", "RIGHT")
        prefix = []
        for step in range(6):
            observation = f"episode={episode_id};step={step};prefix={prefix}"
            events.append(_event(episode_id, sequence, "OBSERVATION", {
                "step": step,
                "observable_state_sha256": observable_sha256(observation),
            }))
            sequence += 1
            events.append(_event(episode_id, sequence, "NATIVE_ADMISSIBILITY", {
                "step": step,
                "native_actions": list(actions),
            }))
            sequence += 1
            action = actions[step % 2]
            events.append(_event(episode_id, sequence, "ENVIRONMENT_STEP", {
                "step": step,
                "executed_action": action,
                # Changing this must not affect outcome-blind point selection.
                "reward": 9999.0 * episode_index,
            }))
            sequence += 1
            prefix.append(action)
    (evidence / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    (evidence / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    (evidence / "manifest.json").write_text("{}\n", encoding="utf-8")
    return evidence


def test_plan_is_episode_split_and_outcome_blind(tmp_path):
    evidence = _evidence(tmp_path)
    plan = build_visual_intervention_plan(
        evidence,
        game="game",
        snapshots_per_episode=1,
        minimum_prefix_steps=1,
        maximum_prefix_steps=4,
        max_episode_steps=6,
    )
    snapshots = validate_plan(plan)
    assert plan["split_counts"] == {
        "discovery": 2,
        "held_out": 2,
        "qualification": 2,
    }
    selected = [(row.episode_id, row.step) for row in snapshots]
    events_path = evidence / "events.jsonl"
    events_path.write_text(
        events_path.read_text(encoding="utf-8").replace("9999.0", "-7777.0"),
        encoding="utf-8",
    )
    changed = build_visual_intervention_plan(
        evidence,
        game="game",
        snapshots_per_episode=1,
        minimum_prefix_steps=1,
        maximum_prefix_steps=4,
        max_episode_steps=6,
    )
    assert selected == [
        (row.episode_id, row.step) for row in validate_plan(changed)
    ]


class FakeEnv:
    action_names = ["LEFT", "RIGHT"]

    def __init__(self, game: str, max_steps: int):
        self.game = game
        self.max_steps = max_steps
        self.episode_id = ""
        self.prefix = []
        self.closed = False

    def reset(self, *, seed: int):
        self.episode_id = f"episode-{seed - 100}"
        self.prefix = []
        return self._observation(), {"seed": seed}

    def _observation(self):
        return (
            f"episode={self.episode_id};step={len(self.prefix)};"
            f"prefix={self.prefix}"
        )

    def step(self, action: str):
        self.prefix.append(action)
        return self._observation(), float(action == "RIGHT"), False, False, {
            "action": action,
        }

    def render(self):
        return "data:image/png;base64," + base64.b64encode(PNG).decode()

    def close(self):
        self.closed = True


def test_collection_forks_all_actions_and_binds_frames(tmp_path):
    plan = build_visual_intervention_plan(
        _evidence(tmp_path),
        game="game",
        snapshots_per_episode=1,
        minimum_prefix_steps=1,
        maximum_prefix_steps=4,
        max_episode_steps=6,
    )
    manifest = collect_plan_split(
        plan,
        split="discovery",
        output_dir=tmp_path / "collected",
        env_factory=FakeEnv,
        workers=2,
    )
    assert manifest["all_interventions_observed"]
    assert manifest["before_frame_consistent_per_snapshot"]
    assert manifest["jobs_expected"] == 4
    assert manifest["action_counts"] == {"LEFT": 2, "RIGHT": 2}
    receipts = [
        json.loads(line)
        for line in (tmp_path / "collected/receipts.jsonl").read_text().splitlines()
    ]
    assert {row["intervention_action"] for row in receipts} == {"LEFT", "RIGHT"}
    assert all(row["before_observable_sha256"] == row["expected_observable_sha256"]
               for row in receipts)
    png_hash = hashlib.sha256(PNG).hexdigest()
    assert (tmp_path / f"collected/frames/{png_hash}.png").read_bytes() == PNG


def test_collection_fails_closed_on_tampered_plan(tmp_path):
    plan = build_visual_intervention_plan(
        _evidence(tmp_path), game="game", snapshots_per_episode=1,
        minimum_prefix_steps=1, maximum_prefix_steps=4, max_episode_steps=6,
    )
    plan["snapshots"][0]["step"] += 1
    with pytest.raises(ValueError, match="plan hash mismatch"):
        validate_plan(plan)


def test_frame_store_rejects_non_png(tmp_path):
    store = ContentAddressedFrameStore(tmp_path / "frames")
    bad = "data:image/png;base64," + base64.b64encode(b"not-png").decode()
    with pytest.raises(ValueError, match="not a PNG"):
        store.write_data_url(bad)
