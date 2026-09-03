from __future__ import annotations

import base64
import json
from pathlib import Path

from motif_transfer.causal_effect_options import (
    CLASS_CONTEXTUAL,
    CLASS_NULL,
    CLASS_STABLE,
    build_causal_effect_option_artifact,
    validate_causal_effect_option_artifact,
)
from motif_transfer.visual_intervention_receipts import (
    build_visual_intervention_plan,
    collect_plan_split,
    observable_sha256,
)


def _event(episode_id, kind, payload):
    return {"episode_id": episode_id, "kind": kind, "payload": payload}


def _build_evidence(tmp_path: Path) -> Path:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    events = []
    episodes = []
    actions = ("NULL", "STABLE", "CONTEXT")
    for index in range(18):
        episode = f"ep-{index:02d}"
        episodes.append({"episode_id": episode})
        events.append(_event(episode, "RESET", {"requested_seed": 100 + index}))
        for step in range(2):
            obs = f"{episode}:{step}"
            events.extend((
                _event(episode, "OBSERVATION", {
                    "step": step,
                    "observable_state_sha256": observable_sha256(obs),
                }),
                _event(episode, "NATIVE_ADMISSIBILITY", {
                    "step": step, "native_actions": list(actions),
                }),
                _event(episode, "ENVIRONMENT_STEP", {
                    "step": step, "executed_action": actions[index % 3],
                }),
            ))
    (evidence / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    (evidence / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    (evidence / "manifest.json").write_text("{}\n", encoding="utf-8")
    return evidence


class EffectEnv:
    action_names = ["NULL", "STABLE", "CONTEXT"]

    def __init__(self, game, max_steps):
        self.index = 0
        self.episode = ""
        self.last_action = "NULL"

    def reset(self, *, seed):
        self.index = 0
        self.episode = f"ep-{seed - 100:02d}"
        self.last_action = "NULL"
        return f"{self.episode}:0", {}

    def step(self, action):
        self.last_action = action
        self.index += 1
        return f"{self.episode}:{self.index}", 0.0, False, False, {}

    def render(self):
        episode_index = int(self.episode.split("-")[1])
        visible = self.last_action == "STABLE" or (
            self.last_action == "CONTEXT" and episode_index % 2 == 0
        )
        marker = self.last_action.encode() if visible else b"NULL"
        raw = b"\x89PNG\r\n\x1a\n" + marker
        return "data:image/png;base64," + base64.b64encode(raw).decode()

    def close(self):
        pass


def test_discovery_induces_anonymous_effect_classes_and_deranged_control(tmp_path):
    plan = build_visual_intervention_plan(
        _build_evidence(tmp_path), game="game", snapshots_per_episode=1,
        minimum_prefix_steps=1, maximum_prefix_steps=1, max_episode_steps=2,
    )
    discovery = tmp_path / "discovery"
    collect_plan_split(
        plan, split="discovery", output_dir=discovery,
        env_factory=EffectEnv, workers=2,
    )
    artifact = build_causal_effect_option_artifact(
        plan, discovery, stable_effect_min_rate=0.75,
        null_effect_max_rate=0.0, minimum_snapshots=6,
    )
    validate_causal_effect_option_artifact(artifact)
    assert artifact["source_grounding"]["action_classes"] == {
        "CONTEXT": CLASS_CONTEXTUAL,
        "NULL": CLASS_NULL,
        "STABLE": CLASS_STABLE,
    }
    authentic = artifact["source_grounding"]["action_classes"]
    shuffled = artifact["shuffled_control"]["action_classes"]
    assert all(authentic[action] != shuffled[action] for action in authentic)
    assert artifact["lifecycle"] == "DISCOVERY_CANDIDATE_NOT_SOURCE_QUALIFIED"
