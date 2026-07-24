from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .contracts import stable_hash
from .runtime import EpisodeResult


def episode_artifact(
    result: EpisodeResult,
    *,
    episode_id: str,
    environment_id: str,
    policy_identity: dict[str, Any],
    seed: int | None,
) -> dict[str, Any]:
    records = [asdict(record) for record in result.records]
    body = {
        "schema_version": 1,
        "episode_id": episode_id,
        "environment_id": environment_id,
        "policy_identity": policy_identity,
        "seed": seed,
        "records": records,
        "final_observation": asdict(result.final_observation),
    }
    return {**body, "artifact_sha256": stable_hash(body)}


def write_episode_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_episode_artifact(artifact: dict[str, Any]) -> bool:
    body = dict(artifact)
    expected = body.pop("artifact_sha256", None)
    return expected == stable_hash(body)
