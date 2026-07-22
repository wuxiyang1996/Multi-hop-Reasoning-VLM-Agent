from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .phase1_assets import sha256_file


ROLE_TO_ADAPTER = {
    "segment": "segment",
    "binding": "contract",
    "review": "curator",
}


def build_coevolved_specs(checkpoint_root: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_root)
    games: dict[str, Any] = {}
    for game_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        roles: dict[str, Any] = {}
        base_models: set[str] = set()
        for runtime_role, adapter_role in ROLE_TO_ADAPTER.items():
            adapter_dir = game_dir / "adapters" / "skillbank" / adapter_role
            config_path = adapter_dir / "adapter_config.json"
            weights_path = adapter_dir / "adapter_model.safetensors"
            if not config_path.exists() or not weights_path.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            base_models.add(str(config.get("base_model_name_or_path", "")))
            roles[runtime_role] = {
                "adapter_role": adapter_role,
                "adapter_path": str(adapter_dir),
                "config_sha256": sha256_file(config_path),
                "weights_sha256": sha256_file(weights_path),
                "rank": config.get("r"),
            }
        games[game_dir.name] = {
            "base_models": sorted(base_models - {""}),
            "roles": roles,
            "complete": set(roles) == set(ROLE_TO_ADAPTER),
        }
    return {
        "schema_version": 1,
        "frozen": True,
        "authority": "UNTRUSTED_MOTIF_PROPOSAL_ONLY",
        "games": games,
    }
