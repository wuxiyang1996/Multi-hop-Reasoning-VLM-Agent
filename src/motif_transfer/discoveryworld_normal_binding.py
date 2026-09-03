"""Normal-only target binding normalization.

The validated DiscoveryWorld Easy route remains pinned to its byte-exact V23
adapter. Proteomics Normal exposed one additional target-native representation:
neural binders may return ``same_location`` with a JSON-null distance, whose
only coherent symbolic interpretation is distance zero. Keep that Normal
development behavior outside the frozen Easy adapter.
"""

from __future__ import annotations

import json
from typing import Any

from motif_transfer.discoveryworld_sokoban_transfer import (
    DiscoveryWorldTargetBinding,
    parse_target_binding as parse_frozen_easy_target_binding,
)


def parse_normal_target_binding(
    raw: str,
    observation: Any,
) -> DiscoveryWorldTargetBinding:
    """Normalize the unique zero-distance encoding, then use the frozen parser."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return parse_frozen_easy_target_binding(raw, observation)
    if (
        isinstance(payload, dict)
        and payload.get("commit_subject_relation_to_target") == "same_location"
        and payload.get("target_distance") is None
    ):
        payload["target_distance"] = 0
        raw = json.dumps(payload, sort_keys=True)
    return parse_frozen_easy_target_binding(raw, observation)


__all__ = ["parse_normal_target_binding"]
