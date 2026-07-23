from __future__ import annotations

import hashlib
from typing import Iterable


def id_digest(identifier: str, *, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}\0{identifier}".encode("utf-8")).hexdigest()


def freeze_one_shot_split(
    adaptation_ids: Iterable[str],
    test_ids: Iterable[str],
    *,
    namespace: str,
) -> dict[str, object]:
    """Select one adaptation ID without inspecting content or outcomes."""
    adaptation = sorted({str(value) for value in adaptation_ids}, key=lambda x: (id_digest(x, namespace=namespace), x))
    test = sorted({str(value) for value in test_ids}, key=lambda x: (id_digest(x, namespace=namespace), x))
    if not adaptation:
        raise ValueError("adaptation pool is empty")
    shot = adaptation[0]
    # Some benchmarks expose only one public split. The selected shot must
    # never leak back into the held-out list.
    heldout = [value for value in test if value != shot]
    if not heldout:
        raise ValueError("held-out test pool is empty after removing the shot")
    return {
        "selection_rule": "lowest_sha256(namespace\\0sample_id)",
        "namespace": namespace,
        "adaptation_id": shot,
        "adaptation_id_sha256": id_digest(shot, namespace=namespace),
        "adaptation_pool_size": len(adaptation),
        "test_ids": heldout,
        "test_pool_size": len(heldout),
        "content_or_outcome_used_for_selection": False,
    }
