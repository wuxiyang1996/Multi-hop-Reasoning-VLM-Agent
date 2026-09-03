"""Resolve frozen repository artifacts without trusting machine-local prefixes.

Historical receipts intentionally retain the absolute path that was recorded on
the machine that produced them.  That path is provenance, not a requirement
that a reproduction use the same mount point.  This module maps paths rooted at
the repository directory name back into the active checkout while leaving
unrelated absolute paths alone.

Always remap a matching repository anchor, even when the historical path exists.
That property is important for clean-room bundle tests on the original cluster:
otherwise missing files in the extracted bundle would be silently read from the
author's full working tree.
"""

from __future__ import annotations

from pathlib import Path


def resolve_repo_artifact(value: str | Path, repo_root: Path) -> Path:
    """Return the checkout-local location for a frozen repository artifact."""

    root = repo_root.resolve()
    path = Path(value)
    if not path.is_absolute():
        return root / path

    anchor = root.name
    indices = [index for index, part in enumerate(path.parts) if part == anchor]
    if indices:
        relative_parts = path.parts[indices[-1] + 1 :]
        return root.joinpath(*relative_parts)
    return path
