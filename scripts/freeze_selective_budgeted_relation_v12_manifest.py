#!/usr/bin/env python3
"""Freeze V12 identities through the audited V11 manifest freezer."""

from __future__ import annotations

from freeze_budgeted_relation_edge_v11_manifest import main


if __name__ == "__main__":
    raise SystemExit(main(default_manifest_version="v12"))
