"""Retired-but-preserved subsystems.

Each subpackage here is kept as design provenance. Nothing in ``legacy/``
should be imported by production code on new code paths; existing call
sites that still resolve here go through soft-retirement shims at the
old import paths (e.g. ``crafter`` -> ``legacy.crafter``).
"""
