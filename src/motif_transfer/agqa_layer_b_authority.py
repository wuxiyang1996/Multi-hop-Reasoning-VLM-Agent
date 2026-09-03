"""Lightweight authority checks shared by AGQA Layer-B runtime entrypoints."""

from __future__ import annotations


def cohort_crossed_authority(cohort: dict) -> bool:
    """Accept any explicitly outcome-free public projection schema.

    Authority is a data contract, not a hard-coded schema-name allowlist.  New
    immutable cohorts may add evaluator-side manifest metadata while exposing
    the same question-only row projection to the parser.
    """
    common_projection_keys = {"answers_projected", "functional_programs_projected"}
    scene_graph_key = next(
        (
            key
            for key in (
                "scene_graph_grounding_projected",
                "official_scene_graph_grounding_projected",
            )
            if key in cohort
        ),
        None,
    )
    if common_projection_keys <= set(cohort) and scene_graph_key is not None:
        return not all(
            cohort[key] is False
            for key in (*sorted(common_projection_keys), scene_graph_key)
        )
    return (
        cohort.get("answers_read") is not False
        or cohort.get("scene_graphs_read") is not False
        or cohort.get("functional_program_visible_at_runtime") is not False
    )


__all__ = ["cohort_crossed_authority"]
