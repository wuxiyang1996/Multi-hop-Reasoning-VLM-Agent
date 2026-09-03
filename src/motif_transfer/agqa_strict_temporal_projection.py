"""Order-preserving projection from native video time to sampled-frame time.

Nearest-neighbour projection is unsafe at strict temporal boundaries: the
native frame immediately after an action can map back onto the final sampled
frame inside that action.  These helpers use a ceiling for lower bounds and a
floor for upper bounds.  An empty sampled intersection is represented by a
lower index greater than the upper index and must fail closed downstream.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import replace
from typing import Sequence


def project_native_window_strict(
    sampled_native_indices: Sequence[int], native_lower: int, native_upper: int,
) -> tuple[int, int]:
    """Project an inclusive native interval without crossing either boundary."""

    sampled = tuple(int(value) for value in sampled_native_indices)
    if not sampled or tuple(sorted(set(sampled))) != sampled:
        raise ValueError("sampled native indices must be non-empty, unique, and ordered")
    lower = int(native_lower)
    upper = int(native_upper)
    if lower > upper:
        return 1, 0
    left = bisect_left(sampled, lower)
    right = bisect_right(sampled, upper) - 1
    if left >= len(sampled) or right < 0 or left > right:
        return 1, 0
    return left, right


def strict_track_for_label(
    tracks, label: str, lower: int, upper: int, *, canonicalize,
) -> tuple[str, int] | None:
    """Choose a label-matched track only when it has in-window pixel evidence."""

    if int(lower) > int(upper):
        return None
    values = [row for row in tracks if row.canonical_label == canonicalize(label)]
    midpoint = (int(lower) + int(upper)) / 2
    ranked = []
    for row in values:
        evidence = [
            int(frame) for frame in row.evidence_frames
            if int(lower) <= int(frame) <= int(upper)
        ]
        if not evidence:
            continue
        chosen = min(evidence, key=lambda frame: (abs(frame - midpoint), frame))
        ranked.append((float(row.confidence), -abs(chosen - midpoint), row.track_id, chosen))
    if not ranked:
        return None
    best = max(ranked)
    return str(best[2]), int(best[3])


def action_localization_native_view(obligation: dict) -> tuple[int, ...]:
    """Prefer the explicitly separated dense action-localization view."""
    values = obligation.get("localization_native_frame_index_view")
    if values is None:
        values = obligation.get("native_frame_index_view", ())
    return tuple(int(value) for value in values)


def recenter_action_anchor_events(
    events, plan: dict, selected_native_indices: Sequence[int],
):
    """Represent clip-classifier action anchors at their receptive-field center.

    The full clip remains the evidence used to bind the typed patient track.
    Its center is the conventional temporal point estimate used by the
    symbolic BEFORE/AFTER executor.  Only parser-requested action obligations
    are changed; root answer events and relation events are untouched.
    """
    by_slot = {}
    for obligation in plan.get("action_obligations", ()):
        view = action_localization_native_view(obligation)
        if view:
            by_slot[str(obligation["slot_id"])] = view
    selected = tuple(int(value) for value in selected_native_indices)
    if not selected or tuple(sorted(set(selected))) != selected:
        raise ValueError("selected native indices must be non-empty, unique, and ordered")
    output = []
    count = 0
    for event in events:
        if "patient" not in event.role_map and "theme" not in event.role_map:
            output.append(event)
            continue
        views = [by_slot[slot] for slot in event.semantic_slot_ids if slot in by_slot]
        if len(views) != 1:
            output.append(event)
            continue
        view = views[0]
        native_center = round((min(view) + max(view)) / 2)
        center = min(
            range(len(selected)),
            key=lambda index: (abs(selected[index] - native_center), index),
        )
        recentered = replace(
            event, start_frame=center, end_frame=center, evidence_frames=(center,),
        )
        output.append(recentered)
        if recentered != event:
            count += 1
    return tuple(output), count


def recenter_degenerate_boundary_action_events(
    events, plan: dict, selected_native_indices: Sequence[int],
):
    """Use a clip center only when boundary clipping makes time empty.

    A BEFORE query cannot use the lower edge of an action receptive field
    when that field is clipped at native frame zero; likewise an AFTER query
    cannot use an upper edge clipped at the final frame.  In only those two
    degenerate cases, treat the classifier's receptive-field center as its
    point estimate.  This rule depends on sampling geometry and parsed target
    semantics, never on an answer or execution outcome.
    """
    temporal = str(plan.get("temporal_operator", "VIDEO")).strip().upper()
    last = int(plan.get("source_frame_count", 0)) - 1
    if temporal not in {"BEFORE", "AFTER"} or last < 0:
        return tuple(events), 0
    eligible_slots = set()
    for obligation in plan.get("action_obligations", ()):
        view = action_localization_native_view(obligation)
        if not view:
            continue
        if (temporal == "BEFORE" and min(view) == 0) or (
            temporal == "AFTER" and max(view) == last
        ):
            eligible_slots.add(str(obligation["slot_id"]))
    if not eligible_slots:
        return tuple(events), 0
    restricted = {
        **plan,
        "action_obligations": [
            row for row in plan.get("action_obligations", ())
            if str(row["slot_id"]) in eligible_slots
        ],
    }
    return recenter_action_anchor_events(events, restricted, selected_native_indices)


def rebind_nested_action_patients(events, tracks, semantic):
    """Bind ``ToAction`` patients to their nested reference entity.

    The compact target semantics represents ``action_reference(verb, query)``
    as an ACTION node whose first child is the verb literal and whose second
    child is the entity-producing query.  This function follows that typed
    edge; it never consults an answer or a functional program.
    """

    by_id = {row.slot_id: row for row in semantic.slots}
    track_by_id = {row.track_id: row for row in tracks}

    def descendants(slot_id: str) -> frozenset[str]:
        output = {slot_id}
        for child in by_id[slot_id].children:
            output.update(descendants(child))
        return frozenset(output)

    action_by_literal = {
        row.children[0]: row
        for row in semantic.slots
        if row.kind == "ACTION" and len(row.children) >= 2
    }
    rebound = []
    count = 0
    for event in events:
        action = next((
            action_by_literal[slot_id] for slot_id in event.semantic_slot_ids
            if slot_id in action_by_literal
        ), None)
        if action is None:
            rebound.append(event)
            continue
        reference_slots = descendants(action.children[1])
        references = []
        for candidate in events:
            if candidate.event_id == event.event_id or not (
                set(candidate.semantic_slot_ids) & reference_slots
            ):
                continue
            role_map = candidate.role_map
            track_id = next((role_map.get(name) for name in (
                "relation_object", "patient", "theme", "destination", "instrument",
            ) if role_map.get(name)), None)
            if track_id in track_by_id:
                references.append((float(candidate.confidence), str(track_id)))
        if not references:
            rebound.append(event)
            continue
        reference_id = max(references)[1]
        label = track_by_id[reference_id].canonical_label
        compatible = []
        for track in tracks:
            if track.canonical_label != label:
                continue
            in_window = [
                frame for frame in track.evidence_frames
                if event.start_frame <= frame <= event.end_frame
            ]
            if in_window:
                compatible.append((float(track.confidence), track.track_id))
        if not compatible:
            rebound.append(event)
            continue
        selected = max(compatible)[1]
        roles = tuple(
            (name, selected if name in {"patient", "theme"} else track_id)
            for name, track_id in event.roles
        )
        if roles != event.roles:
            event = replace(event, roles=roles)
            count += 1
        rebound.append(event)
    return tuple(rebound), count


__all__ = [
    "action_localization_native_view", "project_native_window_strict",
    "rebind_nested_action_patients", "recenter_action_anchor_events",
    "recenter_degenerate_boundary_action_events",
    "strict_track_for_label",
]
