"""Compact, answer-free text view of one frozen CLEVRER NS-DR graph."""

from __future__ import annotations

from typing import Any


def _event(row: dict[str, Any]) -> str:
    objects = ",".join(str(x) for x in row.get("object", ()))
    return f"{row.get('type')}({objects})@{int(row.get('frame', -1))}"


def compact_event_graph(explicit_executor: Any) -> str:
    """Serialize shared neural facts, without question/program/answer fields."""

    sim = explicit_executor.sim
    lines = ["OBJECTS:"]
    for object_id in sorted(explicit_executor.all_objs):
        attrs = sim.get_static_attrs(object_id)
        lines.append(
            f"- id={object_id} color={attrs['color']} material={attrs['material']} "
            f"shape={attrs['shape']}"
        )
    existing = [row for row in explicit_executor.existing_events if row.get("type") not in {"start", "end"}]
    unseen = list(explicit_executor.unseens)
    lines.append("OBSERVED_EVENTS: " + ("; ".join(_event(x) for x in existing) or "none"))
    lines.append("PREDICTED_FUTURE_EVENTS: " + ("; ".join(_event(x) for x in unseen) or "none"))
    for removed_id, events in sorted(sim.cf_events.items()):
        values = [row for row in events if row.get("type") not in {"start", "end"}]
        lines.append(
            f"COUNTERFACTUAL_REMOVE_{removed_id}: "
            + ("; ".join(_event(x) for x in values) or "none")
        )
    # State rows are derived from the same predicted trajectories.  Include all
    # available five-frame samples so public temporal anchors remain answerable.
    observed = next(row for row in sim.preds if int(row["what_if"]) == -1)
    lines.append("MOTION_STATES (frame:id=moving|stationary):")
    for index, frame in enumerate(observed["trajectory"]):
        values = []
        for obj in frame.get("objects", ()):
            object_id = int(obj["id"])
            moving = bool(sim.is_moving(object_id, ann_idx=index))
            values.append(f"{object_id}={'moving' if moving else 'stationary'}")
        lines.append(f"- {int(frame['frame_index'])}:" + ",".join(values))
    return "\n".join(lines)


__all__ = ["compact_event_graph"]
