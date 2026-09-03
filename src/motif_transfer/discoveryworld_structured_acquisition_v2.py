"""Outcome-blind target-native acquisition grounding for Proteomics Easy."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .discoveryworld_policy import native_action_from_decision
from .phase3_discoveryworld_grounding import exact_action_catalog
from .phase3_discoveryworld_transfer import outcome_blind_target_native_facts


ACQUISITION_SYSTEM_PROMPT = """You are a target-native DiscoveryWorld acquisition grounder.
The payload contains one exact phase3_acquisition_action_catalog entry compiled
from the task text, current native affordances, and raw instrument-evidence
coverage. Copy its action and arguments exactly. This is grounding, not free-form
planning. Never use a score, task completion flag, evaluator, source game, or
formal outcome. Return exactly one JSON object:
{"action":"...","arg1":...,"arg2":...,"reason":"..."}
Omit arguments not present in the catalog action. No prose or Markdown."""

_STATUE_RE = re.compile(r"^statue of (?:a |an )?(?P<subject>.+)$", re.IGNORECASE)


def _object_rows(observation) -> tuple[Mapping[str, Any], ...]:
    return tuple(exact_action_catalog(observation).get("object_facts") or ())


def _find_named(rows, predicate):
    matches = [row for row in rows if predicate(str(row.get("name") or "").lower())]
    return sorted(matches, key=lambda row: (str(row.get("name") or ""), int(row["uuid"])))


def phase3_acquisition_action_catalog(
    observation, *, measured_subjects: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    """Compile exactly one legal next acquisition action, or empty when ready."""

    native = exact_action_catalog(observation)
    objects = _object_rows(observation)
    inventory = set(native.get("inventory_uuids") or ())
    accessible = set(native.get("accessible_uuids") or ())
    visible = set(native.get("visible_uuids") or ())
    meters = _find_named(objects, lambda name: "proteomics meter" in name)
    flags = _find_named(objects, lambda name: "flag" in name)
    statues = _find_named(objects, lambda name: _STATUE_RE.match(name) is not None)
    subjects = sorted({
        _STATUE_RE.match(str(row.get("name") or "").lower()).group("subject")
        for row in statues
        if _STATUE_RE.match(str(row.get("name") or "").lower()) is not None
    })
    specimens = {
        subject: _find_named(objects, lambda name, subject=subject: name == subject)
        for subject in subjects
    }
    if not meters or not flags or len(subjects) != 3:
        raise ValueError("target-native Proteomics acquisition objects are incomplete")
    meter_uuid = int(meters[0]["uuid"])
    flag_uuid = int(flags[0]["uuid"])

    proposal: dict[str, Any]
    stage: str
    if meter_uuid not in inventory:
        if meter_uuid in accessible:
            proposal = {"action": "PICKUP", "arg1": meter_uuid}
            stage = "ACQUIRE_INSTRUMENT"
        elif meter_uuid in visible:
            proposal = {"action": "TELEPORT_TO_OBJECT", "arg1": meter_uuid}
            stage = "LOCALIZE_INSTRUMENT"
        else:
            raise ValueError("proteomics meter is not currently groundable")
    else:
        measured = {str(value).lower() for value in measured_subjects}
        missing = [subject for subject in subjects if subject not in measured]
        if missing:
            subject = missing[0]
            if not specimens[subject]:
                raise ValueError(f"specimen is not visible: {subject}")
            specimen_uuid = int(specimens[subject][0]["uuid"])
            if specimen_uuid in accessible:
                proposal = {
                    "action": "USE", "arg1": meter_uuid, "arg2": specimen_uuid,
                }
                stage = "MEASURE_MISSING_SUBJECT"
            elif specimen_uuid in visible:
                proposal = {"action": "TELEPORT_TO_OBJECT", "arg1": specimen_uuid}
                stage = "LOCALIZE_MISSING_SUBJECT"
            else:
                raise ValueError(f"specimen is not currently groundable: {subject}")
        elif flag_uuid not in inventory:
            if flag_uuid in accessible:
                proposal = {"action": "PICKUP", "arg1": flag_uuid}
                stage = "ACQUIRE_COMMIT_SUBJECT"
            elif flag_uuid in visible:
                proposal = {"action": "TELEPORT_TO_OBJECT", "arg1": flag_uuid}
                stage = "LOCALIZE_COMMIT_SUBJECT"
            else:
                raise ValueError("flag is not currently groundable")
        else:
            return ()
    action = native_action_from_decision(proposal, observation)
    return ({
        "stage": stage,
        "action": action,
        "measured_subjects": sorted({str(value) for value in measured_subjects}),
        "formal_outcome_fields_visible": False,
    },)


def acquisition_prompt_payload(
    observation, *, measured_subjects: Sequence[str],
    schema_error: str | None = None,
) -> Mapping[str, Any]:
    catalog = phase3_acquisition_action_catalog(
        observation, measured_subjects=measured_subjects,
    )
    payload = {
        "target_native_facts": outcome_blind_target_native_facts(observation),
        "phase3_acquisition_action_catalog": list(catalog),
        "phase3_acquisition_catalog_is_exhaustive": True,
        "formal_outcome_fields_visible": False,
        "source_program_visible": False,
    }
    if schema_error:
        payload["previous_response_rejected"] = str(schema_error)[:1000]
    return payload


def call_structured_acquisition_grounder(
    *, backend, observation, measured_subjects: Sequence[str], attempts: int,
) -> tuple[Mapping[str, Any], str, list[Mapping[str, Any]], bool]:
    """Neurally copy one compiled action; fail visibly to symbolic fallback."""

    expected = phase3_acquisition_action_catalog(
        observation, measured_subjects=measured_subjects,
    )
    if len(expected) != 1:
        raise ValueError("acquisition grounder called at structurally ready state")
    expected_action = dict(expected[0]["action"])
    audit = []
    schema_error = None
    for attempt in range(attempts):
        payload = acquisition_prompt_payload(
            observation, measured_subjects=measured_subjects,
            schema_error=schema_error,
        )
        raw = backend.complete("acquisition", ACQUISITION_SYSTEM_PROMPT, payload)
        usage = dict(backend.last_usage or {})
        try:
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise TypeError("acquisition response must be an object")
            action = native_action_from_decision(value, observation)
            if dict(action) != expected_action:
                raise ValueError("action must exactly copy acquisition catalog")
            audit.append({
                "attempt": attempt + 1, "accepted": True,
                "cache_hit": bool(usage.get("cache_hit")),
                "formal_outcome_fields_visible": False,
            })
            return action, raw, audit, False
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
            audit.append({
                "attempt": attempt + 1, "accepted": False,
                "error": schema_error,
                "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "cache_hit": bool(usage.get("cache_hit")),
                "formal_outcome_fields_visible": False,
            })
    return expected_action, "", audit, True


__all__ = [
    "ACQUISITION_SYSTEM_PROMPT", "acquisition_prompt_payload",
    "call_structured_acquisition_grounder", "phase3_acquisition_action_catalog",
]
