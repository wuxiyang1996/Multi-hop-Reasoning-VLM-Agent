from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motif_transfer.discoveryworld_normal_transfer import (
    SourceProgramMonitor,
    export_neural_grounder,
    induce_target_only_program,
    positive_binding_count,
    predict_grounding,
    target_grounding_features,
    target_grounding_label,
    trace_conforms,
    typed_trace,
)


REPO = Path(__file__).resolve().parents[1]
SOURCE = json.loads((REPO / "runs/sokoban_goal_acquisition_v1/artifact.json").read_text())


def _facts(*, inventory=(), statue=None):
    salient = []
    if statue is not None:
        salient.append({
            "uuid": 99, "name": statue,
            "relation_from_agent": "north", "distance": 1,
        })
    return {
        "inventory": [
            {"uuid": uuid, "name": name} for uuid, name in inventory
        ],
        "salient_relative_objects": salient,
    }


def _memory():
    return json.dumps({
        "anomaly": "echojelly",
        "measured": {name: [0.1, 0.2] for name in "abcde"},
    })


def _steps():
    empty = _facts()
    tools = _facts(inventory=((1, "proteomics meter"),))
    ready = _facts(inventory=((1, "proteomics meter"), (2, "flag")))
    bound = _facts(
        inventory=((1, "proteomics meter"), (2, "flag")),
        statue="statue of an echojelly",
    )
    return [
        {"action": {"action": "TELEPORT_TO_OBJECT", "arg1": 1}, "action_succeeded": True, "before_target_native_facts": empty, "after_target_native_facts": empty, "memory": "{}"},
        {"action": {"action": "PICKUP", "arg1": 1}, "action_succeeded": True, "before_target_native_facts": empty, "after_target_native_facts": tools, "memory": "{}"},
        {"action": {"action": "TELEPORT_TO_LOCATION", "arg1": "Statue of a echojelly"}, "action_succeeded": True, "before_target_native_facts": ready, "after_target_native_facts": bound, "memory": _memory()},
        {"action": {"action": "DROP", "arg1": 2}, "action_succeeded": True, "before_target_native_facts": bound, "after_target_native_facts": tools, "memory": _memory()},
    ]


def test_normal_trace_has_earlier_acquisition_unique_binding_and_relation():
    steps = _steps()
    labels = [target_grounding_label(step) for step in steps]
    assert labels.count("ACQUISITION_ENTITY") >= 1
    assert labels.count("ACQUISITION_CONTROL") >= 1
    assert labels[-2:] == ["BINDING", "RELATION"]
    assert positive_binding_count(steps[-2], "after") == 1
    assert positive_binding_count(steps[-1]) == 1
    assert trace_conforms(typed_trace(steps, SOURCE), SOURCE)


def test_echojelly_article_normalization_preserves_binding():
    step = _steps()[-2]
    assert step["action"]["arg1"] == "Statue of a echojelly"
    observed = [
        row["name"] for row in step["after_target_native_facts"]["salient_relative_objects"]
        if "statue" in row["name"]
    ]
    assert "statue of an echojelly" in observed
    assert target_grounding_label(step) == "BINDING"


def test_target_only_budget_zero_abstains_and_one_identifies_program():
    sequence = typed_trace(_steps(), SOURCE)
    assert induce_target_only_program([sequence], budget=0)["status"] == (
        "ABSTAIN_NO_COMPLETE_TARGET_TRAJECTORY"
    )
    learned = induce_target_only_program([sequence], budget=1)
    assert learned["status"] == "TARGET_ONLY_PROGRAM_INDUCED"
    assert learned["program"]["binding_operator_type_id"] == sequence[-2]
    assert learned["program"]["relation_operator_type_id"] == sequence[-1]


def test_grounder_rejects_evaluator_fields_and_round_trips():
    step = _steps()[-1]
    with pytest.raises(ValueError, match="outcome field"):
        target_grounding_features({**step, "official_success": True})
    feature_count = len(target_grounding_features(step))
    model = SimpleNamespace(
        classes_=np.asarray(["ACQUISITION_CONTROL", "RELATION"]),
        coefs_=[np.zeros((feature_count, 2))],
        intercepts_=[np.asarray([-10.0, 10.0])],
    )
    artifact = export_neural_grounder(model)
    assert predict_grounding(artifact, step)[0] == "RELATION"


def test_authentic_monitor_accepts_program_and_permuted_fails_closed():
    roles = ["ACQUISITION_CONTROL", "ACQUISITION_ENTITY", "BINDING", "RELATION"]
    authentic = SourceProgramMonitor("authentic_source")
    assert [authentic.authorize(role)[0] for role in roles] == [True] * 4
    assert authentic.phase == "DONE"
    permuted = SourceProgramMonitor("source_permuted")
    assert permuted.authorize("ACQUISITION_CONTROL")[0] is True
    assert permuted.authorize("BINDING")[0] is False
    assert permuted.phase == "ACQUISITION"
