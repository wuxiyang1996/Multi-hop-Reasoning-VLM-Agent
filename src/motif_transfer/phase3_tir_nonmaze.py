"""Phase-3 source-induced typed-effect transfer to non-maze TIR tasks.

The source side is intentionally imported rather than reimplemented.  A TIR
candidate is an opaque, target-native evidence program.  Its successive
wrapper calls induce a belief-state trajectory, so transition 1/4/8 have the
same typed interface as the frozen game programs without pretending that the
pixels themselves change.

This module never calls a model and never sees an image.  It validates target
grounder artifacts, converts outcome-blind program descriptors into typed
effect probabilities, executes the unchanged ``AnonymousAttemptRuntime`` over
those probabilities, and scores already-collected matched forks.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

from .contracts import stable_hash
from .phase3_attempt_runtime import AnonymousAttemptRuntime
from .phase3_source_portfolio import (
    permute_selected_effect_binding,
    select_source_program_portfolio,
)
from .phase3_typed_effect_induction import (
    IMMEDIATE_EFFECT,
    MEDIUM_EFFECT,
    PERSISTENCE_EFFECT,
    SHORT_EFFECT,
    TYPED_EFFECTS,
)


NEURAL_ONLY = "neural_only"
SOURCE_INDUCED = "source_induced"
SOURCE_PERMUTED = "source_permuted"
GENERIC_SCAFFOLD = "generic_scaffold"
TARGET_NATIVE_CEILING = "target_native_ceiling"
CONDITIONS = (
    NEURAL_ONLY,
    SOURCE_INDUCED,
    SOURCE_PERMUTED,
    GENERIC_SCAFFOLD,
    TARGET_NATIVE_CEILING,
)

EFFECT_HORIZONS = {
    IMMEDIATE_EFFECT: 1,
    SHORT_EFFECT: 4,
    MEDIUM_EFFECT: 8,
    PERSISTENCE_EFFECT: 8,
}
ANSWER_SLOTS = tuple("ABCDEF")
ROUTING_CLASSES = (
    "answer", "compare", "count", "identity", "ocr", "ratio", "spatial",
    "verify",
)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def _finite_probability(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    output = float(value)
    if not math.isfinite(output) or not 0.0 <= output <= 1.0:
        raise ValueError(f"{label} is outside [0,1]")
    return output


def _normalized_box(action: Mapping[str, Any]) -> tuple[float, float, float, float]:
    values = action.get("normalized_box")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("TIR action omitted normalized_box")
    if len(values) != 4:
        raise ValueError("normalized_box must contain x,y,w,h")
    x, y, width, height = map(float, values)
    if not all(math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError("normalized_box contains a nonfinite value")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("normalized_box is invalid")
    if x + width > 1.000001 or y + height > 1.000001:
        raise ValueError("normalized_box exceeds the image")
    return x, y, width, height


def _prefix_geometry(
    actions: Sequence[Mapping[str, Any]], horizon: int,
) -> dict[str, float]:
    selected = list(actions[:horizon])
    if len(selected) != horizon:
        raise ValueError(f"candidate omitted transition-{horizon} actions")
    boxes = [_normalized_box(row) for row in selected]
    areas = [width * height for _, _, width, height in boxes]
    centers = [(x + width / 2.0, y + height / 2.0) for x, y, width, height in boxes]
    mean_x = sum(row[0] for row in centers) / len(centers)
    mean_y = sum(row[1] for row in centers) / len(centers)
    dispersion = sum(
        (x - mean_x) ** 2 + (y - mean_y) ** 2 for x, y in centers
    ) / len(centers)

    # Deterministic 16x16 occupancy is a target-interface feature, not an
    # answer-derived label.  It captures program coverage and redundancy.
    occupied: set[tuple[int, int]] = set()
    for x, y, width, height in boxes:
        for gy in range(16):
            cy = (gy + 0.5) / 16.0
            if not y <= cy <= y + height:
                continue
            for gx in range(16):
                cx = (gx + 0.5) / 16.0
                if x <= cx <= x + width:
                    occupied.add((gx, gy))
    tools = [str(row.get("tool") or "") for row in selected]
    return {
        "prefix_mean_area": sum(areas) / len(areas),
        "prefix_min_area": min(areas),
        "prefix_max_area": max(areas),
        "prefix_coverage": len(occupied) / 256.0,
        "prefix_dispersion": dispersion,
        "prefix_unique_box_fraction": len(set(boxes)) / len(boxes),
        "prefix_extract_colors_fraction": tools.count("extract_colors") / len(tools),
        "prefix_zoom_fraction": tools.count("zoom_region") / len(tools),
    }


def candidate_feature_map(
    candidate: Mapping[str, Any], *, effect_type: str,
    image_size: Sequence[int], routing: Mapping[str, Any],
) -> dict[str, float]:
    """Return the fixed, outcome-blind target-grounder feature map."""

    if effect_type not in TYPED_EFFECTS:
        raise ValueError(f"unsupported typed effect: {effect_type}")
    if len(image_size) != 2:
        raise ValueError("TIR image_size must have width and height")
    width, height = map(float, image_size)
    if width <= 0 or height <= 0:
        raise ValueError("TIR image_size is invalid")
    raw = candidate.get("raw_typed_effect_probabilities")
    if not isinstance(raw, Mapping) or set(raw) != set(TYPED_EFFECTS):
        raise ValueError("candidate raw typed effects do not match frozen IR")
    actions = candidate.get("actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise ValueError("candidate omitted target-native action program")
    if len(actions) != 8:
        raise ValueError("TIR Phase-3 candidate must contain exactly 8 actions")
    horizon = EFFECT_HORIZONS[effect_type]
    geometry = _prefix_geometry(actions, horizon)
    classes = {str(value) for value in routing.get("classes") or ()}
    features = {
        "raw_effect_probability": _finite_probability(
            raw[effect_type], label=f"raw {effect_type}",
        ),
        "planner_score": _finite_probability(
            candidate.get("planner_score", 0.0), label="planner_score",
        ),
        "log_image_pixels": math.log1p(width * height) / 20.0,
        "absolute_log_aspect": abs(math.log(width / height)) / 5.0,
        "horizon_fraction": horizon / 8.0,
        **geometry,
    }
    for name in ROUTING_CLASSES:
        features[f"route_{name}"] = float(name in classes)
    return features


def grounder_feature_names() -> tuple[str, ...]:
    dummy = {
        "planner_score": 0.5,
        "raw_typed_effect_probabilities": {name: 0.5 for name in TYPED_EFFECTS},
        "actions": [
            {
                "tool": "zoom_region",
                "normalized_box": [0.0, 0.0, 1.0, 1.0],
            }
            for _ in range(8)
        ],
    }
    return tuple(sorted(candidate_feature_map(
        dummy, effect_type=IMMEDIATE_EFFECT,
        image_size=(100, 100), routing={"classes": []},
    )))


FEATURE_NAMES = grounder_feature_names()

# These values are available only after the selected target-native evidence
# program has actually reached its source-requested horizon.  They therefore
# ground the symbolic HIGH/LOW observation without leaking an unexecuted fork
# or evaluator gold into the controller.
OBSERVATION_FEATURE_NAMES = (
    "answer_changed_from_baseline",
    "answer_changed_from_previous_endpoint",
    "baseline_confidence",
    "contradiction_risk",
    "endpoint_confidence",
    "endpoint_entropy",
    "endpoint_margin",
    "horizon_fraction",
    "local_detail_sufficient",
    "nonredundant_fraction",
    "planner_score",
    "previous_endpoint_confidence",
    "question_coverage",
    "raw_effect_probability",
    "referent_visible",
)

BASELINE_FEATURE_NAMES = (
    "absolute_log_aspect",
    "answer_confidence",
    "baseline_contradiction_risk",
    "baseline_local_detail_sufficient",
    "baseline_question_coverage",
    "baseline_referent_visible",
    "log_image_pixels",
    "maximum_candidate_planner_score",
    "maximum_candidate_raw_effect_probability",
    "mean_candidate_planner_score",
    "mean_candidate_raw_effect_probability",
    "route_answer",
    "route_compare",
    "route_count",
    "route_identity",
    "route_ocr",
    "route_ratio",
    "route_spatial",
    "route_verify",
    "verifier_contradiction_probability",
    "verifier_overview_sufficiency_probability",
    "verifier_support_probability",
)


def target_native_program_bank() -> tuple[dict[str, Any], ...]:
    """Return a fixed basis over the wrapper's native region action space.

    This is candidate enumeration, not a transferred controller: no source
    game, effect type, question answer, or success label chooses the programs.
    A target neural model predicts their typed effects for each image/question,
    while the frozen source program decides which opaque operand to execute.
    """

    def action(tool: str, box: Sequence[float]) -> dict[str, Any]:
        return {"tool": tool, "normalized_box": list(map(float, box))}

    programs = [
        {
            "descriptor": "global-to-local multiscale color evidence",
            "actions": [
                action("extract_colors", (0, 0, 1, 1)),
                action("zoom_region", (0, 0, 1, 1)),
                action("extract_colors", (.125, .125, .75, .75)),
                action("zoom_region", (.25, .25, .5, .5)),
                action("extract_colors", (0, 0, .5, .5)),
                action("extract_colors", (.5, 0, .5, .5)),
                action("extract_colors", (0, .5, .5, .5)),
                action("extract_colors", (.5, .5, .5, .5)),
            ],
        },
        {
            "descriptor": "eight horizontal regional measurements",
            "actions": [
                action("extract_colors" if index % 2 == 0 else "zoom_region",
                       (0, index / 8, 1, 1 / 8))
                for index in range(8)
            ],
        },
        {
            "descriptor": "eight vertical regional measurements",
            "actions": [
                action("extract_colors" if index % 2 == 0 else "zoom_region",
                       (index / 8, 0, 1 / 8, 1))
                for index in range(8)
            ],
        },
        {
            "descriptor": "two-by-four local grid measurements",
            "actions": [
                action(
                    "extract_colors" if index % 2 == 0 else "zoom_region",
                    ((index % 4) / 4, (index // 4) / 2, 1 / 4, 1 / 2),
                )
                for index in range(8)
            ],
        },
    ]
    output = []
    for row in programs:
        body = {
            "schema_version": "phase3-tir-target-native-evidence-program-v1",
            "descriptor": row["descriptor"],
            "actions": row["actions"],
        }
        output.append(body | {"candidate_id": stable_hash(body)})
    if len({row["candidate_id"] for row in output}) != 4:
        raise RuntimeError("target-native TIR program bank is not unique")
    return tuple(output)


def validate_grounder_artifact(artifact: Mapping[str, Any]) -> None:
    body = dict(artifact)
    claimed = str(body.pop("artifact_sha256", ""))
    if not claimed or stable_hash(body) != claimed:
        raise ValueError("TIR grounder artifact hash mismatch")
    if artifact.get("schema_version") != "phase3-tir-nonmaze-grounder-v2":
        raise ValueError("unsupported TIR grounder artifact schema")
    if artifact.get("formal_outcome_read_for_training_or_calibration") is not False:
        raise ValueError("TIR grounder does not attest formal-outcome isolation")
    if artifact.get("source_program_updated") is not False:
        raise ValueError("TIR grounder changed the frozen source program")
    heads = artifact.get("heads")
    if not isinstance(heads, Mapping) or set(heads) != set(TYPED_EFFECTS):
        raise ValueError("TIR grounder heads do not match frozen typed effects")
    for effect_type, head in heads.items():
        if tuple(head.get("feature_names") or ()) != FEATURE_NAMES:
            raise ValueError(f"{effect_type} grounder feature schema mismatch")
        for key in ("means", "scales", "weights"):
            values = head.get(key)
            if not isinstance(values, Sequence) or len(values) != len(FEATURE_NAMES):
                raise ValueError(f"{effect_type} grounder {key} mismatch")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{effect_type} grounder {key} is nonfinite")
        if not math.isfinite(float(head.get("intercept"))):
            raise ValueError(f"{effect_type} grounder intercept is nonfinite")
    observation_head = artifact.get("observation_head")
    if not isinstance(observation_head, Mapping):
        raise ValueError("TIR grounder omitted observed-effect head")
    if tuple(observation_head.get("feature_names") or ()) != OBSERVATION_FEATURE_NAMES:
        raise ValueError("TIR observed-effect feature schema mismatch")
    for key in ("means", "scales", "weights"):
        values = observation_head.get(key)
        if (
            not isinstance(values, Sequence)
            or len(values) != len(OBSERVATION_FEATURE_NAMES)
            or not all(math.isfinite(float(value)) for value in values)
        ):
            raise ValueError(f"TIR observed-effect {key} mismatch")
    if not math.isfinite(float(observation_head.get("intercept"))):
        raise ValueError("TIR observed-effect intercept is nonfinite")
    baseline_head = artifact.get("baseline_head")
    if not isinstance(baseline_head, Mapping):
        raise ValueError("TIR grounder omitted baseline head")
    if baseline_head.get("calibration") == "DIRECT_INDEPENDENT_VERIFIER_SUPPORT_V1":
        if baseline_head.get("verifier_field") != "support_probability":
            raise ValueError("TIR direct verifier baseline field mismatch")
        if baseline_head.get("target_outcome_used_for_calibration") is not False:
            raise ValueError("TIR direct verifier used target outcome calibration")
    elif "feature_names" in baseline_head:
        if tuple(baseline_head.get("feature_names") or ()) != BASELINE_FEATURE_NAMES:
            raise ValueError("TIR baseline feature schema mismatch")
        for key in ("means", "scales", "weights"):
            values = baseline_head.get(key)
            if (
                not isinstance(values, Sequence)
                or len(values) != len(BASELINE_FEATURE_NAMES)
                or not all(math.isfinite(float(value)) for value in values)
            ):
                raise ValueError(f"TIR baseline {key} mismatch")
        if not math.isfinite(float(baseline_head.get("intercept"))):
            raise ValueError("TIR baseline intercept is nonfinite")
    elif not all(
        math.isfinite(float(baseline_head.get(key)))
        for key in ("slope", "intercept")
    ):
        raise ValueError("legacy TIR baseline calibration is nonfinite")
    for key in (
        "baseline_commit_confidence", "evidence_high_probability",
        "minimum_predicted_advantage",
    ):
        _finite_probability(artifact.get("thresholds", {}).get(key), label=key)


def predict_candidate_effects(
    candidate: Mapping[str, Any], *, artifact: Mapping[str, Any],
    image_size: Sequence[int], routing: Mapping[str, Any],
) -> dict[str, float]:
    validate_grounder_artifact(artifact)
    output = {}
    for effect_type in TYPED_EFFECTS:
        head = artifact["heads"][effect_type]
        features = candidate_feature_map(
            candidate, effect_type=effect_type,
            image_size=image_size, routing=routing,
        )
        vector = [features[name] for name in FEATURE_NAMES]
        standardized = [
            (value - float(mean)) / float(scale)
            for value, mean, scale in zip(vector, head["means"], head["scales"])
        ]
        logit = float(head["intercept"]) + sum(
            float(weight) * value
            for weight, value in zip(head["weights"], standardized)
        )
        output[effect_type] = _sigmoid(logit)
    return output


def _probabilities(row: Mapping[str, Any]) -> dict[str, float]:
    values = row.get("probabilities")
    if not isinstance(values, Mapping):
        raise ValueError("answer row omitted probabilities")
    output = {
        slot: _finite_probability(values.get(slot, 0.0), label=f"probability {slot}")
        for slot in ANSWER_SLOTS
    }
    total = sum(output.values())
    if total <= 0:
        raise ValueError("answer probabilities have zero mass")
    return {key: value / total for key, value in output.items()}


def _answer_confidence(row: Mapping[str, Any]) -> float:
    values = _probabilities(row)
    answer = str(row.get("answer") or "")
    if answer not in ANSWER_SLOTS:
        answer = max(values, key=lambda key: (values[key], key))
    return values[answer]


def baseline_feature_map(receipt: Mapping[str, Any]) -> dict[str, float]:
    """Return outcome-blind features for baseline evidence sufficiency."""

    quality = receipt.get("baseline", {}).get("evidence_quality")
    if not isinstance(quality, Mapping):
        raise ValueError("baseline omitted evidence quality")
    expected_quality = {
        "contradiction_risk", "local_detail_sufficient",
        "question_coverage", "referent_visible",
    }
    if set(quality) != expected_quality:
        raise ValueError("baseline evidence-quality schema mismatch")
    verification = receipt.get("baseline_verification")
    if not isinstance(verification, Mapping):
        raise ValueError("baseline omitted independent verification")
    expected_verification = {
        "contradiction_probability", "overview_sufficiency_probability",
        "support_probability", "reason",
    }
    if set(verification) != expected_verification:
        raise ValueError("baseline verification schema mismatch")
    if len(receipt.get("image_size") or ()) != 2:
        raise ValueError("baseline receipt omitted image size")
    width, height = map(float, receipt["image_size"])
    if width <= 0 or height <= 0:
        raise ValueError("baseline receipt image size is invalid")
    candidates = list(receipt.get("candidates") or ())
    if not candidates:
        raise ValueError("baseline receipt omitted target candidates")
    planner_scores = [
        _finite_probability(row.get("planner_score"), label="planner_score")
        for row in candidates
    ]
    raw_effects = [
        _finite_probability(value, label="candidate raw effect")
        for row in candidates
        for value in (row.get("raw_typed_effect_probabilities") or {}).values()
    ]
    if len(raw_effects) != len(candidates) * len(TYPED_EFFECTS):
        raise ValueError("baseline candidate raw-effect schema mismatch")
    classes = {str(value) for value in receipt["wrapper_routing"].get("classes") or ()}
    features = {
        "absolute_log_aspect": abs(math.log(width / height)) / 5.0,
        "answer_confidence": _answer_confidence(receipt["baseline"]),
        "baseline_contradiction_risk": _finite_probability(
            quality["contradiction_risk"], label="baseline contradiction_risk",
        ),
        "baseline_local_detail_sufficient": _finite_probability(
            quality["local_detail_sufficient"],
            label="baseline local_detail_sufficient",
        ),
        "baseline_question_coverage": _finite_probability(
            quality["question_coverage"], label="baseline question_coverage",
        ),
        "baseline_referent_visible": _finite_probability(
            quality["referent_visible"], label="baseline referent_visible",
        ),
        "log_image_pixels": math.log1p(width * height) / 20.0,
        "maximum_candidate_planner_score": max(planner_scores),
        "maximum_candidate_raw_effect_probability": max(raw_effects),
        "mean_candidate_planner_score": sum(planner_scores) / len(planner_scores),
        "mean_candidate_raw_effect_probability": sum(raw_effects) / len(raw_effects),
        "verifier_contradiction_probability": _finite_probability(
            verification["contradiction_probability"],
            label="verifier contradiction_probability",
        ),
        "verifier_overview_sufficiency_probability": _finite_probability(
            verification["overview_sufficiency_probability"],
            label="verifier overview_sufficiency_probability",
        ),
        "verifier_support_probability": _finite_probability(
            verification["support_probability"],
            label="verifier support_probability",
        ),
    }
    for name in ROUTING_CLASSES:
        features[f"route_{name}"] = float(name in classes)
    if tuple(sorted(features)) != BASELINE_FEATURE_NAMES:
        raise RuntimeError("baseline feature schema drift")
    return features


def predict_baseline_correctness(
    receipt: Mapping[str, Any], *, artifact: Mapping[str, Any],
) -> float:
    """Predict whether target-native baseline evidence is already sufficient."""

    head = artifact["baseline_head"]
    if head.get("calibration") == "DIRECT_INDEPENDENT_VERIFIER_SUPPORT_V1":
        verification = receipt.get("baseline_verification") or {}
        return _finite_probability(
            verification.get("support_probability"),
            label="baseline verifier support_probability",
        )
    if "feature_names" not in head:
        confidence = _answer_confidence(receipt["baseline"])
        epsilon = 1e-6
        logit = math.log(
            max(epsilon, confidence) / max(epsilon, 1.0 - confidence)
        )
        return _sigmoid(
            float(head.get("intercept", 0.0))
            + float(head.get("slope", 1.0)) * logit
        )
    features = baseline_feature_map(receipt)
    standardized = [
        (features[name] - float(mean)) / float(scale)
        for name, mean, scale in zip(
            BASELINE_FEATURE_NAMES, head["means"], head["scales"],
        )
    ]
    return _sigmoid(float(head["intercept"]) + sum(
        float(weight) * value
        for weight, value in zip(head["weights"], standardized)
    ))


def observation_feature_map(
    receipt: Mapping[str, Any], candidate: Mapping[str, Any], *,
    effect_type: str,
) -> dict[str, float]:
    """Describe one *executed* endpoint without reading its correctness.

    H4 may use H1 and H8 may use H4 because those states are prefixes of the
    same executed program.  No endpoint from another candidate is visible.
    """

    if effect_type not in TYPED_EFFECTS:
        raise ValueError(f"unsupported observed effect: {effect_type}")
    horizon = EFFECT_HORIZONS[effect_type]
    endpoint = _endpoint(candidate, horizon)
    endpoint_probabilities = _probabilities(endpoint)
    ordered = sorted(endpoint_probabilities.values(), reverse=True)
    entropy = -sum(
        value * math.log(max(value, 1e-12))
        for value in endpoint_probabilities.values()
    ) / math.log(len(ANSWER_SLOTS))
    if horizon == 1:
        previous = receipt["baseline"]
    elif horizon == 4:
        previous = _endpoint(candidate, 1)
    else:
        previous = _endpoint(candidate, 4)
    quality = endpoint.get("evidence_quality")
    if not isinstance(quality, Mapping):
        raise ValueError("executed endpoint omitted neural evidence quality")
    required_quality = {
        "contradiction_risk", "local_detail_sufficient",
        "question_coverage", "referent_visible",
    }
    if set(quality) != required_quality:
        raise ValueError("executed endpoint evidence-quality schema mismatch")
    transitions = candidate.get("transitions") or ()
    prefix = list(transitions[:horizon])
    if len(prefix) != horizon:
        raise ValueError("candidate omitted executed transition prefix")
    nonredundant = sum(
        bool((row.get("effect") or {}).get("nonredundant")) for row in prefix
    ) / horizon
    raw = candidate.get("raw_typed_effect_probabilities") or {}
    features = {
        "answer_changed_from_baseline": float(
            str(endpoint["answer"]) != str(receipt["baseline"]["answer"])
        ),
        "answer_changed_from_previous_endpoint": float(
            str(endpoint["answer"]) != str(previous["answer"])
        ),
        "baseline_confidence": _answer_confidence(receipt["baseline"]),
        "contradiction_risk": _finite_probability(
            quality["contradiction_risk"], label="contradiction_risk",
        ),
        "endpoint_confidence": _answer_confidence(endpoint),
        "endpoint_entropy": entropy,
        "endpoint_margin": ordered[0] - ordered[1],
        "horizon_fraction": horizon / 8.0,
        "local_detail_sufficient": _finite_probability(
            quality["local_detail_sufficient"],
            label="local_detail_sufficient",
        ),
        "nonredundant_fraction": nonredundant,
        "planner_score": _finite_probability(
            candidate.get("planner_score", 0.0), label="planner_score",
        ),
        "previous_endpoint_confidence": _answer_confidence(previous),
        "question_coverage": _finite_probability(
            quality["question_coverage"], label="question_coverage",
        ),
        "raw_effect_probability": _finite_probability(
            raw.get(effect_type), label=f"raw {effect_type}",
        ),
        "referent_visible": _finite_probability(
            quality["referent_visible"], label="referent_visible",
        ),
    }
    if tuple(sorted(features)) != OBSERVATION_FEATURE_NAMES:
        raise RuntimeError("observed-effect feature schema drift")
    return features


def predict_observed_effect(
    receipt: Mapping[str, Any], candidate: Mapping[str, Any], *,
    effect_type: str, artifact: Mapping[str, Any],
) -> float:
    """Predict HIGH after, and only after, a target intervention executes."""

    validate_grounder_artifact(artifact)
    features = observation_feature_map(
        receipt, candidate, effect_type=effect_type,
    )
    head = artifact["observation_head"]
    vector = [features[name] for name in OBSERVATION_FEATURE_NAMES]
    standardized = [
        (value - float(mean)) / float(scale)
        for value, mean, scale in zip(vector, head["means"], head["scales"])
    ]
    return _sigmoid(float(head["intercept"]) + sum(
        float(weight) * value
        for weight, value in zip(head["weights"], standardized)
    ))


def _endpoint(candidate: Mapping[str, Any], horizon: int) -> Mapping[str, Any]:
    row = (candidate.get("endpoints") or {}).get(str(horizon))
    if not isinstance(row, Mapping):
        raise ValueError(f"candidate omitted H{horizon} endpoint")
    if str(row.get("answer") or "") not in ANSWER_SLOTS:
        raise ValueError(f"candidate H{horizon} answer is not A--F")
    _probabilities(row)
    return row


def _baseline_probability(receipt: Mapping[str, Any]) -> float:
    """Calibrated target-native baseline reliability estimate."""

    confidence = _answer_confidence(receipt["baseline"])
    calibration = receipt.get("baseline_calibration") or {}
    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    epsilon = 1e-6
    logit = math.log(max(epsilon, confidence) / max(epsilon, 1.0 - confidence))
    return _sigmoid(intercept + slope * logit)


def attach_grounding(
    receipt: Mapping[str, Any], *, artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach predictions while proving no answer label enters the grounder."""

    validate_grounder_artifact(artifact)
    candidates = []
    for candidate in receipt.get("candidates") or ():
        row = dict(candidate)
        row["typed_effect_probabilities"] = predict_candidate_effects(
            row, artifact=artifact, image_size=receipt["image_size"],
            routing=receipt["wrapper_routing"],
        )
        candidates.append(row)
    output = dict(receipt)
    output["candidates"] = candidates
    baseline_head = artifact.get("baseline_head") or {}
    output["baseline_calibration"] = {
        "schema": (
            str(baseline_head.get("calibration"))
            if baseline_head.get("calibration") else
            "TARGET_NATIVE_VERIFIER_FEATURE_HEAD_V1"
            if "feature_names" in baseline_head else
            "LEGACY_CONFIDENCE_LOGIT"
        ),
        "artifact_bound": True,
    }
    output["baseline_predicted_correctness"] = predict_baseline_correctness(
        receipt, artifact=artifact,
    )
    grounding_body = {
        "sample_id": str(receipt["sample_id"]),
        "grounder_artifact_sha256": artifact["artifact_sha256"],
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "candidate_effects": [row["typed_effect_probabilities"] for row in candidates],
        "baseline_predicted_correctness": output["baseline_predicted_correctness"],
        "target_outcome_read": False,
        "source_identity_used_as_feature": False,
    }
    output["target_grounding_receipt"] = grounding_body | {
        "target_grounding_sha256": stable_hash(grounding_body),
    }
    return output


def _neural_decision(
    receipt: Mapping[str, Any], *, artifact: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = receipt["baseline"]
    thresholds = artifact["thresholds"]
    if float(receipt["baseline_predicted_correctness"]) >= float(
        thresholds["baseline_commit_confidence"]
    ):
        return {
            "answer": str(baseline["answer"]),
            "candidate_id": None,
            "horizon": 0,
            "attempts": [],
            "reason": "TARGET_NATIVE_CONFIDENT_COMMIT",
        }
    rows = []
    for candidate in receipt["candidates"]:
        effects = candidate["typed_effect_probabilities"]
        for effect_type in (IMMEDIATE_EFFECT, SHORT_EFFECT, MEDIUM_EFFECT):
            rows.append((
                float(effects[effect_type]),
                -EFFECT_HORIZONS[effect_type],
                str(candidate["candidate_id"]),
                effect_type,
                candidate,
            ))
    best = max(rows)
    predicted, _, candidate_id, effect_type, candidate = best
    advantage = predicted - float(receipt["baseline_predicted_correctness"])
    if advantage < float(thresholds["minimum_predicted_advantage"]):
        return {
            "answer": str(baseline["answer"]),
            "candidate_id": None,
            "horizon": 0,
            "attempts": [],
            "reason": "TARGET_NATIVE_ADVANTAGE_ABSTENTION",
            "predicted_advantage": advantage,
        }
    horizon = EFFECT_HORIZONS[effect_type]
    endpoint = _endpoint(candidate, horizon)
    return {
        "answer": str(endpoint["answer"]),
        "candidate_id": candidate_id,
        "horizon": horizon,
        "effect_type": effect_type,
        "attempts": [{"candidate_id": candidate_id, "horizon": horizon}],
        "reason": "TARGET_NATIVE_NEURAL_TYPED_EFFECT",
        "predicted_advantage": advantage,
    }


def _generic_decision(
    receipt: Mapping[str, Any], *, artifact: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = receipt["baseline"]
    if float(receipt["baseline_predicted_correctness"]) >= float(
        artifact["thresholds"]["baseline_commit_confidence"]
    ):
        return {
            "answer": str(baseline["answer"]), "candidate_id": None,
            "horizon": 0, "attempts": [],
            "reason": "TARGET_NATIVE_CONFIDENT_COMMIT",
        }
    candidate = max(receipt["candidates"], key=lambda row: (
        sum(float(row["typed_effect_probabilities"][name]) for name in TYPED_EFFECTS)
        / len(TYPED_EFFECTS),
        str(row["candidate_id"]),
    ))
    # This fixed source-free scaffold has a four-transition gather/commit
    # schedule.  It is a control, never the transferred operator.
    endpoint = _endpoint(candidate, 4)
    return {
        "answer": str(endpoint["answer"]),
        "candidate_id": str(candidate["candidate_id"]),
        "horizon": 4,
        "effect_type": "SOURCE_FREE_MEAN_EFFECT",
        "attempts": [{"candidate_id": str(candidate["candidate_id"]), "horizon": 4}],
        "reason": "GENERIC_SOURCE_FREE_FOUR_STEP_SCAFFOLD",
    }


def _source_decision(
    receipt: Mapping[str, Any], *, artifact: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]], permuted: bool,
) -> dict[str, Any]:
    baseline = receipt["baseline"]
    if float(receipt["baseline_predicted_correctness"]) >= float(
        artifact["thresholds"]["baseline_commit_confidence"]
    ):
        return {
            "answer": str(baseline["answer"]), "candidate_id": None,
            "horizon": 0, "attempts": [], "source_admitted": False,
            "reason": "TARGET_NATIVE_CONFIDENT_COMMIT",
        }
    candidates = list(receipt["candidates"])
    ids = [str(row["candidate_id"]) for row in candidates]
    effects = [dict(row["typed_effect_probabilities"]) for row in candidates]
    grounding_sha = receipt["target_grounding_receipt"]["target_grounding_sha256"]
    portfolio = select_source_program_portfolio(
        source_artifacts, candidate_ids=ids, candidate_effects=effects,
        target_grounding_sha256=grounding_sha,
    )
    selected_sha = portfolio["selected_artifact_sha256"]
    if selected_sha is None:
        fallback = _neural_decision(receipt, artifact=artifact)
        return fallback | {
            "source_admitted": False,
            "reason": "SOURCE_PORTFOLIO_ABSTAINED_TO_NEURAL",
            "portfolio_receipt": portfolio,
        }
    source = next(
        row for row in source_artifacts if row["artifact_sha256"] == selected_sha
    )
    bound_effects = effects
    control = None
    if permuted:
        bound_effects, control = permute_selected_effect_binding(
            source["typed_effect_program"], candidate_ids=ids,
            candidate_effects=effects,
        )
        bound_effects = list(bound_effects)
    effect_type = str(source["typed_effect_program"]["selected_effect_type"])
    horizon = EFFECT_HORIZONS[effect_type]
    maximum = max(float(row[effect_type]) for row in bound_effects)
    advantage = maximum - float(receipt["baseline_predicted_correctness"])
    if advantage < float(artifact["thresholds"]["minimum_predicted_advantage"]):
        fallback = _neural_decision(receipt, artifact=artifact)
        return fallback | {
            "source_admitted": False,
            "reason": "TARGET_NATIVE_SOURCE_APPLICABILITY_ABSTENTION",
            "selected_effect_type": effect_type,
            "selected_source_artifact_sha256": selected_sha,
            "predicted_advantage": advantage,
            "portfolio_receipt": portfolio,
            "effect_binding_control_receipt": control,
        }

    runtime = AnonymousAttemptRuntime(
        artifact=source, candidate_ids=ids, candidate_effects=bound_effects,
        target_grounding_sha256=grounding_sha,
    )
    decision = runtime.start()
    by_id = {str(row["candidate_id"]): row for row in candidates}
    attempts = []
    budget = 8
    current_answer = str(baseline["answer"])
    while decision.kind == "TRIAL" and decision.candidate_id is not None:
        if horizon > budget:
            break
        candidate = by_id[decision.candidate_id]
        endpoint = _endpoint(candidate, horizon)
        current_answer = str(endpoint["answer"])
        confidence = _answer_confidence(endpoint)
        observed_effect_probability = predict_observed_effect(
            receipt, candidate, effect_type=effect_type, artifact=artifact,
        )
        attempts.append({
            "candidate_id": decision.candidate_id,
            "horizon": horizon,
            "answer": current_answer,
            "confidence": confidence,
            "observed_effect_probability": observed_effect_probability,
        })
        budget -= horizon
        effect = (
            "HIGH" if observed_effect_probability >= float(
                artifact["thresholds"]["evidence_high_probability"]
            ) else "LOW"
        )
        decision = runtime.observe(effect)
        if decision.kind == "TRIAL" and horizon > budget:
            break
    first = attempts[0]["candidate_id"] if attempts else None
    return {
        "answer": current_answer,
        "candidate_id": first,
        "horizon": sum(int(row["horizon"]) for row in attempts),
        "attempts": attempts,
        "source_admitted": bool(attempts),
        "reason": (
            "SOURCE_INDUCED_RUNTIME" if attempts
            else "SOURCE_RUNTIME_BUDGET_ABSTENTION"
        ),
        "runtime_terminal_decision": decision.kind,
        "selected_effect_type": effect_type,
        "selected_program_sha256": source["typed_effect_program"]["program_sha256"],
        "selected_source_artifact_sha256": selected_sha,
        "predicted_advantage": advantage,
        "portfolio_receipt": portfolio,
        "effect_binding_control_receipt": control,
    }


def _ceiling_decision(receipt: Mapping[str, Any]) -> dict[str, Any]:
    gold = str(receipt["gold_answer"])
    rows = [(receipt["baseline"], None, 0)]
    for candidate in receipt["candidates"]:
        for horizon in (1, 4, 8):
            rows.append((_endpoint(candidate, horizon), candidate["candidate_id"], horizon))
    selected = next((row for row in rows if str(row[0]["answer"]) == gold), rows[0])
    return {
        "answer": str(selected[0]["answer"]),
        "candidate_id": selected[1],
        "horizon": selected[2],
        "attempts": [],
        "reason": "EVALUATOR_ONLY_TARGET_NATIVE_FORK_CEILING",
        "formal_outcome_used_for_policy": True,
    }


def execute_condition(
    receipt: Mapping[str, Any], *, condition: str,
    grounder_artifact: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported TIR Phase-3 condition: {condition}")
    grounded = attach_grounding(receipt, artifact=grounder_artifact)
    if condition == NEURAL_ONLY:
        decision = _neural_decision(grounded, artifact=grounder_artifact)
    elif condition == GENERIC_SCAFFOLD:
        decision = _generic_decision(grounded, artifact=grounder_artifact)
    elif condition == SOURCE_INDUCED:
        decision = _source_decision(
            grounded, artifact=grounder_artifact,
            source_artifacts=source_artifacts, permuted=False,
        )
    elif condition == SOURCE_PERMUTED:
        decision = _source_decision(
            grounded, artifact=grounder_artifact,
            source_artifacts=source_artifacts, permuted=True,
        )
    else:
        decision = _ceiling_decision(grounded)
    body = {
        "condition": condition,
        "sample_id": str(receipt["sample_id"]),
        "answer": decision["answer"],
        "success": decision["answer"] == str(receipt["gold_answer"]),
        "decision": decision,
        "grounder_artifact_sha256": grounder_artifact["artifact_sha256"],
        "source_ir_implementation": (
            "motif_transfer.phase3_attempt_runtime.AnonymousAttemptRuntime"
            if condition in {SOURCE_INDUCED, SOURCE_PERMUTED} else None
        ),
    }
    return body | {"condition_receipt_sha256": stable_hash(body)}


def _paired(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, int]:
    wins = losses = ties = 0
    for row in rows:
        left_value = bool(row["conditions"][left]["success"])
        right_value = bool(row["conditions"][right]["success"])
        if left_value and not right_value:
            wins += 1
        elif right_value and not left_value:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def evaluate_matched_receipts(
    receipts: Sequence[Mapping[str, Any]], *,
    grounder_artifact: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any], role: str,
) -> dict[str, Any]:
    validate_grounder_artifact(grounder_artifact)
    if len({str(row["sample_id"]) for row in receipts}) != len(receipts):
        raise ValueError("duplicate TIR sample receipts")
    rows = []
    for receipt in receipts:
        if receipt.get("formal_outcome_exposed_to_neural_calls") is not False:
            raise ValueError("TIR receipt does not attest outcome-blind neural calls")
        conditions = {
            condition: execute_condition(
                receipt, condition=condition,
                grounder_artifact=grounder_artifact,
                source_artifacts=source_artifacts,
            )
            for condition in CONDITIONS
        }
        rows.append({"sample_id": str(receipt["sample_id"]), "conditions": conditions})
    successes = {
        condition: sum(bool(row["conditions"][condition]["success"]) for row in rows)
        for condition in CONDITIONS
    }
    source_neural = _paired(rows, SOURCE_INDUCED, NEURAL_ONLY)
    source_permuted = _paired(rows, SOURCE_INDUCED, SOURCE_PERMUTED)
    source_generic = _paired(rows, SOURCE_INDUCED, GENERIC_SCAFFOLD)
    source_effects = Counter()
    selected_programs = Counter()
    source_changes = 0
    permuted_contrasts = 0
    source_admissions = 0
    for row in rows:
        source = row["conditions"][SOURCE_INDUCED]["decision"]
        permuted = row["conditions"][SOURCE_PERMUTED]["decision"]
        neural = row["conditions"][NEURAL_ONLY]["decision"]
        if source.get("source_admitted"):
            source_admissions += 1
            source_effects[str(source.get("selected_effect_type"))] += 1
            selected_programs[str(source.get("selected_program_sha256"))] += 1
        source_signature = (
            source.get("candidate_id"), source.get("horizon"), source.get("answer")
        )
        neural_signature = (
            neural.get("candidate_id"), neural.get("horizon"), neural.get("answer")
        )
        permuted_signature = (
            permuted.get("candidate_id"), permuted.get("horizon"), permuted.get("answer")
        )
        source_changes += source_signature != neural_signature
        permuted_contrasts += source_signature != permuted_signature
    negative = source_neural["losses"]
    expected = int(gates["expected_tasks"])
    checks = {
        "expected_task_count": len(rows) == expected,
        "target_native_ceiling_capable": successes[TARGET_NATIVE_CEILING]
        >= int(gates["minimum_ceiling_successes"]),
        "source_changes_target_policy": source_changes
        >= int(gates["minimum_source_action_contrasts"]),
        "authentic_differs_from_permuted": permuted_contrasts
        >= int(gates["minimum_permuted_action_contrasts"]),
        "multiple_source_effect_types_selected": len(source_effects)
        >= int(gates["minimum_selected_effect_types"]),
        "maximum_negative_transfer": (
            negative / max(1, len(rows))
            <= float(gates["maximum_negative_transfer_rate"])
        ),
        "source_not_below_neural": source_neural["losses"] == 0,
        "source_strictly_beats_neural": source_neural["wins"] > 0,
        "source_strictly_beats_permuted": (
            successes[SOURCE_INDUCED] > successes[SOURCE_PERMUTED]
        ),
        "source_strictly_beats_generic": (
            successes[SOURCE_INDUCED] > successes[GENERIC_SCAFFOLD]
        ),
    }
    required = list(gates.get("required_gate_names") or checks)
    unknown = sorted(set(required) - set(checks))
    if unknown:
        raise ValueError(f"unknown required TIR gates: {unknown}")
    status = (
        f"TIR_PHASE3_{str(role).upper()}_PASSED"
        if all(checks[name] for name in required)
        else f"TIR_PHASE3_{str(role).upper()}_FAILED"
    )
    body = {
        "schema_version": "phase3-tir-nonmaze-matched-report-v1",
        "status": status,
        "role": str(role),
        "tasks": len(rows),
        "successes": successes,
        "paired": {
            "source_vs_neural": source_neural,
            "source_vs_permuted": source_permuted,
            "source_vs_generic": source_generic,
        },
        "behavior": {
            "source_admissions": source_admissions,
            "source_neural_action_contrasts": source_changes,
            "source_permuted_action_contrasts": permuted_contrasts,
            "selected_effect_types": dict(source_effects),
            "selected_programs": dict(selected_programs),
        },
        "gates": checks,
        "required_gate_names": required,
        "rows": rows,
        "same_frozen_source_ir": True,
        "only_target_native_grounder_replaced": True,
        "formal_outcome_exposed_to_neural_or_source_selection": False,
        "grounder_artifact_sha256": grounder_artifact["artifact_sha256"],
        "source_artifact_sha256s": sorted(
            str(row["artifact_sha256"]) for row in source_artifacts
        ),
    }
    return body | {"report_sha256": stable_hash(body)}


__all__ = [
    "ANSWER_SLOTS", "BASELINE_FEATURE_NAMES", "CONDITIONS",
    "EFFECT_HORIZONS", "FEATURE_NAMES",
    "GENERIC_SCAFFOLD", "NEURAL_ONLY", "SOURCE_INDUCED",
    "OBSERVATION_FEATURE_NAMES",
    "SOURCE_PERMUTED", "TARGET_NATIVE_CEILING", "attach_grounding",
    "baseline_feature_map", "candidate_feature_map", "evaluate_matched_receipts",
    "execute_condition",
    "grounder_feature_names", "observation_feature_map",
    "predict_baseline_correctness", "predict_candidate_effects",
    "predict_observed_effect",
    "target_native_program_bank", "validate_grounder_artifact",
]
