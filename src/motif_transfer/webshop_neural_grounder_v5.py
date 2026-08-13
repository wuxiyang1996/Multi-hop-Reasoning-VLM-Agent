from __future__ import annotations

import ast
import math
import re
from typing import Mapping, Sequence

import numpy as np


ACTION_VERBS = (
    "click",
    "fill",
    "press",
    "scroll",
    "go_back",
    "go_forward",
    "noop",
    "other",
)
URL_PHASES = (
    "landing",
    "search_results",
    "item_page",
    "item_sub_page",
    "done",
    "other",
)
ELEMENT_ROLES = ("textbox", "button", "link", "radio", "heading", "other")
TARGET_FEATURE_NAMES = (
    *(f"action_{verb}" for verb in ACTION_VERBS),
    *(f"url_{phase}" for phase in URL_PHASES),
    "step_fraction",
    "remaining_budget_fraction",
    "action_repeated",
    "action_has_bid",
    "action_string_length_tanh",
    "observation_line_count_tanh",
    "observation_bid_count_tanh",
    "goal_action_token_overlap",
    *(f"element_role_{role}" for role in ELEMENT_ROLES),
    "goal_element_token_overlap",
    "element_string_length_tanh",
    "element_checked_or_selected",
    "element_required",
)


def action_verb(action: str) -> str:
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", action)
    verb = match.group(1) if match else "other"
    return verb if verb in ACTION_VERBS else "other"


def url_phase(url: str) -> str:
    if "/search_results/" in url:
        return "search_results"
    if "/item_sub_page/" in url:
        return "item_sub_page"
    if "/item_page/" in url:
        return "item_page"
    if "/done/" in url:
        return "done"
    if re.search(r"/fixed_\d+/?$", url):
        return "landing"
    return "other"


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def action_bid(action: str) -> str | None:
    try:
        expression = ast.parse(action, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call) or not expression.args:
        return None
    first = expression.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def element_text_for_bid(observation_text: str, bid: str | None) -> str:
    if bid is None:
        return ""
    live_pattern = re.compile(rf"^\s*\[{re.escape(bid)}\]\s+", re.MULTILINE)
    canonical_pattern = re.compile(rf"\bbid={re.escape(bid)}(?:\D|$)")
    for line in observation_text.splitlines():
        if live_pattern.search(line) or canonical_pattern.search(line):
            return line.strip()
    return ""


def element_role(element_text: str) -> str:
    lowered = element_text.lower()
    for role in ELEMENT_ROLES[:-1]:
        if re.search(rf"\b{role}\b", lowered):
            return role
    return "other"


def target_action_features(
    *,
    observation_text: str,
    url: str,
    goal: str,
    action: str,
    step_index: int,
    maximum_steps: int,
    previous_action: str | None,
) -> tuple[float, ...]:
    verb = action_verb(action)
    phase = url_phase(url)
    goal_tokens = _tokens(goal)
    action_tokens = _tokens(action)
    overlap = len(goal_tokens & action_tokens) / max(1, len(action_tokens))
    element = element_text_for_bid(observation_text, action_bid(action))
    role = element_role(element)
    element_tokens = _tokens(element)
    element_overlap = len(goal_tokens & element_tokens) / max(1, len(element_tokens))
    element_lower = element.lower()
    values = (
        *(float(verb == candidate) for candidate in ACTION_VERBS),
        *(float(phase == candidate) for candidate in URL_PHASES),
        step_index / max(1, maximum_steps - 1),
        (maximum_steps - step_index) / maximum_steps,
        float(previous_action is not None and action == previous_action),
        float(bool(re.search(r"['\"]\d+['\"]", action))),
        float(np.tanh(len(action) / 100.0)),
        float(np.tanh(len(observation_text.splitlines()) / 100.0)),
        float(np.tanh(len(re.findall(r"\bbid\b", observation_text)) / 50.0)),
        overlap,
        *(float(role == candidate) for candidate in ELEMENT_ROLES),
        element_overlap,
        float(np.tanh(len(element) / 200.0)),
        float("checked" in element_lower or "selected" in element_lower),
        float("required" in element_lower),
    )
    if len(values) != len(TARGET_FEATURE_NAMES):
        raise AssertionError("WebShop target feature contract drift")
    return tuple(map(float, values))


def validate_browsergym_action(action: str, valid_bids: set[str]) -> bool:
    if not action or "\n" in action or len(action) > 500:
        return False
    try:
        expression = ast.parse(action, mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        return False
    verb = expression.func.id
    if verb not in {"click", "fill", "press", "scroll", "go_back", "go_forward", "noop"}:
        return False
    if expression.keywords:
        return False
    if verb in {"go_back", "go_forward", "noop"}:
        return len(expression.args) == 0
    if verb == "scroll":
        return len(expression.args) == 2 and all(
            isinstance(value, ast.Constant) and isinstance(value.value, (int, float))
            for value in expression.args
        )
    required = 1 if verb == "click" else 2
    if len(expression.args) != required:
        return False
    first = expression.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return False
    if first.value not in valid_bids:
        return False
    return all(isinstance(value, ast.Constant) for value in expression.args[1:])


def bids_from_axtree(text: str) -> set[str]:
    patterns = (
        r"\bbid[=:'\"\s]+([A-Za-z0-9_-]+)",
        r"\[([A-Za-z0-9_-]+)\]\s+",
    )
    output: set[str] = set()
    for pattern in patterns:
        output.update(re.findall(pattern, text))
    return output


def mlp_predict(artifact: Mapping[str, object], features: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    mean = np.asarray(artifact["input_scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(artifact["input_scaler"]["scale"], dtype=np.float64)
    hidden = (matrix - mean) / scale
    coefficients = artifact["mlp"]["coefficients"]
    intercepts = artifact["mlp"]["intercepts"]
    for index, (weights, bias) in enumerate(zip(coefficients, intercepts, strict=True)):
        hidden = hidden @ np.asarray(weights, dtype=np.float64) + np.asarray(bias, dtype=np.float64)
        if index + 1 < len(coefficients):
            hidden = np.maximum(hidden, 0.0)
    lower = np.asarray([0, 0, -1, -1, 0, -1, 0], dtype=np.float64)
    upper = np.ones(7, dtype=np.float64)
    return np.clip(hidden, lower, upper)


def nearest_source_options(
    effects: Sequence[Sequence[float]], source_candidate: Mapping[str, object]
) -> np.ndarray:
    matrix = np.asarray(effects, dtype=np.float64)
    mean = np.asarray(source_candidate["effect_scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(source_candidate["effect_scaler"]["scale"], dtype=np.float64)
    centers = np.asarray(source_candidate["cluster_centers"], dtype=np.float64)
    normalized = (matrix - mean) / scale
    distances = np.sum((normalized[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def source_option_values(
    option_ids: Sequence[int],
    *,
    source_candidate: Mapping[str, object],
    context_features: Sequence[float],
    previous_option: int | None,
    corruption: str | None = None,
) -> np.ndarray:
    options = np.asarray(option_ids, dtype=np.int64)
    cluster_count = int(source_candidate["cluster_count"])
    if corruption == "phase_permuted":
        options = (options + 1) % cluster_count
        if previous_option is not None:
            previous_option = (previous_option + 1) % cluster_count
    elif corruption is not None:
        raise ValueError(f"unknown source corruption: {corruption}")
    current = np.eye(cluster_count, dtype=np.float64)[options]
    previous_index = cluster_count if previous_option is None else int(previous_option)
    previous = np.repeat(
        np.eye(cluster_count + 1, dtype=np.float64)[previous_index][None, :],
        len(options),
        axis=0,
    )
    context = np.repeat(np.asarray(context_features)[None, :], len(options), axis=0)
    design = np.column_stack((context, current, previous))
    coefficients = np.asarray(source_candidate["value_model"]["coefficients"], dtype=np.float64)
    intercept = np.asarray(source_candidate["value_model"]["intercept"], dtype=np.float64)
    return design @ coefficients.T + intercept


def finite_json(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    return True
