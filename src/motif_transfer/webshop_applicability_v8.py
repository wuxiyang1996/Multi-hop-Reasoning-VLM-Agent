from __future__ import annotations

import re
from typing import Sequence

from motif_transfer.webshop_neural_grounder_v5 import (
    action_bid,
    action_verb,
    element_role,
    element_text_for_bid,
    url_phase,
)


COMMIT_PATTERNS = (
    "buy now",
    "checkout",
    "place order",
    "submit order",
    "confirm purchase",
)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9.]+", text.lower()) if len(token) > 1}


def candidate_semantics(
    *, observation_text: str, url: str, goal: str, action: str
) -> dict:
    verb = action_verb(action)
    element = element_text_for_bid(observation_text, action_bid(action))
    lowered = element.lower()
    role = element_role(element)
    paired_constraint_bid = None
    paired_constraint_text = ""
    bid = action_bid(action)
    if role == "other" and "labeltext" in lowered and bid and bid.isdigit():
        prior_bid = str(int(bid) - 1)
        prior_element = element_text_for_bid(observation_text, prior_bid)
        if element_role(prior_element) == "radio":
            paired_constraint_bid = prior_bid
            paired_constraint_text = prior_element
    element_tokens = _tokens(element)
    if paired_constraint_text:
        element_tokens |= _tokens(paired_constraint_text)
    goal_tokens = _tokens(goal)
    overlap_tokens = sorted(element_tokens & goal_tokens)
    is_commit = any(pattern in lowered for pattern in COMMIT_PATTERNS)
    is_constraint = paired_constraint_bid is not None or role in {"radio", "checkbox"} or any(
        token in lowered for token in ("combobox", "option ", "select ")
    )
    # Accessibility trees encode both true and false boolean attributes.  A
    # substring test therefore turns ``checked='false'`` into a false positive
    # and blocks the paired-label recovery that WebShop sometimes requires.
    is_selected = bool(re.search(
        r"(?:checked|selected)\s*=\s*['\"]?(?:true|1)['\"]?",
        lowered,
    ))
    is_navigation = verb in {"scroll", "go_back", "go_forward"} or (
        verb == "click" and role in {"link", "button"} and not is_commit
    )
    return {
        "verb": verb,
        "url_phase": url_phase(url),
        "element_role": role,
        "element_text": element,
        "action_bid": bid,
        "paired_constraint_bid": paired_constraint_bid,
        "paired_constraint_text": paired_constraint_text,
        "goal_overlap_tokens": overlap_tokens,
        "goal_overlap": len(overlap_tokens) / max(1, len(element_tokens)),
        "is_commit": is_commit,
        "is_constraint": is_constraint,
        "is_goal_constraint": is_constraint and bool(overlap_tokens),
        "is_selected": is_selected,
        "is_navigation": is_navigation,
        "is_noop": verb == "noop",
    }


def exact_stall(
    *,
    previous_before_hash: str | None,
    previous_after_hash: str | None,
    rank_zero_action: str,
    previous_action: str | None,
) -> bool:
    return bool(
        previous_before_hash
        and previous_after_hash
        and previous_action
        and previous_before_hash == previous_after_hash
        and rank_zero_action == previous_action
    )


def safe_recovery_indices(
    semantics: Sequence[dict], *, rank_zero_index: int = 0
) -> tuple[tuple[int, ...], str | None]:
    rank_zero = semantics[rank_zero_index]
    if rank_zero["is_commit"]:
        return (), "preserve_target_commit"
    rank_zero_bid = rank_zero.get("action_bid")
    paired_recoveries = tuple(
        index for index, row in enumerate(semantics)
        if index != rank_zero_index
        and rank_zero_bid is not None
        and row.get("paired_constraint_bid") == rank_zero_bid
        and row.get("is_goal_constraint", False)
        and not row.get("is_selected", False)
    )
    if rank_zero["is_constraint"]:
        return (
            (paired_recoveries, None)
            if paired_recoveries
            else ((), "preserve_target_constraint_action")
        )

    safe = []
    for index, row in enumerate(semantics):
        if index == rank_zero_index or row["is_commit"] or row["is_noop"]:
            continue
        if row["is_constraint"]:
            if row["is_goal_constraint"] and not row["is_selected"]:
                safe.append(index)
            continue
        if row["is_navigation"]:
            safe.append(index)
    return tuple(safe), None if safe else "no_safe_recovery_candidate"


__all__ = ["candidate_semantics", "exact_stall", "safe_recovery_indices"]
