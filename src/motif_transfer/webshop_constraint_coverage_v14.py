"""Target-native prerequisite coverage for safe WebShop commits.

The transferred source contract is binary (prepare versus commit), while a
WebShop item may have several simultaneous option requirements.  This module
keeps the symbolic source binary but grounds readiness as a target-native set
coverage problem.  Only an observed state-changing constraint action counts as
verified; merely proposing or clicking a no-op control does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9.]+", str(text).lower())
        if len(token) > 1
    }


def visible_goal_constraint_label_actions(axtree: str, goal: str) -> tuple[str, ...]:
    """Return paired LabelText clicks for unchecked goal-matching radios.

    The installed WebShop frequently exposes a radio BID whose direct click is
    a no-op, followed by a clickable LabelText BID.  This parser uses only the
    live target accessibility tree and goal; it does not inspect reward or a
    source artifact.
    """

    goal_tokens = _tokens(goal)
    lines = axtree.splitlines()
    actions = []
    radio_pattern = re.compile(
        r"^\s*\[(\d+)\]\s+radio\s+'([^']+)'[^\n]*"
        r"checked\s*=\s*['\"]?(false|0)",
        re.IGNORECASE,
    )
    label_pattern = re.compile(r"^\s*\[(\d+)\]\s+LabelText\b", re.IGNORECASE)
    for index, line in enumerate(lines):
        match = radio_pattern.search(line)
        if not match or not (_tokens(match.group(2)) & goal_tokens):
            continue
        radio_bid = int(match.group(1))
        # BrowserGym's tree normally places the associated label immediately
        # after the radio.  Limit the search to two lines and require BID+1 to
        # avoid binding an unrelated label.
        for following in lines[index + 1:index + 3]:
            label = label_pattern.search(following)
            if label and int(label.group(1)) == radio_bid + 1:
                action = f"click('{label.group(1)}')"
                if action not in actions:
                    actions.append(action)
                break
    return tuple(actions)


def augment_with_constraint_labels(
    candidates: Sequence[str], *, axtree: str, goal: str,
) -> tuple[str, ...]:
    """Append deterministic target-native recovery actions without reranking."""

    output = list(dict.fromkeys(candidates))
    for action in visible_goal_constraint_label_actions(axtree, goal):
        if action not in output:
            output.append(action)
    return tuple(output)


def constraint_signature(row: Mapping[str, Any]) -> str | None:
    """Map a grounded target constraint to a stable semantic signature."""

    if not row.get("is_goal_constraint"):
        return None
    tokens = sorted({str(token).strip().lower() for token in row.get(
        "goal_overlap_tokens", []
    ) if str(token).strip()})
    return "|".join(tokens) or None


@dataclass
class ConstraintCoverage:
    """Episode-local required/verified prerequisite ledger."""

    required: set[str] = field(default_factory=set)
    verified: set[str] = field(default_factory=set)
    pending_signature: str | None = None

    def begin_decision(
        self,
        semantics: Sequence[Mapping[str, Any]],
        *,
        prior_action_had_no_effect: bool,
    ) -> None:
        """Resolve the previous transition and ingest currently visible needs."""

        if self.pending_signature is not None and not prior_action_had_no_effect:
            self.verified.add(self.pending_signature)
        self.pending_signature = None
        self.required.update(
            signature for row in semantics
            if (signature := constraint_signature(row)) is not None
        )

    def record_selected(self, row: Mapping[str, Any]) -> None:
        self.pending_signature = constraint_signature(row)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(sorted(self.required - self.verified))

    @property
    def commit_authorized(self) -> bool:
        return bool(self.required) and not self.missing

    def preferred_missing_index(
        self, semantics: Sequence[Mapping[str, Any]],
    ) -> int | None:
        """Return the first currently executable missing prerequisite action."""

        missing = set(self.missing)
        for index, row in enumerate(semantics):
            if constraint_signature(row) in missing and not row.get("is_selected"):
                return index
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "required": sorted(self.required),
            "verified": sorted(self.verified),
            "missing": list(self.missing),
            "commit_authorized": self.commit_authorized,
            "pending_signature": self.pending_signature,
        }


def audit_receipt_commits(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Replay a historical receipt and identify coverage-unsafe commits."""

    ledger = ConstraintCoverage()
    commits = []
    previous_no_effect = False
    for step in receipt.get("steps", []):
        semantics = list(step.get("candidate_semantics", []))
        ledger.begin_decision(
            semantics, prior_action_had_no_effect=previous_no_effect,
        )
        selected_index = int(step.get("selected_index", 0))
        if 0 <= selected_index < len(semantics):
            selected = semantics[selected_index]
            if selected.get("is_commit"):
                commits.append({
                    "step": step.get("step"),
                    "selected_action": step.get("selected_action"),
                    "authorized": ledger.commit_authorized,
                    "coverage": ledger.as_dict(),
                })
            ledger.record_selected(selected)
        previous_no_effect = step.get("before_hash") == step.get("after_hash")
    return commits


__all__ = [
    "ConstraintCoverage",
    "augment_with_constraint_labels",
    "audit_receipt_commits",
    "constraint_signature",
    "visible_goal_constraint_label_actions",
]
