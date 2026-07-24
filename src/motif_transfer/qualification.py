from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SourceOutcome:
    condition: str
    pair_id: str
    initial_state_hash: str
    prefix_hash: str
    policy_hash: str
    budget_hash: str
    official_success: bool
    official_score: float


@dataclass(frozen=True)
class SourceQualificationReport:
    supported: bool
    reason: str
    outcomes: tuple[SourceOutcome, ...]
    success_rates: Mapping[str, float] = field(default_factory=dict)
    mean_scores: Mapping[str, float] = field(default_factory=dict)


class SourceQualifier:
    REQUIRED = frozenset(
        {
            "authentic_skill_loaded",
            "skill_disabled",
            "generic_protocol",
            "shuffled_topology",
            "other_source",
        }
    )

    def evaluate(self, outcomes: Iterable[SourceOutcome]) -> SourceQualificationReport:
        rows = tuple(outcomes)
        pairs: dict[str, list[SourceOutcome]] = {}
        for row in rows:
            pairs.setdefault(row.pair_id, []).append(row)
        if not pairs:
            return SourceQualificationReport(False, "no paired source outcomes", rows)
        identity_fields = ("initial_state_hash", "prefix_hash", "policy_hash", "budget_hash")
        for pair_id, pair_rows in pairs.items():
            names = {row.condition for row in pair_rows}
            missing = self.REQUIRED - names
            if missing:
                return SourceQualificationReport(
                    False, f"pair {pair_id} missing conditions: {sorted(missing)}", rows
                )
            if len(names) != len(pair_rows):
                return SourceQualificationReport(False, f"pair {pair_id} has duplicate conditions", rows)
            if any(len({getattr(row, field) for row in pair_rows}) != 1 for field in identity_fields):
                return SourceQualificationReport(False, f"pair {pair_id} identity mismatch", rows)

        count = len(pairs)
        rates = {
            condition: sum(row.official_success for row in rows if row.condition == condition) / count
            for condition in self.REQUIRED
        }
        scores = {
            condition: sum(row.official_score for row in rows if row.condition == condition) / count
            for condition in self.REQUIRED
        }
        authentic = "authentic_skill_loaded"
        controls = self.REQUIRED - {authentic}
        separates_success = rates[authentic] > max(rates[name] for name in controls)
        separates_score = scores[authentic] > max(scores[name] for name in controls)
        if separates_success or separates_score:
            return SourceQualificationReport(
                True,
                "authentic source condition separates from every registered control in this pilot",
                rows,
                rates,
                scores,
            )
        return SourceQualificationReport(
            False,
            "authentic source condition has no attributable separation",
            rows,
            rates,
            scores,
        )
