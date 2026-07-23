from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .contracts import Lifecycle


REQUIRED_TARGET_CONDITIONS = (
    "target_only",
    "generic_reasoning",
    "authentic_game_source",
    "renamed_game_source",
    "shuffled_game_source",
    "other_game_source",
)


@dataclass(frozen=True)
class TransferExperimentSpec:
    experiment_id: str
    target_cell: str
    target_manifest_sha256: str
    source_candidate_id: str
    source_candidate_sha256: str
    source_lifecycle: Lifecycle
    decision_model: str
    harness_model: str
    judge_model: str
    tool_contract_sha256: str
    action_or_tool_budget: int
    conditions: tuple[str, ...] = REQUIRED_TARGET_CONDITIONS

    @property
    def confirmatory_ready(self) -> bool:
        return (
            self.source_lifecycle == Lifecycle.SOURCE_SUPPORTED
            and tuple(self.conditions) == REQUIRED_TARGET_CONDITIONS
            and bool(self.source_candidate_id and self.source_candidate_sha256)
        )

    def validate(self, *, diagnostic_only: bool = False) -> None:
        if tuple(self.conditions) != REQUIRED_TARGET_CONDITIONS:
            raise ValueError("transfer experiment must preserve all six frozen conditions in order")
        if self.action_or_tool_budget <= 0:
            raise ValueError("transfer experiment needs a positive matched budget")
        if not diagnostic_only and not self.confirmatory_ready:
            raise ValueError("confirmatory transfer requires a frozen SOURCE_SUPPORTED source candidate")

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_lifecycle"] = self.source_lifecycle.value
        payload["confirmatory_ready"] = self.confirmatory_ready
        return payload


@dataclass(frozen=True)
class TargetConditionOutcome:
    sample_id: str
    condition: str
    initial_asset_hash: str
    decision_model_hash: str
    harness_model_hash: str
    judge_model_hash: str
    tool_contract_hash: str
    budget_hash: str
    apr_pass: bool
    ars: float
    tool_calls: int
    invalid_calls: int
    max_output_tokens: int = 0
    temperature: float = 0.0
    source_advisories: int = 0
    live_supported_advisories: int = 0
    fallback_step: int | None = None


def evaluate_transfer_matrix(outcomes: Iterable[TargetConditionOutcome]) -> dict[str, Any]:
    rows = tuple(outcomes)
    by_sample: dict[str, list[TargetConditionOutcome]] = {}
    for row in rows:
        by_sample.setdefault(row.sample_id, []).append(row)
    if not by_sample:
        raise ValueError("target transfer matrix is empty")
    identity_fields = (
        "initial_asset_hash",
        "decision_model_hash",
        "harness_model_hash",
        "judge_model_hash",
        "tool_contract_hash",
        "budget_hash",
        "max_output_tokens",
        "temperature",
    )
    for sample_id, sample_rows in by_sample.items():
        conditions = [row.condition for row in sample_rows]
        if len(conditions) != len(set(conditions)):
            raise ValueError(f"sample {sample_id} has duplicate conditions")
        if set(conditions) != set(REQUIRED_TARGET_CONDITIONS):
            missing = sorted(set(REQUIRED_TARGET_CONDITIONS) - set(conditions))
            extra = sorted(set(conditions) - set(REQUIRED_TARGET_CONDITIONS))
            raise ValueError(f"sample {sample_id} condition mismatch missing={missing} extra={extra}")
        for field in identity_fields:
            if len({getattr(row, field) for row in sample_rows}) != 1:
                raise ValueError(f"sample {sample_id} has unmatched {field}")
        if any(row.ars < 0 or row.ars > 1 for row in sample_rows):
            raise ValueError(f"sample {sample_id} has invalid ARS")
        if any(row.max_output_tokens <= 0 for row in sample_rows):
            raise ValueError(f"sample {sample_id} has invalid max_output_tokens")
        if any(row.live_supported_advisories > row.source_advisories for row in sample_rows):
            raise ValueError(f"sample {sample_id} claims more supported than emitted advisories")

    count = len(by_sample)
    means: dict[str, Mapping[str, float]] = {}
    for condition in REQUIRED_TARGET_CONDITIONS:
        selected = [row for row in rows if row.condition == condition]
        means[condition] = {
            "apr": sum(row.apr_pass for row in selected) / count,
            "ars": sum(row.ars for row in selected) / count,
            "tool_calls": sum(row.tool_calls for row in selected) / count,
            "invalid_calls": sum(row.invalid_calls for row in selected) / count,
        }
    authentic = means["authentic_game_source"]
    destructive_controls = ("generic_reasoning", "shuffled_game_source", "other_game_source")
    alpha_renamed = means["renamed_game_source"]
    authentic_rows = [row for row in rows if row.condition == "authentic_game_source"]
    supported = sum(row.live_supported_advisories for row in authentic_rows)
    renamed_supported = sum(
        row.live_supported_advisories for row in rows if row.condition == "renamed_game_source"
    )
    if supported == 0:
        status = "NO_LIVE_SUPPORT"
    elif authentic["ars"] < means["target_only"]["ars"]:
        status = "NEGATIVE_TRANSFER_PILOT"
    elif authentic["ars"] > max(means[name]["ars"] for name in destructive_controls):
        if renamed_supported == 0:
            status = "ALPHA_RENAMED_NO_LIVE_SUPPORT"
        elif alpha_renamed["ars"] > max(means[name]["ars"] for name in destructive_controls):
            status = "STRUCTURE_TRANSFER_PILOT"
        else:
            status = "LEXICAL_OR_IDENTITY_DEPENDENT_PILOT"
    elif authentic["ars"] <= max(means[name]["ars"] for name in destructive_controls):
        status = "GENERIC_OR_CONTROL_EXPLAINS_EFFECT"
    else:  # pragma: no cover - exhaustive numeric ordering
        status = "INCONCLUSIVE"
    return {
        "schema_version": 1,
        "status": status,
        "sample_count": count,
        "means": means,
        "authentic_live_supported_advisories": supported,
        "alpha_renamed_live_supported_advisories": renamed_supported,
        "estimands": {
            "authentic_minus_target_only_ars": authentic["ars"] - means["target_only"]["ars"],
            "authentic_minus_generic_ars": authentic["ars"] - means["generic_reasoning"]["ars"],
            "alpha_renamed_minus_authentic_ars": alpha_renamed["ars"] - authentic["ars"],
            "authentic_minus_best_destructive_control_ars": authentic["ars"] - max(
                means[name]["ars"] for name in ("shuffled_game_source", "other_game_source")
            ),
        },
        "claim_limit": "Pilot status is not statistical evidence. Alpha-renaming is an invariance probe, not a destructive control; confirmatory structural claims require a preregistered equivalence margin and uncertainty intervals.",
    }
