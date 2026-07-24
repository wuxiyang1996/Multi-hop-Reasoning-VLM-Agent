from __future__ import annotations

from .contracts import Advisory, AdvisoryVerdict


class NeutralMotifAgent:
    """No-source control: preserve the two-agent interface without source advice."""

    def review(self, proposal, observation, binding, history):
        return Advisory(
            AdvisoryVerdict.ADMIT,
            "target-only control; no source-derived advisory",
        )
