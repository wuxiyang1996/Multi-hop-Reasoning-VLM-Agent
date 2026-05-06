"""Per-domain `FailureTrace` synthesizers for the offline reflect pipeline.

Each submodule exposes one function:

    from_sample(sample: dict, *, domain: str, sample_id: str,
                max_failures: int) -> List[FailureTrace]

That function takes the on-disk cold-start record (e.g. the per-sample
JSON written by ``cold_start/generate_cold_start_actor_visual_reasoning.py``
for VR / video, or the per-task ``episode_000.json`` written by
``cold_start/generate_cold_start_actor_browsergym.py`` for AB) and
synthesises a bounded list of ``FailureTrace`` records.

The dispatch table at the bottom is consumed by
``labeling_supplement/reflect_per_episode_gpt54.py`` when ``--domain``
is set to a transfer-target domain (see PLAN-SKILL-BANK §0.4 +
``common.enums.TRANSFER_TARGET_DOMAINS``). The legacy ``gymv``
synthesizer remains in ``reflect_per_episode_gpt54._synthesize_failures``
because it has access to the per-step bank-mgmt context that this
package does not need.

All synthesizers respect the same severity ordering convention:
*OUTCOME-level failures first, signal-level failures next, then
schema-level diagnostic noise* — so a low ``max_failures`` cap retains
the most informative trace.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from data_structure.extensions.failure_trace import FailureTrace

from labeling_supplement._failure_synth import visual_reasoning as _vr

SynthFn = Callable[..., List[FailureTrace]]

DOMAIN_SYNTHESIZERS: Dict[str, SynthFn] = {
    "visual_reasoning": _vr.from_sample,
    # Future: "browser": _browser.from_episode,
    #         "video":   _video.from_sample,
    #         "osworld": _osworld.from_episode,
}


def get_synthesizer(domain: str) -> SynthFn:
    """Return the synthesiser for ``domain``; raise on unknown."""
    fn = DOMAIN_SYNTHESIZERS.get(domain)
    if fn is None:
        raise KeyError(
            f"no failure synthesiser registered for domain={domain!r}; "
            f"known={sorted(DOMAIN_SYNTHESIZERS.keys())}"
        )
    return fn


__all__ = ["DOMAIN_SYNTHESIZERS", "SynthFn", "get_synthesizer"]
