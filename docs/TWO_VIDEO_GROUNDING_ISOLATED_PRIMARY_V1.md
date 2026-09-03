# AGQA2 + CLEVRER grounding-isolated primary result

## Outcome

The two video benchmarks now have an auditable primary evaluation in which
visual-model grounding cannot create the matched-arm performance difference.
AGQA2 uses the official STSG through budgeted typed tools. CLEVRER uses one
frozen local paired event-graph proof receipt per task. Every controller arm
uses the same target-native structured backend or receipt. No generative
captioner, external VLM, answer, or official functional program is available
to the runtime comparison.

| Benchmark | Unit | Neural/generic reference | Source-induced | Paired result | Grounding losses |
|---|---:|---:|---:|---:|---:|
| AGQA2 fresh V3 | 900 questions / 300 videos | generic 322 | 386 | 64W / 0L | 0 |
| CLEVRER prospective V15 | 360 questions | neural-only 236 | 252 | 16W / 0L | 0 |

AGQA2 also beats its temporal-permuted control by `55W/0L`. CLEVRER beats its
source-permuted control by `16W/4L` and has a positive but non-significant
`13W/6L` comparison against the generic scaffold. The target-written
isomorphic AGQA controller is equal to source-induced by construction; this
is disclosed rather than used as evidence of source-provenance necessity.
The primary paired one-sided exact p-values are `2^-64 = 5.42e-20` for AGQA2
against generic and `2^-16 = 1.53e-5` for CLEVRER against neural-only.

## Why captions are not the primary grounder

A generated caption is another neural perception prediction: it can omit a
short event, hallucinate a broad action, or collapse two temporally distinct
events into one conjunction. For the primary skill-transfer estimand, a shared
benchmark-native structured interface is both stronger and cheaper. AGQA uses
official STSG; CLEVRER uses frozen local PropNet-derived proof receipts shared
by all arms. If a text interface is required for the 9B harness, the event
graph should be rendered deterministically into canonical typed facts, not
summarized by a VLM.

The raw-video Qwen/Gemini experiments remain useful secondary robustness
evidence, but they are not mixed into this primary table. Their negative or
unstable results therefore cannot change the controller-transfer estimate.
The newest frozen-router/unchanged-V65 Qwen replication is documented in
[`AGQA2_ROUTER_V65_FORMAL_V1_RESULTS.md`](AGQA2_ROUTER_V65_FORMAL_V1_RESULTS.md):
it improves `45/80` to `53/80` (`14W/6L`) but misses the preregistered exact
test (`p=.057659`) and loss bound, so it remains a failed confirmatory raw-video
result.

A subsequent 91-video multi-class-router Qwen diagnostic also failed: the
question-only route agreed with the evaluator-only official route on 78/91
questions, which triggered the frozen global fail-closed policy and produced
zero harness authorizations. Because this was post-V74, it is exploratory and
cannot overturn the earlier negative raw-video verdict. See
[`AGQA2_MULTICLASS_ROUTER_V65_FORMAL_V2_RESULTS.md`](AGQA2_MULTICLASS_ROUTER_V65_FORMAL_V2_RESULTS.md).

## Claim boundary

This validates controller/skill transfer **conditional on benchmark-native
structured grounding**. It does not claim raw-pixel video-QA SOTA. CLEVRER
absolute accuracy still depends on its frozen neural proof grounder; sharing
the exact receipt across arms removes that grounder from the paired delta but
does not make the grounder itself perfect. CLEVRER
predictive and counterfactual questions additionally require authorized
dynamics rollouts; factual simulator annotations alone authorize only the
factual/explanatory track.

Run the portable audit with:

```bash
PYTHONPATH=src:. python scripts/audit_two_video_grounding_isolated_primary_v1.py \
  --output docs/results/two_video_grounding_isolated_primary_v1.json
```

The consolidated audit, including the two failed raw-Qwen diagnostics, is:

```bash
PYTHONPATH=src:. python scripts/audit_two_video_grounding_isolated_primary_v2.py \
  --output docs/results/two_video_grounding_isolated_primary_v2.json
```
