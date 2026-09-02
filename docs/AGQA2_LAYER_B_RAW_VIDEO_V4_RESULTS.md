# AGQA2 Layer B raw-video qualification V4

## Status

`qualification_v4` is permanently consumed. The preregistered all-gates decision is
**FAILED**, because `source_induced` did not exceed the unrestricted
`generic_scaffold`. No untouched formal reserve is authorized by this protocol.

This is not a null result. Under exactly matched raw-video grounding,
`source_induced` produced a statistically significant gain over both
`neural_only` and `source_permuted`, while satisfying the preregistered
negative-transfer bound. The correct claim is therefore **safe/risk-constrained
game-to-video transfer**, not accuracy superiority over arbitrary generic symbolic
reasoning.

## Frozen pipeline

The reserve contains 512 task- and video-disjoint questions, balanced over eight
semantic roots (64 each), with zero semantic-parser supervision overlap. Before any
answer was opened, the following were frozen:

```text
raw Charades video
  -> Qwen3-VL-32B uniform 48f and 96f event grounders
  -> frozen SlowFast-R50 Charades action tool (dense 10x32)
  -> fill-missing merge and source-blind router
  -> Qwen3-VL-32B scoped SUPPORTED/REFUTED/UNKNOWN claims
  -> operator-free Flan-T5 semantic slots
  -> five Harness arms
  -> shared typed executor and Qwen3.5-9B fallback
  -> answer
```

All arms shared the same videos, presented frames/hashes, frame budget, event and
claim receipts, parser, executor, fallback, normalization, and evaluator. Only the
symbolic Harness differed. The pre-outcome receipt recorded source execution
coverage `232/512 = 45.31%`, above the frozen 40% gate.

## Primary results

| Arm | Correct | Accuracy | Symbolic commits |
|---|---:|---:|---:|
| neural-only | 238/512 | 46.48% | 0 |
| source-permuted | 238/512 | 46.48% | 0 |
| generic scaffold | 264/512 | 51.56% | 352 |
| game source-induced | 258/512 | 50.39% | 232 |
| target-written isomorphic | 258/512 | 50.39% | 232 |

Paired comparisons:

- source vs neural/source-permuted: `44W/24L`, exact two-sided `p=.02053`;
- source gain: `+20/512 = +3.91 pp`;
- source negative-transfer losses: `24/512 = 4.69%`, passing the 5% gate;
- source vs generic: `17W/23L`, `p=.42959`; source is lower by 6 correct;
- generic vs neural: `67W/41L`, `p=.01575`, but generic losses are
  `41/512 = 8.01%`, failing the same 5% safety bound;
- target-written-isomorphic/source prediction equivalence: 100%.

Thus source-induced is the only tested symbolic arm that combines a significant
gain with the preregistered negative-transfer constraint. However, unrestricted
generic symbolic execution obtains higher raw accuracy, so V4 does not establish
that source acquisition is necessary for accuracy improvement.

## Provenance and cost

- public cohort SHA: `3ce5cab90a1bb5fcdc22b42158c256560dfe12b7732d25e07b1b3832c3267d55`
- preregistration file SHA: `45809e66593e13841b1e5f16268229acc23a9cabb7bdaf391b05735009d4d508`
- routed grounding report SHA: `c73faef5812c8ee6449021bcba3cb5fb3afb3ff8c0bea927420765440b079885`
- atomic-claim report SHA: `8553d644b80cc6596ad30531397f1c5b4a12a72a4adaa38de992eff7d05eb9b1`
- shared fallback report SHA: `79496d0ae08cafe3bd19b1c1359dcb309b822b7502e286a4e58edd01a25a561a`
- pre-outcome receipt SHA: `2d51bb548844994eb0c7f567601d5294dc1e061146ca45d88283bf189e8c6ad5`
- outcome report SHA: `c030229390ea080e7b4b803853ef454c261fe53f1071242ce7700d81385864cd`
- event-grounding API cost: approximately `$0.6204`;
- atomic-claim API cost: `$0.1769`;
- claim states: 241 SUPPORTED, 90 REFUTED, 66 UNKNOWN; 18 provider/contract
  failures were deterministically converted to UNKNOWN.

The final Slurm job exits 1 intentionally because the all-gates protocol failed;
this is not an infrastructure, GPU, or OOM failure.

## Interpretation and next legal experiment

The dominant unresolved comparison is not neural grounding. It is the objective:
unrestricted generic execution buys six more correct answers but incurs 17 more
negative-transfer losses than source-induced. A follow-up must be preregistered as
one of two distinct claims:

1. **risk-constrained transfer**: compare methods under the same `loss <= 5%`
   constraint; V4 already provides a positive qualification result for this claim;
2. **unconstrained accuracy superiority**: learn a better applicability/utility
   policy strictly from source interventions, freeze it, and require it to exceed
   unrestricted generic on a new reserve.

V4 may be used only as consumed diagnostic data. It cannot be retuned or relabeled
as an all-gates pass, and untouched formal remains closed under the original
protocol.
