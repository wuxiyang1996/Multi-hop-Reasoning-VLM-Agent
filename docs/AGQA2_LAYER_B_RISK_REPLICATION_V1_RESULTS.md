# AGQA2 Layer B risk-constrained replication V1

## Decision

The fully history-disjoint replication is permanently consumed and **failed its
preregistered significance gates**. The direction remained positive and the
negative-transfer bound passed, but the paired effect was too small:

| Arm | Correct | Accuracy | Symbolic commits |
|---|---:|---:|---:|
| neural-only | 168/384 | 43.75% | 0 |
| source-permuted | 168/384 | 43.75% | 2 |
| generic scaffold | 187/384 | 48.70% | 273 |
| game source-induced | 174/384 | 45.31% | 190 |
| target-written isomorphic | 174/384 | 45.31% | 190 |

Source versus neural-only and matched source-permuted was `24W/18L`, exact
two-sided `p=.44080`. Losses were `18/384 = 4.69%`, within the frozen 5% budget.
Generic was higher in raw accuracy, but had `29/384 = 7.55%` losses versus
neural-only and was therefore infeasible under the same risk constraint.
Target-written/source prediction equivalence was 100%.

The final Slurm job exited 1 intentionally because two significance gates failed;
there was no infrastructure, GPU, schema, or dependency failure.

## Freshness and matched pipeline

The cohort had 384 questions and 384 videos, balanced over eight semantic roots
(48 each). It had zero task/video overlap with every prior Layer-B acquisition,
development, and qualification cohort, and zero task overlap with semantic-parser
supervision. Selection used deterministic bipartite capacity matching before any
provider call or outcome read.

All five arms shared the same 48f/96f Qwen3-VL-32B event graphs, SlowFast action
evidence, source-blind router, scoped three-valued visual claims, Flan-T5 semantic
slots, Qwen3.5-9B fallback, executor, and evaluator. The source-permuted control
preserved the exact 15-operator inventory and 32-edge count while applying a fixed
derangement to source composition lineage.

Pre-outcome source coverage was `190/384 = 49.48%`, above the frozen 40% gate.
Raw event grounding produced 768 receipts for `$0.4539`; 13 provider failures
(1.69%) failed closed and remained shared. Atomic claims cost `$0.1276`.

## Failure decomposition

| Semantic root | Neural | Source | W | L | Net |
|---|---:|---:|---:|---:|---:|
| alternatives | 23 | 22 | 0 | 1 | -1 |
| duration choice | 23 | 28 | 5 | 0 | +5 |
| duration extremum | 12 | 19 | 12 | 5 | +7 |
| equality condition | 21 | 19 | 2 | 4 | -2 |
| exclusive condition | 23 | 20 | 2 | 5 | -3 |
| goal | 9 | 9 | 0 | 0 | 0 |
| joint condition | 23 | 23 | 3 | 3 | 0 |
| presence question | 34 | 34 | 0 | 0 | 0 |

The transferable signal is concentrated in the source's high-support ordered
effect and duration structure: the two duration roots contribute `17W/5L` and
`+12` net. A static type audit also found that the Layer-B adapter mapped target
semantic equality onto the source's effect-ranking `COMPARE` operator. Those are
not the same typed operation. The broad result therefore mixes a strong compatible
transfer family with unsupported or weakly supported operator bindings.

This decomposition is consumed diagnostic evidence. It cannot be used to tune and
rerun this reserve.

## Legal next claim

The next experiment must be a new, video-disjoint, preregistered **typed
source-compatible replication**. Its cohort is selected before outcomes by exact
IR compatibility with the source's `EFFECT_RANKING + ORDERED_ENDPOINTS` evidence,
not by per-task correctness. It may test duration choice and duration extremum,
while semantic equality must require a distinct `SEMANTIC_EQUALS` capability that
the source artifact does not contain. The resulting claim is selective:

> Game-acquired ordered-effect skills transfer to raw-video temporal/duration
> reasoning under a shared frozen grounder.

It is not a claim of full-distribution AGQA2 improvement. New official Charades
videos are required because the current local video pool is exhausted for another
strictly history-disjoint balanced replication.

## Immutable artifacts

- preregistration: `configs/agqa2_layer_b_risk_replication_v1_preregistration.json`
- public cohort SHA: `8b5fa41462fa2080916ac7f3eabe36e90830533c7eb4acfe2baff3a711e7dbc2`
- routed grounding SHA: `e50aa455b28b5bc2d8d57c5fccf3373ef6575ff4b44ec525de3dc6abfc8eeeb8`
- claims SHA: `3ea4755c6de0081a2d25d2b1b8df1a47474cf4ad95551cffa9aa7e7169f56a09`
- fallback SHA: `71773d346f409f29fb30e4152f0841f21a1c555e84e8e6537d87326d140c507b`
- pre-outcome receipt SHA: `701e08c2be834512f23df3260f1c78432cbe2c070e3d52f608f1cffb8236e9f3`
- outcome report SHA: `c1f2beef88f1dbbac685804f77e2016db592bb2a37629492f8ef015c20acd236`
