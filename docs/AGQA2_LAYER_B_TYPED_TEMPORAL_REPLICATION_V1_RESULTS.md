# AGQA2 Layer B typed-temporal raw-video replication V1

## Decision

This fresh, history-disjoint raw-video replication provides confirmatory evidence
for the preregistered **selective transfer** claim:

> Game-acquired ordered-effect skills transfer to raw-video temporal/duration
> reasoning under shared frozen perception.

The source-induced Harness improved final QA accuracy from `106/256 = 41.41%`
to `135/256 = 52.73%` (`+29` correct, `+11.33 pp`). The paired comparison was
`34W/5L`, exact two-sided `p=2.43e-6`. Negative-transfer losses were
`5/256 = 1.95%`, below the frozen 5% limit. The matched source-permuted arm was
identical to neural-only, so source-induced also beat that control by the same
`34W/5L` margin.

The run did **not** pass the additional `source_is_best_feasible_symbolic_arm`
gate: the unrestricted generic target-VM scaffold reached `149/256 = 58.20%`.
Consequently this result supports transfer and source-lineage causality, but not
superiority over a broader target-engineered symbolic controller.

| Arm | Correct | Accuracy | Symbolic commits |
|---|---:|---:|---:|
| neural-only | 106/256 | 41.41% | 0 |
| source-permuted | 106/256 | 41.41% | 0 |
| generic scaffold | 149/256 | 58.20% | 208 |
| game source-induced | 135/256 | 52.73% | 107 |
| target-written isomorphic | 135/256 | 52.73% | 107 |

The Slurm finalizer exited 1 intentionally because the extra best-feasible-arm
gate failed. Grounding, parser, SlowFast, fallback, merge, and evaluator jobs all
completed; this was not an infrastructure failure.

## What is causally identified

All five arms used the same 256 raw videos, exact Qwen3-VL-32B 48/96-frame
grounding receipts, SlowFast action evidence, Flan-T5 operator-free semantic
slots, source-blind route, typed executor, Qwen3.5-9B fallback outputs, answer
normalizer, and evaluator. The only changed component was the symbolic Harness.

The source-permuted control preserved the source artifact's 15-operator inventory
and 32-edge count, while applying a fixed derangement to composition lineage. Its
zero commits and neural-equivalent score show that source identity or operator
inventory alone was insufficient. The authentic source composition committed on
107 tasks and delivered a highly significant paired gain. The target-written
isomorphic clone had 100% prediction/action equivalence with source-induced; it is
an implementation ceiling, not independent provenance evidence.

The generic scaffold was deliberately unrestricted: it eagerly used all target
VM primitives and committed on 208 tasks. It gained more correct answers than the
selective source policy, while also incurring `11/256 = 4.30%` negative-transfer
losses versus the source policy's `5/256 = 1.95%`. This explains the failed extra
gate without weakening the matched source-versus-permuted causal result.

## Scope and per-family result

The cohort was selected outcome-blind from source capability evidence
`EFFECT_RANKING + ORDERED_ENDPOINTS`, with `GUARDED_BRANCH` additionally required
for duration choice. It contained two balanced semantic families:

| Semantic family | Neural | Source | Source commits | W/L vs neural |
|---|---:|---:|---:|---:|
| duration choice | 68/128 | 75/128 | 39 | 9/2 |
| duration extremum | 38/128 | 60/128 | 68 | 25/3 |

This is not evidence for full-distribution AGQA2, arbitrary video questions, or
target-free perception. It specifically validates game-to-video transfer of a
typed ordered-effect/duration structure. The broader eight-root replication
remains a separately reported negative result.

## Freshness, runtime boundary, and cost

- 256 tasks and 256 videos; 128 examples per semantic family.
- Zero task/video overlap with 1,408 prior Layer-B examples.
- Zero overlap with semantic-parser supervision.
- No official answer, STSG, or functional program was visible to parser,
  grounder, Harness, executor, router, or fallback before final evaluation.
- Source execution coverage was frozen before outcomes at
  `107/256 = 41.80%`, above the 35% gate.
- Route counts were 242 tasks on the 48-frame candidate and 14 on the 96-frame
  candidate; the maximum shared presentation budget was 464 frames per task.
- Qwen grounding made 512 provider calls for `$0.29133`; eight provider errors
  failed closed and remained identical across arms.
- One parser output had a single surplus trailing parenthesis. A frozen,
  syntax-only amendment removed the unique surplus token before any video call or
  outcome read; it changed no semantic token.

## Immutable artifacts

- preregistration:
  `configs/agqa2_layer_b_typed_temporal_replication_v1_preregistration.json`
- parser amendment:
  `configs/agqa2_layer_b_typed_temporal_replication_v1_parser_amendment.json`
- public cohort semantic SHA:
  `7274d64939ae875ca5cae4731afcc9da8ea9a0c32e04620a25e7739f5d367179`
- repaired semantic runtime file SHA:
  `54e5bd19fb2a7ddc57c46746ac0db04265d1c0649986ebddd2d6cd8dcac3f3c6`
- routed grounding semantic SHA:
  `35b0a206d94c187a19e476d36029a3cbd74a04e0cc6bfa24e827b48903e141e9`
- frozen fallback semantic SHA:
  `1f6f7872b532fee747eb894a0c9fee9647314f2d422a4656ccea5e4212a3a8cc`
- pre-outcome receipt semantic SHA:
  `498efc6a9e60399a811fb5f6ede86d93b9422e7c75d95cc22c0866adeb9cbaa9`
- outcome report semantic SHA:
  `9b2e955a5daed242dc8077dfb60adf448de15ae92f8b50b939414dc83d02e6b3`

The semantic SHA values above are the canonical hashes embedded in the immutable
artifacts; whole-file SHA values differ because each JSON also stores its own
canonical report hash.
