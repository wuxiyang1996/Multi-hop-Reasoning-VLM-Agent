# AGQA Query Grounder V2 typed-evidence V13/V14

## Decision

V13 passed the frozen grounding qualification, but the subsequent untouched V14
five-arm formal did **not** validate incremental game-to-video skill transfer.
The failed formal is retained as a negative result and must not be tuned or
rerun on the same cohort.

This result identifies a method-design failure rather than a perception failure:
the query-conditioned grounder reduced each supported question to an almost
answer-complete event.  The shared Qwen3.5-9B neural actor could read the same
event directly, leaving almost no decision for the symbolic Harness to change.

## V13 qualification

V13 used 800 fresh videos and 1,600 query-object tasks.  Runtime acquisition was
answer-blind and used only raw frames, the public target ontology, off-the-shelf
TEMPURA SGDET, off-the-shelf Charades SlowFast, stable tracks, and a deterministic
typed-evidence verifier.  It made no provider calls.

| Metric | Result | Frozen gate |
|---|---:|---:|
| supported candidates | 460/1600 | at least 256 |
| supported coverage | 28.75% | at least 25% |
| supported precision | 67.61% | diagnostic |
| Wilson 95% lower bound | 63.20% | at least 60% |
| entity candidate-pool recall | 90.63% | pass |
| typed role fidelity | 84.77% | pass |
| cross-frame dedup fidelity | 100% | pass |
| source commit coverage | 25.06% | at least 24% |
| matched-permuted commits | 0% | at most 5% |

The qualification status is
`QUERY_GROUNDER_V2_STRICT_BOUNDARY_QUALIFIED`.  Qualification outcomes were
opened only after grounding and pre-outcome decisions were frozen; the cohort
was development evidence, not transfer evidence.

## V14 untouched formal

V14 selected 512 new videos / 1,024 tasks from the official balanced-train split.
All selected videos were disjoint from V13 and from every prior raw-video runtime.
The five arms shared identical videos, frame budgets, parser, SGDET and SlowFast
receipts, typed executor, and frozen 9B fallback.  Only the symbolic Harness arm
changed.  All five-arm decisions were frozen before the evaluator first read an
answer.

| Arm | Correct | Accuracy | Symbolic commits |
|---|---:|---:|---:|
| neural-only | 245/1024 | 23.93% | 0 |
| source-permuted | 245/1024 | 23.93% | 0 |
| generic scaffold | 246/1024 | 24.02% | 285 |
| game source-induced | 246/1024 | 24.02% | 285 |
| target-written isomorphic | 246/1024 | 24.02% | 285 |

Source versus neural/permuted was only `1W/0L`, exact two-sided `p=1.0`.
Negative transfer was `0/1024`, and target-written isomorphic equivalence was
100%, but both preregistered significance gates failed.  The secondary `>55%`
target also failed.

The key anti-collapse audit is:

- source committed on 285 tasks;
- source and neural predictions were identical on 283 of those tasks;
- only 2/1,024 total predictions differed;
- one disagreement recovered the answer, while the other left both arms wrong.

Thus V14 does not show that the transferred controller improves QA.  Candidate
precision alone was insufficient because it measured whether the grounder had
already recovered the answer, not whether the Harness added reasoning value.

## Protocol correction

Future formals now include an outcome-blind pre-outcome gate requiring source and
neural predictions to differ on at least 5% of the frozen cohort.  This is a
minimum non-trivial decision-change opportunity, not a success metric.  Had it
been present in V14, `2/1024 = 0.20%` would have stopped the run before answers
were opened.

The next scientifically valid AGQA route must expose a question-blind multi-event
graph and require temporal/compositional execution.  A query-conditioned grounder
may localize evidence, but it must not collapse the target task to one final
candidate before the Harness.  Development must establish both grounding fidelity
and incremental source-versus-neural decision opportunity before another fresh
reserve is allocated.

The earlier AGQA typed-temporal replication remains separate positive evidence:
`135/256` source versus `106/256` neural, `34W/5L`, `p=2.43e-6`.  V14 neither
overwrites nor strengthens that claim; it is an explicit negative single-hop
grounder ablation.

## Cost and immutable artifacts

V14 used 3,822 local GPU-seconds in total: 1,777 RTX A4000 seconds, 1,770 RTX
A5000 seconds, and 275 RTX A6000 seconds.  Provider calls and provider cost were
zero.  No local-GPU dollar cost is claimed.

- V13 qualification file SHA256:
  `af5418b42015c1c329fc9896e2a75062813f500c09f0582ff29996a50a622d40`
- V14 protocol file SHA256:
  `48288a445e8b168523f4a45073149673948b372c455668e431e4bd76cbb69543`
- V14 formal report semantic SHA256:
  `03630cb9c86f63d772671fb275b1d66fb418adfe24db4358cf428f2341c5754c`
- V14 formal report file SHA256:
  `9c531b6368263f66056b9e6ac566650f611493064870b685bac0c1f7f6ddf26a`
- corrected cost receipt semantic SHA256:
  `749996ef4d70adf8fe1c3b8488ffbb6383db713184d0c32b68a6c711a842ff3e`
- compact paper bundle semantic SHA256:
  `2d408d664fde3ffdee28cb4f2849fe70879d2adc0ad3c471a6d8eece26b2bea1`
