# ALFWorld weak receipt-grounded knowledge teacher pilot

## Status

The canonical four-task pilot completed successfully on Slurm job `7122688`.
All four array jobs exited with code zero, all six conditions were present, every
condition within a task had the same initial-state hash, and no runtime condition
reported an error.

This is a discovery pilot, not held-out evidence and not yet a one-shot result.
The Harness teacher received four canonical ALFWorld adaptation artifacts. One
artifact is a partial episode: it contains eight valid official transitions and
then ends because the collection actor proposed an out-of-range action. This
lineage is stored explicitly in every output artifact.

## Question

The pilot asks whether target test-time reasoning can benefit from weak,
receipt-grounded source knowledge. It does not claim to transfer a game policy
or executable game skill.

The source object is discovery-provisional knowledge induced from real receipts
from two Phase-1 games. The target teacher can instantiate a target-side
hypothesis from this source object and target adaptation receipts, but it cannot
select a target action. The Qwen Decision Agent remains the only component with
action authority.

## Frozen conditions

Each condition received an exactly token-matched context:

1. `target_only`
2. `generic_reasoning`
3. `source_receipts_only`
4. `authentic_weak_control_prior`
5. `shuffled_evidence_prior`
6. `other_game_abstain`

GPT-5-mini initialized and audited the weak hypothesis. Qwen3.5-35B-A3B selected
only ALFWorld-native actions. Every executed action was checked using the live
ALFWorld transition and official outcome.

Source receipt hashes were exposed to the teacher only through short aliases.
The code mapped valid aliases back to immutable receipt hashes. Unknown verdicts
were conservatively converted to `ADMIT` and recorded as protocol violations;
they were never interpreted through a semantic synonym table.

## Canonical results

| Condition | Official success | Mean steps | Mean repeated actions |
|---|---:|---:|---:|
| target only | 2/4 | 20.75 | 13.00 |
| generic reasoning | 2/4 | 18.50 | 13.50 |
| source receipts only | 2/4 | 18.50 | 12.75 |
| authentic weak control prior | **3/4** | **14.50** | **7.50** |
| shuffled evidence prior | 2/4 | 18.50 | 13.25 |
| empty/abstain control | 2/4 | 21.25 | 14.00 |

Per-task official outcomes and steps:

| Task | Target | Generic | Raw receipts | Authentic | Shuffled | Empty/abstain |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | fail/30 | fail/30 | fail/30 | **success/16** | fail/30 | fail/30 |
| 1 | success/4 | success/5 | success/7 | success/4 | success/4 | success/4 |
| 2 | success/19 | success/9 | **success/7** | success/8 | success/10 | success/21 |
| 3 | fail/30 | fail/30 | fail/30 | fail/30 | fail/30 | fail/30 |

The final canonical run cost approximately USD 1.31: USD 0.96 estimated for
GPT-5-mini and USD 0.35 reported by OpenRouter for the Decision Agent. This does
not include earlier interface-debugging runs.

## Concrete interpretation

Task 0 is the only clean structure-specific positive example in this pilot.
The authentic condition succeeded in 16 steps while all five controls exhausted
the 30-step budget. Its initialization had valid source receipt support; it had
no protocol violation, no source fallback, and no forced Harness replan. The
Decision Agent consumed a weak advisory about escaping a repeated zero-reward
observation loop, stopped repeating desk/lamp inspection, navigated to another
desk, acquired the relevant object, and completed the official task.

This is promising because raw receipts, a generic loop-escape instruction,
shuffled clause-to-receipt evidence, and an empty teacher all failed on the same
initial state. It is nevertheless only one task and one stochastic actor
rollout. The generic, authentic, and shuffled target claims were also
linguistically similar. The result can therefore still be a wording-sensitive
actor trajectory rather than a repeatable effect of correct source structure.

Task 2 provides a weaker result. Authentic was faster than target-only, generic,
shuffled, and empty controls, but raw source receipts were one step faster than
authentic. It supports possible value from source context, not incremental value
from the induced knowledge object.

Task 1 is a ceiling case. Every condition succeeded, and authentic did not beat
the target-only or empty controls. Task 3 is a floor case. Every condition
failed; moreover, the authentic initializer abstained after it could not produce
a valid source-supported alias set, so this was not an active treatment.

The correct current claim is:

> A strong no-action-authority Harness can instantiate and safely expose
> receipt-grounded source knowledge to target test-time reasoning. One of four
> qualification tasks shows a clean, structure-specific positive trajectory,
> but the pilot does not yet establish a repeatable transfer effect.

## Frozen task-0 replication

The task-0 hypothesis artifact from the canonical run was frozen byte-for-byte
(SHA-256
`3f4e524374b958ac17784b8e05275d66823a5c1027366278488c2c03c587b705`).
Ten actor seeds (`92000`--`92009`) reused the same ALFWorld environment seed
(`81000`), initial-state hash, source artifacts, adaptation receipts, hypotheses,
prompts, token budget, action budget, and six conditions. Hypothesis
initialization was loaded from the frozen artifact and made no new teacher call.

One raw-receipt run at seed `92004` produced malformed Decision JSON at step 27.
The retry logic had been issuing an identical memoized request, so it could not
repair invalid JSON. The retry request now carries the previous parse error and
an explicit schema-repair instruction. Only that pre-declared
seed/condition pair was rerun with the same frozen inputs; no outcome-based rerun
was performed. All ten final replicates pass the lineage and hash checks.

| Condition | Official success | Mean steps | Mean repeated actions |
|---|---:|---:|---:|
| target only | 0/10 | 30.0 | 26.4 |
| generic reasoning | **2/10** | **26.5** | 20.2 |
| source receipts only | 0/10 | 30.0 | 24.5 |
| authentic weak control prior | 1/10 | 27.6 | **19.6** |
| shuffled evidence prior | 0/10 | 30.0 | 23.9 |
| empty/abstain control | 0/10 | 30.0 | 26.1 |

Authentic versus target-only has one win, zero losses, and nine success ties
(one-sided exact binomial `p=0.5`). Authentic versus generic has zero wins, one
loss, and nine ties (`p=1.0` in the hypothesized direction). The sole authentic
success occurred at seed `92006` in six steps, but generic reasoning also
succeeded at that seed in 14 steps. Generic also succeeded at seed `92000`,
where authentic failed.

The frozen replication therefore does **not** reproduce a source-specific
benefit. The canonical task-0 trajectory is best treated as
prompt/sampling-sensitive trajectory variance, not as evidence that authentic
source receipts transferred useful knowledge. Fewer repeated actions under
authentic are diagnostic only: they did not produce a success advantage and
were not unique to authentic.

The selected, integrity-checked replication artifacts cost an estimated USD
5.38: USD 3.84 for GPT-5-mini teacher calls and USD 1.55 reported by OpenRouter
for the Decision Agent. This excludes the failed raw-receipt attempt, smoke
runs, and earlier debugging, so it is not total project spend.

## Bugs found and fixed during the pilot

- Long source hashes caused fabricated receipt citations. Source identifiers are
  now short aliases with mechanical reverse mapping.
- The teacher sometimes evaluated ALFWorld task semantics and suggested target
  commands. Review is now limited to the supplied abstract control precondition
  and observable transition prediction.
- Verification previously credited evidence without checking whether the
  hypothesis precondition held. Missing or partial preconditions now force
  `INCONCLUSIVE`.
- Teacher outputs such as `PASS` and `ACCEPT` caused whole-episode source
  fallback. Unknown verdicts now safely become `ADMIT` and are audited as
  protocol violations.
- Adaptation discovery accidentally matched
  `task_2.failed_schema_256tokens.json`. Selection now accepts only exact
  `task_<integer>.json` filenames and records artifact hashes and collection
  errors.

The invalid and partial runs remain available as debugging receipts but must not
be mixed with the canonical result.

## Decision after replication

Do not extend this unchanged task-0 protocol to 20 seeds, do not reduce it to a
one-shot claim, and do not train a Harness LoRA from its trajectories. More
samples of the same mechanism would estimate a near-zero effect more precisely
without addressing the mechanism failure.

The next admissible experiment must be a new, pre-registered mechanism rather
than another prompt adjustment on task 0. It should first diagnose, on a
disjoint adaptation split, whether a target-side Harness can learn a
receipt-verifiable failure detector or information need that predicts future
official progress. Source knowledge may then be added as a separately frozen
prior and must beat the same target-only and generic controls on held-out target
tasks. If it does not reduce examples-to-success or environment steps on
ALFWorld and a second far domain, the source-derived component should be
dropped; a target-trained adaptive Harness may still be studied, but it is no
longer evidence of cross-domain source benefit.

## Artifacts

- Canonical episode artifacts:
  `runs/receipt_knowledge_teacher_pilot_v1/alfworld/full_v5_canonical/`
- Machine-readable summary:
  `runs/receipt_knowledge_teacher_pilot_v1/alfworld/pilot_summary_v5_canonical.json`
- Frozen replication episodes:
  `runs/alfworld_task0_frozen_replication_v1/`
- Frozen replication summary:
  `runs/alfworld_task0_frozen_replication_v1/summary.json`
- Source knowledge:
  `runs/receipt_knowledge_teacher_pilot_v1/source/joint_v4.json`
- Runner:
  `scripts/run_alfworld_weak_knowledge_teacher_pilot.py`
- Summarizer:
  `scripts/summarize_alfworld_weak_knowledge_teacher_pilot.py`
- Frozen replication summarizer:
  `scripts/summarize_alfworld_task0_frozen_replication.py`
- Slurm launcher:
  `cluster/run_alfworld_weak_knowledge_teacher_pilot_v1.sbatch`
- Frozen replication launcher:
  `cluster/run_alfworld_task0_frozen_replication_v1.sbatch`
