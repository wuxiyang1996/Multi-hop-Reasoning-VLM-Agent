# Relation-edge intervention value V13: insufficient and heterogeneous

Date: 2026-08-12

## Verdict

V13 replaced whole-trajectory labels with matched intervention forks at the
same consumed ALFWorld state.  It reconstructed 62 first actionable source-edge
opportunities from disjoint V9--V12 task groups and ran 124 cells:

```text
same task + seed + exact action prefix + verified fork-state hash
  ├─ execute one target-grounded source RELATE edge
  └─ abstain once to the target fallback
then use the identical source-edge-disabled continuation policy
```

All fork-state, source/control action, feature, and paired-cell invariants
passed.  Only 10/62 opportunities changed the native action.  The other 52
source-edge decisions selected exactly the target fallback action and therefore
contained no treatment contrast.

Among the ten informative task forks, three had positive utility, five had
negative utility, and two were neutral.  Success effects were exactly balanced:
two rescues and two harms.  A frozen ridge value head improved aggregate
cross-fitted utility but still selected one held-out success loss and failed
the data-sufficiency and per-version safety gates.

The consumed audit stopped.  No fresh V13 manifest was frozen, the preserved
confirmation split was not opened, and ALFWorld `valid_unseen` remains unread.

## Why this estimand is stronger

V9--V12 compared complete closed-loop policies.  When authentic and node-only
outcomes differed, the difference could combine source edges, target slot
safety, later interventions, and path-dependent continuation.  V13 isolates
one source edge at one observed state and gives both branches the same policy
after that action.

The frozen task-level unit is the first target-actionable relation edge within
a 60-step endpoint.  Opportunity selection used only consumed decision fields,
prefix actions, identities, and target-native grounding availability.  It did
not inspect fork outcomes or episode official success.  The pre-action feature
schema was fixed before collection and contains:

- budget and episode position;
- completed and remaining slot fractions;
- source/fallback neural policy, completion, binding, and applicability scores;
- source-versus-fallback score margins;
- realization score and native action-set size;
- fallback effect class.

No official outcome, after-state, target task content, task identity, or source
game label is a model feature.

## Infrastructure corrections before evidence

Two fail-closed retries occurred before any fork treatment action was executed:

1. The first runner assumed ALFWorld would reset in the input list order.
   ALFWorld exposes the actual game through `extra.gamefile` but may internally
   reorder the batch.  The runner was fixed to map the actual path to exactly
   one frozen task independently within each treatment.
2. The first plan selected a routed edge even when no target-native RELATE
   action had passed grounding/scoring.  Such a state has no executable source
   treatment.  Opportunity selection was tightened to the first edge carrying
   both realization and target-policy-ratio receipts.

After each correction, a new plan bound the changed implementation hashes.  The
successful plan was frozen before any outcome-producing branch.  Failed plans
and attempts wrote no result report.

## Matched causal results

The ten genuine action contrasts were:

| Version | Task | Step | Source | Abstain | Effect |
|---|---|---:|---:|---:|---|
| V9 | Pencil -> Drawer | 31 | fail@60 | fail@60 | neutral/negative progress |
| V9 | Potato -> Fridge | 15 | success@39 | fail@60 | source rescue |
| V9 | RemoteControl -> ArmChair | 9 | success@28 | success@31 | source +3 steps |
| V10 | Plate -> CoffeeTable | 13 | success@45 | success@44 | source -1 step |
| V10 | Bread -> Fridge | 9 | success@53 | fail@60 | source rescue |
| V10 | TissueBox -> Drawer | 6 | fail@60 | success@49 | source harm |
| V11 | Lettuce -> Fridge | 14 | fail@60 | success@38 | source harm |
| V11 | Candle -> Cabinet | 5 | success@55 | success@51 | source -4 steps |
| V12 | Pencil -> Drawer | 15 | fail@60 | fail@60 | neutral |
| V12 | DishSponge -> Cart | 18 | fail@60 | fail@60 | neutral |

Every source branch executed a native `move object to receptacle` action and
observed `RELATE_SLOT_CLOSED`; each paired abstention used the frozen target
fallback and observed `IGNORE` at the fork.  Thus the treatment manipulation
was operational even when eventual utility was zero or negative.

The most important correction is V11 Lettuce.  Complete V12 replay previously
made authentic look three steps faster than node-only.  The matched one-edge
fork instead shows that the isolated edge changes success to failure.  Later
source decisions and path dependence, not this single edge alone, produced the
whole-trajectory result.  Whole-episode outcome is therefore not a valid label
for intervention-level applicability.

## Frozen value audit

The model was standardized linear ridge regression with `L2=1`, a zero
admission threshold, and leave-one-version-out evaluation.  Utility made
official success primary, with bounded normalized-step and slot-progress tie
breaks.

| Policy on 10 informative forks | Selected | Success wins/losses | Success delta | Utility |
|---|---:|---:|---:|---:|
| admit all | 10 | 2 / 2 | 0 | -0.0117 |
| V12 step >= 9 | 8 | 2 / 1 | +1 | 1.0133 |
| cross-fitted value head | 6 | 2 / 1 | +1 | 1.0200 |

Cross-fitting selected only Bread on held-out V10 and obtained a rescue.  On
held-out V11 it selected only Lettuce and caused a success loss.  Held-out V12
was neutral; held-out V9 had one rescue plus a step improvement.  The value
head therefore does not provide calibrated cross-version safety.

Seven preregistered gates failed:

- informative task forks at least 32: observed 10;
- positive-utility tasks at least four: observed three;
- selected tasks at least eight: observed six;
- aggregate selected success delta at least two: observed one;
- nonnegative selected success delta and utility in every held-out version;
- zero selected success losses.

The head did beat admit-all and the step-9 rule on aggregate utility, but those
secondary gates cannot override insufficient data or a held-out success harm.

## Bitter lesson and next target

The main bottleneck is now upstream of value-model architecture.  A discovered
symbolic edge is not automatically a useful training intervention:

```text
62 target-actionable source-edge opportunities
  -> 52 choose the same native action as target fallback
  -> 10 genuine action contrasts
  -> 3 positive, 5 negative, 2 neutral utilities
```

Adding a larger neural head would overfit ten heterogeneous examples.  Adding
more source-game rollouts supporting the same `BIND -> RELATE` fact would not
create target action contrast.  This relation edge should not receive another
fresh ALFWorld split.

The next experiment should transfer a structure with more causal leverage and
more frequent target disagreement, then require a contrast-rate gate before
any expensive value collection.  Strong candidates are:

1. subgoal ordering for multi-object tasks;
2. recovery after a failed or contradictory postcondition;
3. information-gathering actions when the neural fallback is `examine`/search;
4. budget-aware commitment versus continued exploration.

For each candidate, perform outcome-blind opportunity enumeration first.  Do
not proceed unless at least 32 task-independent states yield distinct authentic
and target-control native actions across at least three source/task groups.
Only then collect matched outcomes and fit a selective value model.

## Audit hashes

- frozen plan: `1abe65c97ae3d566e3913e4dadafab10fc96168247f6df415bdf774c6ab3013c`
- frozen plan file: `ab830df8ec01e1893b7f64c9742b6f15d1f1d71d299f0754668475c785318ac0`
- matched fork report: `9fe79c9f0f7dda9cb6cb2d6d19a32785ac25d061adf940a5864b04a3cb5534cf`
- grouped value audit: `bff0740617592bfbf3bf9dc35b22392fc4da522863706c1c080c23a859a2344b`

The frozen outcome-blind plan is
`configs/relation_edge_intervention_fork_plan_v13.json`.  The compact result is
`docs/results/relation_edge_intervention_value_v13_summary.json`; full consumed
fork trajectories remain under the ignored `runs/relation_edge_value_v13/`
directory.
