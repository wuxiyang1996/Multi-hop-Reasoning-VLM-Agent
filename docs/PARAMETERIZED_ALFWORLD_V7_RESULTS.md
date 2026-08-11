# Parameterized real-source IR to ALFWorld: V7 fresh confirmation

## Result

V7 is a clean negative result.  It fixes the V5 action-ranking confound, preserves
source edge context, adds target-native neural property routing, and replaces a
generic carrier state with role-aware target receipts.  These changes make the
transfer safer and sparser, but they do not improve official success on a fresh
24-task confirmation split.

| Condition | Official success | Mean steps | Changed effects | Changed tasks |
|---|---:|---:|---:|---:|
| target-only | 16/24 | 58.500 | 0 | 0 |
| authentic parameterized IR | 16/24 | 57.625 | 12 | 5 |
| edge-permuted IR | 16/24 | 59.000 | 1 | 1 |
| property-permuted router | 16/24 | 57.792 | 16 | 6 |

Authentic versus target-only is 1 win / 22 ties / 1 loss, or zero net wins.
Authentic and property-permuted have identical outcomes on all 24 tasks.  The
result status is `FRESH_CONFIRMATION_NEGATIVE_STOP`; report SHA256 is
`cf9d200567438a819ab98bb349bb9e094d5f245ad3437262ead9a4113abe7972`.

This is nonzero behavior transfer, not useful success-rate transfer.  All 12
authentic changed effects were `POSITION -> BIND`; no fresh action changed because
of the parameterized `MUTATE` or `RELATE` continuation.  Consequently the fresh
experiment does not establish value for correct property semantics.

## Fresh-data boundary

The confirmation manifest was frozen before any selected-task reset.  ALFWorld's
134 `valid_unseen` tasks had only four task IDs absent from old logs, and all 140
`valid_seen` IDs were already present.  V7 therefore uses a clearly labeled
in-distribution confirmation set drawn from previously unseen train instances:

- six task families, four tasks per family;
- selection uses relative path names and SHA256 ranking only;
- 332 task IDs found in existing configs, runs, and docs were excluded;
- selected-task intersection with prior logs is zero;
- manifest SHA256:
  `b3600e9387786d75ad9b3cfd0d4835072ffca9659c996019503eb026ff6c5d6d`.

This is not a `valid_unseen` OOD claim.  The existing 24-task valid-unseen heldout
remains unread and is prohibited by both the Harness and runner.

## Parameterized neural-symbolic mechanism

Parent real-source IR:
`5a7a48a6cc083a9b8caffbcb02fa721971a28d2052f276c847bb9296641942ea`.
V7 preserves its source receipt lineage and parameterizes its structure as:

```text
BIND(goal_object)
  --[target neural unary goal unsatisfied]-->
ACHIEVE_UNARY_GOAL(goal_object, required_property)
  --[target neural unary goal satisfied]-->
RELATE(goal_object, goal_receptacle)

BIND(goal_object)
  --[no unary goal or already satisfied]-->
RELATE(goal_object, goal_receptacle)
```

Parameterized IR SHA256:
`25d3cce470c479b56db8d04174a65e29541dab081ae3b2664e94a08f9686fd18`.
No source action, coordinate, source task ID, mission, or source model weight is
available at target runtime.

The target-native property router is a small MLP trained on 36 target adaptation
expert trajectories.  Its inputs are hashed target goal unigrams and bigrams;
labels are mechanically extracted from expert target actions as
`NONE/CLEAN/HEAT/COOL/LIGHT`.  It scored 11/12 on the fresh adaptation gate.
Concrete ALFWorld actions are still chosen by the same frozen target policy as the
target-only condition, restricted to role- and property-compatible candidates.

Final Harness SHA256:
`5d80aa99b355aaf4f2de515c9cdca2c4b966f2b6c0a39d774998f0dcdb96d47b`.
Frozen thresholds are property confidence 0.8, role binding 0.5, realization
score 0.1, and target-policy ratio 0.05.  Mutating the forbidden target
`required_option` diagnostic changed zero authentic decisions.

## Pre-confirmation gate history

All changes below occurred before the 24 confirmation tasks were reset.

1. A first adaptation-only candidate used a state-rate nontriviality requirement.
   Because hundreds of SEARCH states dominate the denominator, it selected an
   unsafe zero realization threshold and failed effect noninferiority.
2. Replacing state rate with sparse intervention counts selected realization 0.1,
   but an arbitrary conservative tie-break chose policy ratio 0.95 and reduced
   validation changes to zero.
3. A new outcome-blind 12-task adaptation gate was frozen, disjoint from both old
   logs and confirmation.  The final 0.1/0.05 candidate changed exactly two
   actions on two independent tasks; both were correct expert `RELATE` actions
   with policy ratios above 0.998.  Effect accuracy, action top-1, and non-position
   recall were noninferior or better than target-only.

The adaptation gate initially required changes across two target task families.
That was revised to two independent target tasks because the causal unit is a
task-level intervention, while family breadth is an evaluation result.  This
revision also occurred before confirmation.  The fresh confirmation report is the
first and only outcome read from its 24 selected tasks.

## Why V7 still does not work

The remaining failure is role granularity, not property classification.

The Bowl look-task rescue came from taking a target-type Bowl rather than
continuing navigation.  The PepperShaker regression also came from `BIND`: after
one object had already been handled, the Harness took a PepperShaker from shelf 1
instead of following the target policy toward shelf 3.  A role score that says
"this is a PepperShaker" cannot determine whether it fills an unmet goal slot or
reopens an already satisfied slot.

Thus `goal_object` is still too coarse for repeated-object tasks.  V7 tracks
object type and unary property, but not object identity, count, satisfied relation,
or remaining goal slots.  Its successful speed changes and its failure share the
same mechanism: a generic high-confidence `BIND` prior.  The property-permuted
control reproducing all authentic outcomes confirms that correct property routing
was not the value-producing component.

## Required next experiment

V8 should not rerun V7 on another split unchanged.  It needs an
intervention-grounded target progress predicate:

```text
BIND(x, slot)
guard: UNSATISFIED_GOAL_SLOT(slot)
postcondition: binding x reduces the predicted unsatisfied-slot count

RELATE(x, y, slot)
postcondition: the target-native transition closes slot without reopening another
```

The target grounder must predict marginal goal-predicate closure, not merely
lexical object binding.  The monitor should advance from observed postconditions,
including object instance and count, rather than action type alone.  Before a new
online run, adaptation gates must require nonzero changed decisions for every
claimed transferred continuation (`BIND`, unary-goal achievement, and `RELATE`),
with admission-matched controls.  A new outcome-blind confirmation split must then
be frozen; the existing valid-unseen heldout should remain reserved.
