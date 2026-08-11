# Real-source typed IR to ALFWorld: V5 result and V6 stop

## Bottom line

The transfer is real and nontrivial at the intervention level, but it does **not**
improve target success.  On the already-consumed 24-task ALFWorld qualification
split, target-only solved 16 tasks and the authentic real-source typed IR solved
13.  The paired result was 1 win, 19 ties, and 4 losses.  The target heldout split
was not read.

This is useful negative evidence.  The authentic graph was much better than two
structurally invalid graphs (both 0/24), so graph semantics matter.  Nevertheless,
"better than corrupted symbols" is not evidence of useful transfer when the
authentic graph is worse than the target-only neural policy.

## What was transferred

The source artifact is V4 IR
`5a7a48a6cc083a9b8caffbcb02fa721971a28d2052f276c847bb9296641942ea`.
It was induced from intervention forks in four real simulator tasks:

- MiniGrid DoorKey and UnlockPickup support `BIND -> MUTATE` under
  `CARRIER_BOUND`.
- MiniGrid PutNear and MiniWorld PutNext support `BIND -> RELATE` under the same
  guard.

No source action ordinal, coordinate, environment ID, source probe weight, or
source oracle policy entered ALFWorld.  The source selected only typed effects;
the adaptation-only masked ALFWorld grounder mapped them to target-native actions.
Mutating the forbidden `required_option` diagnostic left every authentic decision
unchanged.

## Frozen V5 result

| Condition | Official success | Mean steps | Changed-effect rate | Source admission |
|---|---:|---:|---:|---:|
| target-only | 16/24 | 42.00 | 0.00% | 0.00% |
| authentic typed IR | 13/24 | 46.75 | 4.28% | 11.94% |
| edge-permuted IR | 0/24 | 70.00 | 10.18% | 0.00% |
| wrong-guard IR | 0/24 | 70.00 | 10.83% | 0.00% |

The authentic-to-target-only paired comparison is 1 win / 19 ties / 4 losses,
or -3 net wins.  The qualification report status is
`DEVELOPMENT_DIAGNOSTIC_NEGATIVE_STOP`, and its hash is
`5268a7d8f15120ebeb8827be80ec8460c26f85af77e7d715cf3070c78e6157fd`.

## Trace diagnosis

V5 exposed two separate problems.

First, the concrete target action was ranked by a new product of applicability,
completion, and binding scores after the source selected an effect.  That changed
12 actions even when the source-selected effect already equaled the target-only
effect.  The experiment therefore mixed a symbolic effect intervention with a
different concrete-action objective.

Second, admission used only an absolute realization threshold.  Among the 52
source-admitted decisions that changed a concrete action, 28 selected an action
whose target-policy score was below 10% of the target-only fallback score; 21 were
below 1%.  The Vase regression is the cleanest example: at step 39, the Harness
replaced `close cabinet 1` (target-policy score 0.9493) with
`take box 1 from cabinet 2` (score `4.38e-9`) because the generic `BIND` score
passed the absolute threshold.

The remaining failure is deeper than admission.  V4 merges two task-conditioned
source continuations under the same `CARRIER_BOUND` guard.  In a cool-Mug task,
the authentic Harness used the direct `BIND -> RELATE` edge and moved an uncooled
mug to the coffeemachine.  In another cool-Mug trace, the generic `MUTATE` node
accepted heat and clean operations.  Thus `BIND`, `MUTATE`, and `RELATE` are still
too coarse: they omit goal roles, mutation subtype, and the context that selects
one source edge rather than the other.

## V6 causal fix and why it did not run online

V6 fixes the concrete-action confound:

1. the source graph selects only an effect;
2. the same target policy used by target-only selects the concrete action within
   that effect; and
3. a source effect can override the fallback only if its target-policy score has
   sufficient relative support.

The old V5 behavior remains the default and reproduces its frozen Harness byte for
byte.  The V6 threshold and policy ratio are selected using adaptation-train only.
The selected pair was realization threshold 0.05 and policy ratio 0.95.  It looked
nontrivial on adaptation-train (20 changed effects in 991 states), but on the
separate adaptation-validation set it changed only 2 of 306 effects (0.65%).  Its
effect accuracy and action top-1 were exactly equal to target-only.  The
nontriviality gate therefore blocked V6 before any online reset.  Harness hash:
`211afc460c597717a5b08314e61dd8eeb9b87cf0dc2a89b79b695d8bb6249028`.

This is the correct stop: the safe version collapsed toward the null policy, so
running qualification and calling it transfer would be misleading.

## Next falsifiable experiment

Do not add more tasks that only reproduce the same coarse nodes.  Preserve source
edge context and induce parameterized predicates:

```text
BIND(x : goal_object)
ACHIEVE_UNARY_GOAL(x, property)  [guard: unary goal unsatisfied]
RELATE(x, y : goal_receptacle)   [guard: required unary goals satisfied]
```

The real-source lineage should keep DoorKey/UnlockPickup's guarded unary-state
continuation separate from PutNear/PutNext's direct relational continuation.  A
target-native neural edge router, trained only on target adaptation trajectories,
should ground `x`, `y`, the required property, and which source continuation is
applicable.  Its output must be calibrated against target-only expected value and
must abstain when uncertain.

Before evaluating that candidate, freeze a new outcome-blind ALFWorld confirmation
split disjoint from adaptation and both existing qualification/heldout lists.  The
current heldout remains untouched, but it is no longer a clean final test for any
candidate designed after observing V5 qualification.
