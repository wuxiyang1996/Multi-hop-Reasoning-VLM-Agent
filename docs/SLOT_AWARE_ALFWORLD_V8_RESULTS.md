# Slot-aware ALFWorld V8: negative confirmation and source-IR audit

Date: 2026-08-11

## Verdict

V8 does **not** demonstrate cross-domain neural-symbolic skill transfer.

The frozen relational adaptation gate improved official success from `14/16`
to `15/16`, but the one-shot fresh confirmation produced `17/24` in every
condition.  Authentic transfer admitted 11 source effects but changed zero
actions and zero effects.  All 24 paired success outcomes tied target-only.

A post-confirmation code audit found a stronger reason to reject the transfer
claim: the V8 decision function validates `source_ir`, but does not execute its
nodes or edges.  The requested `BIND`, `MUTATE`, or `RELATE` effect is generated
by a target-side hard-coded slot controller.  The relational adaptation result
therefore remains useful evidence for the slot ledger and intervention harness,
but it is not evidence that source-game symbolic structure transferred.

The existing ALFWorld `valid_unseen` split was not reset or read.

## Frozen protocol

V8 froze task identities before reset, excluded task IDs found in prior logs,
kept adaptation and confirmation disjoint, and bound the source report, target
grounder, manifest, thresholds, runner, and slot-controller code by hash.
Official success was recorded only after an episode and was never available to
action selection.

The original manifest reserved 18 adaptation and 24 confirmation tasks across
six ALFWorld families.  After two consumed full-graph development gates, a
relation-only revision selected 16 new adaptation tasks from
`pick_and_place_simple` and `pick_two_obj_and_place`, while preserving the same
unread 24-task confirmation split.

The relation-only claim was:

```text
source-game BIND(instance, slot) -> RELATE(instance, receptacle, slot)
    -> target-native goal/slot state
    -> target-native neural action grounding
    -> observed target postcondition receipt
```

`MUTATE` and property-transfer claims were explicitly excluded after the two
full-graph gates failed.

## Results

| Phase | target-only | authentic | edge control | property control | authentic changed effects | Decision |
|---|---:|---:|---:|---:|---:|---|
| Full gate 1, 18 tasks | 12 | 13 | 12 | 13 | 3 | failed; 2 reopened slots |
| Full gate 2, 30 fresh tasks | 17 | 17 | 16 | 18 | 13 | failed; no changed `MUTATE` |
| Relational gate, 16 fresh tasks | 14 | 15 | 14 | 14 | 4 | passed the preregistered gate |
| Fresh confirmation, 24 tasks | 17 | 17 | 17 | 17 | 0 | **negative stop** |

On the relational adaptation gate, authentic changed one `BIND` and three
`RELATE` decisions on four tasks.  It rescued one SprayBottle task
(`120`-step target failure to a `95`-step success) and shortened one Spoon task
from 56 to 42 steps, while making two already-successful tasks one and four
steps longer.  These are real target action interventions, but the audit below
shows that their symbolic routing came from target-side code rather than the
bound source graph.

Fresh confirmation produced:

- official success: `17/24` for all four conditions;
- authentic versus target-only: `0` wins, `24` ties, `0` losses;
- authentic source admissions: 6 `BIND` plus 5 `RELATE`;
- authentic action/effect changes: `0`;
- authentic source admission rate: `0.720%`;
- completed-slot reopenings: 2 authentic, 3 target-only;
- selected postcondition failures: 0;
- required-option invariance: 1.0.

The property-permuted control changed three `BIND` actions on two light tasks,
but still achieved `17/24`.  It was faster on average (`61.75` versus `64.42`
target-only steps) only because the permutation maps `LIGHT` to the active
`NONE` relation scope.  This is a control leakage diagnostic, not authentic
transfer.

## Why fresh confirmation failed

### 1. The source graph had no operational authority

`choose_slot_aware_action()` calls `validate_slot_source_ir(source_ir)`, then
chooses the requested effect with the target-side `_requested_effect(ledger)`.
No source node, edge, guard, or supporting receipt is read when routing the
effect.  The edge control is also a hard-coded effect permutation rather than
execution of a transformed source graph.

This is the central failure.  A serialized IR is not a transferred skill merely
because it is present in the artifact.  Changing or deleting a source edge must
change an eligible decision, and this must be tested before any expensive
rollout.

### 2. Admissions mostly certified target-policy agreement

All 11 authentic admissions selected the same action and effect as the target
fallback.  They show that the source label was compatible with already-correct
target actions, not that the transferred structure improved control.

The adaptation split happened to contain four states where the target fallback
had a different effect and the slot controller intervened.  The confirmation
split contained no such authentic state.  An adaptation-only intervention count
therefore did not establish a reusable transfer mechanism.

### 3. Router confidence disabled the invariant on hard relation tasks

The inherited minimum property confidence was `0.8`.  Four of the eight
relation-family confirmation tasks fell below it.  This included the only
failed relation task, DishSponge (`0.706`), where the raw policy took a completed
object back out of the target receptacle.  Because safety was coupled to router
confidence, the completed-slot shield was disabled and the ledger recorded a
reopened slot.

The goal grammar already establishes that these tasks require relation with no
unary property.  Neural confidence may govern action grounding, but it should
not disable a directly observed completed-slot invariant.

### 4. Scope metadata was not enforced as a runtime family boundary

`primary_target_families` was descriptive metadata.  Runtime activation used
only the predicted property.  Under the property permutation, light goals were
mapped to `NONE` and entered the relation controller, producing the only three
changed actions in confirmation.

## What V8 did establish

V8 still leaves reusable infrastructure:

- an instance-aware, count-aware target slot ledger;
- state updates gated by observed target-native postconditions;
- completed-object protection and explicit reopening metrics;
- hash-bound manifests, candidates, gates, and one-shot confirmation;
- paired authentic, target-only, edge-permuted, and router-permuted traces;
- an auditable distinction among admission, action change, effect change, and
  official success.

These are prerequisites for a valid test, not a positive transfer result.

## Required V9 fixes

V9 should not tune V8 on the same confirmation set.  That set is now consumed
development data.  A new candidate must be frozen before new task resets and
must satisfy all of the following before confirmation is authorized:

1. Compile the bound source IR into an executable symbolic transition table.
   Start effects and successor effects must be obtained from source nodes and
   guarded source edges, not from a target-coded effect sequence.
2. Make edge ablation/permutation operate on that exact compiled graph.  Unit
   tests must prove that removing `BIND -> RELATE` removes a relation decision.
3. Keep goal-role parsing and all action realization target-native.  Neural
   target scores may bind a symbolic effect to a native action, but may not
   manufacture the source transition.
4. Apply the completed-slot safety invariant from observed relation state even
   when the property router is uncertain.  Report this target-native shield
   separately from source transfer.
5. Enforce task/scope compatibility from parsed target goal structure at
   runtime; do not let a property-control permutation turn a light task into a
   relation task.
6. Require action-changing interventions on multiple fresh adaptation tasks,
   positive paired success, and superiority to graph controls.  Agreement-only
   admissions do not count.

## Audit hashes

- original manifest: `f7f211717fb0b952592ac02ba7d2a2eb86539c71f14798244d5954caf2976f17`
- relational revision manifest: `b80717d1fd19fb4013d0f2a2b46c21ff5af38fe5958f4af81ad98a426b7a1973`
- relational candidate: `7b710937f9aeda9fd97b1b5a85869144962d5366ae5a8cbe02d3c353193d63fa`
- authorized relational harness: `c6835fdaa98cf6f1652b6bd88cc594fc541c51f33899d46d2cf6f4b2ab464bca`
- relational adaptation report: `57adfba48ff01c0ecc31ef5c3bf575fb702271b01b32fc95ac13ab8dd7771bd1`
- fresh negative report: `1a40a9db97bc072723d144a568c858da2353f9f25826d8cccd56b2f1019e42dd`

The detailed runtime reports remain under `runs/slot_aware_alfworld_v8/` and
are intentionally not committed because they contain full step traces.  The
compact machine-readable result is
`docs/results/slot_aware_alfworld_v8_summary.json`.
