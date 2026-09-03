# Selective budgeted relation-edge V12: fresh failure

Date: 2026-08-11

## Verdict

V12 learned a conservative applicability rule from already-consumed V9--V11
adaptation traces and passed both grouped cross-version audit and closed-loop
replay.  It then failed on the one fresh V12 adaptation split.

Authentic source-graph transfer achieved `12/16`, exactly the same as the
source-node-only and safety-only controls.  The authentic graph changed two
target actions through its guarded `BIND -> RELATE` edge, but both affected
tasks still failed.  The one gain over raw target (`11/16`) was shared by all
three safety/slot-aware conditions and therefore cannot be attributed to the
source relation edge.

The adaptation gate stopped.  The preserved 24-task confirmation split and
the existing ALFWorld `valid_unseen` heldout remain unread.

## Selective rule

The previous gates showed that source-edge utility was sparse and could be
negative when the edge fired too early.  V12 used only consumed V9--V11 traces
to audit the first action-changing source edge on each affected task.  Its
utility was preregistered as paired official-success delta plus `0.1` times
normalized step saving, keeping success lexicographically primary.

All three leave-one-version-out folds independently selected the rule:

> admit an authentic source `RELATE` edge only at target episode step 9 or
> later; otherwise abstain to the exact target fallback.

The audit covered ten changed-edge tasks.  On consumed data, the rule retained
both observed success rescues, rejected the one observed success loss, and had
nonnegative held-out utility in every fold.  It did not use fresh V12 outcomes,
the preserved confirmation tasks, or `valid_unseen`.

## Consumed closed-loop replay

Before freezing fresh V12 identities, the fixed step-9 rule was executed in
closed loop on the consumed V9--V11 adaptation tasks:

| Replay | Authentic | Node-only | Safety-only | Raw target | Authentic mean steps | Node-only mean steps |
|---|---:|---:|---:|---:|---:|---:|
| V9, 120 steps | 14/16 | 14/16 | 12/16 | 11/16 | 43.6250 | 48.0000 |
| V10, 60 steps | 15/16 | 13/16 | 13/16 | 13/16 | 26.5625 | 29.8125 |
| V11, 60 steps | 13/16 | 13/16 | 13/16 | 13/16 | 32.2500 | 32.4375 |

The rule produced 52 early-edge abstentions, no paired loss to node-only in
any replay, no reopened completed slot, and no failed selected postcondition.
It repaired the known early V10 TissueBox loss and removed the inefficient V11
Candle intervention.  This development result authorized exactly one fresh
V12 adaptation freeze; it was not counted as fresh evidence.

## Fresh result

The V12 manifest froze 16 new adaptation identities, eight from each target
family, after excluding all previously consumed identities.  It preserved the
same unread 24-task confirmation split and had zero overlap with both prior
logs and confirmation.

| Condition | Success | Mean steps | Changed effects | Source-edge changes |
|---|---:|---:|---:|---:|
| raw target | 11/16 | 35.0000 | 0 | 0 |
| safety-only | 12/16 | 33.6250 | 0 | 0 |
| source-node-only | 12/16 | 33.6250 | 0 | 0 |
| authentic selective graph | 12/16 | 33.6250 | 2 | 2 |

Authentic versus node-only was 0 wins, 16 ties, and 0 losses.  The two genuine
source-edge action changes were:

- Pencil -> Drawer at step 15: `close drawer` became `move pencil to drawer`;
  the episode still failed at 60 steps.
- DishSponge -> Cart at step 18: `examine cart` became `move dishsponge to
  cart`; the episode still failed at 60 steps.

Both actions came from the authentic compiled graph and target-native neural
realization, matched `TARGET_NATIVE_SLOT_READY_FOR_RELATION`, and received the
target relation postcondition.  The edge-permuted graph removed both changes.
This is behaviorally real neural-symbolic execution, but it supplied no fresh
task-level benefit.

The only raw-target rescue was HandTowel -> CounterTop (`60` to `38` steps).
Safety-only, node-only, and authentic all produced the same rescue, so the
causal component is shared slot safety/target handling rather than the source
edge.

Two preregistered gates failed:

- authentic success strictly superior to source-node-only;
- at least four tasks changed by source execution (observed two).

All safety, postcondition, graph-metamorphic, invariance, and target-only
noninferiority gates passed.

## What this establishes

V9--V12 establish that game-derived symbolic structure can be compiled,
guarded by target state, grounded by a target-native neural scorer, executed in
ALFWorld, and causally removed by permuting its source edge.  That is a genuine
neural-symbolic transfer mechanism.

They do **not** establish transferable utility.  V10's success gain did not
replicate in V11, and the consumed-trace selector that explained those old
outcomes did not produce an edge-specific gain on fresh V12 tasks.  A step
threshold is therefore an over-coarse applicability feature, not a sufficient
domain-invariant transfer principle.

## Recommendation

Stop cycling fresh splits with this same relation edge and do not open the
confirmation split.  The next experiment must change the information used to
decide applicability, not merely add more source rollouts or tune another step
threshold.

A defensible successor should freeze a target-native value model using richer
pre-action features: source edge identity, remaining budget, slot completion,
neural action-margin, fallback action class, predicted postcondition progress,
and an uncertainty-calibrated abstention score.  It should be evaluated first
on consumed traces with group-held-out tasks and explicit raw/safety/node
controls.  A new fresh gate is justified only if it predicts per-intervention
incremental value rather than replaying episode timing.

An alternative is to transfer a different source skill whose intervention has
more causal leverage than a single placement relation, such as subgoal
ordering, information gathering, or recovery after failed postconditions.

## Audit hashes

- applicability audit: `7b17687badf2f18cd9ffbb8c53022f593aa118da98e88d94135b9efb7caf3ac8`
- consumed closed-loop development gate: `e8731341b80eac422841ef21e7e4eb5b88b9e5da357cee02cee982372432bfb4`
- fresh manifest: `2c4dc440dfb165af2a891914a8a9ee911eb6b4fa74077619d49dea92ecd21b80`
- fresh candidate: `d44ac8d50a4a459ce5912a7ca150f9946ab4d39224f04b8546b21e45678b8320`
- fresh adaptation report: `2ac20d8ac7fd327f519c9e7d1f7ae2fd815ab658d5d718d03ecbc1533859092e`
- authentic compiled graph: `cb7c086a184333660b239b20f846db9287cd8e7a18f0f5920e7a52ce2369ab9e`
- node-only graph: `7114067205507344cf48f518d40ad7694edc5fb7a403a3d3266f474f90e1e2f4`

The compact result is
`docs/results/selective_budgeted_relation_edge_v12_summary.json`; full traces
remain in the ignored `runs/selective_budgeted_relation_edge_v12/` directory.
