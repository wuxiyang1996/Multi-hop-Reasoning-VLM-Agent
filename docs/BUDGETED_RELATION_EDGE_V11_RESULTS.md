# Budgeted relation-edge V11: non-replication

Date: 2026-08-11

## Verdict

V11 corrected V10's claim-coverage mismatch, but the 60-step success advantage
did not replicate on a second fresh adaptation split.  All four conditions
achieved `13/16`; authentic was also slightly slower than every control
(`32.50` versus `32.4375` mean steps).

Authentic executed three action-changing, postcondition-verified
`BIND -> RELATE` source-edge interventions across two tasks.  This again proves
that the source graph is operational and behaviorally nontrivial.  It does not
show a stable success-rate or efficiency benefit.

The gate failed and stopped.  The preserved 24-task confirmation and the
existing ALFWorld `valid_unseen` heldout remain unread.

## What V11 changed—and did not change

V10 achieved `14/16` versus `13/16` for raw target, safety-only, and node-only
controls, but stopped because it required both `BIND` and `RELATE` to change a
target effect.  All seven actual changes were `RELATE` successor-edge actions.

V11 changed only the claimed action-changing effect from `[BIND, RELATE]` to
`[RELATE]`.  It preserved:

- the exact 60-step endpoint;
- target grounder, property router, source IR, and compiled graphs;
- action-realization thresholds;
- raw target, safety-only, node-only, and authentic control semantics;
- success superiority, paired outcome, nontriviality, safety, and receipt gates.

It froze 16 new adaptation identities after excluding all consumed identities
and preserved the same unread confirmation split.

## Result

| Condition | Success | Mean steps | Changed effects | Source-edge changes |
|---|---:|---:|---:|---:|
| raw target | 13/16 | 32.4375 | 0 | 0 |
| safety-only | 13/16 | 32.4375 | 0 | 0 |
| source-node-only | 13/16 | 32.4375 | 0 | 0 |
| authentic source graph | 13/16 | 32.5000 | 3 | 3 |

Authentic versus every control was 0 wins, 16 ties, and 0 losses.  The three
changed `RELATE` effects occurred on two tasks:

- Lettuce -> Fridge: target 38 steps, authentic 35;
- Candle -> Cabinet: target 51 steps, authentic 55.

All three interventions carried the compiled authentic graph hash, matched the
source guard `TARGET_NATIVE_SLOT_READY_FOR_RELATION`, and received a successful
target `RELATE_SLOT_CLOSED` postcondition.  No completed slot reopened and no
selected postcondition failed.

Two preregistered requirements failed:

- authentic success strictly superior to the source-node control;
- changed-task count at least four (observed two).

The corrected `RELATE` coverage requirement passed.

## Combined interpretation

The sequence now separates three claims:

1. **Can a source symbolic edge be executed in ALFWorld using target-native
   neural grounding?** Yes.  V9–V11 produce exact graph/guard/action/
   postcondition receipts, and graph metamorphic controls remove those actions.
2. **Can it sometimes improve resource-bounded success?** Yes, on the V10 fresh
   adaptation split: `14/16` versus `13/16` for every control.
3. **Is that success improvement reliably transferable?** Not yet.  V11 is a
   direct fresh non-replication: `13/16` everywhere.

Therefore the current result is a valid neural-symbolic transfer mechanism with
unstable utility, not a demonstrated success-rate improvement.

## Recommendation

Do not consume the confirmation split and do not keep cycling fresh gates with
the same source edge.  The intervention is too sparse and its utility depends
on the target fallback being near a useful relation action.

The next method should learn a target-native applicability/utility model from
consumed adaptation traces.  Its input should include the source edge identity,
slot state, target neural score margin, remaining budget, and predicted
postcondition value.  It should choose among authentic edge execution and
target abstention, with safety-only and node-only controls retained.  Only after
that model passes cross-validation on already-consumed V9–V11 traces should a
new, final manifest be frozen.

More source games alone are unlikely to solve this failure: the source relation
edge is already well-supported by MiniGrid PutNear and MiniWorld PutNext.  The
bottleneck is selective target applicability, not lack of source evidence.

## Audit hashes

- manifest: `3fe6806df95c4d406dcb6b60bfb995e7c1c1e21043825f120afbac62fac0424e`
- candidate: `bde081c501d389b6bb340b0deed136d6450b05ce21a60527939751125b00e2ac`
- adaptation report: `56ed27678481795e030ab953281bddb6ce640fa17845e800bd2befac1fda754d`
- authentic compiled graph: `cb7c086a184333660b239b20f846db9287cd8e7a18f0f5920e7a52ce2369ab9e`
- node-only graph: `7114067205507344cf48f518d40ad7694edc5fb7a403a3d3266f474f90e1e2f4`

The compact result is
`docs/results/budgeted_relation_edge_v11_summary.json`; full traces remain in
the ignored `runs/budgeted_relation_edge_alfworld_v11/` directory.
