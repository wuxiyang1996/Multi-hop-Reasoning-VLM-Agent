# Phase 3 source-induced neural-symbolic transfer

## Bottom line

The strongest claim now supported is:

> Source-only intervention rollouts can induce source-specific, typed symbolic programs, and those unchanged programs produced a large, prospective success improvement on untouched DiscoveryWorld tasks when paired with a target-native neural grounder.

The second-target ALFWorld replication did **not** validate. It passed development qualification but failed the frozen formal gates. This distinction is important: the project now has one positive untouched cross-domain result, a working source-specific IR, and one informative same-IR replication failure—not a general multi-target success claim.

The compact machine-readable result is in [`results/phase3_neurosymbolic_transfer_v1_summary.json`](results/phase3_neurosymbolic_transfer_v1_summary.json).

## What was actually induced from source

The source learner consumes matched `(state, action, effect, next_state)` intervention receipts. It exports:

- anonymous typed operators induced from observed ledger state deltas;
- one selected transition-effect type per qualified game lineage;
- a fail-closed abstention rule for missing, tied, non-finite, or unqualified bindings;
- no source-native action token and no target data.

The strings `EXPLORE_UNTRIED`, `BACKTRACK_REPLAN`, and `COMMIT_VERIFY` occur only in a forbidden-token audit field. Validation rejects an artifact if they occur anywhere else; they are not operator templates.

On the third held-out source reserve, the qualified programs achieved `18/22` correct candidate selections, versus `1/22` for shuffled effect binding and `8/22` when programs were permuted across source games. Three games induced qualified but different programs:

| Source game | Induced effect |
|---|---|
| GymV Columns | `EFFECT_BY_TRANSITION_4` |
| GymV Thunder Force III | `EFFECT_BY_TRANSITION_8` |
| Tetris | `EXECUTABLE_TRANSITION_PERSISTENCE` |

Candy Crush, Streets of Rage 2, and Strider induced abstention after independent source calibration. Thus runtime behavior is predicted by source intervention content, not a source identity feature.

## Untouched DiscoveryWorld: validated

The DiscoveryWorld acquisition fix was development-only. On 29 qualification tasks it collected 262 native actions with:

- schema fallback `0/262`;
- repair `0/262`;
- invalid native actions `0`;
- no evaluator finalization and no formal outcome read.

The grounding threshold was then frozen. On 24 new formal seeds, the five preregistered arms achieved:

| Arm | Success |
|---|---:|
| neural-only | 3/24 |
| source-induced | **22/24** |
| source-permuted | 11/24 |
| generic scaffold | 1/24 |
| target-native ceiling | 24/24 |

Source-induced versus neural-only was `19 wins / 0 losses / 5 ties`, exact two-sided sign-test `p = 3.81e-6`; negative transfer was zero. Source-induced also beat source-permuted by `11 wins / 0 losses`. All 15 frozen gates passed. Formal schema fallback was 3.55%, below the frozen bound; binder and grounder repair rates were zero.

This is the current positive cross-domain neural-symbolic transfer result.

## Why the first ALFWorld adapters failed

The early exact-action and option adapters used the wrong causal label. They predicted whether an action or option would occur later in an expert continuation. The source type means something different: what happens if the candidate intervention is executed **now**, measured at transition 1, 4, or 8.

This mismatch selected actions such as `close safe` or `examine` over an immediately useful `place` action. Offline top-1 agreement looked acceptable while long-horizon success fell from neural `5/11` to source `2/11`. A singleton option also crashed the permuted control; it is now explicitly recorded as an identity control because a singleton cannot be permuted.

## ALFWorld intervention-grounded repair

The repaired target grounder used only target development tasks:

- 36 train tasks: 107 snapshots and 237 option forks;
- 12 validation tasks: 36 snapshots and 77 option forks;
- each fork executed one target-native option, then the frozen target neural policy continued to H1/H4/H8;
- labels used transition-grounded target progress and persistence;
- formal success was not read, and no qualification or formal target was reset.

Validation AUCs were `1.000`, `0.938`, `0.833`, and `0.792` for H1, H4, H8, and persistence. V9 remained blocked because source mean H8 utility (`0.4722`) fell more than the frozen tolerance below neural (`0.5000`). V10 froze a policy-support threshold of `0.9` using adaptation-train only; it removed validation H8 losses but failed an inherited 20% state-level control rate. V11 retained the same threshold and corrected the control unit to at least 10% offline state contrast plus at least four online task contrasts. It did not change a neural head, source program, action, or success gate.

The macro executor also now respects type horizons. An H4 program executes one target-native option and observes its result only after three target-neural continuation actions. Arbitrary observation change is no longer treated as H4 success.

## ALFWorld qualification: passed

With a common 180-step transfer budget on 11 consumed development tasks:

| Arm | Success |
|---|---:|
| neural-only | 6/11 |
| source-induced | **7/11** |
| source-permuted | 6/11 |
| generic scaffold | 5/11 |

Source versus neural and permuted was `1 win / 0 losses / 10 ties`. Five concrete actions differed from neural, eight tasks had authentic/permuted behavioral contrasts, and all three qualified source effect types were selected. Every frozen qualification gate passed.

The ALFWorld hand-coded expert proved unstable across processes and can time out when given larger budgets. Target capability is therefore audited as an evaluator-only, source-free per-task union of official expert, neural-only, and generic policies. Source-induced is explicitly excluded from that union.

## ALFWorld formal replication: failed

The formal run used the complete installed 24-task `valid_seen` multiplicity population. It may have historical target exposure, so even a pass would have been an in-distribution same-IR replication, not an untouched result.

| Arm | Success |
|---|---:|
| neural-only | **12/24** |
| source-induced | 11/24 |
| source-permuted | **12/24** |
| generic scaffold | 11/24 |

Source-induced versus neural-only was `0 wins / 1 loss / 23 ties`; versus source-permuted it was `1/2/21`; versus generic it was `4/4/16`. Negative transfer against neural was 4.17%. Source changed 21 actions, selected Columns/H4 194 times, Thunder/H8 94 times, and Tetris/persistence 138 times, with 18 task-level authentic/permuted contrasts. Thus the mechanism was active and source-specific, but it was not useful enough to pass the success gates.

The sole neural loss is especially diagnostic. In the Newspaper→Drawer task:

- at step 42, Columns/H4 correctly changed `close drawer 8` to `move newspaper 2 to drawer 8`, producing positive progress;
- at step 120, Thunder/H8 changed the neural `move creditcard 1 to sidetable 1` to `go to dresser 1` with target-policy support ratio `0.9864`;
- that near-tied but time-critical detour exhausted the budget; neural succeeded at step 167, while source failed at 180.

The bitter lesson is that high policy support and predicted positive horizon effect do not estimate **downside at a deadline**. In a sequential embodied task, one late wrong option can dominate several earlier useful interventions. The target grounder needs state-specific irreversibility, remaining-budget, and missed-commitment risk—not another source-side controller template.

## Claim boundary and next experiment

Supported:

1. source-only anonymous operator and typed-effect induction;
2. source-specific applicability on held-out game interventions;
3. untouched DiscoveryWorld cross-domain success transfer;
4. same IR and source programs executing with only a new target-native ALFWorld grounder;
5. a complete, negative ALFWorld replication that identifies the missing risk variable.

Not supported:

1. universal transfer from games to arbitrary agent domains;
2. a positive second-target formal replication;
3. further ALFWorld tuning on the consumed 24-task formal population.

The next valid positive replication must use a new reserve. The preferred route is TIR non-maze with temporally extended target-native tool interventions, because a static crop→answer call does not instantiate H4/H8 transition semantics. Only its target grounder may change; the source programs, effect vocabulary, anonymous ledger, source permutation, and success/negative-transfer gates must remain frozen.
