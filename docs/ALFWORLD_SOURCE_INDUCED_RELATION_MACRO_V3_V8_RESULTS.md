# ALFWorld source-induced relation macro: V3–V8

## Bottom line

This experiment establishes a useful but bounded result:

> A recurrent entity–goal relation program induced only from source-game
> `(state, action, effect, next_state)` tuples can improve ALFWorld multiplicity
> success when a target-native neural grounder exposes the second object and
> relation. It does not solve target-native acquisition when that binding never
> becomes available.

The fail-closed V5 mechanism passed consumed development (`20/24` versus
`17/24`, `3W/0L`) and a two-task execution-untouched qualification (`1/2`
versus `0/2`, `1W/0L`). The separately frozen two-task formal reserve failed:
every condition was `0/2`, authentic admitted the source program zero times,
and no second-cycle action changed. Therefore **prospective ALFWorld transfer
is not validated**. The result is a positive mechanism diagnosis plus a
negative formal replication, not a success claim.

This line induces one Sokoban relation/cardinality program. Its exactly-one
and shuffled-effect controls establish that the transferred recurrence matters,
but it does **not** by itself establish that six different source games induce
different target-useful programs. That broader source-specificity claim remains
dependent on the separate multi-source experiments.

## What was learned from source

`source_goal_relation_induction.py` reads only source receipts and compresses
primitive paths at changes in `entity_goal_relation_coverage`. It induced:

- an anonymous typed `UPDATE ENTITY_GOAL_RELATION` operator;
- an observed positive relation-delta guard;
- a `ONE_OR_MORE` self-loop;
- terminal coverage `== 1.0`;
- fail-closed zero-binding, multiple-binding, non-positive-effect, and
  unobservable-terminal abstentions.

It exports no source action token and does not use the named
`EXPLORE/BACKTRACK/COMMIT` controller template. On fresh source seeds, 63/63
held-out episodes uniquely selected the program; authentic effect binding was
93 and shuffled-effect binding was 0. All source gates passed.

- Source artifact: `ed31de8757a12b57ea6d9a711a9b646646be5ebd0f1176b07320ef601f5fcb35`
- Fresh source report: `0b9151dfe8c10d77c099d178cc7b7efe781e82b543aad048dfe4d703b9e0da81`

## Why the ALFWorld runtime changed

| Version | Development result | Diagnosis |
|---|---:|---|
| V3 | authentic 20/24, raw 17/24; `5W/2L` | source macro influenced the first target binding before its typed guard was observed |
| V4 | authentic 20/24, raw 17/24; `5W/2L` | first-effect gate fixed early binding, but zero-binding states still ran a custom source search |
| V5 | authentic 20/24, raw/control 17/24; `3W/0L` | source-induced abstention enforced; native admissibility plus neural binding/completion grounds unique `BIND`; recurrence acts only after the first observed relation |

V5 reduced source admissions from 380 in V4 to 39, retained 29 non-trivial
second-cycle action changes, matched the target-native recurrent ceiling, and
reopened zero completed slots. All frozen development gates passed. The exact
two-sided sign test is only `p=.25`; this is development evidence, not a
confirmatory statistical result.

The important correction is semantic, not lexical. When the source artifact
says `zero_target_bindings -> ABSTAIN`, the runtime now returns control to the
target-native policy. It no longer invents a source-side search policy. When a
unique admissible pickup exists, neural entity binding and completion select
its arguments; a separately trained option-applicability head is not allowed
to veto an action whose executability is already established by ALFWorld's
native action set.

## Reserve audit and execution order

The valid-unseen multiplicity population contained 24 identities. A scan of
all five historical repositories left seven identities with no prior textual
reference. All seven lacked compiled `game.tw-pddl` files; three sliced tasks
are explicitly unsupported by ALFWorld. The four non-sliced identities were
frozen by SHA-256 into two qualification and two formal tasks before
compilation or policy reset.

The hand-coded compiler rejected all four. A separately recorded planner
compiler cross-check produced runnable TextWorld games. Compiler
solvability/walkthrough data was never exposed to the transfer policy and did
not affect role selection. The final role manifests and generated game hashes
were frozen before the first policy reset.

Execution order was enforced in code:

1. formal identities fixed before qualification;
2. qualification executed once;
3. formal runner verified the qualification report self-hash, config hash,
   pass status, and every gate;
4. formal executed once with unchanged source artifact, target grounders,
   thresholds, conditions, and evaluator.

## Untouched qualification: passed, but tiny

| Condition | Success |
|---|---:|
| raw target only | 0/2 |
| authentic source recurrence | **1/2** |
| exactly-one source control | 0/2 |
| effect-binding permutation | 0/2 |
| generic single-relation scaffold | 0/2 |
| target-native recurrent ceiling | **1/2** |

Authentic was `1W/0L/1T`, executed one changed second-cycle action, and had no
negative transfer. All mechanism gates passed, authorizing formal execution.
With one discordant pair, `p=1.0`; qualification is only a mechanism gate.

Report SHA-256:
`e9ce474d0744d072978e0b7285d1fbcb4767dbe63ac851efaf64916db0281ce2`.

## Fresh formal: failed

Every arm scored `0/2`. In both tasks, the policy established the first typed
relation, then spent the rest of the 60-step budget without exposing a unique
second-object binding:

- ToiletPaper→Cabinet: one `BIND_INSTANCE`, one `RELATE_SLOT_CLOSED`, 58
  `IGNORE` receipts;
- KeyChain→Safe: one `BIND_INSTANCE`, one `RELATE_SLOT_CLOSED`, 58 `IGNORE`
  receipts.

Authentic therefore made zero admissions and zero second-cycle action changes.
The `nontrivial_second_cycle_action_change`, `success_gain_over_raw`, and
`strict_source_control_superiority` gates failed. Zero negative transfer and
the tie with the target-native recurrent ceiling are vacuous here because the
recurrence was never grounded.

Formal report SHA-256:
`c893d36cc1b197440ca25bb8a03641f1528ba31c6543b0f431d060ae8fba2833`.

## Bitter lesson and next valid experiment

Transferring only the terminal recurrent relation operator is insufficient for
tasks whose bottleneck is finding the next entity. The earlier V4 search could
rescue such tasks, but it was not licensed by the induced program and caused
negative transfer. The correct next source-side advance is not another target
heuristic. It is source-only induction of a typed acquisition subprogram such
as an anonymously learned `SEEK -> BIND -> RELATE` transition, with its own
observed effects and abstention controls, followed by a new target-native
grounder and a genuinely new target reserve.

The installed ALFWorld data now contains no remaining compiled, execution-
untouched multiplicity reserve: all 813 compiled train tasks occur in prior run
artifacts, the valid-seen population was consumed previously, and the last
runnable valid-unseen identities were spent here. The formal result must not be
retuned and rerun as confirmatory evidence.

## Reproduction pointers

- Source induction: `src/motif_transfer/source_goal_relation_induction.py`
- Fail-closed target runtime: `src/motif_transfer/alfworld_goal_relation_macro_v5.py`
- Matched evaluator / reserve guard: `scripts/run_alfworld_goal_relation_macro_v6.py`
- V5 development report: `runs/alfworld_goal_relation_macro_v5_development/report.json`
- V8 qualification report: `runs/alfworld_goal_relation_macro_v8_qualification/report.json`
- V8 formal report: `runs/alfworld_goal_relation_macro_v8_formal/report.json`
