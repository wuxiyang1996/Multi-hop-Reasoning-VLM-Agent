# Source-induced acquisition transfer to ALFWorld: V1–V10

> **Successor result:** V13 has now completed an independent, preregistered
> 24-task `valid_train` replication under the unified harness: 20/24 versus
> 13/24 raw and controls, 7W/0L/17T, exact `p=0.015625`, matching the 20/24
> target-native ceiling. The V1–V10 text below remains the historical
> consumed-development record. See
> [`PHASE8_FOUR_DOMAIN_UNIFIED_NEUROSYMBOLIC_TRANSFER_RESULTS.md`](PHASE8_FOUR_DOMAIN_UNIFIED_NEUROSYMBOLIC_TRANSFER_RESULTS.md).

## Result

The missing acquisition mechanism can be induced from source interventions and
grounded in ALFWorld without exporting Sokoban actions or adding a target
object-location table.

On the consumed 24-task ALFWorld multiplicity development matrix, the final
typed and handle-preserving runtime achieved:

| Arm | Success | Mean steps |
|---|---:|---:|
| neural target only | 17/24 | 39.46 |
| authentic source-induced acquisition + relation | **24/24** | **25.54** |
| source exactly-one control | 17/24 | 39.46 |
| source effect-permutation control | 17/24 | 39.46 |
| generic single-relation scaffold | 17/24 | 39.46 |
| target-native acquisition ceiling | **24/24** | **25.54** |

Authentic was `7W/0L/17T` against raw and against each source control. The exact
two-sided paired sign-test value is `p=0.015625`. It executed 48 observed
relation updates, reopened zero completed slots, and produced zero wrong-handle
effects while the transferred recurrence was active.

This is strong **consumed-development mechanism evidence**, not a new
prospective ALFWorld confirmation. The V8 qualification/formal identities were
not reset or rerun, and no compiled execution-untouched multiplicity reserve
remains in the installed ALFWorld data.

## What was induced from source

`source_goal_acquisition_induction.py` starts from the numeric terminal relation
feature already selected by the source relation inducer. At each source state,
it enumerates native interventions and measures:

```text
positive_relation_intervention_cardinality
```

It then types every observed source transition from `(state, action, effect,
next_state)` and induces four anonymous operator families:

```text
UPDATE CONTROL_STATE / POSITION
UPDATE ENTITY_RELATION / POSITION
UPDATE POSITIVE_EFFECT_BINDING / CANDIDATE_CARDINALITY
UPDATE ENTITY_GOAL_RELATION / RELATION_COVERAGE
```

The learned program is not a supplied `EXPLORE/BACKTRACK/COMMIT` controller. Its
key data-derived rule is:

```text
binding cardinality == 0
    -> repeat a source-observed anonymous state update
    -> observe binding cardinality 0 -> 1
    -> execute the relation update
otherwise
    -> abstain or remeasure under the learned cardinality contract
```

Discovery contained 68 trajectories and 450 successful primitive transitions.
The binding-onset-to-relation edge had 85 observations and precision 1.0.

The inducer and runner were then frozen before procedural Sokoban seeds
299001–299024 were opened. Fresh confirmation retained 69 trajectories and 476
transitions:

| Fresh source gate | Result |
|---|---:|
| authentic program conformance | 476/476 |
| shuffled-effect conformance | 320/476 |
| binding onset followed by relation | 85/85 |
| authentic exact effect binding | 476/476 |
| shuffled exact effect binding | 0/476 |

All nine source gates passed.

- Source acquisition artifact:
  `7ff3e950f3eebaf75cca015df88ab6a01f2f364fe246102984a3cb4ee095f0d7`
- Fresh source report:
  `f02fec96a448ea549e95ffa3fff3f9c5adf71ff202a89fb170a90ba55a257285`

## Target-native neural grounding

The source artifact exports anonymous operator types only. ALFWorld's frozen
target-native grounder supplies its own `SEARCH` actions and neural policy
scores. The adapter binds the source `UPDATE CONTROL_STATE/POSITION` operator
to target actions only when:

1. one target relation has already been observed;
2. the current typed positive-effect binding count is zero;
3. the target workflow independently says `SEARCH` is required;
4. the carried entity, if any, has the goal entity type; and
5. a relation-stage state update preserves the already observed target
   receptacle handle.

The source controls never receive this binding. The target-native ceiling uses
the same target grounder and matches authentic behavior, while reporting no
source admissions.

## V9 failure and the useful bitter lesson

The first executable adapter was intentionally evaluated before refinement.
It scored only 16/24 versus raw 17/24 (`4W/5L`) despite 321 acquisition
groundings. Two mistakes explain the negative transfer:

- it interpreted a recurrent source state update as near-exhaustive traversal
  of the target action list;
- after finding the correct second object, it could navigate to a different
  instance of the destination type, violating the relation argument bound by
  the first successful relation.

V10 fixed the semantics rather than adding task heuristics:

- action selection uses the already-trained target neural policy head, not a
  hand-composed structured score;
- a carried distractor triggers typed-entity abstention;
- `RELATE` acquisition preserves the exact bound target handle.

These changes reduced acquisition groundings from 321 to 199, changed the
result from 16/24 to 24/24, and eliminated all seven raw failures without
introducing a loss.

## Report-only gate correction

The original V10 rollout report had 24/24 success but marked one extra gate
false. Its counter included every `RELATE_NO_PROGRESS` in an episode, including
one raw first-cycle move at step 23 before the source recurrence was enabled.

A separately hash-linked deterministic analysis corrected the scope to
`completed_count_before >= 1`. It changed no rollout, action, or outcome:

- all-cycle `RELATE_NO_PROGRESS`: 1;
- source-active `RELATE_NO_PROGRESS`: 0.

The corrected development report passes every gate.

- Original V10 rollout report:
  `f9c70389a75a7668e9222d0fd8dfe998180d002b8836deaef6cce0457139bc29`
- Scope-correction analysis report:
  `87d45f213c1e57428f028e56f1ff87ef7313a4a4bd23cd4e37fce76cfbc1d378`

## What this establishes—and what remains

Established:

- source-only induction of acquisition operators, transition edges, and
  cardinality abstention;
- fresh source confirmation with shuffled-effect controls;
- executable source IR with target-native neural grounding;
- a nontrivial success-rate gain over neural-only and three source controls on
  consumed ALFWorld development;
- typed argument preservation is necessary to avoid negative transfer.

Not yet established:

- prospective ALFWorld replication on a new untouched reserve;
- transfer to a second target with the same induced IR and only a new
  target-native grounder;
- source-specific differences among programs induced from multiple games.

The next valid confirmatory step requires newly obtained ALFWorld tasks or a
second target domain. The consumed V8 formal reserve must remain closed.

## Reproduction pointers

- Source inducer: `src/motif_transfer/source_goal_acquisition_induction.py`
- Fresh source freeze/run:
  `scripts/freeze_sokoban_goal_acquisition_v1.py`,
  `scripts/induce_sokoban_goal_acquisition_v1.py`
- V9 diagnostic runtime: `src/motif_transfer/alfworld_goal_acquisition_v9.py`
- V10 runtime: `src/motif_transfer/alfworld_goal_acquisition_v10.py`
- V10 matched runner: `scripts/run_alfworld_goal_acquisition_v10.py`
- Gate-scope analysis: `scripts/analyze_alfworld_goal_acquisition_v10.py`
- Source summary: `runs/sokoban_goal_acquisition_v1/summary.json`
- V10 rollout report:
  `runs/alfworld_goal_acquisition_v10_development/report.json`
- Corrected V10 analysis:
  `runs/alfworld_goal_acquisition_v10_development/analysis_report.json`
