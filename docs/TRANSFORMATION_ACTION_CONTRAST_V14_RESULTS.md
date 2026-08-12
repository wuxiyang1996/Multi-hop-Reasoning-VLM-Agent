# Transformation action-contrast V14: stop before outcomes

Date: 2026-08-12

## Verdict

V14 tested whether a different and structurally richer source program had
enough target action disagreement to justify matched outcome collection:

```text
BIND(object)
  -- target slot requires unary property --> MUTATE(object, property)
  -- observed property postcondition -----> RELATE(object, receptacle)
```

The source graph was induced from MiniGrid DoorKey/UnlockPickup causal
structure and grounded entirely through target-native ALFWorld goal, slot,
property, and neural action scores.  The main environment executed a
source-disabled target safety policy; authentic graph decisions were shadow
queries and never changed the rollout.

Across 64 newly frozen train tasks, 28 tasks reached at least one target-
actionable source edge and produced 63 edge opportunities.  Only seven tasks
contained any distinct authentic-versus-target native action, for eight state
contrasts total.  The preregistered gate required 32 tasks, including 16 with a
`MUTATE` contrast, eight with a `RELATE` contrast, and breadth across at least
three families.

Observed:

- tasks with any edge action contrast: `7/64`;
- `MUTATE` contrast tasks: `2`;
- `RELATE` contrast tasks: `5`;
- clean: `3/16`, heat: `4/16`, cool: `0/16`, light: `0/16`;
- identity/receipt failures: zero;
- rewards or official outcomes recorded: zero.

All four contrast gates failed.  V14 stopped before matched forks, value
training, success evaluation, or another fresh split.

## Frozen data boundary

The pool selected 16 path-ranked train identities from each of four families:

- `look_at_obj_in_light`;
- `pick_clean_then_place_in_recep`;
- `pick_cool_then_place_in_recep`;
- `pick_heat_then_place_in_recep`.

Selection excluded 544 task identities found in the five historical
workspaces' frozen configs and compact result artifacts.  It read 113 files
totalling 1,758,166 bytes and persisted the complete exclusion snapshot.  The
64 selected identities had zero intersection with that snapshot.

Resetting them consumed the identities for development.  They are not called a
fresh evaluation set.  The runner deliberately discarded reward and never
serialized or branched on official success.  It stopped each task only on the
environment terminal flag or the frozen 60-step limit.

The initial broad repository scanner was terminated before producing a pool:
the inherited helper attempted to read multi-gigabyte raw log files over NFS.
The corrected freezer scans only protocol identity artifacts under `configs/`
and `docs/results/`, rejects any exclusion artifact above 16 MB, and supports
byte-identical reproduction from the persisted snapshot.  No task had been
reset before this correction.

## What actually disagreed

Five `RELATE` contrasts preferred immediate target placement over target
fallbacks such as closing/examining the receptacle or `look`:

- heat Tomato -> Fridge: two contrast states;
- clean Cloth -> Drawer;
- heat Apple -> DiningTable on two tasks;
- heat Egg -> SideTable.

The two `MUTATE` contrasts were two separate clean-ButterKnife tasks.  In both,
the source graph proposed `clean butterknife with sinkbasin`, while the target
fallback proposed `slice egg with butterknife`.  These are semantically
interesting intervention candidates, but both appeared very late (steps 51
and 57) and are too few and too correlated to support a value model.

Cool tasks produced no action contrast.  Light tasks produced no target-
actionable graph edge at all under the source-disabled target trajectory.  The
current symbolic/grounding path therefore lacks usable coverage for LIGHT,
even though `LIGHT` exists in the target property vocabulary.

## Combined V13--V14 lesson

V13 showed that the relation-only program yielded 10 informative tasks from 62
actionable opportunities.  V14 showed that expanding to the complete
property program yielded only seven contrast tasks from 64 tasks.  More source
nodes and edges did not create more target intervention authority.

The failure is now measurable before outcomes:

```text
source-supported symbolic structure
  -> target can execute the structure
  -> target policy often already chooses the same native action
  -> no matched treatment contrast
  -> no identifiable transfer value
```

This is stronger than another negative success-rate run because it isolates why
the experiment would be underpowered.  A skill library can be large while its
incremental target action support is nearly empty.

## Recommendation

Do not collect outcomes for V14 and do not add more instances of the same
placement/property program.  Before any future transfer experiment, make
outcome-blind contrast enumeration a mandatory stage:

1. freeze target development identities;
2. execute only the target control policy;
3. shadow-query the authentic source skill;
4. count task-independent native-action disagreements by family and effect;
5. proceed only after the fixed contrast-rate gate passes.

The next source skill should change *which subgoal is attempted*, not merely
the concrete action inside a subgoal the target already recognizes.  Better
candidates remain budget-aware exploration, recovery after contradictory
postconditions, and ordering multiple unresolved subgoals.  Their preflight
must include a structurally matched control and at least 32 disagreements
before any reward is read.

## Audit hashes

- frozen pool: `7c1ff1978ec90b963ab3e874053ecec8be87b04fb3ef3206f76be8a3074a7c1a`
- frozen pool file: `e2ae506b1c1dedfa2117c7c01ac279cdd78aae08cdae34193e5e75c469e675f6`
- outcome-blind contrast report: `40fe79c874fc64de3194a84446cc3b34e44bfa50ef089d4c7995ee7f9e45ebe6`

The frozen pool is `configs/transformation_action_contrast_pool_v14.json`.
The compact result is
`docs/results/transformation_action_contrast_v14_summary.json`; the full
outcome-free task receipts remain under the ignored
`runs/transformation_action_contrast_v14/` directory.
