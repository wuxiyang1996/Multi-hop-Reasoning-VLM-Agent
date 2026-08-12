# Executable source graph V9: adaptation result

Date: 2026-08-11

## Verdict

V9 fixes the central V8 mechanism error: the bound source-game IR is compiled
and executed, and a matched source edge now supplies the requested target
effect.  The fresh relation-only adaptation gate nevertheless **failed and
stopped before confirmation**.

Authentic execution improved 120-step official success from `11/16` to `14/16`,
but the reversed-edge control also reached `14/16`.  Authentic was more
efficient (`43.625` versus `48.0` mean steps) and made four action-changing
`BIND -> RELATE` edge interventions.  The transferred edge therefore affected
behavior and efficiency, but it did not add success beyond the control that
retained the source `BIND` node and target-native completed-slot safety.

The 24-task V9 confirmation split and the existing ALFWorld `valid_unseen`
heldout remain unread.

## What was fixed

V8 validated `source_ir` but generated effects with a target-coded state
machine.  V9 instead performs:

```text
target observation -> target-native slot ledger -> applicable source guard
    -> exact compiled source edge -> symbolic target effect
    -> target-native neural action realization -> observed postcondition
```

For example, after target observation binds a goal object, the ledger exposes
`TARGET_NATIVE_SLOT_READY_FOR_RELATION`.  Only the compiled source edge

```text
BIND --TARGET_NATIVE_SLOT_READY_FOR_RELATION--> RELATE
```

can request a `RELATE` action.  The decision receipt includes the compiled graph
hash, source edge, guard, aggregate source support, grounded native action, and
target postcondition.

The edge control reverses every bound source edge before routing.  It is not a
post-hoc label permutation.  Metamorphic tests verify that reversing or changing
the destination of `BIND -> RELATE` removes or changes the target relation
decision.

V9 also separates the target-native completed-slot shield from neural property
confidence.  ALFWorld states the relation operator explicitly in the goal, so a
low auxiliary router score can no longer authorize taking a completed object
back out of its goal receptacle.  Runtime structural scope prevents the old
light-task-to-relation leakage.

## Frozen boundary

Before any selected reset, the manifest scanned 4,127 text artifacts across the
five historical workspaces and excluded 456 previously mentioned ALFWorld task
IDs.  It froze 16 adaptation tasks and 24 disjoint confirmation tasks from only
`pick_and_place_simple` and `pick_two_obj_and_place`.  Selection used path
identities and seeded SHA256 ranking, never target outcomes or task contents.

The candidate then bound:

- the intervention-grounded source IR and its compiled authentic/control graphs;
- the frozen target-native neural action grounder and property router;
- the manifest and consumed V8 negative report;
- the V9 graph executor, slot ledger, runner, and shared reporting helpers;
- inherited action-realization thresholds and preregistered gates.

## Adaptation result

| Condition | Success | Mean steps | Changed effects | Source edge changes | Reopened slots |
|---|---:|---:|---:|---:|---:|
| target-only | 11/16 | 55.875 | 0 | 0 | 7 |
| authentic source graph | 14/16 | 43.625 | 6 | 4 | 0 |
| reversed-edge graph | 14/16 | 48.000 | 2 | 0 | 0 |
| property/scope control | 11/16 | 55.875 | 0 | 0 | 0 |

Authentic versus target-only was 3 wins, 13 ties, and 0 losses.  Authentic
admitted 23 source `BIND` nodes and 18 source edges, changing two `BIND` and four
`RELATE` effects on four tasks.  All selected effect postconditions succeeded,
the required-option diagnostic was invariant on every step, and the completed
slot ledger never reopened.

Only one preregistered requirement failed:

```text
authentic_success_superior_to_edge_control = false
```

All other ten requirements passed, including noninferiority to target-only,
both claimed changed effects, four changed tasks, four changed source edges,
authentic changes exceeding the edge control, zero reopened slots, and zero
failed selected postconditions.

## Paired trace attribution

The three target-only failures rescued by both authentic and reversed-edge were:

| Task | target | authentic | reversed edge | Attribution |
|---|---:|---:|---:|---|
| Pencil -> Drawer | fail at 120 | success at 73 | success at 75 | authentic edge saves 2 steps; safety supplies success rescue |
| Potato -> Fridge | fail at 120 | success at 36 | success at 101 | authentic edge saves 65 steps; both finish by 120 |
| Egg -> Microwave | fail at 120 | success at 102 | success at 102 | source `BIND` start-node changes, not the relation edge |

RemoteControl was already a target success at 75 steps; authentic completed it
in 28 versus 31 for reversed-edge.  The four authentic edge interventions all
had the exact source receipt
`BIND --TARGET_NATIVE_SLOT_READY_FOR_RELATION--> RELATE` and successful target
`RELATE_SLOT_CLOSED` postconditions.

Thus the source relation edge has a strong resource-efficiency signal, most
clearly Potato `101 -> 36`, but the preregistered 120-step success endpoint is
unable to distinguish it from the safety/control trajectory.

## Next valid experiment

Do not authorize the reserved V9 confirmation from this failed gate.  A valid
next candidate may treat V9 adaptation as consumed development evidence and
preregister a resource-bounded endpoint before touching new tasks.  A 60-step
budget is a defensible next hypothesis because the intended skill is a
procedural shortcut, but choosing it is development-driven and therefore must
be evaluated only on a newly gated split.

The next controls should explicitly separate:

1. exact target-only policy;
2. target-native completed-slot safety without any source graph;
3. source `BIND` node plus safety with the relation edge removed/reversed;
4. the full authentic `BIND -> RELATE` source graph plus the same safety.

The confirmation gate should require authentic budgeted success to exceed all
three controls, positive paired net wins, multiple source-edge action changes,
and zero negative transfer.  Full 120-step success should remain a secondary
safety endpoint.

## Audit hashes

- manifest: `5256dde934dde1e87718180d797ccfe8a2c4f9252e1529df5c47c6f46a1990d5`
- candidate: `d4a3e3e1f10e67eeb87e8987412bf0730fe466160dda5052e6556bbca4a9f979`
- source IR: `5522d11484dfbac4c6dc82a7b2f4cc6369ae771cf6dff780a708e20c9cf411fd`
- authentic compiled graph: `cb7c086a184333660b239b20f846db9287cd8e7a18f0f5920e7a52ce2369ab9e`
- reversed-edge graph: `7114067205507344cf48f518d40ad7694edc5fb7a403a3d3266f474f90e1e2f4`
- adaptation report: `be28eec79e448ba90b8be1e3a62768af7214d1fa37b3422158289bd765cede94`

The compact result is
`docs/results/executable_source_graph_alfworld_v9_summary.json`; full traces
remain in the ignored `runs/executable_source_graph_alfworld_v9/` directory.
