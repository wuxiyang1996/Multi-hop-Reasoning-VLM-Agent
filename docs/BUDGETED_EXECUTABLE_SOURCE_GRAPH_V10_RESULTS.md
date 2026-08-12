# Budgeted executable source graph V10

Date: 2026-08-11

## Verdict

V10 found the first fresh success-rate signal attributable to the executed
source relation edge, but its adaptation gate still **failed and stopped**.

With a preregistered 60-step budget, authentic source-graph execution achieved
`14/16`; raw target, target-native safety-only, and source-node-only controls
each achieved `13/16`.  Authentic made seven action-changing source-edge
interventions across five tasks, all with successful target postconditions.

The only failed gate required action-changing interventions for both claimed
effects, `BIND` and `RELATE`.  All seven V10 changes were `RELATE`.  This was a
candidate-specification mistake: V10's isolated hypothesis was the executable
`BIND -> RELATE` successor edge, and entry through a source `BIND` node need not
change the target action.  The gate cannot be changed after the run, so the
reserved 24-task confirmation remains unread.

## Why a 60-step endpoint

The consumed V9 gate showed authentic and reversed-edge both at `14/16` under
120 steps, while authentic was faster (`43.625` versus `48.0`) and the largest
paired difference was Potato `36` versus `101` steps.  V10 therefore froze a
resource-bounded success hypothesis before selecting or resetting any V10 task.

This is not a retrospective V9 win.  The 60-step endpoint, max steps, runner
seed, four control semantics, implementation files, manifest, and source graph
were all candidate-hashed before the new V10 adaptation reset.

## Controls

| Stored condition | Operational meaning | Success | Mean steps | Source-edge changes |
|---|---|---:|---:|---:|
| `target_only` | exact raw target policy | 13/16 | 30.625 | 0 |
| `property_permuted_router` | completed-slot safety only, no source execution | 13/16 | 29.8125 | 0 |
| `edge_permuted_ir` | source `BIND` node + safety, source edges reversed | 13/16 | 29.8125 | 0 |
| `authentic_slot_ir` | full source graph + identical safety | 14/16 | 27.125 | 7 |

The old condition keys are retained for runner compatibility, but the frozen
candidate records the operational meanings above.  In particular,
`property_permuted_router` is no longer a property permutation in V10; it is a
true safety-only control in the same structural relation task.

Authentic versus each control was 2 wins, 13 ties, and 1 loss, for one net win.
It rescued Spoon (`60 -> 22`) and Bread (`60 -> 48`) but lost TissueBox
(`49 -> 60`).  The result is positive in aggregate but exposes continuing
negative-transfer risk.

## Gate result

Ten of eleven requirements passed:

- authentic success noninferior to raw target: pass;
- authentic success strictly superior to node-only: pass;
- paired net win nonnegative: pass (`+1`);
- five changed tasks: pass;
- seven changed source edges: pass;
- authentic changes exceed node-only: pass;
- source admission rate within the frozen range: pass (`10.14%`);
- zero reopened completed slots: pass;
- zero selected postcondition failures: pass;
- required-option invariance: pass;
- each claimed effect action-changing: **fail**, because `BIND=0`, `RELATE=7`.

## Next valid step

V11 may fix only the claim mismatch: declare the transferred unit as the
guarded source successor edge `BIND -> RELATE`, and require changed `RELATE`
edge receipts rather than a changed entry-node `BIND`.  It must use another
fresh adaptation split and may preserve the still-unread confirmation IDs.

No threshold, 60-step endpoint, target grounder, source graph, or success gate
should change.  V11 should still require authentic budgeted success to exceed
the node-only control, positive paired net wins, multiple changed source edges,
zero reopened slots, and zero failed postconditions.

## Audit hashes

- manifest: `a066cea5191d51710091307eea76ad1269931f3a1d5b9b56b3aaa647d2a2cb77`
- candidate: `c066b21a097f9c9c9803d1e65e2ee157c21ce5735d3f35cc6863a020ef10cfee`
- adaptation report: `eabaf136b27bdbf73f8eea5647bc78d5f77c91f439111511936648bc1dca3147`
- authentic compiled graph: `cb7c086a184333660b239b20f846db9287cd8e7a18f0f5920e7a52ce2369ab9e`
- node-only graph: `7114067205507344cf48f518d40ad7694edc5fb7a403a3d3266f474f90e1e2f4`

The compact result is
`docs/results/budgeted_executable_source_graph_v10_summary.json`; full traces
remain under ignored `runs/budgeted_executable_source_graph_alfworld_v10/`.
