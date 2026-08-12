# Subgoal option contrast V15: leverage found, source-specific breadth failed

Date: 2026-08-12

## Verdict

V15 moved the intervention boundary from a concrete relation/property action to
the abstract option attempted next.  It reused the historically frozen V4
controller, which had passed a controlled multi-source-to-ALFWorld held-out
experiment, and tested it only as a shadow policy on 64 new two-object ALFWorld
train identities:

```text
controlled source workflows -> frozen option values
target observation/actions   -> target-native neural grounding
source authority             -> choose SEARCH/ACQUIRE/TRANSFORM/PLACE/VERIFY
target authority             -> choose the concrete native action within an option
```

The main trajectory always executed the source-disabled target-native policy.
Reward was discarded and official success was never read or serialized.

The strict option-level preflight found substantial causal leverage but failed
the preregistered source-specific breadth gate:

- all `64/64` tasks contained an authentic-versus-target option contrast;
- all `64/64` contained a contrast after at least one object-placement cycle;
- `22/64` contained an option where authentic differed from both target and the
  phase-permuted source control;
- those 22 tasks covered all eight destination strata, but only Cabinet,
  Drawer, and Shelf had at least four such tasks; the gate required four
  destination groups;
- no outcomes, matched treatment forks, value model, confirmation tasks, or
  ALFWorld `valid_unseen` tasks were opened.

Final status:
`STRICT_OUTCOME_BLIND_OPTION_CONTRAST_GATE_FAILED_STOP`.

## Why the first gate was quarantined

The first frozen preflight correctly established native-action disagreement,
but its primary gate counted all action contrasts.  It reported 1,435 such
contrasts.  Post-run audit showed that 718 were `SEARCH -> SEARCH`: different
native navigation/examination actions inside the same abstract option.  That is
not the subgoal-level intervention V15 was meant to test.

Before reading any outcome, a second plan was frozen over the same already
consumed development identities.  It tightened the opportunity definition to:

```text
authentic abstract option != target-control abstract option
```

and the source-specific definition to:

```text
authentic option != target option
and authentic option != phase-permuted option
```

The strict replay reproduced exactly 680 option-level contrasts from the broad
report's per-task option-pair counts.  The broad action-level status is retained
for audit but does not authorize outcome collection.

## Strict outcome-blind result

Across 64 tasks, the strict audit recorded:

| Metric | Observed | Required |
|---|---:|---:|
| Tasks with any option contrast | 64 | 32 |
| Tasks with a source-specific option contrast | 22 | 16 |
| Tasks with a second-cycle option contrast | 64 | 16 |
| Destination groups with >=4 source-specific tasks | 3 | 4 |
| Identity/receipt failures | 0 | 0 |
| Outcomes recorded | 0 | 0 |

There were 680 option contrasts, including 645 after at least one placement
cycle.  Only 69 were source-specific relative to the phase-permuted control.
The dominant patterns were:

| Target -> authentic; phase control | Count | Source-specific |
|---|---:|---:|
| `SEARCH -> PLACE; phase=PLACE` | 443 | 0 |
| `SEARCH -> ACQUIRE; phase=ACQUIRE` | 126 | 0 |
| `SEARCH -> PLACE; phase=SEARCH` | 47 | 47 |
| `ACQUIRE -> SEARCH; phase=SEARCH` | 19 | 0 |
| `PLACE -> SEARCH; phase=SEARCH` | 19 | 0 |
| `PLACE -> SEARCH; phase=PLACE` | 10 | 10 |
| `ACQUIRE -> SEARCH; phase=ACQUIRE` | 9 | 9 |

Thus most of the high option contrast was shared by authentic and corrupted
source controllers.  The non-trivial authentic-only residue exists, but it is
clustered by task/destination rather than broad enough for the registered fork
experiment.

## Mechanistic interpretation

V15 confirms the V14 recommendation that a skill changing the next subgoal has
far more intervention support than another relation/property edge:

```text
V13 relation edge:       10 informative task forks / 62 opportunities
V14 property program:     7 contrast tasks / 64 tasks
V15 option controller:   64 option-contrast tasks / 64 tasks
```

However, intervention support is not source-specific value.  In 569 of the 680
option contrasts, authentic and phase-permuted controllers chose the same
`PLACE` or `ACQUIRE` option against target `SEARCH`.  Running success outcomes
on those states would mostly test a generic source-induced option prior.

The 69 source-specific opportunities also expose risk that must be evaluated
causally before deployment.  Some authentic choices place the correct second
object; others attempt to place unrelated held objects or switch back to search.
Action legality and option disagreement therefore do not establish utility.

## Decision and next experiment

Do not run full trajectories or a new confirmation split from V15.  The safe
continuation is to enlarge **source-specific option support**, not merely add
more target instances:

1. return to source discovery and learn a control that distinguishes when
   `PLACE` or `ACQUIRE` is preferable to continued `SEARCH`, using matched
   source interventions rather than phase-observational correlation;
2. require that authentic and phase/wrong-source controls disagree at the
   abstract option level across at least four independent target strata;
3. only then freeze one first source-specific opportunity per task and execute
   exact-state target/authentic/phase matched forks with a shared continuation;
4. require official-success non-inferiority and positive utility before a
   source controller is allowed online.

This remains controlled synthetic multi-source neural-symbolic transfer.  It is
not evidence that the failed real-game `PROGRESS/STALL -> SWITCH/PERSIST`
microcontroller became transferable.

## Audit artifacts

- broad pool: `configs/subgoal_option_contrast_pool_v15.json`
- strict audit plan: `configs/strict_subgoal_option_audit_plan_v15.json`
- broad report (ignored): `runs/subgoal_option_contrast_v15/report.json`
- strict report (ignored):
  `runs/subgoal_option_contrast_v15/strict_audit_report.json`
- compact result: `docs/results/subgoal_option_contrast_v15_summary.json`

Content hashes:

- broad pool stable hash:
  `c6c2cdea492c0ba3edd002024d7991ac352bceb7973513d3d538c5edfab76bed`
- broad report stable hash:
  `e398195387e9b07ce6c6b603c4455a7569a549fbbfa43b49b539686aca266c95`
- strict plan stable hash:
  `b68a31a6ef5c17b2bcc9720558e40ef9134c3a8873162ca633368a5f953a8838`
- strict report stable hash:
  `84f2b9c15a8cdba51c8d0a5957a023571eed47030862ed2345ac4c78dd4a5976`
