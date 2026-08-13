# WebShop neural-symbolic applicability V8

> Historical note: this document records the V8 safety-only result. Credentials
> were subsequently restored, the counterfactual target grounder and
> all-visible-constraint fix were completed in V9/V10, and an independent V11
> fresh-goal replication passed the frozen transfer gates. See
> `docs/WEBSHOP_NEURAL_SYMBOLIC_TRANSFER_V9_V11_RESULTS.md`.

## Current result

V8 fixes the demonstrated safety failure but does not yet demonstrate transfer
benefit. Its scientific status is
`SAFETY_FIX_ONLY_NO_TRANSFER_EVIDENCE`.

On exact-request replay of the eight V7 diagnostic tasks, target-only,
safe neural-only, and safe authentic-source conditions all obtained `1/8`
strict success, `6/8` pass success, mean reward `0.573`, and mean length `7.0`.
V8 made zero interventions. This recovers the strict success that V7 lost, but
it is abstention rather than a success-rate improvement.

The grouped adaptation/calibration/confirmation run was not executed because
both available Decision API credentials returned HTTP 401. Failed zero-step
attempts are retained as infrastructure receipts and are excluded from all
scientific metrics. The untouched held-out partition was not read or run.

## Semantic split correction

The V7 task-ID partition contained only 12 unique ASIN groups among 24 consumed
tasks. Seven of eight reserve task goals duplicated an adaptation or
qualification goal exactly. V8 groups by `goal.asin` and quarantines every ASIN
seen in the eight V7 diagnostic tasks.

The four remaining consumed groups are deterministically assigned to:

| role | groups | representative tasks |
|---|---:|---|
| adaptation | 2 | `webshop.29`, `webshop.43` |
| calibration | 1 | `webshop.12` |
| confirmation | 1 | `webshop.7` |
| diagnostic | 8 | previously evaluated V7 groups |

No semantic group crosses roles. This is a small development split, not a
replacement for a larger group-held-out evaluation.

## Applicability shield

V8 admits source reranking only when all of the following hold:

1. the canonical environment state exactly equals the preceding post-action
   state;
2. target rank zero repeats the preceding action;
3. target rank zero is not a product-constraint action;
4. the alternative is target-native and reversible, never `Buy Now` or another
   commit action;
5. its predicted repeat probability is at least `0.2` below rank zero; and
6. authentic source value gives it positive advantage over rank zero.

A matched safe neural-only condition applies the same predicates without
source values.

In the V7 diagnostic replay, exact stalls occurred only on constraint radios:
pecan on task 1, black on task 28, and orange on task 49. The shield therefore
preserved target rank zero at all four opportunities. All 192 Decision requests
were exact cache hits, all 24 episode receipts completed, and condition initial
state hashes matched.

## Target-native candidate interventions

All candidates at the four exact stalls were executed once from independently
reconstructed sessions. All 18 branch state hashes matched and no branch
failed.

| symbolic candidate class | branches | state changed | terminated |
|---|---:|---:|---:|
| constraint | 6 | 0 | 0 |
| commit | 4 | 4 | 4 |
| navigation | 7 | 4 | 0 |
| other | 1 | 0 | 0 |

Every commit action terminated immediately, with mean reward `0.646`. This is
the early-purchase failure that reduced V7 strict success.

The old observational grounder was badly miscalibrated on these
counterfactuals: state-change MSE was `0.544` and termination MSE was `0.222`.
In particular, it predicted termination probability `0.0` for all four commit
actions even though all four terminated. Aggregate validation on logged actions
therefore did not measure the off-policy predictions used for reranking.

## What is and is not established

Established:

- V7 negative transfer came from an applicability and counterfactual-grounding
  failure, not an environment crash.
- Exact-stall, constraint-preservation, and irreversible-commit predicates
  prevent all observed harmful interventions.
- Candidate-level target interventions are necessary; observational effect MSE
  is not a sufficient gate.

Not established:

- V8 does not improve success or efficiency because it never intervenes.
- There is no confirmation-task result.
- There is still no evidence that authentic game source values beat a matched
  target-native recovery rule.

## Required continuation

After restoring a valid Decision credential, run only the two grouped
adaptation tasks, collect candidate interventions, fit/calibrate transfer
utility on adaptation plus `webshop.12`, freeze the result, and then run the
single consumed confirmation task `webshop.7`. Do not train on the 18 diagnostic
branches and do not open the original held-out tasks.

Authoritative artifacts:

- `configs/webshop_grouped_development_v8.json`
- `configs/webshop_neurosymbolic_applicability_v8.json`
- `runs/webshop_neurosymbolic_applicability_v8/diagnostic_replay8/summary.json`
- `runs/webshop_neurosymbolic_applicability_v8/candidate_interventions/diagnostic_report.json`
- `runs/webshop_neurosymbolic_applicability_v8/applicability_report.json`
