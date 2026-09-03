# AGQA2 multi-class-router + unchanged-V65 Qwen diagnostic V2

## Result

The 91-video run completed with 91 immutable runtime receipts and `$0.14929539`
of accepted provider calls. It **failed** and cannot validate raw-video AGQA2
transfer.

| Check | Result |
|---|---:|
| Videos disjoint from V1 | 91/91 |
| Runtime rows | 91/91 |
| Question-only route vs evaluator-only official route | 78/91 (85.7%) |
| Unified harness authorizations | 0/91 |
| Neural-only / source-induced | 20/91 / 20/91 |
| Source-permuted abstentions | 91/91 |
| Target-written dynamics matches | 91/91 |

The program router was trained and thresholded only on AGQA train/development
questions. Selection used only public question text and required agreement
between the learned classifier and deterministic parser. After all visual and
direct-model receipts froze, official functional programs revealed 13 route
errors. The historical V65 runtime therefore correctly failed closed globally;
it did not permit a post-outcome per-row salvage.

All 13 disagreements have the same form: the public-text parser predicted
`RELATION_RECURRENT`, while the evaluator-only functional program classified
the task as `NO_EXACT_SOURCE_TYPE`. The four temporal-pair and one
temporal-single routes were all correct. Thus this is a false-authorization
boundary problem, not confusion between the two temporal operators.

This means the V2 failure is not evidence that the source symbolic executor is
wrong. It is evidence that a question-only route classifier is an avoidable
confounder for the transfer estimand. The benchmark-native functional program
can supply operator type without supplying video facts or the answer, but that
constitutes a different, explicitly oracle-routed evaluation track.

## Scientific status

V2 was prospectively frozen relative to its own 91-video cohort, but it was run
after the earlier V74 stop policy. It is therefore labeled
`POST_V74_EXPLORATORY_FAILED`; it cannot overturn the preregistered negative
raw-video verdict. It also cannot be repaired on the same cohort. The raw Qwen
track stops here.

The paper-primary video result instead uses shared benchmark-native structured
grounding for every matched arm:

- AGQA2: official STSG behind identical budgeted typed tools;
- CLEVRER: one identical frozen event-graph proof receipt per task.

In that primary estimand, grounding can affect absolute task accuracy but
cannot create the paired source-vs-control delta. See
[`TWO_VIDEO_GROUNDING_ISOLATED_PRIMARY_V1.md`](TWO_VIDEO_GROUNDING_ISOLATED_PRIMARY_V1.md)
and the V2 consolidated audit.

## Outcome-neutral runtime repairs

Malformed provider strings were never cached and were retried unchanged. One
task cached two schema-invalid receipts (`UNOBSERVED` plus pixel intervals);
they were moved byte-for-byte into a content-hashed quarantine, then the same
requests were repeated. After all 91 receipts froze, a finalizer-only optional
program-hash check was applied because privacy-preserving selection had not
opened program metadata. None of these repairs changed model, prompt, frames,
validator, prediction, or gate semantics.

## Reproduce the audits

```bash
PYTHONPATH=src:. python scripts/evaluate_agqa2_multiclass_router_v65_formal_v2.py \
  --protocol configs/agqa2_multiclass_router_v65_formal_v2_protocol.json \
  --config configs/agqa2_multiclass_router_v65_formal_v2.json \
  --selection configs/agqa2_multiclass_router_formal_v2_selection.json \
  --prior-selection configs/agqa2_router_heldout_formal_v1_selection.json \
  --manifest configs/agqa2_multiclass_router_v65_formal_v2_manifest.json \
  --report runs/agqa2_multiclass_router_v65_formal_v2/base_report.json \
  --output runs/agqa2_multiclass_router_v65_formal_v2/formal_evaluation.json

PYTHONPATH=src:. python scripts/audit_two_video_grounding_isolated_primary_v2.py \
  --output docs/results/two_video_grounding_isolated_primary_v2.json
```
