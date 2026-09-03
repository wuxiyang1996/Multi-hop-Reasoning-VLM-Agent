# AGQA2 frozen-router + unchanged-V65-grounder formal V1

## Outcome

This is the requested raw-video test of a question-only target-native router,
the source-induced symbolic IR, and the off-the-shelf
`qwen/qwen3-vl-235b-a22b-instruct` visual grounder.  The point estimate is
positive, but the preregistered confirmatory result is **failed**.

| Arm | Exact correct |
|---|---:|
| Neural-only matched direct | 45/80 |
| **Source-induced** | **53/80** |
| Source-permuted | 45/80 |
| Target-written equivalent | 53/80 |

Source-induced versus neural-only is `14W/6L/60T`, net `+8` questions or
`+10.0 pp`.  The one-sided exact binomial value is `p=0.057659`.  The frozen
protocol required at least 6 wins, net gain at least 5, at most 4 losses, and
`p <= .05`.  It therefore fails both the success-gain and negative-transfer
gates.  The result must not be reported as validated AGQA transfer.

## What did pass

- 80 questions come from 80 videos disjoint from router train/validation and
  all runtime exposure known when selection was frozen.
- The question-only router was trained only on AGQA official-train
  train/development partitions.  Its video-disjoint validation had 1.0
  precision and recall on 16,685 rows.
- The provider-facing runtime uses the historical V65 collector and grounder
  module blobs exactly (`c845...` and `87a4...`), with Qwen235 untrained and
  off the shelf.  No post-V65 grounder prompt, threshold, or weight is used.
- Route accuracy is 80/80; source-permuted abstains 80/80; target-written
  dynamics match source-induced 80/80; 74/80 source executions are
  authorized.
- Runtime has no answer, functional-program, scene-graph, source-identity, or
  competing-operand access.  Gold access begins only after all 80 runtime
  receipts freeze.
- Accepted-call reported provider cost is `$0.12977558`, below the `$0.75`
  preregistered cap.  Malformed provider attempts that never produced an
  accepted usage receipt are disclosed separately and are not included in
  that reported-cost number.

## Failure diagnosis

All six paired losses are visual-grounding decisions, not router or symbolic
control failures:

- four false negatives: recurrent scans incorrectly concluded an event was
  unobserved (`carrying`, `drink from`, and two referential `touching` cases);
- two false positives: the grounder hallucinated `lean on bicycle` or treated
  a person as the carried object.

The same executor also corrected 14 direct-model errors, so this is not a
zero-value mechanism.  It is an insufficiently safe raw-video grounder: the
gain is not yet statistically confirmatory and the loss count violates the
frozen safety bound.

## Outcome-neutral transport repairs

Two infrastructure failures were repaired without reading target outcomes or
changing a prediction:

1. V65 cached two receipts before semantic validation.  They explicitly said
   `UNOBSERVED` with null frames but had an inconsistent
   `observability=OBSERVED` field.  The files were moved unchanged into a
   content-hashed quarantine, then the identical requests were retried.  The
   validator, prompt, model, and input hashes did not change.
2. After all 80 runtime receipts froze, the V65 evaluator expected a
   preregistered `program_sha256`.  The privacy-preserving selection had
   intentionally not read program metadata.  A finalizer-only transport copy
   changed this assertion to “verify if present”; the original V65 collector
   remains the file used for grounder identity.  No provider, grounding,
   prediction, or gate semantics changed.

Both fixes have machine-readable receipts in the run directory.

## Subsequent multi-route diagnostic

There was no second untouched cohort compatible with the binary
`RELATION/EXISTS` router. A later, separately disclosed multi-class router
unlocked 91 previously video-unseen questions from additional temporal route
families. That post-V74 exploratory run failed route agreement (78/91), caused
the frozen runtime to abstain globally, and cannot change V1 or V74. See
[`AGQA2_MULTICLASS_ROUTER_V65_FORMAL_V2_RESULTS.md`](AGQA2_MULTICLASS_ROUTER_V65_FORMAL_V2_RESULTS.md).

## Reproduce and audit

```bash
PYTHONPATH=src:. python scripts/evaluate_agqa2_router_v65_grounder_formal_v1.py \
  --protocol configs/agqa2_router_v65_grounder_formal_v1_protocol.json \
  --config configs/agqa2_router_v65_grounder_formal_v1.json \
  --selection configs/agqa2_router_heldout_formal_v1_selection.json \
  --manifest configs/agqa2_router_v65_grounder_formal_v1_manifest.json \
  --report runs/agqa2_router_v65_grounder_formal_v1/base_report.json \
  --output runs/agqa2_router_v65_grounder_formal_v1/formal_evaluation.json

PYTHONPATH=src:. python scripts/audit_agqa2_router_v65_formal_v1.py \
  --output docs/results/agqa2_router_v65_formal_v1_audit.json
```

Primary machine-readable audit:
[`results/agqa2_router_v65_formal_v1_audit.json`](results/agqa2_router_v65_formal_v1_audit.json).
