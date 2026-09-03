# Sokoban → CLEVRER event-graph transfer V14

## Result

The independent formal route passed its frozen gate on 720 prospective CLEVRER
validation questions:

| Condition | Correct | Accuracy | Recoveries |
|---|---:|---:|---:|
| Target explicit-relation dynamics | 489/720 | 67.92% | 0 |
| Authentic Sokoban proof + target neural grounding | **511/720** | **70.97%** | 59 |
| Target-only trajectory dynamics | 430/720 | 59.72% | 720 |
| Target base-receipt recovery | 506/720 | 70.28% | 68 |
| Permuted uplift control | 484/720 | 67.22% | 122 |
| Shuffled proof binding | 503/720 | 69.86% | 69 |
| Inverted source effect | 406/720 | 56.39% | 144 |
| Shuffled source action binding | 485/720 | 67.36% | 59 |
| Family-matched marginal | 480/720 | 66.67% | 59 |

Authentic versus the primary target-only representation has 27 paired wins, 5
losses, and 688 ties (net +22; exact two-sided sign-test
`p=0.0001130742`). All frozen lineage, compiler, non-triviality, superiority,
proof-binding, phase, inverted-effect, and marginal-control gates passed.

## Neural-symbolic boundary

The source contributes only the confirmed control contract:

```text
COMMIT -> VERIFY_EXPECTED_EFFECT
EXPECTED_EFFECT_REFUTED -> REPLAN_OR_ABSTAIN
```

CLEVRER supplies its own paired neural dynamics representations. A typed postfix
compiler and symbolic executor generate step-level event-graph proof receipts;
the learned target-native uplift grounder decides whether to keep the explicit
relation representation or recover with the trajectory representation. Source
actions, coordinates, game tokens, video tokens, official functional programs,
and gold answers are not runtime inputs.

This is therefore a formal event-graph route, not the failed free-text event
ledger used in earlier Video-Holmes/STAR/NExT-QA experiments.

## Boundary

The result is adaptation-based and specific to the pinned local CLEVRER setup.
It does not establish zero-shot ontology induction, Video-Holmes, STAR, or
NExT-QA transfer. It also does not prove that Sokoban provenance is necessary
relative to an extensionally identical target-written controller; it shows that
the source-qualified structure transfers and is more useful than the specified
destructive and target-only controls.

Run the portable evidence audit with:

```bash
PYTHONPATH=src:. python scripts/audit_video_event_graph_v14.py
```
