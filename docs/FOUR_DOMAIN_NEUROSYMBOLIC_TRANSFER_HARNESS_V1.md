# Four-domain neural-symbolic transfer harness V1

## Result

The harness now has a source-qualified, target-native, fresh-formal route for each requested target domain:

| Target route | Transfer condition | Target comparator | Paired change | Evidence |
|---|---:|---:|---:|---|
| WebShop product search/commit | 18/32 | target-only 11/32 | 7 wins, 0 losses | fresh formal, `p=0.015625` |
| ALFWorld text household workflow | 22/24 | target-only 14/24 | 9 wins, 1 loss | fresh held-out procedural-game suite, `p=0.021484375` |
| DiscoveryWorld Easy scientific spatial commit | 4/8 eligible | target-native myopic 2/8 | 2 wins, 0 losses | fresh formal, `p=0.5` |
| TIR single-image maze sequence | 23/24 | raw neural 14/24 | 9 wins, 0 losses | fresh formal, `p=0.00390625` |

The unified audit status is `FOUR_DOMAIN_FRESH_FORMAL_SKILL_DISPATCH_VALIDATED`. This means four exact target routes are both executable and evidence-qualified. All four routes now have game-source evidence, but they do not share one universal source game or one universal skill: WebShop, DiscoveryWorld, and TIR use distinct Sokoban-derived structures, while ALFWorld uses a controlled procedural-game suite.

## Harness contract

The dispatcher performs exact matching over:

```text
target domain + target interface + required capabilities + minimum evidence tier
```

Its only legal outputs are `SELECT_SKILL` and `ABSTAIN`. It cannot emit a target action. Each selected route binds:

- an immutable source artifact and source-confirmation receipt;
- a formal target-evidence receipt;
- hash-bound target adapter files;
- a target-native neural grounder;
- a target-native executor that retains action authority.

Unknown domains, unsupported interfaces, missing grounding capabilities, insufficient evidence, artifact drift, and any route that gives the source direct target-action authority fail closed.

## Registered skills

### Sokoban positive-effect guard

Transfers `positive effect -> COMMIT -> VERIFY; otherwise POSITION -> RECOMPUTE`. It is used by WebShop and DiscoveryWorld, but each target has a different neural grounder and native executor. Source coordinates, objects, action names, and durations do not transfer.

### Procedural-game typed workflow value

Transfers matched-intervention value over the typed `SEARCH -> ACQUIRE -> TRANSFORM -> PLACE -> VERIFY` procedure to ALFWorld. Four controlled finite-horizon game surfaces train the value ensemble; two disjoint game surfaces test its source-side generalization. Source-native actions are independently alpha-renamed, while ALFWorld-native neural heads bind symbolic options to admissible target actions. This is game-suite-to-ALFWorld procedural transfer, not Sokoban-only, arbitrary-game, or zero-shot transfer.

### Sokoban anonymous topology executor

Transfers bind/execute/refute/verify/commit/abstain over an anonymous graph to TIR maze tasks. Target neural binding supplies direction semantics and endpoint colors; target pixel grounding supplies the graph. It does not cover TIR rotation, RefCOCO, or visual search.

## Important negative boundaries

- The earlier binary Sokoban-to-ALFWorld transfer regressed and remains negative; the successful ALFWorld route uses a richer typed workflow value learned from a designed procedural-game suite.
- The ALFWorld ontology and target symbolic parser are designed, and its neural grounder uses target adaptation data. The result does not establish unsupervised ontology discovery. The one held-out two-object task regressed, exposing a multiplicity-state gap.
- DiscoveryWorld V22 Normal had zero eligible forks and remains failed. V23 validates two Easy interfaces only.
- The broad four-family TIR effect V7 qualification scored 4/12 for every condition and remains failed. The validated route is TIR maze only.
- Video QA is intentionally absent from this registry and dispatches to `ABSTAIN`.

These failures are why the library contains multiple source-qualified skills and exact target routes instead of one high-level `POSITION/COMMIT` heuristic.

The ALFWorld final evidence is an exact frozen deterministic replay after a foreground transport interruption. The first invocation wrote no final report; no scientific input changed before replay. See `docs/PROCEDURAL_GAME_ALFWORLD_NEUROSYMBOLIC_V1_RESULTS.md` for the full disclosure and factorization.

## Reproduction

```bash
PYTHONPATH=src python scripts/audit_four_domain_neurosymbolic_library.py \
  --registry configs/neurosymbolic_skill_library_v1.json \
  --output docs/results/four_domain_neurosymbolic_harness_v1_summary.json

PYTHONPATH=src:. pytest -q \
  tests/test_neurosymbolic_skill_library.py \
  tests/test_four_domain_neurosymbolic_registry.py
```

The registry is `configs/neurosymbolic_skill_library_v1.json`. Compact target evidence is under `docs/results/`; full local receipts remain under `runs/`.
