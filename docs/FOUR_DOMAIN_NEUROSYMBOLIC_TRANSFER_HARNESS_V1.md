# Four-domain neural-symbolic transfer harness V1

Web overview: [`NEUROSYMBOLIC_TRANSFER_FOUR_DOMAIN_STATUS.html`](NEUROSYMBOLIC_TRANSFER_FOUR_DOMAIN_STATUS.html).

## Result

The original four exact routes passed their frozen route-specific gates. We then
ran independent reserves under a separately frozen directional aggregate gate:

| Independent target reserve | Transfer | Comparator | Paired change | Route-level interpretation |
|---|---:|---:|---:|---|
| WebShop product search/commit | 14/32 | target-only 9/32 | 6W/1L/25T, `p=.125` | positive direction; the stronger per-route confirmation gate did **not** pass |
| ALFWorld text household workflow | 31/32 | target-only 22/32 | 9W/0L/23T, `p=.00390625` | independently confirmed |
| DiscoveryWorld Easy spatial commit | 13/16 | target-native myopic 10/16 | 3W/0L/13T, `p=.25` | independently validated by its frozen route gates; small discordant sample |
| TIR single-image maze sequence | 41/48 | raw neural 19/48 | 22W/0L/26T, `p=4.768e-7` | independently confirmed |

All four deltas are positive and every domain has more paired wins than losses.
The predeclared equal-domain aggregate gate therefore passed with status
`FOUR_DOMAIN_INDEPENDENT_REPLICATION_VALIDATED`: equal-domain mean success-rate
delta `+27.08 pp`; pooled descriptive counts `40W/1L/87T`
(`p=3.82e-11`). Task units and success semantics differ, so the pooled statistic
is descriptive, not a replacement for each domain's test. In particular, this
aggregate result must not be reported as a successful strong WebShop
replication.

The original unified dispatch audit remains
`FOUR_DOMAIN_FRESH_FORMAL_SKILL_DISPATCH_VALIDATED`. All four routes have
game-source evidence, but they do not share one universal source game or skill:
WebShop, DiscoveryWorld, and TIR use distinct Sokoban-derived structures, while
ALFWorld uses a controlled procedural-game suite.

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

Transfers bind/execute/refute/verify/commit/abstain over an anonymous graph to TIR maze tasks. Target neural binding supplies direction semantics and endpoint colors; target pixel grounding supplies the graph. It does not cover RefCOCO or visual search. A separate Tetris inverse-group program now covers the exact TIR `rotation_game` interface.

## Seven-step extension audit

The follow-on plan has now been executed through step 7. “Complete” below means
the experiment was run and audited; it does not imply that every hypothesis was
positive.

| Step | Experiment | Result | Scientific reading |
|---:|---|---|---|
| 1 | Portable four-domain audit | complete | compact evidence and exact registry dispatch are clean-checkout auditable; three external runtimes/caches are still not vendored |
| 2 | Independent four-domain reserves | aggregate gate passed | ALFWorld and TIR strongly confirm; DiscoveryWorld is directionally positive/small; WebShop's strong individual gate fails |
| 3 | ALFWorld multiplicity | 4/6 vs 1/6, 3W/0L, `p=.25` | identity-aware target state fixes the observed count-two failure on this small reserve; not arbitrary-count validation |
| 4 | Non-maze TIR + DiscoveryWorld Normal | TIR rotation 19/21 vs 4/21, 15W/0L; DW Normal 0/1 vs 1/1 | target-native counterfactual grounding works for rotation; the old DW relation vocabulary cannot express adjacency and negatively transfers at a ceiling fork |
| 5 | Learned latent ontology | not confirmed | authentic held-out MSE `8.57` is worse than marginal `8.22`; universal pooled latent roles are not supported |
| 6 | Online transfer utility | 5 `SELECT_SKILL`, 4 `ABSTAIN` | calibrated route-level Beta lower bound prevents use of weak/negative/structurally invalid routes; this is not yet state-level learned applicability |
| 7 | CLEVRER event graph | 511/720 vs 489/720, 27W/5L, `p=.000113` | formal synthetic-video route passes; STAR, NExT-QA and Video-Holmes remain unvalidated |

## Important negative boundaries

- The earlier binary Sokoban-to-ALFWorld transfer regressed and remains negative; the successful ALFWorld route uses a richer typed workflow value learned from a designed procedural-game suite.
- The ALFWorld ontology and target symbolic parser are designed, and its neural grounder uses target adaptation data. The multiplicity extension is positive on only six count-two tasks; it does not establish arbitrary counts or unsupervised ontology discovery.
- The learned cross-game latent ontology failed fresh confirmation. This is evidence against replacing target-native grounding with one pooled universal ontology in the current implementation.
- DiscoveryWorld V22 Normal had zero eligible forks. The later source-blind acquisition reached a usable interface, but the consumed matched diagnosis produced negative transfer because `same_location` could not represent `beside`; a fresh Normal reserve was deliberately not opened.
- The broad four-family TIR effect V7 qualification remains failed. Maze and rotation are two separate executable programs. The rotation result does not establish source-provenance necessity because a target-written isomorphic controller tied authentic transfer.
- CLEVRER is the only validated video route. It uses a synthetic event graph; STAR, NExT-QA, Video-Holmes, and broad natural-video transfer remain unsupported.

These failures are why the library contains multiple source-qualified skills and exact target routes instead of one high-level `POSITION/COMMIT` heuristic.

The ALFWorld final evidence is an exact frozen deterministic replay after a foreground transport interruption. The first invocation wrote no final report; no scientific input changed before replay. See `docs/PROCEDURAL_GAME_ALFWORLD_NEUROSYMBOLIC_V1_RESULTS.md` for the full disclosure and factorization.

## What remains

The seven-step plan is experimentally complete. The remaining gaps are about
strengthening the claim, not unfinished executions hidden as successes:

- independently repeat WebShop until its frozen per-route confirmation criterion is either passed or rejected; the current reserve is positive but inconclusive;
- learn target-state applicability before acting. The current utility selector is route-level and post-replication, uses designed structural predicates, and cold-start abstains;
- replace the failed pooled latent ontology with a narrower induction target, such as causal roles inside one executable family, and validate on unopened games before target use;
- redesign DiscoveryWorld Normal around a target-native adjacency predicate and a non-ceiling decision point, then freeze a new reserve; do not reuse the consumed seed2 diagnosis as confirmation;
- test source-provenance necessity where feasible. TIR rotation currently shows useful structure, not that Tetris is the unique way to obtain it;
- extend video beyond synthetic CLEVRER to STAR/NExT-QA/Video-Holmes only after target event/identity grounding is stable;
- complete a full four-target clean-room replay package. WebShop, DiscoveryWorld, and TIR still depend on external runtimes/caches;
- add equal-budget target memory/retrieval, target-written workflow, and black-box target-policy comparisons.

Portable audit and ALFWorld replay instructions are in [`FOUR_DOMAIN_NEUROSYMBOLIC_RELEASE_V1.md`](FOUR_DOMAIN_NEUROSYMBOLIC_RELEASE_V1.md).

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

Additional machine-readable audits:

- `docs/results/four_domain_replication_v1_summary.json`
- `docs/results/online_transfer_utility_v1_audit.json`
- `docs/results/alfworld_multiplicity_v1_formal_summary.json`
- `docs/results/tir_rotation_counterfactual_v2_summary.json`
- `docs/results/discoveryworld_proteomics_normal_v24_summary.json`
- `docs/results/real_game_latent_options_v6_formal_summary.json`
- `artifacts/video_event_graph_v14/formal_report.json.gz`
