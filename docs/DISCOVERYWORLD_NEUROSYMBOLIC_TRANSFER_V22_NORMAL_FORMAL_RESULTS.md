# DiscoveryWorld neural-symbolic transfer: V22 Normal formal result

> Reproduction note (2026-08-14): the `same_location`/null normalization found
> in the later consumed V24 diagnosis now lives in
> `src/motif_transfer/discoveryworld_normal_binding.py` and the Normal-only
> launcher `scripts/run_discoveryworld_commit_recovery_normal_v24.py`. This
> preserves the byte-exact V23 Easy adapter required by the validated registry.

## Verdict

V22 **does not validate held-out game-to-DiscoveryWorld transfer**. The formal
run stopped after three completed Normal target trajectories because all three
contained zero predeclared `DROP`/`PUT` actions. With only three unopened tasks
remaining, the frozen minimum of four eligible forks became mathematically
impossible.

No matched transfer arm was run on Normal data. This is a clean coverage failure,
not a negative-transfer result. V21's consumed-Easy mechanism rescue remains
valid, but it cannot be promoted to held-out generalization.

The exact record is
`docs/results/discoveryworld_v22_normal_formal_early_stop.json`.

## Frozen before opening

Commit `eac08378344b1f8cb2b295b3d4820a1ae4f03fa8` fixed the source program,
target policy, neural binder/grounder prompts, symbolic predicates, shared local
spatial realizer, five matched arms, and tests. Commit `01d895f` then froze:

- Space Sick and Proteomics Normal seeds2--4;
- a 96-step oracle-free target acquisition horizon;
- outcome-blind first `DROP`/`PUT` fork selection;
- an eight-step matched recovery horizon;
- minimum four eligible forks, both themes, zero negative transfer, at least one
  authentic success gain over myopic, and strict wins over availability and
  always-position controls.

The pre-open receipt self-hash is `4c4ee380cc75e92c6488535b26809b08b3bfb78f48d098abce26aeed3bb2880f`.

## Opened target trajectories

| Task | Steps | Success | Score | Invalid actions | Schema fallbacks | DROP/PUT fork |
|---|---:|---:|---:|---:|---:|---:|
| Space Sick Normal seed2 | 96 | no | 0.0526 | 3 | 7 | no |
| Proteomics Normal seed2 | 96 | no | 0.5000 | 0 | 6 | no |
| Space Sick Normal seed3 | 96 | no | 0.1053 | 6 | 5 | no |

All episode self-hashes validate and the policy saw no official scorecard. Fork
eligibility was computed without reading action success, terminal outcomes,
evaluation, or scorecards. Each trajectory received
`NO_PREDECLARED_COMMIT_ACTION`.

Space Sick repeatedly failed NPC accessibility and then cycled around the
cafeteria. Proteomics successfully acquired the meter and measured several
animals, but spent the remaining horizon in manual navigation and never reached
the flag/statue placement stage. The main limitation is therefore target-state
acquisition before the transferred effect guard can act.

There is also a more fundamental protocol error: difficulty does not preserve
the intervention interface. Space Sick Easy ends by putting a diagnosed food
item into a jar, whereas Space Sick Normal asks the agent to make colonists eat
ten safe mushrooms through Chef/dialog dynamics. The latter has no required
`DROP`/`PUT` commit at all. V22 selected a target by scenario name instead of by
task-native action/effect schema, making its Space Sick fork rule structurally
inapplicable before any rollout began.

## Early-stop proof

After three trajectories:

`observed eligible (0) + unopened tasks (3) < required eligible (4)`.

The formal conclusion could no longer change, so Proteomics Normal seed3 and
both seed4 tasks were left unopened. Running more tasks or weakening the gate
would only spend reserve after the pre-registered claim had already failed.

## What is actually established

The strongest honest statement is:

> On consumed Easy first-commit forks, source-qualified effect structure caused
> one matched success rescue with no negative transfer. Under the frozen Normal
> target policy, the agent did not reach any eligible commit fork in the first
> three tasks, so held-out transfer efficacy could not be evaluated and formal
> validation failed.

The next research problem is not another selector heuristic. DiscoveryWorld
needs two prerequisites before another reserve is opened: an outcome-blind
applicability audit over the target task's actual action/effect schema, and a
stronger **target-native acquisition/tool controller** that can reliably reach
scientific decision states on compatible tasks such as Proteomics. Both must be
qualified independently of the source program.

## V24 sequel: acquisition fixed, transfer still negative

V24 implemented the missing source-blind, target-native acquisition layer for
Proteomics Normal. It exhaustively surveys the public teleport locations,
measures five distinct species with the native proteomics meter, computes the
robust protein-vector outlier, picks up the flag, and returns to the matching
statue. It reads neither the source skill nor the official scorecard. On the two
consumed development seeds it reached `DROP` and officially succeeded 2/2
(steps 24 and 27), so the V22 coverage blocker is resolved for this one
compatible interface.

An outcome-blind fork was then frozen immediately before seed2's first `DROP`.
The matched diagnostic completed with valid receipts and no oracle use:

| Condition | Success | Recovery steps |
|---|---:|---:|
| Recorded target baseline | 1/1 | 1 |
| Target-native myopic | 1/1 | 1 |
| Authentic Sokoban effect guard | **0/1** | 8 |
| Commit-availability control | 1/1 | 1 |
| Inverted-effect control | 1/1 | 1 |
| Position-prior control | 0/1 | 8 |

This is a genuine consumed-development negative-transfer diagnosis, not a fresh
formal result. The target policy simply committed and was already at ceiling.
The authentic effect guard instead rejected `DROP` for all eight steps. The
immediate cause is also informative: the natural goal says “directly beside,”
but the binder encoded `same_location, distance=0`, while exact target facts
placed the statue one square north. The current symbolic relation vocabulary
cannot represent undirected adjacency. Availability-based selection ignores
that failed witness and succeeds; the authentic exact-effect guard correctly
refuses to claim an effect it cannot prove.

Therefore V24 does **not** justify opening another Normal reserve. A future
version should add a target-native `adjacent` predicate during development and
test an earlier scientific decision point where the source-blind target policy
is below ceiling. Merely forcing `DROP` here would erase the mechanism rather
than demonstrate transfer utility. The compact record is
`docs/results/discoveryworld_proteomics_normal_v24_summary.json`.
