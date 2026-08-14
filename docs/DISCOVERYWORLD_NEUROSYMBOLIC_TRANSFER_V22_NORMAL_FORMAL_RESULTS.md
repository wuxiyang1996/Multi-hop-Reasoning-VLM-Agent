# DiscoveryWorld neural-symbolic transfer: V22 Normal formal result

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
