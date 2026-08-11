# Source visual intervention v1

## Why this collection is necessary

The existing fresh Thunder Force III receipts are replayable, but they retain
only textual observations and coarse score/lives fields.  They do not retain
the frame or object-level state needed to identify lower-level effects such as
reposition, engage, evade, or verify.  The previously tested
`PROGRESS/STALL -> PERSIST/SWITCH` controller also failed its random-policy
control, so it must not be promoted to a transferable skill.

This protocol returns to the native source transitions.  It reuses the exact
prefixes chosen by the no-human-hint game agent, but makes no new LLM calls.
At each frozen state it replays every native action and records the actual
before/after frame.

## Frozen design

- Split unit: whole episode.
- Split rule: sort episode IDs, then round-robin discovery, qualification,
  held-out.
- Point selection: hash only `(episode_id, step)` after applying structural
  prefix bounds.  Reward, future return, observation content, action identity,
  and skill prose are not inputs to selection.
- Coverage: two states per episode and every native action at every state.
- Replay identity: seed + complete native-action prefix + the source logger's
  canonical-JSON observable SHA-256 (not a raw-text byte hash).
- Visual identity: PNGs are content-addressed by SHA-256.  Every fork binds its
  before and after PNG to the intervention receipt.
- Current claim boundary: these are causal visual receipts, not yet a source
  skill and not transfer evidence.

The initial run consumes only the discovery split.  Qualification and held-out
snapshot identities are already frozen in the plan, but their interventions
must remain uncollected until a candidate visual option vocabulary and source
gate are frozen.

## Run

```bash
sbatch cluster/collect_source_visual_intervention_v1.sbatch
```

For a one-snapshot mechanics smoke, use the same collector with
`--snapshot-limit 1`.  Such a truncated run has `selection_complete=false` and
cannot be used for a source gate.

## Next gate

Discovery images may be used to define target-agnostic effect variables (for
example action-conditioned motion, state change, damage, or score change).
Any symbolic option induced from them must then beat, on untouched source
episodes:

1. a shuffled option/effect assignment,
2. a marginal action prior,
3. a hash-random native policy, and
4. the original source action where relevant.

Target-native grounding starts only if the source option is predictable and
causally valuable on qualification and held-out source episodes.  Target pixels
or action names are never copied from the source artifact.

## Discovery result (2026-08-11)

The complete Thunder discovery run collected 96/96 valid forks (eight frozen
states times 12 native actions).  Every snapshot had one identical before-frame
hash across all treatments.  Exact counterfactual frame equivalence induced the
following anonymous source grounding:

- `PERSISTENT_NULL_EFFECT`: `MODE`, `START`, `X`, `Y`, `Z` (effect rate 0/8);
- `STABLE_CAUSAL_EFFECT`: `A` (6/8), `B` (7/8), `C` (7/8);
- `CONTEXTUAL_CAUSAL_EFFECT`: `UP` (4/8), `DOWN` (5/8), `LEFT` (2/8),
  `RIGHT` (5/8).

These native names are retained only to test the candidate back in Thunder.
The proposed transfer object is the anonymous three-way effect partition and
the symbolic rule “exclude the persistent nullspace, alternate stable-effect
basis actions with contextual probes, and verify the predicted effect using a
target-native grounder.”  The artifact remains
`DISCOVERY_CANDIDATE_NOT_SOURCE_QUALIFIED`.

Qualification uses untouched episodes, h=8 rollouts, and two estimands.  Its
five paired conditions are authentic effect structure, cardinality-preserving
fully deranged structure, all-action hash random, discovery action marginal,
and repeated original source action.  The primary endpoint is cumulative game
reward under the full treatment regime.  Qualification must beat shuffled,
random, and marginal controls both in mean paired reward and net paired wins;
otherwise held-out source and every target run stay blocked.

## Qualification result: rejected

Job `7237914` completed all 80 planned trajectories with no protocol failures.
The primary full-regime h=8 results were:

| condition | mean reward | positive rate | life-loss rate |
| --- | ---: | ---: | ---: |
| authentic effect structure | 0.0 | 0.000 | 0.250 |
| shuffled effect structure | 50.0 | 0.375 | 0.000 |
| all-action hash random | 0.0 | 0.000 | 0.000 |
| discovery action marginal | 0.0 | 0.000 | 0.125 |
| repeat source action | 12.5 | 0.125 | 0.125 |

Using source episode as the independent paired unit (four episodes), authentic
versus shuffled had zero wins, one tie, and three losses; its mean margin was
-50.  Authentic tied random and marginal at zero, so it also failed both of
those strict gates.  A second execution after the episode-unit and runtime-code
receipt fixes produced byte-identical trajectory receipts.  Held-out source and
all targets were therefore not run.

The important diagnosis is not “neural-symbolic transfer is impossible.”  It is
that **visual effect existence is not option value**.  Actions in the apparent
visual nullspace can implement useful waiting while the world continues to
evolve, and a stable visual effect can move the agent into danger.  In this run,
the shuffled controller accidentally composed `DOWN`, `B`, and wait-like keys;
three qualification episodes gained reward, while the authentic controller's
value-blind effect cycling gained none and lost lives in two snapshots.

A defensible v2 must add source-native delayed valence and composition before
transfer: learn when an effect is useful from the current frame, represent
wait/observe as an option instead of filtering it, and test matched sequences
such as actuate -> wait/observe -> verify and risk-conditioned reposition.  The
symbolic sequence may transfer, but source action names, pixels, and value
predictions may not; those require a target-native neural grounder.  This v1
candidate must not be sent to WebShop, visual reasoning, or video tasks.
