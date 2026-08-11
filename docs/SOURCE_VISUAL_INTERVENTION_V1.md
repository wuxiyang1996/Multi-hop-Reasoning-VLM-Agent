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
