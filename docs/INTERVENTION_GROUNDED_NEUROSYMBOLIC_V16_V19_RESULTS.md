# Intervention-grounded neural-symbolic transfer V16–V19

## Outcome

The current mechanism is rejected.  V19 failed the success gate, and a post-hoc
contract audit found that V18 did not establish neural-symbolic transfer.

Mechanically, V18 showed an outcome-blind option-change contrast on 64
already-consumed ALFWorld train-development tasks.  The authentic source controller
changed the target fallback's abstract option on 41 tasks and produced 1,279
source-specific contrasts.  Four destructive controls did not reproduce this
behavior, and the task-cluster bootstrap lower bounds were 0.252–0.266.  However,
the target number bound into the source `causal_effect` slots was a class-balanced
expert-action imitation score.  It was neither a causal successor-event probability
nor calibrated on the source controller's support.  The V18 gate is therefore
post-hoc invalidated; it demonstrates numerical control-flow dependence, not valid
transfer evidence.

V19 then executed matched full-rollout forks under a plan frozen before those
outcomes were read.  Every condition achieved 0/64 official successes and mean
return 0:

| condition | successes | mean source interventions | mean steps |
|---|---:|---:|---:|
| target neural baseline | 0/64 | 0.00 | 60.0 |
| authentic intervention effect | 0/64 | 14.59 | 60.0 |
| source-effect permutation | 0/64 | 0.00 | 60.0 |
| target-effect permutation | 0/64 | 0.11 | 60.0 |

The pre-registered V19 transfer gate therefore failed.  No confirmation or
`valid_unseen` task was read or run.

## What V18 actually wired together

The factorization was:

```text
controlled synthetic source intervention receipts
    -> pairwise relative abstract-option advantage + conformal admission
    -> target-native expert-action imitation ranking
    -> abstract option override
    -> target-native representative action
```

The wiring did satisfy several useful negative controls:

- source raw tokens and target ALFWorld tokens were not shared;
- required-option, workflow phase/progress, held/transformed state, reward,
  official success, and completion labels were absent from target inference;
- a source relative option-value model altered decisions made from a target-native
  sigmoid action-ranking head;
- cyclic source-effect, shuffled-value, effect-blind, and target-effect controls
  were frozen before the V18 shadow;
- authentic decisions were selective rather than an always-on option prior.

These properties are insufficient for neural-symbolic skill transfer because the
neural score did not have the semantics required by the symbolic feature it
populated.  It is also not evidence that a completed game skill was transferred.
V17's source is a controlled synthetic multi-surface workflow generator.  Surface
names such as `game`, `science`, and `maintenance` do not turn those samples into
real game rollouts.

## Oracle-shortcut and optimizer audits

V16 initially passed its source gate almost perfectly.  A post-hoc audit found that
`matches_required_option` and its interactions were still available to the pairwise
model.  The result was quarantined before target use.

V17 rebuilt source inputs from matched intervention-effect receipts and zeroed every
required-phase, match, workflow-progress, urgency, and match-interaction feature.  On
2,048 held-out controlled source states it admitted 1,521 interventions, all with
positive exact-DP utility; each destructive control abstained.  This establishes the
controlled source mechanism, not real-game transfer.

The first two oracle-free target MLP fits reached their LBFGS iteration limits.  They
were quarantined before shadow execution.  V18 reduced the neural head to a convex
sigmoid unit and required both authentic and permuted-label optimizers to converge.
It passed on 266 isolated adaptation-validation states:

- authentic pairwise AUC: 0.847;
- permuted-expert control pairwise AUC: 0.291;
- authentic top-1: 0.312 versus random 0.032;
- task-cluster lower bound, top-1 minus random: 0.173;
- task-cluster lower bound, pairwise AUC minus control: 0.527.

These checks prevented a hand-written phase oracle and an unconverged model from
being mistaken for transfer evidence, but they did not check score semantics or
source-target calibration support.  Passing them was necessary, not sufficient.

## Why V18 contrast did not become V19 success

The outcome logs expose three concrete failures.

First, the target field called `neural_effect_probability` was actually expert-action
likelihood.  It was trained from action identity only.  It did not predict a causal
change between the before and after target state.  A good imitation ranking on
isolated adaptation states is not an intervention-effect model.

This also caused a severe support mismatch.  In source train/calibration states,
normally exactly one of five options had effect score above 0.1 and the other four
were near `1/18 = 0.056`.  In all 1,279 target contrast states, two or three available
options exceeded 0.1; target scores ranged 0.439–0.997.  The authentic override
always moved away from the imitation fallback: selected-minus-fallback score ranged
from -0.433 to -0.077.  Source-only conformal residuals provide no coverage guarantee
under this shift.

Second, eliminating hand-written binding by setting unknown source binding fields to
the same zero was safe against leakage but insufficient for execution.  The
controller could choose a valid abstract option bound to the wrong target object or
tool.  Examples in the frozen V19 log include:

- three toilet tasks executing `look` 59 times after repeated
  `ACQUIRE -> SEARCH` overrides;
- desk placement tasks executing `use desklamp 1` 55–56 times after
  `ACQUIRE -> TRANSFORM` overrides, even though transformation was not the task;
- a CD-to-drawer task alternating 29 `take cellphone` and 29 `move cellphone`
  actions.

Third, the transferred object was an open-loop option preference.  It had no
target-native successor-state update, termination predicate, or recovery edge.  The
authentic condition's adjacent exact-action repeat rate rose from 0.267 to 0.448,
and its median longest identical-action run was 17.5 steps.  More interventions were
not more procedural competence.

The right interpretation is:

> V18 verifies only that the source controller can create a source-specific option
> contrast under the old numerical wiring.  The score-semantics and support audit
> invalidates that contrast as evidence of neural-symbolic transfer.  V19 separately
> rejects the mechanism as a success-improving controller.

## Bitter lesson

Do not repair this failure with a hand-written phase parser, a cooldown, a larger
repeat penalty, or an option-order prompt.  Those changes can suppress the visible
loops while restoring exactly the task-specific heuristic leakage the experiment was
designed to remove.

The implementation now fails closed at the evidence boundary.  Legacy imitation
artifacts may still rank the target baseline's native actions, so the agent remains
runnable, but they produce no source option features and every source decision
abstains.  Even a future causal target scorer must also present a joint source-target
support receipt before source conformal admission can run.

The transferable unit must include a causal transition contract:

```text
(learned precondition event,
 native intervention,
 predicted successor-event delta,
 learned termination predicate,
 recovery transition)
```

The source should supply the relational automaton and relative transition values.
The target should supply neural grounding for every event and binding from its own
counterfactual transition receipts.  Unknown bindings must cause abstention, not a
zero-filled guess.

## V20 recommendation

The next experiment should be frozen around atomic transition receipts rather than
high-level option names:

1. Collect real source forks from Sokoban, Thunder, and additional games.  Each
   receipt must contain observable state before/after an intervention, action-set
   change, reversibility/no-op evidence, and repeated-intervention decay.  Do not
   export raw domain tokens.
2. On target adaptation tasks only, collect matched native-action counterfactual
   forks.  Train target-native neural heads to predict successor observation/action
   embeddings and causal event deltas.  Do not use reward, official success, a
   required-option label, or a hand-written workflow phase.
3. Learn target-native object/destination binding from which entity-conditioned
   interventions produce the predicted event delta.  Bind or abstain; never choose a
   representative action using option score alone.
4. Transfer a small closed-loop automaton over event deltas.  Advance an automaton
   edge only after the target neural observer detects its expected successor event;
   terminate or recover using learned predicates.
5. Require held-out source-game transition prediction, isolated target adaptation
   effect calibration, a joint source-target covariate-support receipt, outcome-blind
   target state-change contrast, and destructive source/target binding controls
   before another consumed-development success fork.  Source conformal admission
   must fail closed when the joint support gate is absent.
6. Keep confirmation and `valid_unseen` sealed until a pre-registered development
   outcome gate passes.

This is more demanding than option transfer, but it directly addresses the observed
failure without adding a task-specific workflow heuristic.

## Reproducibility boundary

Compact authoritative metrics are in
`docs/results/intervention_grounded_neurosymbolic_v16_v19_summary.json`.
The reproducible post-hoc contract audit is implemented by
`scripts/audit_intervention_grounded_contract_v20.py`; its compact result is
`docs/results/intervention_grounded_contract_audit_v20.json`.
The large V18/V19 raw reports remain under ignored `runs/` paths:

- `runs/intervention_grounded_transfer_shadow_v18/report.json`
  (`report_sha256` `9dad3a5d...e360`);
- `runs/intervention_grounded_outcome_forks_v19/report.json`
  (`report_sha256` `deb10758...2996`).

Frozen plans and portable candidates are committed under `configs/`.  The raw target
adaptation receipt is intentionally referenced by hash and remains a local run
artifact; no confirmation or held-out artifact is needed to reproduce the code-level
and compact-report checks.
