# DiscoveryWorld neural-symbolic transfer: V17 formal result

## Verdict

V17 **did not validate** game-to-DiscoveryWorld transfer. The run is technically
clean and contains a causal mechanism signal, but it fails the pre-registered
coverage, success-gain, success-rate-gain, and source-control-separation gates.

The exact machine-readable result is in
`docs/results/discoveryworld_v17_formal_summary.json`.

## Frozen evaluation

Before opening the reserve, V17 froze six Easy instances from two interfaces:
Space Sick and Proteomics seeds 2--4. It also froze the source receipt, target
model, prompts, first-baseline-commit eligibility rule, five matched conditions,
six-step recovery horizon, and all success gates. Normal-difficulty seeds remain
unopened.

The target-only baseline completed all six instances with no invalid native
actions and no policy access to evaluator scorecards:

| Metric | Result |
|---|---:|
| Official successes | 1 / 6 |
| Nonzero-progress tasks | 6 / 6 |
| Mean normalized score | 0.6389 |
| Invalid native actions | 0 |
| Oracle scorecard use | 0 |

## Outcome-blind fork coverage

The freezer read no reward, terminal outcome, evaluation, or scorecard field.
Only Space Sick seed2 contained a target-baseline `DROP`/`PUT` proposal; the other
five episodes were excluded with `NO_PREDECLARED_COMMIT_ACTION`.

| Frozen gate | Required | Observed | Pass |
|---|---:|---:|---:|
| Eligible forks | >= 4 | 1 | no |
| Both target themes represented | yes | Space Sick only | no |

This is the dominant protocol failure: eligibility depended on the target
baseline already reaching and proposing a commit, so it excluded the Proteomics
episodes that had made 4/6 or 5/6 official progress but had not dropped the flag.

## Matched intervention result

The one eligible fork was exactly matched by both policy-state and hidden-world
hashes. All selection receipts validate, all five arms completed without runtime
errors, and none saw the evaluator scorecard.

| Condition | Success | Recovery behavior |
|---|---:|---|
| Target-native myopic | yes | correct `PUT` at step 1 |
| Authentic Sokoban effect + target grounding | yes | same correct `PUT` at step 1 |
| Commit-availability control | yes | same correct `PUT` at step 1 |
| Inverted-effect control | no | alternated target-bound `POSITION` for 6 steps |
| Position-prior control | no | alternated target-bound `POSITION` for 6 steps |

This is valid evidence that the transferred effect polarity changes the policy:
authentic commits where inverted effect does not. It is **not** efficacy evidence,
because target-native myopic and availability-only selection already succeed.

## Pre-registered conclusion

V17 has zero authentic negative transfer and passes all runtime, receipt, and
matched-fork checks. It nevertheless has:

- 1 rather than 4 eligible forks;
- zero authentic success gain rather than at least 2;
- zero success-rate gain rather than at least 0.25; and
- a 1--1 tie with the commit-availability source control.

Therefore the formal claim fails. Normal seeds must not be opened under V17.

## Bitter lesson and next protocol

The transferable program is an **applicability-gated decision rule**:
`COMMIT` only when an exact target-native positive-effect witness exists;
otherwise `POSITION` and recompute. A fork rule based on the baseline's first
commit proposal observes mostly states where the baseline already knows the
right commit. It both misses hard near-complete cases and suppresses the
contrast that the symbolic guard is designed to create.

The next version should use consumed Easy seeds only for adaptation and define an
outcome-blind **first applicability/disagreement state**. A state is eligible only
when target-native grounding supplies a valid bound `COMMIT` and target-bound
`POSITION`, target-native myopic prefers the commit, and the exact symbolic
effect witness rejects it. This selects on policy applicability, not task
outcome. After measuring its coverage on consumed data, the rule, model, horizon,
conditions, and gates must be frozen before any Normal-difficulty reserve is
opened.
