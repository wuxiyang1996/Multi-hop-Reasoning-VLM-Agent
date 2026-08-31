# WebShop selective neural-symbolic transfer V6

> Superseded by `WEBSHOP_NEUROSYMBOLIC_CAUSAL_V7_RESULTS.md`. V7 fixes
> cross-restart goal nondeterminism, completes all conditions, adds a matched
> neural-only baseline, and executes exact-state causal intervention forks.

## Audited outcome

The post-audit runner fixes source-state tracking for
`selective_other_game_source`. In the fixed same-round comparison, selective
Sokoban/Tetris/2048 transfer matched target-only success and reward while using
fewer steps on the five complete task pairs. Three of eight tasks had a shared
Decision-model infrastructure failure in all four conditions, so the target
held-out partition remains unauthorized and unread.

This is evidence for selective recovery efficiency, not for an improved
WebShop success rate.

## Bug found and fixed

The original V6 action selector correctly used the Candy/Mario artifact for
`selective_other_game_source`, but its recurrent `previous_option` update used
the Sokoban/Tetris/2048 artifact. The artifacts have different anonymous option
spaces, so the original Candy/Mario control was invalid.

The runner now routes both selection and recurrent state updates through one
`_source_for_condition` helper. A regression test covers selective and
unrestricted other-game conditions. The old Candy/Mario result of 2/7 strict
successes must not be used.

## Setup

- The target Decision model proposes only legal BrowserGym actions.
- The target-native neural grounder predicts anonymous action effects.
- Frozen game-derived symbolic option dynamics can only rerank those actions.
- V6 opens reranking only when rank zero has predicted repeat probability at
  least 0.5 and the source-selected alternative lowers it by at least 0.2.
- All four conditions in the fixed run used the same runner hash and shared a
  per-split exact-request cache.
- Initial accessibility-tree hashes matched across conditions within every
  task. They changed across WebShop server restarts, so old caches did not
  guarantee cross-round replay.

## Fixed-runner complete-case results

Complete tasks were `webshop.6`, `webshop.13`, `webshop.28`, `webshop.34`, and
`webshop.49`. Tasks `webshop.1`, `webshop.24`, and `webshop.31` were excluded
because all four paired conditions encountered the same output/schema failure.

| condition | strict | pass | mean reward | mean steps | interventions |
|---|---:|---:|---:|---:|---:|
| target only | 1/5 | 4/5 | 0.650 | 7.6 | 0/38 |
| selective authentic game source | 1/5 | 4/5 | 0.650 | **6.4** | 2/32 (6.25%) |
| selective phase-permuted source | 1/5 | 4/5 | 0.650 | 7.6 | 0/38 |
| selective Candy/Mario source | 1/5 | 4/5 | 0.650 | 7.8 | 1/39 (2.56%) |

Authentic transfer changed strict success, pass success, and mean reward by
zero. It reduced mean trajectory length by 1.2 steps.

## Mechanistic cases

On `webshop.28`, all conditions reached reward 0.5. Target-only,
phase-permuted, and fixed Candy/Mario required 12 steps. Authentic transfer
detected a repeated click, intervened once, and reached the same reward in 5
steps. This is a seven-step source-specific recovery improvement.

On `webshop.49`, target-only and phase-permuted reached reward 0.75 in 9 steps.
Authentic and fixed Candy/Mario each intervened once and reached the same reward
in 10 steps. This is a one-step efficiency regression.

The net fixed-runner result is therefore an efficiency signal with both a
positive and a negative intervention case, not a uniform benefit.

## Gate decision

On the five complete pairs, all method gates pass: intervention rate, no
success reduction, improved reward-or-steps, and success metrics no worse than
the destructive controls. An explicit fail-closed operational safeguard
requires at least seven of eight complete pairs; only five completed. Held-out
execution is therefore not authorized.

The previous seven-pair run remains useful as historical diagnostics for the
authentic and phase conditions, but its Candy/Mario condition is invalid due to
the fixed state-tracking bug and its aggregate report is superseded.

## Next action

Stabilize the Decision backend on already consumed tasks by increasing the
output budget and recording it in every receipt, then demonstrate at least
seven of eight paired completions without changing the transfer policy. After
that reliability check, freeze a new runtime receipt before considering the
untouched 24-task held-out partition.

The authoritative machine-readable fixed-runner report is
`runs/real_game_multitarget_neurosymbolic_v6_fixed/qualification_report.json`.
