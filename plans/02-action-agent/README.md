# 02-action-agent — Stage 2: Action Agent

Two-level MDP decision agent. Consumes the structured `<state>` from [Stage 1](../01-visual-grounding/) and selects environment actions guided by skills retrieved from [Stage 3](../03-skill-bank/).

| Document | Purpose |
|----------|---------|
| [`PLAN-ACTION-AGENT.md`](PLAN-ACTION-AGENT.md) | Two-level MDP (outer environment + inner reasoning hops), inner-action alphabet `GROUND / CHECK / RETRIEVE / COMMIT / EXECUTE`, three-agent role split (Actor / Skill-Use / Synthesis-Reflection), co-evolution and GRPO decomposition, uncertainty-driven `GROUND` triggering, tiered model architecture (Tier 0 / 1 / 2), reward shaping `r_env + r_follow + r_cost`. |

Cross-cutting consumers: [Harness](../05-harness/PLAN-HARNESS.md) (the Actor's online policy that consumes Harness-filtered `eligible_skills`), [Pipeline Orchestrator](../06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) (`InnerHopRecord` / `ActionRecord` artifacts), [First Eval Target](../00-system/PLAN-EVAL-FIRST-TARGET.md) (`reasoning_trace_summary`, Avg Hops).

Back to [plans/README.md](../README.md).
