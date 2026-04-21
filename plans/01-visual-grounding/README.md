# 01-visual-grounding — Stage 1: Visual Grounding

Pixels → structured `<state>` schema. The first stage of the canonical pipeline. Everything downstream consumes the schema produced here.

| Document | Purpose |
|----------|---------|
| [`PLAN-VISUAL-GROUNDING.md`](PLAN-VISUAL-GROUNDING.md) | Design of the VLM parser, canonical `<state>` schema, three grounding heads (heuristic / vision / OmniParser / tool loop), domain adapters (Gym-V, BrowserGym, OSWorld, video), schema completeness guarantee (§12), Qwen3-VL-8B training plan. |
| [`PLAN-VISUAL-GROUNDING-MILESTONES.md`](PLAN-VISUAL-GROUNDING-MILESTONES.md) | Concrete week-by-week build-out: 5-stage inference pipeline, routing policy (Path A / B / C), training phases 0–4, 7 ablations, success criteria. |
| [`PLAN-VISUAL-SKILLS.md`](PLAN-VISUAL-SKILLS.md) | *Optional* — transferable visual grounding **strategies** as skills (multi-step perception programs that compose tools). Sits between perception tools and reasoning skills; uses `belief / binding` effect contracts rather than world-effect contracts. |

Cross-cutting consumers: [Action Agent](../02-action-agent/PLAN-ACTION-AGENT.md) (inner-MDP `GROUND`), [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) (contract learning), [Uncertainty Calibration](../08-cross-cutting/PLAN-UNCERTAINTY-CALIBRATION.md) (per-field uncertainty source).

Back to [plans/README.md](../README.md).
