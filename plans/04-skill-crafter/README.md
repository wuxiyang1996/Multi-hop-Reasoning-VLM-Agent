# 04-skill-crafter — Stage 4: Skill Crafter

Top-down skill creation: composition (chaining skills the actor never tried), generalization (re-binding a skill across domains), hypothesis (proposing skills no rollout has produced). Outputs are **typed proposals only** — the [Unified Skill Gate](../07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) decides admission.

| Document | Purpose |
|----------|---------|
| [`PLAN-SKILL-CRAFTER.md`](PLAN-SKILL-CRAFTER.md) | Composer / generalizer / hypothesizer modes, typed proposals (`PatchProposal | ComposeProposal | TransferProposal | RetireProposal`), failure reflection & counterfactual reasoning, frozen 32B/72B teacher design with phased adaptation policy (§2), frozen-teacher improvement channels, integration with visual-grounding tool traces, Crafter-private `FailurePatternStore` (§6.7). |

Cross-cutting consumers: [Skill Bank](../03-skill-bank/PLAN-SKILL-BANK.md) (writes go into `draft_store` / `candidate_store`), [Harness](../05-harness/PLAN-HARNESS.md) (proposal runtime viability via gates G0–G5), [Failure Routing](../08-cross-cutting/PLAN-FAILURE-ROUTING.md) (consumes failure clusters).

Back to [plans/README.md](../README.md).
