# 00-system — System-level control documents

Top-of-stack documents that govern how the rest of the corpus is read and how releases are judged. They do **not** implement modules — they pin invariants, scoreboards, and ownership.

| Document | Purpose |
|----------|---------|
| [`PLAN-SYSTEM-NORTHSTAR.md`](PLAN-SYSTEM-NORTHSTAR.md) | The single canonical scoreboard (Layer 1 end-task / Layer 2 mechanism / Layer 3 cost), the headline metric (Joint Success Rate), and the binding go/no-go decision rules every other plan must report against. |
| [`PLAN-EVAL-FIRST-TARGET.md`](PLAN-EVAL-FIRST-TARGET.md) | First end-to-end evaluation contract for the project: short-video evidence-grounded reasoning. Defines task contract, six evaluation axes, failure taxonomy `F1`–`F7`, judge protocol, slice plan, phase rollout (E0 → E2). |
| [`DISCUSSION-COMPONENT-RESPONSIBILITIES.md`](DISCUSSION-COMPONENT-RESPONSIBILITIES.md) | Role walkthrough across Skill Bank Agent / Harness / Orchestrator on four short-video scenarios. Onboarding aid; live plan files win on conflict. |

Read these first if you are new to the repo, then jump to the stage / component plan you need.

Back to [plans/README.md](../README.md).
