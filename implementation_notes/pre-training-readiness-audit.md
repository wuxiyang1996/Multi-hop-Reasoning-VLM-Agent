# Pre-training readiness audit — what's missing before fast-loop GRPO launch

> **Status:** audit captured + **revised 2026-05-01 (revision 4 —
> S0 + S1 wrappers landed).** Synthesis of four parallel plan-vs-repo
> audits (Action Agent, Skill Bank, Skill Crafter, Harness, plus the
> cross-cutting visual-grounding and trainer scaffolds), reconciled
> against the on-disk SFT inventory under [`runs/`](../runs/)
> (revision 2), the recorded lane decision in
> [`skill-lane-decision.md`](skill-lane-decision.md) (revision 3 —
> lane (a): skills are retrieval payloads), and the **2026-05-01
> S0/S1 closure** (revision 4 — every S0 code/doc item shipped:
> `enable_protocol_patching` flag, threshold YAMLs, SFT manifest +
> load-smoke + schema-gen probe scripts, offline-promotion-cycle
> wrapper, lane-(a) banners on the four PLAN docs + harness README +
> roles doc; outstanding S0/S1 work is now GPU-bound execution of
> the shipped scripts, not new code or governance). Action items are
> tracked in the Tier 1–4 sections below; sequencing in §5;
> **the consolidated "Not Done" list is in §0**.
> **Last reviewed:** 2026-05-01 (revision 4).
> **Cross-refs:**
> [`runs/sft_schema_gen/schema_gen_20260430_091831/`](../runs/sft_schema_gen)
> (Phase-1 SFT, eval token-acc 0.9838 — see §0.2),
> [`runs/sft_coldstart/`](../runs/sft_coldstart)
> (5 LoRA SFT seeds: `action_taking`, `skill_selection`, `segment`,
> `contract`, `curator` — see §0.2),
> [`implementation_notes/skill-lane-decision.md`](skill-lane-decision.md)
> (**decided 2026-05-01:** lane (a) — skills are retrieval payloads,
> not runnable programs; closes T1.3),
> [`implementation_notes/crafter-harness-orchestrator-roles.md`](crafter-harness-orchestrator-roles.md)
> (§7 — historical context for the lane decision; superseded by
> `skill-lane-decision.md`),
> [`implementation_notes/harness-usability-and-intra-gymv-transfer.md`](harness-usability-and-intra-gymv-transfer.md)
> (§6.2 keystone — `bank.runnable()` is empty until the offline loop fires once),
> [`implementation_notes/protocol-lift-design.md`](protocol-lift-design.md)
> (lane (b) implementation hook — `labeling/_decorate_skill_records.py`),
> [`implementation_notes/single-vs-two-mdp-tradeoff.md`](single-vs-two-mdp-tradeoff.md)
> (**decided + shipped:** no `hop_select` LoRA, no inner MDP — see T3.6;
> remaining work is plan-doc cleanup only),
> [`plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md`](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md) (§5.0 fast-loop = gymv only),
> [`plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md`](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md) (§13 Phase-1 SFT bar),
> [`plans/05-harness/PLAN-HARNESS.md`](../plans/05-harness/PLAN-HARNESS.md) (§17 single reward sink, §20.2 ablations),
> [`plans/00-system/PLAN-SYSTEM-NORTHSTAR.md`](../plans/00-system/PLAN-SYSTEM-NORTHSTAR.md) (§5 stop/go, §7.3 release scoreboard),
> [`plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md`](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) (§4 eval-suite loader, §9 `skill_gate.yaml`),
> [`plans/08-cross-cutting/PLAN-EXPERIENCE-EXTENSION.md`](../plans/08-cross-cutting/PLAN-EXPERIENCE-EXTENSION.md) (§3 five typed extension records),
> [`plans/08-cross-cutting/PLAN-FAILURE-ROUTING.md`](../plans/08-cross-cutting/PLAN-FAILURE-ROUTING.md),
> [`plans/10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md`](../plans/10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md) (§3.2 HopTrace, §4.7 Crafter input contract),
> [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) (stale — refresh required, see T2.6).

This memo records the synthesis of four parallel plan-vs-repo audits run
on 2026-05-01 to answer one question: **"what are we still missing
before we start to run the training?"**

It exists because the same gaps kept being re-discovered in independent
audits (visual-grounding readiness, trainer co-evolution scaffold, Crafter
lane decision, Harness wire-up). Consolidating them here means the next
contributor opening `trainer/coevolution/` does not re-litigate which
items are blockers, which are deferred-by-design, and which are content
gaps masquerading as structural ones.

---

## 0. Not done — consolidated checklist (post-SFT discovery)

The first audit pass (revision 1) had `schema_gen` SFT as the single
biggest blocker. Inspection of [`runs/`](../runs/) on 2026-05-01
showed all five SFT adapters (`schema_gen` + 4 cold-start LoRAs) are
**already trained and on disk**. This revision drops T1.1 from the
"hard blocker" tier and re-prioritises the remaining work.

### 0.1 What's still not done — ranked by sprint

| ID | Sprint | Item | Why it blocks training | One-line fix |
|---|---|---|---|---|
| ~~T1.2~~ | ~~S1~~ | ~~`bank.runnable()` is empty~~ → **wrapper landed 2026-05-01.** Driver: [`scripts/run_offline_promotion_cycle.sh`](../scripts/run_offline_promotion_cycle.sh) (orchestrates `decide_promotion_gpt54.py` + inline `legacy_writeback.writeback_promotion` + `bank.runnable()` post-condition). **Outstanding:** one GPU-bound run on the labeled corpus to flip ≥1 record to `ACTIVE/SHADOW`. | — |
| ~~T1.3~~ | ~~S0~~ | ~~Lane decision (retrieval payload vs. runnable program) is unrecorded~~ → **closed 2026-05-01: lane (a) — context-only skills.** See [`skill-lane-decision.md`](skill-lane-decision.md). Skills are RAG-style retrieval payloads / procedural guidance for the actor LLM; the harness is an eligibility filter + validator, not an executor. All six sub-items (T1.3a – T1.3f) shipped — see §0.4. | — |
| **T1.4** | S3 | Real env executors for transfer-target adapters | `osworld` / `video` are pure stubs; `browser` / `visual_reasoning` have helpers but the trainer never calls them. Blocks G3a `min_target_domains_verified ≥ 1` ⇒ no skill ever leaves `CANDIDATE` even after promotion fires. | Pick one target (`browser` is plan default), bind `set_executor`, ship a few-shot demo file. |
| **T2.1** | S3 | Few-shot demo libraries for the four transfer targets | Compounds with T1.4 — without target demos `FewShotAdapter.adapt(...)` returns `target_domain_demo_unavailable`. | Mirror `harness/few_shot_demos_gymv.py` for the chosen target. |
| **T2.2** | S2 | `orchestrator/eval_suite.py` (G5 non-regression loader) | NORTHSTAR §5 stop/go cannot fire on automated promotions. | Frozen-suite loader returning the (`pre`, `post`) `EvalSuite` shape `GateService._run_non_regression` already accepts. |
| **T2.3** | S2 | Eval E0 driver + canonical scoreboard | `releases/{release_id}/scoreboard.md` has no producer ⇒ GRPO loop has no headline. | Implement `evaluation/driver.py` + `evaluation/answer_evaluator.py` per NORTHSTAR §7.3. |
| **T2.4** | S2 | `harness/reward_logger.py` not wired into GRPO | Audit channel is forked: shaping flows from `decision_agents.reward_func` instead of the Harness sink → §17 single-sink invariant violated. | Replace the import in `trainer/coevolution/grpo_training.py::_collect_grpo_records`. |
| ~~T2.5~~ | ~~S0~~ | ~~`configs/skill_gate.yaml` + `configs/failure_routing.yaml` missing~~ → **shipped 2026-05-01.** [`configs/skill_gate.yaml`](../configs/skill_gate.yaml) + [`configs/failure_routing.yaml`](../configs/failure_routing.yaml) (versioned policy v1.0.0; lane-(a) failure taxonomy with `lane_b_primary_mode` overrides). | — |
| ~~T2.6~~ | ~~S0~~ | ~~`IMPLEMENTATION-STATUS.md` stale (last 2026-04-21)~~ → **refreshed 2026-05-01** to point at all six checkpoint paths + Day-7→10 wire-up + `legacy_writeback` + `enable_protocol_patching` + threshold YAMLs + SFT manifest + offline-promotion driver. | — |
| T3.1 | S4 | Five typed extension records (`RunRecord`, `GroundingRecord`, `SkillInvocationRecord`, `FailureRoutingRecord`, `AnswerSupportRecord`) | Blocks multi-domain claims; not the fast loop. | Either ship the five dataclasses or update the plans to bless the existing analogs. |
| T3.2 | S4 | Failure router (R1–R5) entirely missing | L2/L4 failures dropped or consumed ad-hoc; contradicts §10 anti-goal #4. | New module `orchestrator/failure_router.py` + queue tree + dashboards. |
| T3.3 | S4 | `HopTrace` artefact + inner-hop logging | Phase B mining (typed `SkillIR`, transferable reasoning skills) has no input. Reconcile with the single-MDP decision (could be offline-only). | Add `HopTrace` as offline log on Action Agent; *do not* re-add online inner-hop LLM. |
| T3.4 | S4 | Canonical cross-domain ontology types (`Agent`, `UIElement`, `EvidenceSpan`, …) | Same blocker as T1.3 lane (b); typed validators can't run. | Module + dataclasses derived from PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS §3.4. |
| T3.5 | S4 | Online runtime adapters (`HarnessSkillProvider`, `RunnerActorAdapter`, `EnvLike`, `legacy_bridge`) | Day-10 `SkillHarnessHook` covers the trainer; these only block live-online deployment. | Defer until post-S2. |
| ~~T3.6~~ | ~~S0 (doc-only)~~ | ~~`hop_select` LoRA absence~~ → **bundled with T1.3e/T1.3f and shipped 2026-05-01.** The four PLAN-doc lane-(a) banners (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS, PLAN-COMPONENTS-IMPLEMENTATION) explicitly mark `hop_select` / `inner_mdp` references obsolete and point at [`single-vs-two-mdp-tradeoff.md`](single-vs-two-mdp-tradeoff.md). PLAN-ACTION-AGENT §5 / §5.5 retain the historical `hop_select` rationale; the lane-(a) banner in the four downstream plans is sufficient governance. | — |
| T4.* | post-launch | 7 open questions (hop-vocab, Q4 ablation, Phase-2 SFT trigger, counterfactuals, K-thresholds, Stage-5 window, Crafter input contract) | Tuning, not blocking. | See §5. |

### 0.2 What's already done — reconciliation against `runs/`

The Phase-1 SFT pass and all four cold-start LoRA seeds **landed
2026-04-30**:

| Adapter | Path | Base | Train rows | Epochs | Final eval loss | Final eval token-acc | Wall-clock |
|---|---|---|---|---|---|---|---|
| `schema_gen` | [`runs/sft_schema_gen/schema_gen_20260430_091831/`](../runs/sft_schema_gen) (final + ckpts 200/400/477) | Qwen3.5-35B-A3B | 477 steps × bs 16 effective | 1 | **0.0861** | **0.9838** | 6 h 30 m |
| `action_taking` | [`runs/sft_coldstart/decision/action_taking/`](../runs/sft_coldstart/decision/action_taking) | Qwen3.5-9B | 30 482 | 2 | 0.0889 | — | 12 h 27 m |
| `skill_selection` | [`runs/sft_coldstart/decision/skill_selection/`](../runs/sft_coldstart/decision/skill_selection) | Qwen3.5-9B | 29 532 | 2 | **0.0685** | — | 12 h 11 m |
| `segment` | [`runs/sft_coldstart/skillbank/segment/`](../runs/sft_coldstart/skillbank/segment) | Qwen3.5-9B | 4 343 | 4 | 0.3234 | — | 4 h 25 m |
| `contract` | [`runs/sft_coldstart/skillbank/contract/`](../runs/sft_coldstart/skillbank/contract) | Qwen3.5-9B | 1 316 | 5 | 0.0748 | — | 1 h 25 m |
| `curator` | [`runs/sft_coldstart/skillbank/curator/`](../runs/sft_coldstart/skillbank/curator) | Qwen3.5-9B | 193 | 15 | 1.218 ⚠️ | — | 24 m |

LoRA shape: `r=16`, `α=32`, dropout 0.05, 9-projection target set
(`q/k/v/o_proj`, `in_proj_qkv`, `out_proj`, `gate/up/down_proj`).
`schema_gen` adapter is 16 MB; the 5 cold-start adapters are 148 MB
each.

#### Quality flags discovered during inventory

* **`schema_gen` eval token-acc 0.9838 (eval loss 0.086).** PLAN-VISUAL-GROUNDING-MILESTONES §13 specifies *exact-match* schema accuracy thresholds (≥ 70 % `gymv`, ≥ 50 % `browser`); the trainer reports `mean_token_accuracy` (token-level CE-derived). The numbers strongly suggest the bar is met but **must be re-verified with an exact-match probe** before pinning `schema_gen_ckpt_id` into `RunRelease`. New T1.1 sub-item → §0.3.
* **`segment` eval≈train (0.32 vs. 0.30).** Either Stage-2 boundary preference is high-entropy or the LR schedule is under-trained. Flag for Phase-F warm-up convergence test; not a blocker.
* **`curator` eval 1.22 vs. train 0.49 over 15 epochs on 193 rows.** Severe overfit on a tiny corpus. Mitigations: (a) treat curator as a tie-breaker only in early GRPO; (b) regenerate corpus with more games; (c) stronger early stopping. **Now the top quality concern.** New T2.7 → §3.
* **`schema_gen` is on the 35B-A3B control-plane base, not the 9B actor.** Matches PLAN-VISUAL-GROUNDING (frozen control-plane), but the 9B actor and 35B-A3B grounding LoRA cannot share a vLLM worker. Confirm the deployment topology (T2.8 → §3).
* **`sft_summary.json` is per-GPU**, not run-wide. `runs/sft_coldstart/sft_summary.json` lists only `action_taking`. Standardise into a run-wide manifest so the offline-promotion driver knows where to find each adapter (T2.9 → §3).
* **`*.partial_20260430_091054` log shards** suggest an earlier 5-GPU run was preempted/crashed. Confirm the recovered final checkpoints are the post-restart artefacts, not stale partials (T2.10 → §3).

#### What this changes in the readiness ledger

* T1.1 (perception SFT) is **closed — pending exact-match verification**.
* All Phase-F GRPO LoRA targets (`skill_select` / `action_execute` / `CONTRACT` / `CURATOR` / `SEGMENT`) have **SFT warm-starts on disk**. Phase F is unblocked on weight inputs.
* The single biggest remaining Tier-0 blocker is now **T1.2** (run the offline promotion loop once), not T1.1.

### 0.3 New / promoted action items from the SFT discovery

* **T1.1′ (closed-pending-verification)** — write a 30-min exact-match probe script that loads `schema_gen` over Qwen3.5-35B-A3B, runs it on a held-out slice of `labeling/output/grounding/{gymv,env_wrappers,browser}/` triples, and reports the schema-match rate against the milestone thresholds. Target file: `evaluation/probe_schema_gen_exact_match.py`.
* **T2.7 (curator overfit mitigation)** — see §3.7.
* **T2.8 (vLLM topology check)** — see §3.8.
* **T2.9 (run-wide SFT manifest)** — see §3.9.
* **T2.10 (audit `*.partial_*` shards)** — see §3.10.

### 0.4 New action items from the lane decision (T1.3 closed → lane (a))

The lane decision recorded in [`skill-lane-decision.md`](skill-lane-decision.md)
opens six follow-ups that replace the original open-ended T1.3:

| ID | Sprint | Item | One-line fix |
|---|---|---|---|
| ~~T1.3a~~ | ~~S0~~ | ~~Default-disable the Repairer in the live trainer Crafter~~ → **shipped 2026-05-01.** `SkillCrafterService(enable_protocol_patching=False)` default + `_crafter_hook.run_crafter_step(enable_protocol_patching=…)` plumb-through + `CoEvolutionConfig.crafter_enable_protocol_patching` + `scripts/run_coevolution.py --enable-protocol-patching` opt-in. New tests at [`tests/test_crafter_lane_a_flag.py`](../tests/test_crafter_lane_a_flag.py) verify the default fall-through to the Hypothesizer; offline driver [`labeling_supplement/reflect_per_episode_gpt54.py`](../labeling_supplement/reflect_per_episode_gpt54.py) opts back in. | — |
| **T1.3b** | S2 | Add `RewriteProposal` (rename `ComposeProposal` → `MergeProposal` if cleaner) in `data_structure/extensions/bank_mutation.py` | New typed proposal subclasses for the lane-(a) Crafter taxonomy |
| **T1.3c** | S2 | Implement `BANK_GAP` / `RETRIEVAL_MISLEAD` / `STALE_DESCRIPTION` `FailureClass` taxonomy in the *live* (not offline-mirror) Crafter path | Replaces the six protocol-edit `RecoveryStrategy` values for live use; the protocol-edit values stay for offline gate diagnostics |
| **T1.3d** | S2 | Replace the multi-domain `ACTIVE` invariant with `min_retrievals_per_skill` in `PromotionOrchestrator` | Unblocks `ACTIVE` promotion on single-domain `gymv` banks |
| ~~T1.3e~~ | ~~S0 (doc-only)~~ | ~~Update [`harness/README.md`](../harness/README.md) §22 + [`crafter-harness-orchestrator-roles.md`](crafter-harness-orchestrator-roles.md) §7.3 / §7.5~~ → **shipped 2026-05-01.** `harness/README.md` §22 lane block + `crafter-harness-orchestrator-roles.md` §7.3 banner + §7.5 supersedure note + new "Post-decision rule" sub-section. | — |
| ~~T1.3f~~ | ~~S0 (doc-only)~~ | ~~Update plan documents (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS, PLAN-COMPONENTS-IMPLEMENTATION)~~ → **shipped 2026-05-01.** Lane-(a) banner blocks at the top of all four PLAN docs, each pointing at [`skill-lane-decision.md`](skill-lane-decision.md) + [`single-vs-two-mdp-tradeoff.md`](single-vs-two-mdp-tradeoff.md) and explicitly marking `hop_select` / `inner_mdp` references obsolete. | — |

S0 closed: every code item opened by the lane decision (T1.3a) and
every doc item (T1.3e/f, T3.6, T2.6) is shipped. The remaining S2
follow-ups (T1.3b/c/d) are scoped for fast-loop bring-up.

---

## 1. TL;DR

The trainer scaffolding runs end-to-end on `gymv`, all five SFT
warm-starts are on disk
(`schema_gen` + `action_taking` + `skill_selection` + `segment` +
`contract` + `curator`; see §0.2), and the **lane decision** is now
recorded — **lane (a): skills are retrieval payloads / procedural
guidance for the actor LLM**, not runnable programs (see
[`skill-lane-decision.md`](skill-lane-decision.md), §0.4). The
remaining tight critical path is **`bank.runnable()` empty until the
offline promotion loop fires once**, default-disabling the Repairer
in the trainer Crafter (T1.3a), four config / policy artefacts, and
the curator-overfit mitigation. Most other "missing" items are
deliberately deferred per the cadence-asymmetry rule
([`PLAN-PIPELINE-ORCHESTRATOR.md` §5.0](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md):
fast loop = `gymv` only; transfer-target adapters land in the slow loop).

What we **can** already exercise:

- Cascaded grounding, gate stages 0–4
- Lifecycle / storage split, promotion + rollback transactions
- G0 invariant on every `SkillEpisode`
- `gymv` few-shot transfer
- The typed `BankMutationProposal` family
- The Day-10 trainer ↔ harness wire-up
- **All five SFT warm-starts** for the Phase-F GRPO targets (`runs/sft_coldstart/{decision,skillbank}/`) plus the Phase-1 `schema_gen` checkpoint (`runs/sft_schema_gen/schema_gen_20260430_091831/`, eval token-acc 0.9838)

The structural plumbing is real and the weight inputs to GRPO Phase F
are trained. What's missing is mostly **wire-up content** (one offline
promotion run, a lane decision, target-domain demos, a few config
YAMLs) — not new architecture or new model training.

---

## 2. Tier 1 — Hard blockers

Training is unsafe or a silent no-op without these.

### T1.1 `schema_gen` Phase-1 SFT checkpoint — **closed (pending exact-match probe)**

> ✅ **Status flipped 2026-05-01 (revision 2).** Phase-1 SFT trained
> 2026-04-30; checkpoint at
> [`runs/sft_schema_gen/schema_gen_20260430_091831/`](../runs/sft_schema_gen)
> with intermediate ckpts at `checkpoint-{200,400,477}`. Final eval
> token-accuracy **0.9838** / eval loss **0.0861** on hold-out
> (`mean_token_accuracy` reported by trainer). Adapter is 16.8 MB,
> r=16 / α=32 over the 7-projection target set on the
> Qwen3.5-35B-A3B base. Training run config in
> `train_config.json` confirms domains `gymv` + `env_wrappers` +
> `browser` + `image_qa` + `video_qa`.

**Outstanding sub-item (T1.1′, blocker for `RunRelease` pinning):**
the trainer reports `mean_token_accuracy`, but
[`PLAN-VISUAL-GROUNDING-MILESTONES.md` §13](../plans/01-visual-grounding/PLAN-VISUAL-GROUNDING-MILESTONES.md)
specifies **exact-match schema-validity rate** thresholds (Path A
≥ 70 % `gymv`, ≥ 50 % `browser`; field accuracy ≥ 85 % / ≥ 75 %).
Token-accuracy 0.9838 strongly suggests the bar is met but is not the
same metric. Build a 30-min probe at
`evaluation/probe_schema_gen_exact_match.py` that loads the LoRA over
the 35B-A3B base, runs it on a held-out slice of
`labeling/output/grounding/{gymv,env_wrappers,browser}/` triples, and
emits the exact-match rate per domain.

**Fix:** run the probe; if it passes, pin `schema_gen_ckpt_id =
"schema_gen_20260430_091831"` into `RunRelease`. If it fails, retrain
with the same scaffold against a stricter loss or more epochs (the
existing `train.py` is reusable as-is).

### T1.2 §17 keystone — `bank.runnable()` is empty until the offline loop fires once

```51:57:Multi-hop-Reasoning-VLM-Agent/skill_bank/repository.py
def runnable(self, *, include_shadow: bool = True) -> List[SkillRecord]:
    out: List[SkillRecord] = []
    for r in self.active.all():
        if r.status == SkillStatus.SHADOW and not include_shadow:
            continue
        out.append(r)
    return out
```

`runnable()` reads only the active store. Cold-start ingest writes
**everything** to `CANDIDATE`. Until `PromotionOrchestrator.promote`
graduates a batch `CANDIDATE → SHADOW`, the trainer's harness hook sees
`[]` and every `filter_candidates` call admits the legacy bank by
missing-fallback — i.e. the harness is silently bypassed. This is the
keystone called out in
[`harness-usability-and-intra-gymv-transfer.md` §6.2](harness-usability-and-intra-gymv-transfer.md).

**Fix:** run
[`labeling_supplement/decide_promotion_gpt54.py`](../labeling_supplement/decide_promotion_gpt54.py)
at least once on the cold-start corpus before launching co-evolution; or
have the trainer's first warm-up step gate on a non-empty `runnable()`.

### T1.3 Lane decision — **closed: lane (a), skills are retrieval payloads**

> ✅ **Status flipped 2026-05-01 (revision 2).** Decision recorded in
> [`skill-lane-decision.md`](skill-lane-decision.md). A skill is a
> *semantic retrieval payload* and *procedural guidance* for the
> actor LLM — name, description, preconditions / effects / role
> labels, optionally NL `protocol`. Skills are **not** runnable
> programs invoked by the harness at training time. The actor's
> live decision loop (`select_skill → update_intention →
> take_action`) and the Day-10 `SkillHarnessHook` (which calls only
> `filter_candidates` + `validate_choice`, never `run_skill`) already
> implement the lane-(a) wire; the decision formalises what was
> implicit.

**What this changes for training:**

* The Repairer's protocol-edit path is parked behind a feature flag
  (T1.3a, S0). Live runs default to lane-(a) Crafter behaviour.
* The lane-(b) machinery that already shipped (`labeling/_protocol_lift.py`
  v2.1, `harness/gymv_executor.py`, `harness/gymv_success.py`,
  `harness/replay_validator.py` action-walk mode, `FewShotAdapter`
  intra-source-domain transfer) **stays in tree** as offline gate /
  diagnostic infrastructure — used by `decide_promotion_gpt54.py`
  and `_phase4_transfer_cycle.py`, not by the live actor.
* Multi-domain `feasible_domains ≥ 2` invariant for `ACTIVE` is
  replaced by `min_retrievals_per_skill` (T1.3d, S2) — unblocks
  `ACTIVE` promotion on single-domain `gymv` banks.
* `FailureClass` taxonomy gains three retrieval-centric classes
  (`BANK_GAP`, `RETRIEVAL_MISLEAD`, `STALE_DESCRIPTION`) for live
  Crafter use (T1.3c, S2).
* Plan documents need a documentation pass — the four PLAN docs
  (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS,
  PLAN-COMPONENTS-IMPLEMENTATION) were written for lane (b) and need
  retrofitting (T1.3f, S0 doc-only).

**Rollback condition:** revisit only if both (i) the actor's
`skill_selection` LoRA + retrieval scoring saturates **and**
(ii) NORTHSTAR §7.3 Joint Success Rate is still below the headline
target after exhausting the
[`single-vs-two-mdp-tradeoff.md` §"Escalation order"](single-vs-two-mdp-tradeoff.md)
(better `strategic_description`, pattern-tag abstraction layer,
two-call inference inside one MDP). Even then the next escalation is
MCTS-with-a-forward-model on games and tool-augmented harness ops on
VR / video — **not lane (b) directly**.

### T1.4 Real env executors for transfer-target adapters

Only `gymv` has a real executor 1. `osworld`, `video` are pure
`StubTransferTargetAdapter`. `browser` and `visual_reasoning` have helper
bindings (`vlm_wrapper/browser_adapter.py`,
`visual_reasoning_wrapper/skill_executor.bind_executor`) but the trainer
never calls them.

For the fast loop this is fine —
[`PLAN-PIPELINE-ORCHESTRATOR.md` §5.0](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)
pins fast-loop = `gymv` only. For G3a transfer, every target except
`gymv` returns `target_domain_demo_unavailable` ⇒
`min_target_domains_verified ≥ 1` is unreachable ⇒ **no skill ever
leaves `CANDIDATE` even after promotion fires.**

**Fix order:** `gymv` only for Phase-A fast loop → revisit one
transfer-target executor + demo library before claiming any cross-domain
win.

---

## 3. Tier 2 — Critical-path infrastructure

Training fires but signal is degraded.

### T2.1 Few-shot demo libraries for the four transfer targets

[`harness/few_shot_demos_gymv.py`](../harness/few_shot_demos_gymv.py)
exists; analogues for `browser` / `osworld` / `video` / `visual_reasoning`
do not. Compounds with T1.4.

### T2.2 `orchestrator/eval_suite.py` — G5 non-regression eval driver

`orchestrator/gate_service.py::_run_non_regression` accepts
caller-supplied `(baseline, post)` or `EvalSuite`. The frozen-suite
loader specified in
[`PLAN-UNIFIED-SKILL-GATE.md` §4](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md)
is missing. Without it the §5 stop/go rules in
[`PLAN-SYSTEM-NORTHSTAR.md`](../plans/00-system/PLAN-SYSTEM-NORTHSTAR.md)
cannot fire on automated promotions.

### T2.3 Eval E0 driver + canonical scoreboard

[`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) lists
`evaluation/driver.py` + `evaluation/answer_evaluator.py` as not yet
delivered.
[`PLAN-SYSTEM-NORTHSTAR.md` §7.3](../plans/00-system/PLAN-SYSTEM-NORTHSTAR.md):
every release **MUST** emit `releases/{release_id}/scoreboard.md` with
the 10-column canonical table + 4 companion tables, all sharing one
`eval_suite_id` + `bank_snapshot_id`. Without this the **Joint Success
Rate** column has no producer and the GRPO loop has no headline.

### T2.4 `harness/reward_logger.py` not wired into the GRPO buffer

Per
[`PLAN-HARNESS.md` §17](../plans/05-harness/PLAN-HARNESS.md) the
harness `RewardLogger` is the canonical sink for
`r_env + r_follow + r_cost + r_transfer + r_adapter`. The actual reward
flowing into `trainer/coevolution/grpo_training.py::_collect_grpo_records`
comes from `decision_agents/reward_func.py::RewardComputer` — **no file
under `trainer/coevolution/` imports `harness.reward_logger`**. The
shaping is real, but the audit channel is forked.

### T2.5 Policy-as-config YAMLs

| Artefact | Required by | Status |
|---|---|---|
| `configs/skill_gate.yaml` | [`PLAN-UNIFIED-SKILL-GATE.md` §9](../plans/07-skill-gate/PLAN-UNIFIED-SKILL-GATE.md) | Missing — current thresholds live in `orchestrator/config.py::GateThresholds`; audit-log drift is invisible without the YAML diff |
| `configs/failure_routing.yaml` | [`PLAN-FAILURE-ROUTING.md`](../plans/08-cross-cutting/PLAN-FAILURE-ROUTING.md) | Missing entirely |

### T2.6 `IMPLEMENTATION-STATUS.md` is stale

Last updated 2026-04-21. Predates the `_crafter_hook` / `_promotion_hook`
/ `_harness_hook` trio (Day-7 → 10),
[`legacy_writeback.py`](../skill_bank/legacy_writeback.py),
`cold_start/evaluation_dataset/`, the GRPO / vLLM lifecycle stack, **and
the entire `runs/sft_*` SFT corpus** — all of which now exist.
**Refresh it before a sprint planning call** so people don't
re-implement what's already there.

### T2.7 `curator` SFT overfit mitigation (added 2026-05-01)

The `curator` adapter at
[`runs/sft_coldstart/skillbank/curator/`](../runs/sft_coldstart/skillbank/curator)
trained 15 epochs on **193 rows** with eval loss **1.218** vs. train
loss **0.491**. The corpus is too small for the LoRA capacity. This
LoRA participates in the medium-timescale Bank GRPO loop alongside
`segment` and `contract`; if its outputs are unreliable on held-out
bank states the medium loop will fight the actor. **Mitigations
(non-blocking, in priority order):**

1. Treat `curator` as a tie-breaker only in early GRPO (low weight in
   the per-step Bank reward composition) — gated via a
   `curator_weight` knob in `trainer/coevolution/config.py`.
2. Regenerate the curator corpus by running
   [`labeling_supplement/decide_skill_crafting_gpt54.py`](../labeling_supplement/decide_skill_crafting_gpt54.py)
   over more games / more episodes, then re-train.
3. Add early-stopping at `eval_loss` plateau detection (currently the
   trainer rides the full 15-epoch budget).

### T2.8 vLLM topology check for split-base inference (added 2026-05-01)

`schema_gen` LoRA targets `Qwen3.5-35B-A3B`; the GRPO LoRAs target
`Qwen3.5-9B`. They cannot share a vLLM worker. Confirm the deployment
plan: (i) one vLLM worker per base, with the actor calling each in
sequence, **or** (ii) `schema_gen` runs offline only on cold-start
ingest and never live. The trainer config currently shows only the 9B
backbone (`model_name = "Qwen/Qwen3.5-9B"` in
[`trainer/coevolution/config.py`](../trainer/coevolution/config.py)), so
option (ii) is the implicit default; document this explicitly so the
visual-grounding plan and the trainer plan agree.

### T2.9 Run-wide SFT manifest (added 2026-05-01)

[`runs/sft_coldstart/sft_summary.json`](../runs/sft_coldstart) is a
**per-GPU** summary (it lists only `action_taking`, the GPU-1 shard).
The offline promotion driver and any future eval driver need a
**run-wide** manifest pointing at all six adapter paths + their meta.
Ship as `runs/sft_coldstart/sft_summary_all.json` (and an analogous
`runs/sft_schema_gen/manifest.json`). Schema:

```json
{
  "schema_gen": {"path": "...", "base": "...", "eval_loss": ...},
  "decision":   {"action_taking": {...}, "skill_selection": {...}},
  "skillbank":  {"segment": {...}, "contract": {...}, "curator": {...}}
}
```

### T2.10 Audit `*.partial_20260430_091054` shards (added 2026-05-01)

Several `runs/sft_coldstart/*.log.partial_20260430_091054` files exist
alongside the final logs, suggesting an earlier 5-GPU shard crashed or
was preempted. Confirm the **final** checkpoints under
`decision/{action_taking,skill_selection}` and
`skillbank/{segment,contract,curator}` are the post-restart artefacts
(timestamps Apr 30 21:39 / 21:22 / 13:37 / 10:37 / 08:53 respectively),
not stale partials. Cheap probe: `peft.PeftModel.from_pretrained(...)`
+ a single forward pass per adapter.

---

## 4. Tier 3 — Architectural gaps

Won't break the fast loop, but block multi-domain claims.

### T3.1 The five typed extension records ([`PLAN-EXPERIENCE-EXTENSION.md` §3](../plans/08-cross-cutting/PLAN-EXPERIENCE-EXTENSION.md))

| Spec'd | In repo? | Closest analog |
|---|---|---|
| `RunRecord` | ❌ | `RunRelease` covers ~60 %, lacks budget summary + per-run promotion / rollback decision |
| `GroundingRecord` | ❌ as a Python dataclass | `<evidence_refs>` schema section + `parse_evidence_refs` exist; typed dataclass missing |
| `SkillInvocationRecord` | ❌ | `SkillEpisode` covers most fields but plan calls for separate per-attempt record |
| `FailureRoutingRecord` | ❌ | `FailureTrace` lacks `severity`, `recoverability`, `route_to`, `parent_failure_id` — the routing-critical fields |
| `AnswerSupportRecord` | ❌ | reconstructed from `Episode.answer_support_chain` ad-hoc |

**Pick one canon:** either ship the five records or update the plans to
bless the existing alternates.

### T3.2 Failure router — entirely missing

[`PLAN-FAILURE-ROUTING.md`](../plans/08-cross-cutting/PLAN-FAILURE-ROUTING.md)
R0–R5 deliverables — none ship:

- `orchestrator/failure_router.py` (R1)
- `orchestrator/failure_router_runner.py` (R2)
- Detector shims on Harness / Grounding / Judge / Budget (R3)
- `failures/<class>/<id>` queue tree
- Two-step write contract (`failures/incoming/` → `failures/<class>/`)
- Target-rate dashboards (R5)

Today every L2 / L4 failure is either consumed ad-hoc by the Crafter or
dropped, contradicting §10 anti-goal #4.

### T3.3 `HopTrace` artefact + inner-hop logging on the Action Agent

`grep -n HopTrace decision_agents/` returns zero. Per
[`PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md` §3.2](../plans/10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md),
Phase B mining (typed `SkillIR`, inner-hop reasoning skill discovery) has
no input substrate until this lands. `SkillEpisodeStep` carries
inner-step info but isn't logged as the typed `HopTrace` artefact
required.

**Note:** [`single-vs-two-mdp-tradeoff.md`](single-vs-two-mdp-tradeoff.md)
deliberately retired the inner-MDP scaffold for latency reasons. The
two notes need to be reconciled — either Phase B of the
transferable-reasoning edits is deferred, or `HopTrace` is reintroduced
as an **offline artefact** (typed log) without restoring the online
inner LLM call.

### T3.4 Canonical cross-domain ontology types

`Agent`, `UIElement`, `EvidenceSpan`, `FrameSpan`, `RegionPatch`,
`IdentityHypothesis`, … — zero hits. Without them,
`SkillRecord.protocol` stays `List[Dict[str, Any]]`, and HARNESS
§10b/c/d typed validators (`slot_binding_validator`,
`ontology_remap_validator`, `adapter_compatibility_checker`) cannot do
typed-level checks. **Same blocker as T1.3 lane (b).**

### T3.5 Online runtime adapters (audit §16.2 – §16.4)

| Spec | In repo? |
|---|---|
| `HarnessSkillProvider` | ❌ — actor still consumes `SkillBankProvider`. Trainer routes harness via the orthogonal `SkillHarnessHook` instead. |
| `RunnerActorAdapter` | ❌ — `EpisodeRunner.run` expects `ActorLike.choose_action(state, eligible)`; current `ActorAgent.run` returns text actions. |
| `EnvLike` shim in `vlm_wrapper/` | ❌ — protocol exists at `orchestrator/runner.py:49`; no production-env bridge. |
| `skill_bank/legacy_bridge.py` | ❌ — only one-way [`legacy_writeback.py`](../skill_bank/legacy_writeback.py) ships. |

These don't block the fast loop (the Day-10 hook bypasses them) but
block any future "live online runtime" execution path.

### T3.6 `hop_select` LoRA absence — **decided: single-MDP, plans need updating**

> ✅ **Status corrected 2026-05-01 (revision 2).** This is **not** an
> open question — the lane has been picked, the code has shipped, and
> the SFT corpus on disk reflects the decision. The only remaining
> work is documentation drift on the plan side.

**What the live actor actually does (one MDP):**

* [`decision_agents/actor_agent.py`](../decision_agents/actor_agent.py) `_run_inner_mdp` was **deleted** (~430 LOC); `_pick_action` chooses directly from `harness.valid_actions(state)` ([`actor_agent.py:24`](../decision_agents/actor_agent.py)).
* [`decision_agents/inner_mdp.py`](../decision_agents/inner_mdp.py) is **deleted entirely** — `HopAction` / `HopPolicy` / `HopStep` / `HopTrace` / `HeuristicHopPolicy` / `parse_hop_action` raise `AttributeError` with a `DeprecationWarning` via the package shim.
* The legacy `hop_policy=` / `max_hops_per_step=` kwargs on `ActorAgent.__init__` (and `run_actor_episode`) only exist as deprecation-warning shims for back-compat ([`actor_agent.py:286-298`](../decision_agents/actor_agent.py)); they are ignored at runtime.
* What were inner-MDP operators are now first-class harness actions: `GROUND` → `VRHarness.step(LOOK)`, `RETRIEVE` → `VRHarness.step(RETRIEVE)`, `CONCLUDE` → `VRHarness.step(NOTE)`, `EXECUTE` → folded into the env-action loop. See [`decision_agents/README.md` §"Migration of inner-MDP operators"](../decision_agents/README.md).
* Five `Harness` implementations (`GymHarness`, `BrowserHarness`, `OSWorldHarness`, `VRHarness`, `VideoHarness`) own the per-task action vocabulary; `info["valid_actions"]` is sourced from `harness.valid_actions(state)`.
* GRPO LoRAs are exactly **two**: `skill_selection` + `action_taking`. The `hop_select` LoRA was explicitly dropped because hop decisions are now action decisions, both rolled into `action_taking`. **The on-disk SFT seed corpus at [`runs/sft_coldstart/decision/`](../runs/sft_coldstart/decision) confirms this — only `action_taking/` and `skill_selection/` are trained, no `hop_select/`.**

**Why the single-MDP decision stands**
([`single-vs-two-mdp-tradeoff.md` §"Escalation order"](single-vs-two-mdp-tradeoff.md)):

1. Latency win — the actor has a non-trivial vLLM TTFT and an inner
   MDP doubles the call count without obvious return.
2. Games specifically — the inner-hop alphabet (`GROUND` / `CHECK` /
   `RETRIEVE`) doesn't fit retro action games; the cold-start
   collectors and SFT corpus would not transfer.
3. Cross-task transfer — one `action_taking` LoRA trained on games +
   web + VR + video (all five harnesses) is the cheapest and most
   transfer-friendly design. Forking into a `hop_select` LoRA breaks
   `harness.action_kind` cost bucketing and the
   "GPT-5.4 → Qwen3.5-9B SFT cold-start works for both LoRAs from the
   same trace" property.

**What's still left to do — plan-side documentation hygiene only**
(the code is settled). All references in `plans/` that imply a
three-LoRA actor (`schema_gen` / `skill_select` / `hop_select` /
`action_execute`) need to be rewritten to reflect the shipped
two-LoRA actor (`skill_selection` / `action_taking`) plus the
SFT-only `schema_gen` adapter on the 35B-A3B base. The list:

| Plan file | Section to update |
|---|---|
| [`plans/02-action-agent/PLAN-ACTION-AGENT.md`](../plans/02-action-agent/PLAN-ACTION-AGENT.md) | §5.3 hop alphabet should reference `Harness.valid_actions` instead of an inner-MDP `hop_select`; §6 phase plan drops `hop_select` from the GRPO target list |
| [`plans/03-skill-bank/PLAN-SKILL-BANK.md`](../plans/03-skill-bank/PLAN-SKILL-BANK.md) | §1.5 hop vocabulary should reconcile with the harness action sets |
| [`plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md`](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md) | Phase F GRPO target list — remove `hop_select` |
| [`IMPLEMENTATION-STATUS.md`](../IMPLEMENTATION-STATUS.md) | Mark the single-MDP pivot as shipped (Patch #13 in `decision_agents/README.md`) |

**Effort:** ~30 min find-and-replace across 4 plan files. **No code
changes required.** This is the same documentation-drift pattern as
T3.6's S0 entry in §0.1; the cleanup belongs in the same governance
slot as T2.6 (refresh `IMPLEMENTATION-STATUS.md`).

---

## 5. Tier 4 — Open questions

Settle before scaling, not before starting.

1. **Hop-vocabulary inconsistency** between
   [`PLAN-SKILL-BANK.md` §1.5](../plans/03-skill-bank/PLAN-SKILL-BANK.md)
   (`{GROUND, CHECK, RETRIEVE, COMMIT, ACT/EXECUTE, VERIFY}` + `CONCLUDE`)
   and
   [`PLAN-ACTION-AGENT.md` §5.3](../plans/02-action-agent/PLAN-ACTION-AGENT.md)
   (`{GROUND, CHECK, RETRIEVE, COMMIT, EXECUTE}`). Matters for G0
   role-vs-hop check.
2. **Q4 ablation unmeasured** — [`PLAN-HARNESS.md` §20.2](../plans/05-harness/PLAN-HARNESS.md):
   until A0 (no Harness) and A4 (full system) are run on the intra-`gymv`
   probe, the architectural claim that the **Actor (not the 72B) is the
   policy** is unverified. The `skill_transfer_test/` framework is
   sketched in
   [`harness-usability-and-intra-gymv-transfer.md` §5](harness-usability-and-intra-gymv-transfer.md).
3. **Phase-2 teacher-SFT trigger is qualitative-only** ("if narrow
   failures persist"); no quantitative threshold.
4. **Counterfactual prediction reliability** — Crafter §6.9
   auto-synthesis at 3+ recurrences, no verifier-quality test.
5. **Numerical thresholds unpinned:** shadow-cycle K,
   `few_shot.k_shot_default` / `max`,
   `transfer_min_target_domains_verified`, slow-loop N episodes,
   NORTHSTAR §5.6 noise bands (blank for first 3 releases).
6. **Stage 5 two-step (`PROVISIONAL → ACTIVE`)** — `PromotionOrchestrator.promote`
   does not visibly enforce the `provisional_active_window_episodes: 200`
   window or `shadow_origin_penalty: 0.30`.
7. **Crafter input contract** (`active_bank_skills`,
   `candidate_protocol_clusters`, `failed_hop_patterns`,
   `failure_cases`, `transfer_mismatch_reports`) per
   [`PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md` §4.7](../plans/10-edits/PLAN-EDITS-TRANSFERABLE-REASONING-SKILLS.md)
   — not present as a typed `CrafterInput` dataclass.

---

## 6. Recommended sequencing (concrete sprint plan)

> **Revised 2026-05-01 (revision 2):** S0 pre-flight is lighter — the
> `schema_gen` Phase-1 SFT is already trained, so the only remaining
> S0-data item is the exact-match probe (T1.1′).

| Sprint | Track | Items | Why now |
|---|---|---|---|
| **S0 — pre-flight** ✅ shipped 2026-05-01 | data | ~~**T1.1′**~~ — probe at [`evaluation/probe_schema_gen_exact_match.py`](../evaluation/probe_schema_gen_exact_match.py); **outstanding:** one GPU-bound run to pin the metric, but the script is complete | converts "0.9838 token-acc" into the spec's exact-match metric; gate for `RunRelease` pinning |
| | code | ~~**T1.3a**~~ — `enable_protocol_patching: bool = False` flag on `SkillCrafterService`, threaded through `_crafter_hook` / `CoEvolutionConfig` / `run_coevolution.py`; tests in [`tests/test_crafter_lane_a_flag.py`](../tests/test_crafter_lane_a_flag.py) | lane-(a) default — Repairer parked behind feature flag |
| | governance | ~~T1.3~~ — **closed**, see [`skill-lane-decision.md`](skill-lane-decision.md). All six follow-ups (T1.3a/e/f shipped here; T1.3b/c/d scoped for S2). | — |
| | governance | ~~T2.6~~ — `IMPLEMENTATION-STATUS.md` refreshed to reference `runs/sft_*`, Day-7 → 10 hooks, lane-(a) decision, threshold YAMLs, SFT manifest, offline-promotion driver | — |
| | governance | ~~**T1.3e + T1.3f + T3.6**~~ — `harness/README.md` §22 lane block + `crafter-harness-orchestrator-roles.md` §7.3 / §7.5 banner+supersedure + four PLAN docs banner (PLAN-SKILL-CRAFTER, PLAN-SKILL-BANK, PLAN-HARNESS, PLAN-COMPONENTS-IMPLEMENTATION); T3.6 obsoletes `hop_select` references in those banners | doc drift closed |
| | governance | ~~**T2.9**~~ — [`scripts/build_sft_manifest.py`](../scripts/build_sft_manifest.py) emits [`runs/sft_coldstart/sft_summary_all.json`](../runs/sft_coldstart/sft_summary_all.json) | — |
| | governance | ~~**T2.10**~~ — load-smoke driver at [`evaluation/smoke_load_sft_adapters.py`](../evaluation/smoke_load_sft_adapters.py); **outstanding:** one GPU run to confirm none of the six adapters has a torn `*.partial_*` shard | cheap, prevents Phase-F starting from a corrupt seed |
| **S1 — fire offline once** ✅ wrapper shipped 2026-05-01 | data | ~~**T1.2**~~ — orchestration wrapper at [`scripts/run_offline_promotion_cycle.sh`](../scripts/run_offline_promotion_cycle.sh) (drives `decide_promotion_gpt54.py` + inline `legacy_writeback.writeback_promotion` + `bank.runnable()` post-condition); **outstanding:** one GPU/API run to flip ≥1 record to `ACTIVE/SHADOW` | converts `bank.runnable() == []` into non-empty; this is the §17 keystone |
| | governance | ~~T2.5~~ — [`configs/skill_gate.yaml`](../configs/skill_gate.yaml) + [`configs/failure_routing.yaml`](../configs/failure_routing.yaml) shipped (policy v1.0.0; lane-(a) failure taxonomy + drift annotations) | — |
| **S2 — fast loop launch** | code | **T1.3b** — `RewriteProposal` (and rename `ComposeProposal` → `MergeProposal` if cleaner) in `data_structure/extensions/bank_mutation.py` | typed proposal subclasses for the lane-(a) Crafter taxonomy |
| | code | **T1.3c** — implement `BANK_GAP` / `RETRIEVAL_MISLEAD` / `STALE_DESCRIPTION` `FailureClass` taxonomy in the *live* Crafter path | replaces protocol-edit `RecoveryStrategy` for live use |
| | code | **T1.3d** — replace multi-domain `ACTIVE` invariant with `min_retrievals_per_skill` in `PromotionOrchestrator` | unblocks `ACTIVE` on single-domain `gymv` banks |
| | training | **T2.7** — gate `curator` LoRA at low weight in early GRPO (config knob in `trainer/coevolution/config.py`) | mitigates the eval-loss 1.22 overfit until the corpus is regenerated |
| | training | T2.4 (wire `harness/reward_logger.py` into `_collect_grpo_records`) | restores [`PLAN-HARNESS.md` §17](../plans/05-harness/PLAN-HARNESS.md) single-sink invariant |
| | training | Launch fast-loop GRPO on `gymv` only ([`PLAN-PIPELINE-ORCHESTRATOR.md` §5.0](../plans/06-orchestrator/PLAN-PIPELINE-ORCHESTRATOR.md)) — actor + bank LoRAs at 5–10 : 1 cadence, all 5 LoRAs warm-started from `runs/sft_coldstart/` | Phase 1 of [`PLAN-ACTION-AGENT.md` §6](../plans/02-action-agent/PLAN-ACTION-AGENT.md) |
| | governance | T2.2 (`orchestrator/eval_suite.py` loader) | unblocks G5 |
| | governance | T2.3 (Eval E0 driver + canonical scoreboard) | NORTHSTAR §7.3 release plumbing |
| | governance | **T2.8** — document the split-base vLLM topology (35B-A3B for `schema_gen`, 9B for actor + bank LoRAs) | resolves the implicit deployment ambiguity |
| **S3 — first transfer probe** | data | T1.4 + T2.1 — pick one transfer target (probably `browser` per the plan's first-arena), build demo library, plug `set_executor` | unblocks G3a → first `ACTIVE` skill possible |
| | governance | A0 vs A4 ablation on intra-`gymv` probe (Q4 + [`harness-usability-and-intra-gymv-transfer.md`](harness-usability-and-intra-gymv-transfer.md)) | answers Q4: is the Actor doing real work? |
| **S4 — multi-domain hardening** | architecture | T3.1 (extension records) + T3.2 (failure router) + T3.3 (`HopTrace`) + T3.4 (ontology types) | required for cross-domain claims, not for `gymv` |

---

## 7. What to **not** block on before training

- **`hop_select` LoRA (T3.6)** — design has been deliberately rejected
  and **the single-MDP code has shipped**
  ([`single-vs-two-mdp-tradeoff.md`](single-vs-two-mdp-tradeoff.md);
  [`decision_agents/actor_agent.py`](../decision_agents/actor_agent.py)
  has no `_run_inner_mdp`; `inner_mdp.py` is deleted; SFT seeds in
  [`runs/sft_coldstart/decision/`](../runs/sft_coldstart/decision)
  are exactly 2 LoRAs). The remaining work is plan-doc cleanup, not
  re-litigation. **Do not re-add `hop_select` to the trainer.**
- **Online runtime adapters (T3.5)** — the Day-10 `SkillHarnessHook`
  covers the trainer surface; the runtime adapters are independent work
  for live deployment.
- **Real executors for `osworld` / `video` / `visual_reasoning`** —
  fast loop is `gymv`-only; these are 4-week projects each per
  [`harness-usability-and-intra-gymv-transfer.md` §2](harness-usability-and-intra-gymv-transfer.md).
- **Phase B of transferable-reasoning edits** — defer until `HopTrace`
  design is reconciled with the single-MDP decision.

---

## 8. Verdict

**Earliest safe training start = end of Sprint S2** (unchanged in
absolute terms, but S0 is materially shorter now). The fast loop
(`gymv` actor + bank GRPO) needs:

- ~~T1.1 (schema SFT to checkpoint)~~ → **closed; on disk at `runs/sft_schema_gen/schema_gen_20260430_091831/`** (T1.1′ exact-match probe is the only S0 verification step)
- T1.2 (offline-loop fired once → non-empty `runnable()`)
- ~~T1.3 (lane picked)~~ → **closed: lane (a)**, see [`skill-lane-decision.md`](skill-lane-decision.md). Only T1.3a (default-disable Repairer flag, S0) is left for fast-loop launch; T1.3b / T1.3c / T1.3d land in S2.
- T2.4 (reward logger wired)
- T2.5 (config YAMLs)
- T2.7 (curator weight gate in early GRPO — mitigation for the 1.22 eval loss)

Everything else can land asynchronously without poisoning the actor's
gradient signal.

The structural plumbing is real, **the SFT warm-starts for all six
GRPO/SFT targets are trained**, and **the lane question is now
decided**. What's missing is mostly **wire-up content** (one offline
promotion run, the Repairer feature flag, a few config YAMLs,
target-domain demos, one weighting knob). Day-7 → 10 work closed the
trainer ↔ harness wire; revision-2 confirmed the weight inputs;
revision-3 closed the lane question; the next sprint feeds runtime
data into them.

---

## 9. Subagent audits — converged findings

The four parallel audits all landed on overlapping conclusions; they are
recorded here so the synthesis above is reproducible without re-running
them.

### 9.1 Plan-discipline audit (Action Agent + Skill Bank + Skill Crafter + Harness)

The plans converge on a strict frozen-first / phase-A-then-phase-B
discipline:

- **(A) Phase order:** GRPO targets the **9B Actor first**. *Plan
  side* lists `schema_gen` / `skill_select` / `hop_select` /
  `action_execute`; *live side* (after the single-MDP pivot — see
  T3.6) is `schema_gen` (SFT-only on the 35B-A3B control plane) +
  `skill_selection` + `action_taking`. The Bank's medium-timescale
  `CONTRACT` / `CURATOR` / `SEGMENT` LoRAs second (5–10 actor
  cycles : 1 bank cycle); only narrow SFT on the frozen 32B / 72B
  Crafter as a last resort.
- **(B) Cold-start prerequisites:** Tier-0 / Tier-1 seed trajectories,
  the five-stage bank pipeline, the unified gate plumbing
  (`SkillLifecycleManager` + four-store split + `skill_gate.yaml`),
  schema-as-input wiring, and Crafter Phase 0 multi-pass procedures
  with the typed-proposal schema.
- **(C) Hard prerequisites:** timescale separation; the G0–G5
  acceptance-gate stack on every episode (G0 evidence-driven invariant
  runs continuously); the frozen game-source-domain non-regression
  slice; the general-protocol invariant (no per-domain sub-banks);
  Layer A guardrail before Layer B GRPO; and Harness Phase 0+1
  (`SkillEpisode` + `SkillHarness` + `AdapterRegistry`) before any new
  trainable model is added.
- **(D) Openly flagged unresolved items:** hop-vocabulary inconsistency
  between Bank (`CONCLUDE`) and Action Agent (`COMMIT` + `VERIFY`); the
  documented 7B / 8B drift risk on free-form `hop_select`; "new skill
  vs. new adapter" ownership; counterfactual prediction reliability;
  the unverified "Actor is the real policy" claim until Harness
  ablations A0–A4 run; and "verifier quality" being the silent
  load-bearing assumption.

### 9.2 Phase-and-artefact audit

- **Phase ordering A → F is strict;** GRPO surface lives in Phase F.
- **Ten architectural invariants** in
  [`PLAN-COMPONENTS-IMPLEMENTATION.md` §7](../plans/09-implementation/PLAN-COMPONENTS-IMPLEMENTATION.md);
  four-way Actor / Harness / Bank / Orchestrator separation.
- **Source-domain (`gymv`) vs. transfer-target asymmetry** is binding;
  NORTHSTAR §5 stop / go rules are binding.
- **Per-phase acceptance gates:** Phases A.5 / B.7 / C.7 in
  PLAN-COMPONENTS; Phase D shadow-first / six-gate; Phase E scoreboard
  reproducibility; eval phases E0 / E1 / E2; NORTHSTAR phase-emphasis
  exits.
- **Required artefacts:** full module file map under
  `src/{harness, crafter, orchestrator, skill_bank, common}`; eleven
  record types incl. `SkillEpisode` / `GateVerdict` / `AuditRecord`;
  `artifacts/*` storage layout; canonical scoreboard with four
  companion tables incl. few-shot transfer; `AuditRecord` six-field
  contract; three judge surfaces; `BudgetController`.
- **Open questions:** unpinned thresholds for K-cycle shadow stability
  and few-shot K-shot; deferred Hypothesizer Best-of-N and
  `PatchProposal` / `RetireProposal`; stub adapters for `browser` /
  `osworld` / `video` / `visual_reasoning`; episode-local-only state
  surface; judge-revision ownership when agreement < 0.85.

**Bottom line:** GRPO co-evolution can only start safely once Phases A–E
are green, the §0a.6 control-path interface is the sole call pattern,
evidence-role G0 is enforced on every `SkillEpisode`,
`PromotionOrchestrator` transactions are atomic and audited, the
canonical scoreboard CI emission contract is operational, and the
cadence asymmetry (fast / medium = `gymv`, slow = few-shot transfer) is
wired.

### 9.3 Plan-vs-repo cross-check (8 plan documents)

- **Visual-grounding plumbing** (cascade, semantic validator,
  `evidence_refs`, labeling scripts) is **shipped**. ~~the Phase-1
  `schema_gen` SFT checkpoint is only scaffolded, not trained~~ →
  **revision 2:** Phase-1 SFT trained 2026-04-30 (`runs/sft_schema_gen/schema_gen_20260430_091831/`,
  eval token-acc 0.9838); the remaining S0 verification step is the
  exact-match probe (T1.1′), not the training run.
- **G0–G5 gate stack is structurally complete**
  (`SkillLifecycleManager`, `GateService` / `GateRunner`,
  `FewShotAdapter`, `PromotionOrchestrator`, source / target asymmetry
  in `common/enums.py`); missing pieces are `configs/skill_gate.yaml`,
  `orchestrator/eval_suite.py`, and few-shot demo libraries for the
  four transfer targets.
- **Failure routing is the largest hole:** none of the five
  PLAN-EXPERIENCE-EXTENSION records (`RunRecord`, `GroundingRecord`,
  `SkillInvocationRecord`, `FailureRoutingRecord`, `AnswerSupportRecord`)
  exist as dataclasses, and `orchestrator/failure_router.py` +
  `configs/failure_routing.yaml` + the `failures/<class>/<id>` queue
  tree are absent.
- **Edits status:** Harness control-plane edits and the lightweight
  visual-grounding edits have largely landed; the
  transferable-reasoning-skills edits are the most incomplete — typed
  `SkillIR`, canonical cross-domain ontology types
  (`Agent` / `UIElement` / `EvidenceSpan` / etc.), `HopTrace` logging,
  and the inner-hop discovery pipeline are all still on paper, which
  directly blocks cross-domain transfer.

### 9.4 Trainer co-evolution scaffold audit

The trainer co-evolution scaffold is real and substantive: orchestrator,
episode runner, GRPO with 5 LoRAs (`skill_selection`, `action_taking`,
`segment`, `contract`, `curator`), vLLM lifecycle / hot-reload, per-game
skill bank pipeline, Crafter + Promotion hooks, harness hook. **All
five GRPO LoRA targets have SFT warm-starts on disk** as of 2026-04-30
(`runs/sft_coldstart/decision/{action_taking,skill_selection}/`,
`runs/sft_coldstart/skillbank/{segment,contract,curator}/`). Blocking
gaps before §6 launch:

- `hop_select` LoRA does not exist — **single-MDP design has shipped**
  (T3.6). `inner_mdp.py` is deleted; only `skill_selection` and
  `action_taking` are in the GRPO / SFT corpus. Remaining work is
  plan-doc cleanup, not re-implementation.
- `schema_gen` is SFT-only (intentional per
  [`PLAN-EDITS-VISUAL-GROUNDING-LIGHTWEIGHT.md`](../plans/10-edits/PLAN-EDITS-VISUAL-GROUNDING-LIGHTWEIGHT.md))
  and not in the GRPO `ADAPTER_MAP`. **Revision 2:** the SFT
  checkpoint is now on disk
  (`runs/sft_schema_gen/schema_gen_20260430_091831/`, base
  Qwen3.5-35B-A3B); needs the exact-match probe + a vLLM topology
  decision (T1.1′ + T2.8).
- `HarnessSkillProvider`, `RunnerActorAdapter`, and the `vlm_wrapper`
  `EnvLike` shim are unimplemented.
- `skill_bank/legacy_bridge.py` is missing (one-way
  [`legacy_writeback.py`](../skill_bank/legacy_writeback.py) ships).
- `harness/reward_logger.py` is **not** wired into GRPO (shaping flows
  from `decision_agents.reward_func` instead).
- Only `gymv` has a real adapter executor (`browser` / `osworld` /
  `video` / `visual_reasoning` ship as stubs with `set_executor` hooks).
- `GateRunner` exposes 5 stages, not the plan's G0–G5.
- `IMPLEMENTATION-STATUS.md` is stale relative to the delivered hook
  trio + writeback **and the entire `runs/sft_*` SFT corpus**.
- **Revision 2 (added 2026-05-01):** the `curator` SFT eval loss
  (1.218) flags an overfit risk that needs gating in early GRPO (T2.7).

---

## 10. Headline

Training is **not** ready to start safely in the strict sense the plans
require, but the gating items are smaller again than the first audit
pass implied. Revision 2 (post-`runs/` discovery) dropped the
`schema_gen` SFT + four cold-start LoRA seeds. Revision 3 (this
update) closes the lane question — **lane (a): skills are retrieval
payloads / procedural guidance for the actor LLM, not runnable
programs** ([`skill-lane-decision.md`](skill-lane-decision.md)). The
remaining S0–S2 critical path is:

1. Verify `schema_gen` exact-match against PLAN-VISUAL-GROUNDING-MILESTONES §13 (T1.1′)
2. ~~Pick a skill lane~~ → done. **Default-disable the Repairer in the live trainer Crafter** (T1.3a)
3. Fire the offline promotion loop once so `bank.runnable()` is non-empty (T1.2)
4. Wire `harness/reward_logger.py` into the GRPO buffer (T2.4)
5. Ship `configs/skill_gate.yaml` + `configs/failure_routing.yaml` (T2.5)
6. Gate `curator` LoRA at low weight in early GRPO (T2.7)

That's still 6 items, all content/wiring, none requiring new training
or new architecture. The Day-7 → 10 work closed the structural wire;
revision-2 confirmed the weights are trained; revision-3 confirmed
the actor consumes skills as **retrieval context**, not as runnable
programs; what's left is plumbing the warm-started LoRAs into a
training step that won't silently bypass the harness, rely on an
over-fit curator, or run a Repairer that has no protocol to repair.
