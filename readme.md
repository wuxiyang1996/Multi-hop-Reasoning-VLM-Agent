# Learning General Visual Reasoning and Agentic Skills via Game-Playing

**TL;DR:** Games teach reusable reasoning-form skills that can be transferred to web agents and visual reasoning.

Our previous codebase is here: https://github.com/wuxiyang1996/Game-AI-Agent. This codebase is modified based on this repo.

We learn reasoning skills from games, then transfer them to web agents and visual reasoning by representing each skill as a reusable multi-hop chain over structured observations.

## Benchmarks

Evaluation uses a **shared mixture pipeline** with **visual grounding**: one observation–action style interface (including pixel or multimodal observations where the benchmark supports them) so the same agent stack can be run on games, browser tasks, and visual reasoning suites.

### Games (video / interactive vision)

- **[Gym-V](https://github.com/ModalMinds/gym-v)** — Unified Gymnasium-style **vision environment** system (**179** procedurally generated visual environments across **10** domains) for training and evaluating agentic vision models. [Paper (arXiv)](https://arxiv.org/abs/2603.15432).
- **[LMGame-Bench](https://github.com/lmgame-org/GamingAgent)** — Benchmark for **LLMs/VLMs playing games** (platformer, puzzle, narrative titles via a unified API). [Paper (arXiv)](https://arxiv.org/abs/2505.15146).

### Web agents

- **[BrowserGym](https://github.com/ServiceNow/BrowserGym)** — Unified **web automation gym** (Playwright-based) that wraps multiple web-agent benchmarks (e.g. MiniWoB++, WebArena, VisualWebArena, WorkArena) under one API; described in *Transactions on Machine Learning Research* (2025). Official leaderboard: [Hugging Face — browsergym-leaderboard](https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard).

### Visual reasoning

- **[HallusionBench](https://github.com/tianyi-lab/HallusionBench)** — Diagnostic suite for **language hallucination** and **visual illusion** in LVLM image–context QA (often what people mean informally by a “hallucination” VLM benchmark). [Paper (arXiv)](https://arxiv.org/abs/2310.14566).
- **[M³-Bench](https://github.com/EtaYang10th/Open-M3-Bench)** (*Multi-Modal, Multi-Hop, Multi-Threaded Tool-Using MLLM Agent Benchmark*) — Evaluates **multimodal tool use** (e.g. MCP-style workflows) with visual grounding and multi-hop dependencies. [Paper (arXiv)](https://arxiv.org/abs/2511.17729).
- **HopChain** — Framework for **multi-hop vision–language reasoning** (synthetic chains that require repeated visual re-grounding). Link: [arXiv:2603.17024](https://arxiv.org/abs/2603.17024) (no public GitHub repo surfaced in search as of Mar 2026).


## Grounding and structured summaries

Observations are grounded through a **three-layer** pipeline that ends in the same **text interface** as Game-AI-Agent (`summary_state`-style key–value summaries).

**Layer A — Native structure first, pixels second**

- **BrowserGym:** **AXTree / DOM / metadata** for semantic identity; **screenshot + vision** only for gaps (canvas, charts, icons, image-only buttons, salience, layout not obvious from DOM). Hybrid grounding matches BrowserGym’s structured observation + image setup and avoids screenshot-only latency and weak identities.
- **Gym-V:** **Pixels primary**; use **wrappers** for captions, rules, or short state summaries (scaffolding strongly affects learning in Gym-V). Good **prototype** environment for the visual-summary schema before noisier web tasks.

**Layer B — Canonical visual summary**

Each step is compressed into a structured summary (same spirit as game summaries). Every entity carries a **`source`** tag (`ax`, `dom`, `vision`, `ocr`, `memory`, …); **uncertainty** is allowed per field so a supervisor can separate **grounding errors** from **reasoning errors**.

```text
domain=browser|gymv|game
task=...
goal=...
scene_type=webpage|grid|diagram|video_frame|puzzle

entities=
  e1[type=button, text="Search", source=ax]
  e2[type=input, label="From", source=ax]
  e3[type=chart_bar, source=vision]
  e4[type=icon, role=warning, source=vision]

relations=
  left_of(e1,e2)
  under(e3,e1)
  linked(e2,form_submit)
  same_group(e1,e4)

salient_targets=
  target_candidate=e2
  blocker_candidate=e4

state_flags=
  dialog_open=true
  input_missing=true
  page_changed=false

uncertainty=
  e3.value=high
  e4.meaning=medium

evidence_slots=
  slot_target=e2
  slot_blocker=e4
  slot_submit=e1
```

**Layer C — Hop trace (skill extraction)**

A **second** text object (for extraction / supervision, not necessarily the acting agent’s full context) records `trigger`, ordered **hops** over grounded entity ids, `intermediate` subgoal, `output`, and `grounding=[…]`. **Segment** these traces into candidate skills; do **not** rely on raw action chunks alone.

Example:

```text
trigger=need to submit a filtered search query
hop1=locate relevant input fields [e2,e5]
hop2=check required constraints from task goal
hop3=detect blocker or missing prerequisite [e4]
hop4=select feasible next action path
intermediate=subgoal=fill missing origin field first
output=click/input action
grounding=[e2,e4,e5]
```

**BrowserGym stack (brief):** (1) compact **AX/DOM** summary of interactives, labels, validation, groups; (2) **screenshot parser** only where text is weak; (3) **temporal diff** across steps (appear/disappear, region change, fields filled, blocker cleared) — web skills are often **state-transition** patterns.

**Gym-V stack (brief):** enable wrapper scaffolding → train the parser toward the **canonical schema** → extract skills in **easier deterministic** visual domains first → reuse the same summary + extraction machinery for BrowserGym (single API for observation design, offline/online).

**Tiny examples**

- **Web:** AX lists inputs “From” / “To” and “Search”; vision adds a **unlabeled icon** or **chart**; after a step, diff shows **validation error** on “From” → update `state_flags` / `uncertainty`.
- **Gym-V:** Frame yields entities/relations (“key at …”, “door north of agent”); wrapper adds partial **rule text**; summary merges both with per-field **source** and **uncertainty**.

---

## Transferable skill protocol

**Do not transfer raw action sequences.** Transfer **reasoning-form** templates: an **abstract** protocol plus **domain instantiation** (game / web / visual QA).

Store each skill with (among others): `skill_name`, `skill_family`, `abstraction_level`, `trigger`, `required_slots`, `observables`, `preconditions`, `hop_chain` (hops as **slot operations**, not fixed UI strings), `intermediate_conclusion`, `output_type`, `output_contract`, success/abort/failure fields, `retrieval_keys`, `domain_tags`, `transfer_notes`, `confidence`, `reusability`.

**Fields that matter most:** `trigger`, `required_slots` (e.g. `target_entity`, `constraint_field`, `candidate_set`, `history_evidence`), `hop_chain`, `intermediate_conclusion` (no direct obs→action jump), `output_contract` (e.g. answer selected, candidate filtered, blocker removed, subgoal updated, navigation advanced), `transfer_notes`.

**Example (cross-domain, compositional)** — `BLOCKER_THEN_REPLAN`: trigger = objective blocked; hops = locate blocked target → identify blocker/prerequisite → minimal prerequisite subgoal → action that satisfies it; conclusion = resolve blocker before original goal; success = blocker cleared and goal executable; abort = blocker ungrounded or no prerequisite; `transfer_notes`: game (deadlock/setup), web (validation/modal/missing field), visual QA (gather missing anchor evidence). **Instantiations** like “click Submit” or “move left” are **not** the stored abstraction.

---

## Transferable skill families (high value)

| Family | Game | Web | Visual reasoning |
|--------|------|-----|------------------|
| **Locate → filter → select** | Candidate moves → constraints → best legal move | UI candidates → relevance/prereqs → control | Objects → attributes/relations → answer target |
| **Blocker → prerequisite → replan** | Deadlock/obstacle → missing setup → new subgoal | Disabled control/error/modal → missing field/step → prerequisite first | Weak evidence → missing anchor/relation → gather evidence |
| **History → hidden state → act** | Votes/dialogue → alliance/threat | Prior pages/forms/results → next step | Prior frames → disambiguate current frame |
| **Compare under future constraint** | Move preserving structure | Path lowering risk/steps | Candidate consistent with all constraints |

---

## Hooking into Game-AI-Agent (brief)

1. **Adapter** on Gym-V / BrowserGym: emit `summary_state` (k=v), `reasoning_goal`, optional `grounding_trace` — **unchanged** decision-agent interface.  
2. **Offline canonicalizer:** trajectory segments → `trigger`, `hop_chain`, `intermediate_conclusion`, `output_contract` → **skill bank**.  
3. **Acting vs extraction:** actor usually sees **short** summary; supervisor / skill module sees **richer hop trace** (cheaper inference).  
4. **Transfer:** **retrieve** abstract templates, **instantiate** slots per domain, **measure** chain fidelity and task score — not “same weights everywhere.”

---

## Gym-V vs BrowserGym

- **Gym-V:** Controlled lab — summary schema design, skill extraction from pixels, ablations on wrappers and chain depth.  
- **BrowserGym:** Deployment target — web transfer, **noisy hybrid** grounding, test whether game-learned **abstract** skills survive realistic UI.

**Shortest grounding recipe:** structure-first web + wrapper-assisted vision games → one canonical k=v summary with **sources** and **uncertainty** → skills from **hop traces**, not raw screenshots.

---

## Skill synthesis / supervision agent

We introduce a **Skill Synthesis / Supervision Agent** to grow the skill bank with **synthetic interaction skills** found from rollout failures, weak existing skills, and missing multi-step compositions. Its job is not to invent arbitrary skills, but to propose **reusable, verifiable, high-utility** routines that improve downstream decision-making in both **game** and **web** environments.

### Motivation

Long-horizon agents often fail not from lack of primitive actions, but from lack of reusable higher-level routines for:

- recurring subgoals,
- unseen multi-step compositions,
- recovery from local failures,
- verification of task completion.

This agent closes those gaps by spotting brittle or missing skills, synthesizing candidate protocols, and **promoting only what helps in practice**.

### Role in the pipeline

The agent sits **between Stage 3 (Contract Learning) and Stage 4 (Bank Maintenance)** in the staged pipeline:

1. Boundary proposal  
2. Segmentation / labeling  
3. Contract learning  
4. **Skill synthesis / supervision** ← insertion point  
5. Bank curation, promotion, merge / split / rollback  

### Core responsibilities

- **Mine skill gaps** from rollouts: repeated failures, excessive replanning, low skill-following reward, missing bank coverage.  
- **Retrieve evidence** from existing skills, successful segments, and failure traces.  
- **Generate candidates** as structured protocols with explicit contracts.  
- **Score quality** with learned supervision from the decision agent and the skill-bank agent.  
- **Promote selectively** via proto-skill **canary** evaluation before anything enters the persistent bank.  

### Initial scope

**First version — two synthesis operators:**

- **Specialize:** narrow a broad skill to a high-value context.  
- **Repair:** fix a brittle or often-retrieved but underperforming skill.  

**Later extensions (examples):**

- **Compose:** merge frequently co-occurring skills into a macro-skill.  
- **Recover:** build recovery routines from repeated failure patterns.  

### Candidate skill format

Each synthetic skill uses the **same schema** as the rest of the bank:

- name / description  
- protocol steps  
- preconditions  
- success / abort criteria  
- outcome contract  
- parent skill references  
- supporting evidence from rollout segments  

### Supervision signals

Instead of hand-tuned heuristics alone, two **learned critics** guide synthesis.

**Decision-side utility**

- How likely is the skill to be **retrieved and selected**?  
- If selected, does it **improve execution**, cut stalling, or raise success?  

**Bank-side acceptance**

- How likely is it to pass **verification and curation**?  
- Will it **survive promotion** and stay useful over time?  

Together these estimate whether a candidate is **used**, **useful when used**, and **worth keeping**.

### Promotion strategy

Synthetic skills are **never** written straight into the persistent bank. Each candidate goes through:

1. Candidate generation  
2. Schema / contract verification  
3. Proto-skill insertion  
4. Canary exposure on a **small** rollout set  
5. Promotion, merge, or rejection  

This limits **bank bloat** from plausible but low-utility proposals.

### Near-term evaluation

Measure whether synthesis improves:

- task success,  
- rollout efficiency,  
- recovery from local failures,  
- verification reliability,  
- skill reuse,  
- bank coverage.  

### Long-term goal

A **self-improving skill bank** where decision-making, skill discovery, and bank supervision **co-evolve**—supporting more robust long-horizon behavior in **interactive games** and **web-based** tasks.

---

## Training

We train visual skills on **lightweight, context-rich video-game environments**, then transfer to **reasoning-heavy visual reasoning benchmarks** and **web-agent arenas**.
