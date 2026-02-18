# World model for experience synthesis

The world model generates synthetic experience sequences from state, historical context, and instructions. It supports **multi-modal** (image editing) and **agentic textual** (LLM) backends, plus **experience planning** to convert natural language descriptions into executable plans.

---

## 1. Multi-modal world model (image editing)

Treats experience synthesis as image editing: given the current frame (image), historical state summary (text), and instructions (action/intended state/skills), outputs edited images as synthetic next states.

**Implementation**: [world_model/multi_modal/](multi_modal/)

### Supported models
- **LongCat** — https://huggingface.co/meituan-longcat/LongCat-Image-Edit
- **Qwen-Edit** — https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- **BAGEL** — https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT (see [BAGEL_DEPLOYMENT.md](multi_modal/BAGEL_DEPLOYMENT.md))

### Model selection
- `WorldModelConfig(model_name="longcat"|"qwen_edit"|"bagel")`
- Env var: `WORLD_MODEL_IMAGE_EDIT_MODEL`
- Custom: `WorldModelConfig(model_id="org/custom-model")`

### Usage
```python
from world_model.multi_modal import WorldModel, WorldModelConfig, SynthesisInput

config = WorldModelConfig(model_name="longcat")
model = WorldModel(config)

# Single step
out = model.synthesize_step(SynthesisInput(
    current_frame=frame,           # PIL.Image or path
    historical_summary="Agent at (2,1). Pot has 2 onions.",
    instructions="Agent picks up onion from dispenser.",
))
next_frame = out.next_frame

# Multi-step sequence
seq = model.synthesize_sequence(
    initial_frame=frame,
    instructions_per_step=["Agent moves to dispenser.", "Agent picks up onion.", "Agent carries to pot."],
    historical_summaries=["..."] * 3,
)
for step in seq.steps:
    frame = step.next_frame
```

### Dependencies
See [multi_modal/requirements.txt](multi_modal/requirements.txt): `torch`, `diffusers`, `Pillow`, `transformers`, `accelerate`

---

## 2. Agentic textual world model

Uses an LLM (via `ask_model`) to synthesize experience sequences from state, historical summaries, and action/goal state/skills. Produces textual next states (no images).

**Implementation**: [world_model/agentic_textual/](agentic_textual/)

### Model selection
- `TextWorldModelConfig(model="gpt-4o-mini")` or any GPT/Claude/Gemini model
- Env var: `WORLD_MODEL_TEXTUAL_MODEL`
- Optional `ask_model_fn` override for testing

### Usage
```python
from world_model.agentic_textual import TextWorldModel, TextWorldModelConfig, TextSynthesisInput

config = TextWorldModelConfig(model="gpt-4o-mini")
model = TextWorldModel(config)

# Single step
out = model.synthesize_step(TextSynthesisInput(
    state="Agent at (2,1). Pot has 2 onions.",
    historical_summary="Agent moved from dispenser to pot.",
    instructions="Agent picks up onion from dispenser.",
))
next_state = out.next_state  # text; also out.action, out.reward

# Multi-step sequence
seq = model.synthesize_sequence(
    initial_state="Agent at (3,0). Pot empty.",
    instructions_per_step=["Pick up onion.", "Carry to pot.", "Place in pot."],
)
# Convert to Experience-like dicts
exps = model.to_experiences(seq, initial_state="...")
```

---

## 3. Experience planning

Converts natural language descriptions of desired episodes/skills into plans (natural language or action language) that drive world model synthesis. Optionally grounds plans using skill_agents (skill bank, action language formats).

**Implementation**: [world_model/experience_planning/](experience_planning/)

### Features
- **Episode planning**: NL description → step-by-step plan
- **Skill planning**: NL skill description → executable steps
- **Skill bank grounding**: Use contract_verification or stage3_mvp skill bank for action language (PDDL, STRIPS, SAS, compact)
- **Integration**: Plans can be fed to multi_modal or agentic_textual world models

### Model selection
- `PlanningConfig(model="gpt-4o-mini")`
- Env var: `WORLD_MODEL_PLANNING_MODEL`

### Usage
```python
from world_model.experience_planning import ExperiencePlanner, EpisodeDescription, SkillDescription

planner = ExperiencePlanner()

# Episode plan
desc = EpisodeDescription(
    description="Agent picks up onion, carries to pot, and places it in.",
    max_steps=5,
    env_hint="overcooked",
)
plan = planner.plan_episode(desc, skill_bank=None)
instructions = plan.to_instructions()  # list of step instructions

# Skill plan
skill_desc = SkillDescription(
    description="Place ingredient into cooking pot.",
    skill_id="place_in_pot",
    goal="Pot contains the ingredient",
)
plan = planner.plan_skill(skill_desc, skill_bank=bank)

# Convert plan to world model inputs
inputs = planner.to_world_model_inputs(
    plan,
    initial_state="Agent at (2,1). Pot empty.",
    initial_historical="",
)
# Feed inputs to TextWorldModel or multi_modal WorldModel
```

---

## End-to-end flow

```
NL description → ExperiencePlanner → SynthesisPlan (instructions)
       ↓                                    ↓
  (optional) skill bank              TextWorldModel / multi_modal WorldModel
       ↓                                    ↓
  action language grounding          synthetic experience sequence
```
 