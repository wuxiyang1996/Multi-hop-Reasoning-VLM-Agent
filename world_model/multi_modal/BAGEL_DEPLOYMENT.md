# BAGEL Deployment for World Model Experience Synthesis

This document describes how to deploy **BAGEL-7B-MoT** (from `bagel/`) as the multi-modal world model for experience synthesis. BAGEL treats experience synthesis as an image-editing task: given the current frame, historical state summary, and action/intended-state/skill instructions, it produces edited images as synthetic next states.

## Prerequisites

- **NVIDIA GPU** with ≥32GB VRAM (e.g. RTX 5090, A100)
- **BAGEL repo** at `ICML2026/bagel/` (sibling to `Game-AI-Agent/`)
- Model weights at `bagel/models/BAGEL-7B-MoT/` (see `bagel/test_bagel.py` for auto-download)

## Prompt Format for Experience Synthesis

The world model takes **one text prompt** that combines context and edit instruction. Use this structure:

```
[Optional: Historical context] <edit instruction>
```

### Template

```
Context: {historical_summary}

Edit instruction: {instructions}
```

- **historical_summary**: Text summary of prior states (e.g. agent positions, inventory, game state). Provides temporal context.
- **instructions**: The action to take, intended next state, or skill to apply. This is the main edit directive.

### Example Prompts (from bagel usage)

**Image editing (test_bagel.py, test_image_editing):**
- `"Transform this image into a watercolor painting style."`
- `"Make this look like it was taken during a beautiful sunset."`

**Experience synthesis (game world model):**
- `"Context: Agent at (2,1) facing east. Pot has 2 onions. No orders pending. Edit instruction: Agent picks up an onion from the dispenser."`
- `"Context: Agent near pot holding onion. Edit instruction: Agent places the onion in the pot."`

## Inference API (from bagel/)

BAGEL uses `InterleaveInferencer` in `bagel/inferencer.py`:

```python
# Image editing
output_dict = inferencer(image=image, text=prompt, **inference_hyper)
edited_image = output_dict["image"]
```

### Recommended Inference Hyperparameters (for editing)

From `bagel/test_bagel.py` and `bagel/inference.ipynb`:

| Parameter          | Value        | Notes                                      |
|--------------------|--------------|--------------------------------------------|
| `cfg_text_scale`   | 4.0          | How strongly to follow the text prompt     |
| `cfg_img_scale`    | 2.0          | Preserve input image details               |
| `cfg_interval`     | [0.0, 1.0]   | Apply CFG over full denoising              |
| `timestep_shift`   | 3.0          | Denoising step distribution                |
| `num_timesteps`    | 50           | Total denoising steps                      |
| `cfg_renorm_min`   | 0.0          | CFG renorm lower bound                     |
| `cfg_renorm_type`  | "text_channel" | For editing (vs "global" for T2I)        |

### Optional: Think / CoT Mode

For harder edits, enable thinking:

```python
output_dict = inferencer(
    image=image, text=prompt,
    think=True,
    max_think_token_n=1000,
    do_sample=False,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)
```

## Deployment Steps

1. **Activate BAGEL environment**
   ```bash
   conda activate bagel
   cd bagel
   ```

2. **Ensure model weights exist**
   ```bash
   python test_bagel.py --model_path models/BAGEL-7B-MoT
   ```
   (Skips download if `ema.safetensors` exists.)

3. **Load model and inferencer**
   ```python
   from test_bagel import load_model
   from inferencer import InterleaveInferencer

   model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
       "models/BAGEL-7B-MoT", max_mem="30GiB"
   )
   inferencer = InterleaveInferencer(
       model=model, vae_model=vae_model, tokenizer=tokenizer,
       vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids
   )
   ```

4. **Build and run synthesis prompt**
   ```python
   from PIL import Image

   current_frame = Image.open("path/to/frame.png")
   historical_summary = "Agent at (2,1). Pot has 2 onions."
   instructions = "Agent picks up onion from dispenser."

   prompt = f"Context: {historical_summary}\nEdit instruction: {instructions}"
   inference_hyper = dict(
       cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.0, 1.0],
       timestep_shift=3.0, num_timesteps=50,
       cfg_renorm_min=0.0, cfg_renorm_type="text_channel",
   )
   output_dict = inferencer(image=current_frame, text=prompt, **inference_hyper)
   next_frame = output_dict["image"]
   ```

## Integration with multi_modal WorldModel

To wire BAGEL into `world_model.multi_modal.WorldModel`, the backend must:

1. Load BAGEL via `bagel/` (model, VAE, tokenizer, transforms)
2. Build the edit prompt from `SynthesisInput(historical_summary, instructions)`
3. Call `inferencer(image=current_frame, text=prompt, **inference_hyper)`
4. Return `SynthesisStepOutput(next_frame=output_dict["image"], edit_prompt=prompt)`

Use `config.model_id` to point at the local BAGEL model path (e.g. `bagel/models/BAGEL-7B-MoT`) when implementing the BAGEL backend.
