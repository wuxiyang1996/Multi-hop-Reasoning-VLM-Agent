# Patches

Tracking fixes and patches applied to the codebase for future reference.

---

## 001 — Candy Crush: inject dynamic action mapping into GymEnvAdapter (2026-03-11)

**File:** `GamingAgent/gamingagent/envs/custom_03_candy_crush/candyCrushEnv.py` (line ~266)

**Problem:**
`game_env_config.json` ships with `"action_mapping": {}` because Candy Crush
has 112 valid swap actions on an 8x8 board — too many to enumerate manually
in the config (unlike 2048/Sokoban which have only 4 directional actions).

Because the adapter's `move_to_action_idx` was always empty, every call to
`_parse_agent_action_str` triggered:

1. `adapter.map_agent_action_to_env_action()` → miss → **warning printed**
2. Fallback regex parse of `((r1,c1),(r2,c2))` → lookup in
   `env_move_to_action_idx` → success

Actions executed correctly, but the per-step warning was noisy and the
unnecessary fallback added overhead.

**Fix:**
After `CandyCrushEnv.__init__` dynamically builds `env_move_to_action_idx`
from the board geometry, inject that mapping into the adapter when the
adapter's own mapping (from config) is empty:

```python
if not self.adapter.move_to_action_idx and self.env_move_to_action_idx:
    self.adapter.move_to_action_idx = {k.lower(): v for k, v in self.env_move_to_action_idx.items()}
    self.adapter.action_idx_to_move = {v: k for k, v in self.env_move_to_action_idx.items()}
```

Now the adapter's first lookup succeeds directly — no warning, no regex
fallback needed.

---

## 002 — Orak: disable screenshot storage for Pokemon Red & Super Mario (2026-03-15)

**Files changed (Orak repo):**
- `Orak/src/mcp_game_servers/pokemon_red/game/pyboy_runner.py` — `take_screenshot()`
- `Orak/src/mcp_game_servers/pokemon_red/game/pokemon_red_env.py` — `PokemonRedEnv`
- `Orak/src/mcp_game_servers/super_mario/game/super_mario_env.py` — `SuperMarioEnv`

**Files changed (Game-AI-Agent):**
- `evaluate_orak/orak_nl_wrapper.py` — `make_orak_env()`
- `evaluate_orak/orak_gym_like.py` — `make_orak_gaming_env()`

**Problem:**
Pokemon Red called `PyBoyRunner.take_screenshot()` on every `initial_obs()`
and `step()`, writing a timestamped PNG to `<log_path>/screenshots/`. Over
long evaluation runs this accumulated gigabytes of unused images. Super Mario
had a similar `save_state_image()` (commented out) and was still converting
every frame to a PIL image unnecessarily.

**Fix:**
1. `pyboy_runner.py`: `take_screenshot()` now accepts `save=True`. When
   `save=False` it returns the PIL image from `pyboy.screen.image` without
   writing to disk.
2. `pokemon_red_env.py`: Added `save_screenshots: bool = True` to `Config`.
   Both `initial_obs()` and `step()` pass `save=self.save_screenshots` to
   `take_screenshot()`.
3. `super_mario_env.py`: Added `save_screenshots: bool = True` to `Config`.
   When False, `initial_obs()` and `step()` skip the `to_pil_image()`
   conversion (sets `image=None`). Also added a `save` guard to
   `save_state_image()`.
4. `orak_nl_wrapper.py` and `orak_gym_like.py`: Both wrappers set
   `cfg.env.save_screenshots = False` for `pokemon_red` / `super_mario`
   (resp. `orak_pokemon_red` / `orak_super_mario`) via `omegaconf.open_dict`.

All defaults remain `True` so other games and direct Orak usage are
unaffected.
