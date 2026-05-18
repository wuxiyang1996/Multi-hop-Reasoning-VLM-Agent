# Per-Game LoRA Adapter Mapping

## Versioning Strategy

- **skill_selection**: uses **v3** SFT data (`sft_per_game_v3`) — trained on new/improved data
- **action_taking**: uses **v1** SFT data (`sft_per_game`) — original training data

## IMPORTANT
Both `skill_selection` and `action_taking` LoRAs are **per-game**.
Before running co-evolution for a game, you MUST update the symlinks in
`decision/skill_selection/` and `decision/action_taking/` to point to
the correct game's adapter.

All adapters: base model `Qwen/Qwen3.5-9B`, LoRA r=16, alpha=32, dropout=0.05, ~166M each.

## Game → LoRA Path Mapping

| Game Slug | SFT Key | skill_selection (v3 — `sft_per_game_v3`) | action_taking (v1 — `sft_per_game`) |
|---|---|---|---|
| `gymv_thunder_force_iii` | `Temporal_ThunderForceIII-v0` | `runs/sft_per_game_v3/Temporal_ThunderForceIII-v0/skill_selection/Temporal_ThunderForceIII-v0__skill_selection/` | `runs/sft_per_game/Temporal_ThunderForceIII-v0/action_taking/Temporal_ThunderForceIII-v0__action_taking/` |
| `candy_crush` | `candy_crush` | `runs/sft_per_game_v3/candy_crush/skill_selection/candy_crush__skill_selection/` | `runs/sft_per_game/candy_crush/action_taking/candy_crush__action_taking/` |
| `tetris` | `tetris` | `runs/sft_per_game_v3/tetris/skill_selection/tetris__skill_selection/` | `runs/sft_per_game/tetris/action_taking/tetris__action_taking/` |
| `super_mario` | `super_mario` | `runs/sft_per_game_v3/super_mario/skill_selection/super_mario__skill_selection/` | `runs/sft_per_game/super_mario/action_taking/super_mario__action_taking/` |
| `twenty_forty_eight` | `twenty_forty_eight` | `runs/sft_per_game_v3/twenty_forty_eight/skill_selection/twenty_forty_eight__skill_selection/` | `runs/sft_per_game/twenty_forty_eight/action_taking/twenty_forty_eight__action_taking/` |
| `gymv_altered_beast` | `Temporal_AlteredBeast-v0` | `runs/sft_per_game_v3/Temporal_AlteredBeast-v0/skill_selection/Temporal_AlteredBeast-v0__skill_selection/` | `runs/sft_per_game/Temporal_AlteredBeast-v0/action_taking/Temporal_AlteredBeast-v0__action_taking/` |
| `gymv_columns` | `Temporal_Columns-v0` | `runs/sft_per_game_v3/Temporal_Columns-v0/skill_selection/Temporal_Columns-v0__skill_selection/` | `runs/sft_per_game/Temporal_Columns-v0/action_taking/Temporal_Columns-v0__action_taking/` |
| `gymv_dynamite_headdy` | `Temporal_DynamiteHeaddy-v0` | `runs/sft_per_game_v3/Temporal_DynamiteHeaddy-v0/skill_selection/Temporal_DynamiteHeaddy-v0__skill_selection/` | `runs/sft_per_game/Temporal_DynamiteHeaddy-v0/action_taking/Temporal_DynamiteHeaddy-v0__action_taking/` |
| `gymv_airstriker` | `Temporal_Airstriker-v0` | `runs/sft_per_game_v3/Temporal_Airstriker-v0/skill_selection/Temporal_Airstriker-v0__skill_selection/` | `runs/sft_per_game/Temporal_Airstriker-v0/action_taking/Temporal_Airstriker-v0__action_taking/` |

### Games WITHOUT skill_selection v3 LoRA (v1 action_taking only)

These 3 games were not included in v3 SFT training. Use v1 skill_selection as fallback, or train v3.

| Game Slug | SFT Key | skill_selection (v1 — `sft_per_game`) | action_taking (v1 — `sft_per_game`) |
|---|---|---|---|
| `gymv_space_harrier_ii` | `Temporal_SpaceHarrierII-v0` | `runs/sft_per_game/Temporal_SpaceHarrierII-v0/skill_selection/Temporal_SpaceHarrierII-v0__skill_selection/` | `runs/sft_per_game/Temporal_SpaceHarrierII-v0/action_taking/Temporal_SpaceHarrierII-v0__action_taking/` |
| `gymv_streets_of_rage_2` | `Temporal_StreetsOfRage2-v0` | `runs/sft_per_game/Temporal_StreetsOfRage2-v0/skill_selection/Temporal_StreetsOfRage2-v0__skill_selection/` | `runs/sft_per_game/Temporal_StreetsOfRage2-v0/action_taking/Temporal_StreetsOfRage2-v0__action_taking/` |
| `gymv_strider` | `Temporal_Strider-v0` | `runs/sft_per_game/Temporal_Strider-v0/skill_selection/Temporal_Strider-v0__skill_selection/` | `runs/sft_per_game/Temporal_Strider-v0/action_taking/Temporal_Strider-v0__action_taking/` |

## Shared Adapters (game-independent, from co-evolution)

| Adapter | Path |
|---|---|
| `segment` | `runs/lora_adapters/decision/segment/` |
| `contract` | `runs/lora_adapters/decision/contract/` |
| `curator` | `runs/lora_adapters/decision/curator/` |

## How to Switch

```bash
# Example: switch to candy_crush
cd runs/lora_adapters/decision/skill_selection/
rm -f adapter_config.json adapter_model.safetensors
ln -s $(pwd)/../../sft_per_game_v3/candy_crush/skill_selection/candy_crush__skill_selection/adapter_config.json .
ln -s $(pwd)/../../sft_per_game_v3/candy_crush/skill_selection/candy_crush__skill_selection/adapter_model.safetensors .

cd ../action_taking/
rm -f adapter_config.json adapter_model.safetensors
ln -s $(pwd)/../../sft_per_game/candy_crush/action_taking/candy_crush__action_taking/adapter_config.json .
ln -s $(pwd)/../../sft_per_game/candy_crush/action_taking/candy_crush__action_taking/adapter_model.safetensors .
```
